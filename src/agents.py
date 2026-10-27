"""Agents for MIND.

Every agent implements the same interface so training and evaluation code
never needs to know which algorithm it is running:

    action = agent.act(obs, greedy=False)      # greedy=True disables exploration
    stats  = agent.update(obs, action, reward, next_obs, terminated, truncated,
                          info=info)            # info (optional): env.step's info dict
    agent.end_episode()                         # advance the exploration schedule
    state  = agent.state_dict(); agent.load_state_dict(state)

`terminated` means a true terminal state (goal, terminal hazard): its future
value is zero. `truncated` means the step limit cut the episode short: the
state still has future value, so we keep bootstrapping from it.

`info` is the environment's info dict for the step. Only mistake-aware replay
reads it (for info["events"]); every agent works without it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.replay import MistakeReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer


@dataclass
class EpsilonSchedule:
    """Linear decay of the exploration rate over episodes.

    epsilon(k) = end + (start - end) * max(0, 1 - k / decay_episodes)
    """

    start: float = 1.0
    end: float = 0.05
    decay_episodes: int = 300

    def value(self, episode: int) -> float:
        if self.decay_episodes <= 0:
            return self.end
        frac = max(0.0, 1.0 - episode / self.decay_episodes)
        return self.end + (self.start - self.end) * frac


def greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with random tie-breaking, so untrained (all-zero) states are not
    biased toward action 0."""
    best = np.flatnonzero(q_values == q_values.max())
    return int(best[rng.integers(len(best))])


MISTAKE_REPLAY_EVENTS = ("collision", "hazard")


def is_mistake(info: Optional[dict], events: Tuple[str, ...] = MISTAKE_REPLAY_EVENTS) -> bool:
    """True if the environment reported one of `events` on this step.

    Read from info["events"], never from the reward: a large negative reward can
    come from the step cost plus a timeout or a revisit, and rewards are
    configurable. Without info nothing counts as a mistake.
    """
    if not info:
        return False
    return any(e in events for e in info.get("events", ()))


class RandomAgent:
    """Baseline: uniform random actions, never learns."""

    name = "random"
    nickname = "Chaos Monkey"

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.episodes = 0

    @property
    def epsilon(self) -> float:
        return 1.0

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, obs, action, reward, next_obs, terminated, truncated,
               info: Optional[dict] = None) -> Optional[dict]:
        return None

    def end_episode(self) -> None:
        self.episodes += 1

    def state_dict(self) -> dict:
        return {"name": self.name, "n_actions": self.n_actions, "episodes": self.episodes}

    def load_state_dict(self, state: dict) -> None:
        self.episodes = int(state["episodes"])


class TabularQAgent:
    """Tabular Q-learning (Watkins, 1989).

    Update after each transition (s, a, r, s'):

        target = r                                  if s' is terminal
        target = r + gamma * max_a' Q(s', a')       otherwise
        Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

    States are the raw bytes of the observation vector, so the agent sees
    exactly what DQN sees. Unvisited states start at Q = 0.
    """

    name = "tabular_q"
    nickname = "The Spreadsheet"

    def __init__(self, n_actions: int, alpha: float = 0.5, gamma: float = 0.99,
                 epsilon: Optional[EpsilonSchedule] = None, seed: Optional[int] = None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.schedule = epsilon if epsilon is not None else EpsilonSchedule()
        self.rng = np.random.default_rng(seed)
        self.q: Dict[bytes, np.ndarray] = {}
        self.episodes = 0
        self.updates = 0

    @property
    def epsilon(self) -> float:
        return self.schedule.value(self.episodes)

    @staticmethod
    def key(obs: np.ndarray) -> bytes:
        return np.asarray(obs, dtype=np.float32).tobytes()

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q(s, .) for an observation; zeros if the state was never seen."""
        values = self.q.get(self.key(obs))
        return values.copy() if values is not None else np.zeros(self.n_actions)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return greedy_action(self.q_values(obs), self.rng)

    def update(self, obs, action, reward, next_obs, terminated, truncated,
               info: Optional[dict] = None) -> dict:
        k = self.key(obs)
        row = self.q.setdefault(k, np.zeros(self.n_actions))
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * float(self.q_values(next_obs).max())
        td_error = target - row[action]
        row[action] += self.alpha * td_error
        self.updates += 1
        return {"td_error": float(td_error)}

    def end_episode(self) -> None:
        self.episodes += 1

    @property
    def n_states(self) -> int:
        return len(self.q)

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": asdict(self.schedule),
            "episodes": self.episodes,
            "updates": self.updates,
            "q": {k: v.copy() for k, v in self.q.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if state["n_actions"] != self.n_actions:
            raise ValueError("checkpoint has a different number of actions")
        self.alpha = float(state["alpha"])
        self.gamma = float(state["gamma"])
        self.schedule = EpsilonSchedule(**state["epsilon"])
        self.episodes = int(state["episodes"])
        self.updates = int(state["updates"])
        self.q = {k: np.array(v, dtype=np.float64) for k, v in state["q"].items()}


@dataclass
class DQNConfig:
    """Hyperparameters for DQNAgent (defaults tuned for 5x5..11x11 grids on CPU)."""

    hidden: Tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 20_000
    learning_starts: int = 500      # env steps collected before the first update
    train_freq: int = 2             # gradient step every N env steps
    target_update: int = 100        # copy online -> target every N gradient steps
    double: bool = True             # Double DQN target (van Hasselt et al., 2016)
    prioritized: bool = False       # prioritized experience replay
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 20_000    # gradient steps to anneal beta to 1
    grad_clip: float = 10.0
    epsilon: EpsilonSchedule = field(default_factory=EpsilonSchedule)
    mistake_replay: bool = False    # keep collisions/hazards in a reserve and replay them
    mistake_fraction: float = 0.25  # share of each batch drawn from the reserve
    mistake_capacity: int = 5_000   # reserve size (circular)
    mistake_events: Tuple[str, ...] = MISTAKE_REPLAY_EVENTS
    dueling: bool = False           # value + advantage heads (Wang et al., 2016)
    n_step: int = 1                 # learn from n rewards before bootstrapping


class QNetwork(nn.Module):
    """MLP mapping an observation vector to one Q-value per action.

    dueling=True (Wang et al., 2016) ends the shared layers in two heads, a
    state value V(s) and per-action advantages A(s, a), combined as

        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

    (subtracting the mean makes V and A identifiable). The plain network is
    built exactly as before, so its initial weights don't change.
    """

    def __init__(self, obs_size: int, n_actions: int, hidden: Tuple[int, ...] = (128, 128),
                 dueling: bool = False):
        super().__init__()
        layers, size = [], obs_size
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        self.dueling = dueling
        if dueling:
            self.net = nn.Sequential(*layers)
            self.value = nn.Linear(size, 1)
            self.advantage = nn.Linear(size, n_actions)
        else:
            layers.append(nn.Linear(size, n_actions))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(x)
        h = self.net(x)
        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=1, keepdim=True)


DQN_NICKNAMES = {"dqn": "Neural Ned", "dqn_per": "Neural Ned (Holds Grudges)",
                 "dqn_mistake": "Neural Ned (Keeps a Diary)",
                 "dqn_per_mistake": "Neural Ned (Holds Grudges, Keeps a Diary)",
                 "dqn_dueling": "Neural Ned (Two-Headed)",
                 "dqn_nstep": "Neural Ned (Looks Ahead)",
                 "dqn_dueling_nstep": "Neural Ned (Two-Headed, Looks Ahead)"}
N_STEP_DEFAULT = 3  # n for the named n-step agents (as in Rainbow, Hessel et al., 2018)


def dqn_name(config: DQNConfig) -> str:
    """'dqn' plus '_per', '_dueling', '_nstep' and '_mistake' for each option on,
    e.g. 'dqn_per_mistake' or 'dqn_dueling_nstep'."""
    return ("dqn_per" if config.prioritized else "dqn") + \
        ("_dueling" if config.dueling else "") + \
        ("_nstep" if config.n_step > 1 else "") + \
        ("_mistake" if config.mistake_replay else "")


class DQNAgent:
    """Deep Q-Network (Mnih et al., 2015) with optional Double DQN and PER.

    A neural network Q(s, a; theta) replaces the table. Each gradient step
    samples a batch from the replay buffer and minimises

        L(theta) = mean_i  w_i * Huber( y_i - Q(s_i, a_i; theta) )

        y_i = r_i                                         if s'_i is terminal
        y_i = r_i + gamma * Q(s'_i, a*; theta_target)     otherwise
        a*  = argmax_a Q(s'_i, a; theta)         (Double DQN)
            = argmax_a Q(s'_i, a; theta_target)  (standard DQN)

    theta_target is a periodically copied snapshot of theta, which keeps the
    regression target stable. w_i are importance weights (all 1 without PER).

    With mistake_replay, transitions whose step reported a collision or hazard
    are also kept in a reserve, and mistake_fraction of every batch is drawn
    from it (see MistakeReplayBuffer). Off by default; when off, the agent is
    exactly the original DQN.

    With n_step = n > 1, each stored transition spans up to n steps:

        y_i = R_i + gamma^k * Q(s_{t+k}, a*; theta_target)  (0 if s_{t+k} is terminal)
        R_i = r_t + gamma r_{t+1} + ... + gamma^(k-1) r_{t+k-1}

    with k = n, or fewer for the last steps of an episode. A timeout still
    bootstraps, with gamma^k. The rewards come from the behaviour policy
    (exploration included) and are not corrected for it, as in Rainbow.
    dueling switches the network to value + advantage heads. Both are off by
    default (n_step = 1, dueling = False) and then the agent is unchanged.
    """

    def __init__(self, obs_size: int, n_actions: int, config: Optional[DQNConfig] = None,
                 seed: Optional[int] = None):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.config = config if config is not None else DQNConfig()
        self.name = dqn_name(self.config)
        self.nickname = DQN_NICKNAMES.get(self.name, DQN_NICKNAMES["dqn"])
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        cfg = self.config
        if cfg.n_step < 1:
            raise ValueError("n_step must be at least 1")
        if cfg.n_step > 1 and cfg.mistake_replay:
            raise ValueError("n-step returns can't be combined with mistake replay: "
                             "it is undefined which of the n steps makes the mistake")
        self.online = QNetwork(obs_size, n_actions, cfg.hidden, cfg.dueling)
        self.target = QNetwork(obs_size, n_actions, cfg.hidden, cfg.dueling)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        if cfg.prioritized:
            self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, obs_size,
                                                  alpha=cfg.per_alpha, seed=seed)
        else:
            self.buffer = ReplayBuffer(cfg.buffer_size, obs_size, seed=seed)
        if cfg.mistake_replay:
            self.buffer = MistakeReplayBuffer(self.buffer, cfg.mistake_capacity,
                                              cfg.mistake_fraction, seed=seed)

        self.episodes = 0
        self.env_steps = 0
        self.updates = 0
        # n-step: the last (obs, action, reward) of the current episode not yet
        # stored. Empty between episodes, so checkpoints never need it.
        self._recent: deque = deque()

    @property
    def epsilon(self) -> float:
        return self.config.epsilon.value(self.episodes)

    @property
    def beta(self) -> float:
        cfg = self.config
        frac = min(1.0, self.updates / max(1, cfg.per_beta_steps))
        return cfg.per_beta_start + (1.0 - cfg.per_beta_start) * frac

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.online(x).squeeze(0).numpy().astype(np.float64)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return greedy_action(self.q_values(obs), self.rng)

    def update(self, obs, action, reward, next_obs, terminated, truncated,
               info: Optional[dict] = None) -> Optional[dict]:
        """Store the transition and, when due, take one gradient step."""
        if self.config.mistake_replay:
            self.buffer.add(obs, action, reward, next_obs, terminated,
                            mistake=is_mistake(info, self.config.mistake_events))
        elif self.config.n_step > 1:
            self._add_n_step(obs, action, reward, next_obs, terminated, truncated)
        else:
            self.buffer.add(obs, action, reward, next_obs, terminated)
        self.env_steps += 1
        cfg = self.config
        if len(self.buffer) < cfg.learning_starts or self.env_steps % cfg.train_freq != 0:
            return None
        return self._learn()

    def _add_n_step(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        """Store the n-step transition starting n steps ago once it is complete,
        and at the end of an episode every shorter one still waiting."""
        n, gamma = self.config.n_step, self.config.gamma
        self._recent.append((obs, action, reward))
        done = terminated or truncated
        while self._recent and (len(self._recent) == n or done):
            ret = sum(gamma ** k * r for k, (_, _, r) in enumerate(self._recent))
            first_obs, first_action, _ = self._recent[0]
            self.buffer.add(first_obs, first_action, ret, next_obs, terminated,
                            discount=gamma ** len(self._recent))
            self._recent.popleft()
            if not done:
                break

    def _learn(self) -> dict:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size, beta=self.beta)
        obs = torch.as_tensor(batch["obs"])
        actions = torch.as_tensor(batch["actions"])
        rewards = torch.as_tensor(batch["rewards"])
        next_obs = torch.as_tensor(batch["next_obs"])
        terminated = torch.as_tensor(batch["terminated"])
        weights = torch.as_tensor(batch["weights"])

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_target = self.target(next_obs)
            if cfg.double:
                best = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = next_target.gather(1, best).squeeze(1)
            else:
                next_q = next_target.max(dim=1).values
            if cfg.n_step > 1:
                discounts = torch.as_tensor(batch["discounts"])
                y = rewards + discounts * (1.0 - terminated) * next_q
            else:
                y = rewards + cfg.gamma * (1.0 - terminated) * next_q

        td = (y - q).detach().numpy()
        loss = (weights * F.smooth_l1_loss(q, y, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.buffer.update_priorities(batch["indices"], td)
        self.updates += 1
        if self.updates % cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.item()), "td_error": float(np.abs(td).mean())}

    def end_episode(self) -> None:
        self.episodes += 1

    def state_dict(self, include_buffer: bool = False) -> dict:
        state = {
            "name": self.name,
            "obs_size": self.obs_size,
            "n_actions": self.n_actions,
            "config": asdict(self.config),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes": self.episodes,
            "env_steps": self.env_steps,
            "updates": self.updates,
        }
        if include_buffer:
            state["buffer"] = self.buffer.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        if state["obs_size"] != self.obs_size or state["n_actions"] != self.n_actions:
            raise ValueError("checkpoint has a different observation or action size")
        cfg = dict(state["config"])
        cfg["hidden"] = tuple(cfg["hidden"])
        cfg["epsilon"] = EpsilonSchedule(**cfg["epsilon"])
        if "mistake_events" in cfg:
            cfg["mistake_events"] = tuple(cfg["mistake_events"])
        if (cfg["hidden"] != tuple(self.config.hidden)
                or cfg["prioritized"] != self.config.prioritized
                or cfg.get("mistake_replay", False) != self.config.mistake_replay
                or cfg.get("dueling", False) != self.config.dueling
                or cfg.get("n_step", 1) != self.config.n_step):
            raise ValueError("checkpoint network, replay type or n-step setting differs "
                             "from this agent")
        self.config = DQNConfig(**cfg)
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.episodes = int(state["episodes"])
        self.env_steps = int(state["env_steps"])
        self.updates = int(state["updates"])
        if "buffer" in state:
            self.buffer.load_state_dict(state["buffer"])


def make_agent(name: str, obs_size: int, n_actions: int, seed: Optional[int] = None, **kwargs):
    """Build an agent by name: 'random', 'tabular_q', or a DQN variant.

    DQN variants: 'dqn', 'dqn_per' (prioritized replay), 'dqn_mistake'
    (mistake-aware replay), 'dqn_per_mistake' (both), 'dqn_dueling' (dueling
    heads), 'dqn_nstep' (3-step returns) and 'dqn_dueling_nstep'. kwargs go to
    TabularQAgent, or become DQNConfig fields for the DQN variants.
    """
    if name == "random":
        return RandomAgent(n_actions, seed=seed)
    if name == "tabular_q":
        return TabularQAgent(n_actions, seed=seed, **kwargs)
    if name in DQN_NICKNAMES:
        kwargs["prioritized"] = name.startswith("dqn_per")
        kwargs["mistake_replay"] = name.endswith("_mistake")
        kwargs["dueling"] = "_dueling" in name
        if "_nstep" in name:
            kwargs.setdefault("n_step", N_STEP_DEFAULT)
        return DQNAgent(obs_size, n_actions, DQNConfig(**kwargs), seed=seed)
    raise ValueError(f"unknown agent {name!r}; choose random, tabular_q or one of "
                     f"{', '.join(DQN_NICKNAMES)}")
