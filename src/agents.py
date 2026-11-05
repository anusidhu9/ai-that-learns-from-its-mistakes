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

Agents with memory (recurrent DQN) also define `agent.begin_episode()`, which
clears the memory; run_episode calls it after every reset when it exists.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.replay import (MistakeReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer,
                        SequenceReplayBuffer)


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
    conv: Tuple[int, ...] = ()      # 3x3 conv channels before the hidden layers (empty: MLP)
    conv_input: Tuple[int, int, int] = (3, 5, 5)  # (channels, rows, cols) of the image part
    recurrent: int = 0              # GRU memory units after the hidden layers (0: no memory)
    seq_length: int = 16            # recurrent: consecutive steps trained on per sequence
    burn_in: int = 16               # recurrent: earlier steps that only rebuild the memory


class QNetwork(nn.Module):
    """MLP mapping an observation vector to one Q-value per action.

    dueling=True (Wang et al., 2016) ends the shared layers in two heads, a
    state value V(s) and per-action advantages A(s, a), combined as

        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

    (subtracting the mean makes V and A identifiable). The plain network is
    built exactly as before, so its initial weights don't change.

    conv=(c1, c2, ...) puts 3x3 convolutions (padding 1, so the size is kept)
    in front of the hidden layers. The first conv_input = (C, H, W) values of
    the observation are read as a C x H x W image (the local view's wall,
    hazard and goal channels); the rest (the direction to the goal) skip the
    convolutions and join the flattened feature maps. The same filters are
    applied at every position, so a corridor or dead end is recognised
    wherever it appears in the window.
    """

    def __init__(self, obs_size: int, n_actions: int, hidden: Tuple[int, ...] = (128, 128),
                 dueling: bool = False, conv: Tuple[int, ...] = (),
                 conv_input: Tuple[int, int, int] = (3, 5, 5)):
        super().__init__()
        size = obs_size
        self.encoder: Optional[nn.Sequential] = None
        if conv:
            channels, rows, cols = conv_input
            self.image_size = channels * rows * cols
            if obs_size < self.image_size:
                raise ValueError(f"observation of size {obs_size} is smaller than the "
                                 f"conv_input image {tuple(conv_input)}")
            self.conv_input = tuple(conv_input)
            convs = []
            for width in conv:
                convs += [nn.Conv2d(channels, width, kernel_size=3, padding=1), nn.ReLU()]
                channels = width
            self.encoder = nn.Sequential(*convs)
            size = channels * rows * cols + obs_size - self.image_size
        layers = []
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
        if self.encoder is not None:
            image = x[:, :self.image_size].reshape(-1, *self.conv_input)
            x = torch.cat([self.encoder(image).flatten(1), x[:, self.image_size:]], dim=1)
        if not self.dueling:
            return self.net(x)
        h = self.net(x)
        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=1, keepdim=True)


class RecurrentQNetwork(nn.Module):
    """Q-network with memory (DRQN, Hausknecht & Stone, 2015).

    Each observation passes through the hidden layers, then a GRU whose state
    carries information from earlier steps of the episode, then a linear layer
    to one Q-value per action:

        h_t = GRU(layers(o_t), h_{t-1}),   h_{-1} = 0,   Q(o_<=t, .) = W h_t + b

    forward runs whole sequences shaped (batch, steps, obs) and returns the
    Q-values and the memory after every step; step continues from given
    memories by one observation each.
    """

    def __init__(self, obs_size: int, n_actions: int, hidden: Tuple[int, ...] = (128,),
                 memory: int = 128):
        super().__init__()
        layers, size = [], obs_size
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        self.net = nn.Sequential(*layers)
        self.gru = nn.GRU(size, memory, batch_first=True)
        self.head = nn.Linear(memory, n_actions)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        out, _ = self.gru(self.net(x), h)
        return self.head(out), out

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Q-values for observations x (..., obs) seen with memories h (..., memory)."""
        shape = x.shape[:-1]
        x = x.reshape(-1, 1, x.shape[-1])
        q, _ = self.forward(x, h.reshape(1, -1, h.shape[-1]).contiguous())
        return q.reshape(*shape, -1)


DQN_NICKNAMES = {"dqn": "Neural Ned", "dqn_per": "Neural Ned (Holds Grudges)",
                 "dqn_mistake": "Neural Ned (Keeps a Diary)",
                 "dqn_per_mistake": "Neural Ned (Holds Grudges, Keeps a Diary)",
                 "dqn_dueling": "Neural Ned (Two-Headed)",
                 "dqn_nstep": "Neural Ned (Looks Ahead)",
                 "dqn_dueling_nstep": "Neural Ned (Two-Headed, Looks Ahead)",
                 "dqn_cnn": "Neural Ned (Sees Patterns)",
                 "dqn_rnn": "Neural Ned (Remembers)"}
N_STEP_DEFAULT = 3  # n for the named n-step agents (as in Rainbow, Hessel et al., 2018)
# Network of the named CNN agent: two 3x3 convolutions (16 and 32 channels) over
# the 5x5 local view, then one hidden layer of 128.
CNN_DEFAULT = dict(conv=(16, 32), hidden=(128,))
# Recurrent agent: one hidden layer of 128, a GRU with 128 memory units, and
# batches of 32 sequences (16 trained steps each, after up to 16 burn-in steps).
RNN_DEFAULT = dict(recurrent=128, hidden=(128,), batch_size=32)


def dqn_name(config: DQNConfig) -> str:
    """'dqn' plus '_per', '_cnn', '_rnn', '_dueling', '_nstep' and '_mistake'
    for each option on, e.g. 'dqn_per_mistake' or 'dqn_dueling_nstep'."""
    return ("dqn_per" if config.prioritized else "dqn") + \
        ("_cnn" if config.conv else "") + \
        ("_rnn" if config.recurrent else "") + \
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
    conv adds convolutional layers in front of the network (see QNetwork); off
    by default (conv = ()).

    recurrent = m > 0 gives the agent a memory: a GRU with m units (see
    RecurrentQNetwork), cleared by begin_episode. It learns from runs of
    seq_length consecutive steps of one episode (batch_size runs per update),
    each after up to burn_in steps that rebuild the memory without being
    trained on; the loss is the mean over the trained steps. It can't be
    combined with PER, mistake replay, n-step returns, dueling heads or
    convolutions.
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
        self.recurrent = cfg.recurrent > 0
        if self.recurrent and (cfg.prioritized or cfg.mistake_replay or cfg.n_step > 1
                               or cfg.dueling or cfg.conv):
            raise ValueError("a recurrent agent can't be combined with PER, mistake replay, "
                             "n-step returns, dueling heads or convolutions")
        if self.recurrent:
            self.online = RecurrentQNetwork(obs_size, n_actions, cfg.hidden, cfg.recurrent)
            self.target = RecurrentQNetwork(obs_size, n_actions, cfg.hidden, cfg.recurrent)
        else:
            self.online = QNetwork(obs_size, n_actions, cfg.hidden, cfg.dueling, cfg.conv,
                                   cfg.conv_input)
            self.target = QNetwork(obs_size, n_actions, cfg.hidden, cfg.dueling, cfg.conv,
                                   cfg.conv_input)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        if self.recurrent:
            self.buffer = SequenceReplayBuffer(cfg.buffer_size, obs_size, seed=seed)
        elif cfg.prioritized:
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
        # recurrent: the memory after the observations acted on so far this
        # episode (None = zero). Cleared by begin_episode; never checkpointed.
        self._memory: Optional[torch.Tensor] = None

    @property
    def epsilon(self) -> float:
        return self.config.epsilon.value(self.episodes)

    @property
    def beta(self) -> float:
        cfg = self.config
        frac = min(1.0, self.updates / max(1, cfg.per_beta_steps))
        return cfg.per_beta_start + (1.0 - cfg.per_beta_start) * frac

    def begin_episode(self) -> None:
        """Clear the memory (recurrent agents); nothing to do otherwise."""
        self._memory = None

    @torch.no_grad()
    def _recurrent_step(self, obs: np.ndarray):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, 1, -1)
        q, memory = self.online(x, None if self._memory is None else self._memory.reshape(1, 1, -1))
        return q.reshape(-1).numpy().astype(np.float64), memory.reshape(-1)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for obs. A recurrent agent answers with its current memory
        and does not remember obs afterwards (only act does)."""
        if self.recurrent:
            return self._recurrent_step(obs)[0]
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.online(x).squeeze(0).numpy().astype(np.float64)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if self.recurrent:
            # every observation enters the memory, exploratory steps included
            q, self._memory = self._recurrent_step(obs)
            if not greedy and self.rng.random() < self.epsilon:
                return int(self.rng.integers(self.n_actions))
            return greedy_action(q, self.rng)
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
        elif self.recurrent:
            self.buffer.add(obs, action, reward, next_obs, terminated, truncated=truncated)
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
        if self.recurrent:
            return self._learn_sequences()
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

    def _learn_sequences(self) -> dict:
        """One gradient step on batch_size replayed sequences (recurrent agents).

        Each step's target uses the memory built over the sequence so far: the
        next observation is fed to the network after the memory at that step,
        which also handles timeouts and the window's last step exactly.
        """
        cfg = self.config
        batch = self.buffer.sample_sequences(cfg.batch_size, cfg.seq_length, cfg.burn_in)
        obs = torch.as_tensor(batch["obs"])               # (batch, steps, obs)
        actions = torch.as_tensor(batch["actions"])
        rewards = torch.as_tensor(batch["rewards"])
        next_obs = torch.as_tensor(batch["next_obs"])
        terminated = torch.as_tensor(batch["terminated"])
        mask = torch.as_tensor(batch["mask"])

        q_all, memory = self.online(obs)
        q = q_all.gather(2, actions.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            _, target_memory = self.target(obs)
            next_target = self.target.step(next_obs, target_memory)
            if cfg.double:
                best = self.online.step(next_obs, memory.detach()).argmax(dim=2, keepdim=True)
                next_q = next_target.gather(2, best).squeeze(2)
            else:
                next_q = next_target.max(dim=2).values
            y = rewards + cfg.gamma * (1.0 - terminated) * next_q

        n = mask.sum()
        loss = (mask * F.smooth_l1_loss(q, y, reduction="none")).sum() / n
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.updates += 1
        if self.updates % cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        td_error = float(((y - q).detach().abs() * mask).sum() / n)
        return {"loss": float(loss.item()), "td_error": td_error}

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
        cfg["conv"] = tuple(cfg.get("conv", ()))
        cfg["conv_input"] = tuple(cfg.get("conv_input", DQNConfig.conv_input))
        if (cfg["hidden"] != tuple(self.config.hidden)
                or cfg["conv"] != tuple(self.config.conv)
                or (cfg["conv"] and cfg["conv_input"] != tuple(self.config.conv_input))
                or cfg["prioritized"] != self.config.prioritized
                or cfg.get("mistake_replay", False) != self.config.mistake_replay
                or cfg.get("dueling", False) != self.config.dueling
                or cfg.get("n_step", 1) != self.config.n_step
                or cfg.get("recurrent", 0) != self.config.recurrent):
            raise ValueError("checkpoint network, replay type or n-step setting differs "
                             "from this agent")
        self.config = DQNConfig(**cfg)
        self.online.load_state_dict(state["online"])
        # Weights-only files (training.export_weights) have no target network or
        # optimizer state: the target starts as a copy, the optimizer fresh.
        self.target.load_state_dict(state.get("target", state["online"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.episodes = int(state["episodes"])
        self.env_steps = int(state["env_steps"])
        self.updates = int(state["updates"])
        if "buffer" in state:
            self.buffer.load_state_dict(state["buffer"])


@dataclass
class PPOConfig:
    """Hyperparameters for PPOAgent: the common published defaults for small
    discrete tasks (Schulman et al., 2017; as in CleanRL and Stable-Baselines3),
    fixed before any experiment and not tuned."""

    hidden: Tuple[int, ...] = (128, 128)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048             # env steps collected per update (may span episodes)
    epochs: int = 4                 # passes over each rollout
    minibatch_size: int = 256
    clip: float = 0.2               # PPO probability-ratio clip
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5
    normalize_advantages: bool = True  # per minibatch


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class PolicyNetwork(nn.Module):
    """MLP with tanh hidden layers and orthogonal initialisation, as in common
    PPO implementations. out_gain 0.01 for the actor makes the initial policy
    nearly uniform; 1.0 for the critic."""

    def __init__(self, obs_size: int, out_size: int, hidden: Tuple[int, ...], out_gain: float):
        super().__init__()
        layers, size = [], obs_size
        for h in hidden:
            layers += [_orthogonal(nn.Linear(size, h), float(np.sqrt(2))), nn.Tanh()]
            size = h
        layers.append(_orthogonal(nn.Linear(size, out_size), out_gain))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOAgent:
    """Proximal Policy Optimization with a clipped objective (Schulman et al., 2017).

    An actor network gives the probability of each move, pi(a | s), and a
    separate critic network estimates the state value V(s). The agent plays
    n_steps environment steps with its current policy (episodes may end and
    start inside a rollout), then learns from them and discards them: no
    replay buffer. Advantages use GAE (Schulman et al., 2016):

        delta_t = r_t + gamma * V(s'_t) * (1 - terminated_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

    so a timeout still bootstraps from V(s'_t) (done but not terminated) and
    the rollout's last step bootstraps too. Each update makes `epochs` passes
    over the rollout in shuffled minibatches and minimises

        L = -mean min(rho A, clip(rho, 1 - c, 1 + c) A)
            + value_coef * mean (V(s) - (A + V_old(s)))^2 - entropy_coef * entropy

    with rho = pi(a|s) / pi_old(a|s). pi_old and V_old are the networks that
    collected the rollout: they do not change while it is collected, so they
    are computed from the stored steps just before the first gradient step.

    It explores by sampling from pi; greedy play takes the most likely move
    (ties broken at random). There is no epsilon: `epsilon` is 0.
    """

    name = "ppo"
    nickname = "Polly the Policy"

    def __init__(self, obs_size: int, n_actions: int, config: Optional[PPOConfig] = None,
                 seed: Optional[int] = None):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.config = config if config is not None else PPOConfig()
        cfg = self.config
        if cfg.n_steps % cfg.minibatch_size != 0:
            raise ValueError("n_steps must be a multiple of minibatch_size")
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.actor = PolicyNetwork(obs_size, n_actions, cfg.hidden, 0.01)
        self.critic = PolicyNetwork(obs_size, 1, cfg.hidden, 1.0)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr)
        self.episodes = 0
        self.env_steps = 0
        self.updates = 0
        self._new_rollout()

    def _new_rollout(self) -> None:
        n, size = self.config.n_steps, self.obs_size
        self.rollout = {"obs": np.zeros((n, size), np.float32),
                        "next_obs": np.zeros((n, size), np.float32),
                        "actions": np.zeros(n, np.int64),
                        "rewards": np.zeros(n, np.float32),
                        "terminated": np.zeros(n, np.float32),
                        "truncated": np.zeros(n, np.float32)}
        self.rollout_size = 0

    @property
    def epsilon(self) -> float:
        return 0.0

    @torch.no_grad()
    def policy_probs(self, obs: np.ndarray) -> np.ndarray:
        """pi(. | obs), the probability of each move."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        probs = torch.softmax(self.actor(x), dim=1).squeeze(0).numpy().astype(np.float64)
        return probs / probs.sum()

    @torch.no_grad()
    def state_value(self, obs: np.ndarray) -> float:
        """The critic's V(obs)."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return float(self.critic(x).item())

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        probs = self.policy_probs(obs)
        if greedy:
            return greedy_action(probs, self.rng)
        return int(self.rng.choice(self.n_actions, p=probs))

    def update(self, obs, action, reward, next_obs, terminated, truncated,
               info: Optional[dict] = None) -> Optional[dict]:
        """Store the step; when the rollout is full, learn from it and clear it."""
        i, r = self.rollout_size, self.rollout
        r["obs"][i], r["next_obs"][i] = obs, next_obs
        r["actions"][i], r["rewards"][i] = action, reward
        r["terminated"][i], r["truncated"][i] = float(terminated), float(truncated)
        self.rollout_size += 1
        self.env_steps += 1
        if self.rollout_size < self.config.n_steps:
            return None
        stats = self._learn()
        self._new_rollout()
        return stats

    def advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """GAE advantages and value targets (A + V_old) for the stored rollout."""
        cfg, r, n = self.config, self.rollout, self.rollout_size
        with torch.no_grad():
            values = self.critic(torch.as_tensor(r["obs"][:n])).squeeze(1).numpy()
            next_values = self.critic(torch.as_tensor(r["next_obs"][:n])).squeeze(1).numpy()
        term, done = r["terminated"][:n], np.maximum(r["terminated"][:n], r["truncated"][:n])
        deltas = r["rewards"][:n] + cfg.gamma * next_values * (1.0 - term) - values
        adv = np.zeros(n, np.float32)
        running = 0.0
        for t in reversed(range(n)):
            running = deltas[t] + cfg.gamma * cfg.gae_lambda * (1.0 - done[t]) * running
            adv[t] = running
        return adv, (adv + values).astype(np.float32)

    def _learn(self) -> dict:
        cfg, r, n = self.config, self.rollout, self.rollout_size
        adv_all, returns_all = self.advantages()
        obs_all = torch.as_tensor(r["obs"][:n])
        actions_all = torch.as_tensor(r["actions"][:n])
        with torch.no_grad():
            old_logp_all = torch.log_softmax(self.actor(obs_all), dim=1) \
                .gather(1, actions_all.unsqueeze(1)).squeeze(1)
        adv_all, returns_all = torch.as_tensor(adv_all), torch.as_tensor(returns_all)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        losses, value_errors = [], []
        for _ in range(cfg.epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = torch.as_tensor(order[start:start + cfg.minibatch_size])
                obs, actions = obs_all[idx], actions_all[idx]
                adv, returns, old_logp = adv_all[idx], returns_all[idx], old_logp_all[idx]
                if cfg.normalize_advantages and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                logp_all = torch.log_softmax(self.actor(obs), dim=1)
                logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
                entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()
                ratio = torch.exp(logp - old_logp)
                policy_loss = -torch.min(ratio * adv,
                                         torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                value = self.critic(obs).squeeze(1)
                value_loss = ((value - returns) ** 2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                self.optimizer.step()
                losses.append(float(loss.item()))
                value_errors.append(float((value - returns).detach().abs().mean()))
        self.updates += 1
        # td_error: mean |V(s) - value target|, the critic's error (logged like DQN's)
        return {"loss": float(np.mean(losses)), "td_error": float(np.mean(value_errors))}

    def end_episode(self) -> None:
        self.episodes += 1

    def state_dict(self, include_buffer: bool = False) -> dict:
        """include_buffer also saves the half-collected rollout, so a resumed
        run continues exactly; without it a resumed run starts a new rollout."""
        state = {"name": self.name, "obs_size": self.obs_size, "n_actions": self.n_actions,
                 "config": asdict(self.config), "actor": self.actor.state_dict(),
                 "critic": self.critic.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "episodes": self.episodes, "env_steps": self.env_steps, "updates": self.updates}
        if include_buffer:
            n = self.rollout_size
            state["rollout"] = {k: v[:n].copy() for k, v in self.rollout.items()}
        return state

    def load_state_dict(self, state: dict) -> None:
        if state["obs_size"] != self.obs_size or state["n_actions"] != self.n_actions:
            raise ValueError("checkpoint has a different observation or action size")
        cfg = dict(state["config"])
        cfg["hidden"] = tuple(cfg["hidden"])
        if cfg["hidden"] != tuple(self.config.hidden) or cfg["n_steps"] != self.config.n_steps:
            raise ValueError("checkpoint network or rollout length differs from this agent")
        self.config = PPOConfig(**cfg)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        # Weights-only files (training.export_weights) have no optimizer state or rollout.
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.episodes = int(state["episodes"])
        self.env_steps = int(state["env_steps"])
        self.updates = int(state["updates"])
        self._new_rollout()
        if "rollout" in state:
            saved = state["rollout"]
            n = len(saved["actions"])
            for k in self.rollout:
                self.rollout[k][:n] = saved[k]
            self.rollout_size = n


def make_agent(name: str, obs_size: int, n_actions: int, seed: Optional[int] = None, **kwargs):
    """Build an agent by name: 'random', 'tabular_q', or a DQN variant.

    DQN variants: 'dqn', 'dqn_per' (prioritized replay), 'dqn_mistake'
    (mistake-aware replay), 'dqn_per_mistake' (both), 'dqn_dueling' (dueling
    heads), 'dqn_nstep' (3-step returns), 'dqn_dueling_nstep', 'dqn_cnn'
    (convolutional network, see CNN_DEFAULT) and 'dqn_rnn' (recurrent network
    with memory, see RNN_DEFAULT), and 'ppo' (PPOAgent). kwargs go to
    TabularQAgent, or become DQNConfig / PPOConfig fields. PPO explores by
    sampling its policy, so an `epsilon` schedule passed to it is ignored.
    """
    if name == "random":
        return RandomAgent(n_actions, seed=seed)
    if name == "tabular_q":
        return TabularQAgent(n_actions, seed=seed, **kwargs)
    if name == "ppo":
        kwargs.pop("epsilon", None)
        return PPOAgent(obs_size, n_actions, PPOConfig(**kwargs), seed=seed)
    if name in DQN_NICKNAMES:
        kwargs["prioritized"] = name.startswith("dqn_per")
        kwargs["mistake_replay"] = name.endswith("_mistake")
        kwargs["dueling"] = "_dueling" in name
        if "_nstep" in name:
            kwargs.setdefault("n_step", N_STEP_DEFAULT)
        if "_cnn" in name:
            for key, value in CNN_DEFAULT.items():
                kwargs.setdefault(key, value)
        if "_rnn" in name:
            for key, value in RNN_DEFAULT.items():
                kwargs.setdefault(key, value)
        return DQNAgent(obs_size, n_actions, DQNConfig(**kwargs), seed=seed)
    raise ValueError(f"unknown agent {name!r}; choose random, tabular_q, ppo or one of "
                     f"{', '.join(DQN_NICKNAMES)}")
