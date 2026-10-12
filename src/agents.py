"""Agents for MIND.

Every agent implements the same interface so training and evaluation code
never needs to know which algorithm it is running:

    action = agent.act(obs, greedy=False)      # greedy=True disables exploration
    stats  = agent.update(obs, action, reward, next_obs, terminated, truncated)
    agent.end_episode()                         # advance the exploration schedule
    state  = agent.state_dict(); agent.load_state_dict(state)

`terminated` means a true terminal state (goal, terminal hazard): its future
value is zero. `truncated` means the step limit cut the episode short: the
state still has future value, so we keep bootstrapping from it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


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


class RandomAgent:
    """Baseline: uniform random actions, never learns."""

    name = "random"

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.episodes = 0

    @property
    def epsilon(self) -> float:
        return 1.0

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, obs, action, reward, next_obs, terminated, truncated) -> Optional[dict]:
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

    def update(self, obs, action, reward, next_obs, terminated, truncated) -> dict:
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
