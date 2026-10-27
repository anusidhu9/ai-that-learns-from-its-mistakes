"""Experience replay buffers for DQN.

ReplayBuffer samples stored transitions uniformly. PrioritizedReplayBuffer
(Schaul et al., 2016, proportional variant) samples transition i with
probability

    P(i) = p_i^alpha / sum_k p_k^alpha,     p_i = |td_error_i| + eps

so surprising transitions (big mistakes in the agent's predictions) are
replayed more often. The bias this introduces is corrected with importance
sampling weights

    w_i = (N * P(i))^(-beta) / max_j w_j

where beta is annealed toward 1 during training. Both buffers return the
same batch dictionary so the agent treats them identically.

MistakeReplayBuffer wraps either one and additionally keeps a copy of every
mistake transition (collision, hazard) in a reserved buffer, so each batch
can mix ordinary experience with remembered mistakes in a fixed proportion.

Every transition can also carry its own bootstrap discount (`discounts`),
which n-step DQN needs because a transition cut short by the end of an
episode spans fewer than n steps. One-step DQN never reads it.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer of (obs, action, reward, next_obs, terminated),
    plus an optional per-transition bootstrap discount (NaN when not given)."""

    def __init__(self, capacity: int, obs_size: int, seed: Optional[int] = None):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.full(capacity, np.nan, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, terminated,
            discount: Optional[float] = None) -> int:
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = float(terminated)
        self.discounts[i] = np.nan if discount is None else discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _batch(self, idx: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "terminated": self.terminated[idx],
            "discounts": self.discounts[idx],
            "weights": weights.astype(np.float32),
            "indices": idx,
        }

    def sample(self, batch_size: int, beta: float = 1.0) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self._batch(idx, np.ones(batch_size))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Uniform replay ignores priorities; present for a shared interface."""

    def state_dict(self) -> dict:
        n = self.size
        return {"capacity": self.capacity, "pos": self.pos, "size": n,
                "obs": self.obs[:n].copy(), "next_obs": self.next_obs[:n].copy(),
                "actions": self.actions[:n].copy(), "rewards": self.rewards[:n].copy(),
                "terminated": self.terminated[:n].copy(),
                "discounts": self.discounts[:n].copy()}

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError("buffer capacity mismatch")
        n = state["size"]
        self.obs[:n] = state["obs"]
        self.next_obs[:n] = state["next_obs"]
        self.actions[:n] = state["actions"]
        self.rewards[:n] = state["rewards"]
        self.terminated[:n] = state["terminated"]
        # buffers saved before n-step returns existed have no discounts
        self.discounts[:n] = state.get("discounts", np.nan)
        self.pos, self.size = state["pos"], n


class SumTree:
    """Binary tree where each parent stores the sum of its children.

    Leaves hold priorities; sampling a value u in [0, total) and descending
    the tree finds leaf i with probability priority_i / total in O(log n).
    """

    def __init__(self, capacity: int):
        self.n_leaves = 1
        while self.n_leaves < capacity:
            self.n_leaves *= 2
        self.depth = int(np.log2(self.n_leaves))
        self.tree = np.zeros(2 * self.n_leaves, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, data_idx: np.ndarray) -> np.ndarray:
        return self.tree[np.asarray(data_idx) + self.n_leaves]

    def set(self, data_idx: np.ndarray, priorities: np.ndarray) -> None:
        nodes = np.asarray(data_idx, dtype=np.int64) + self.n_leaves
        self.tree[nodes] = priorities
        for _ in range(self.depth):
            nodes = np.unique(nodes // 2)
            self.tree[nodes] = self.tree[2 * nodes] + self.tree[2 * nodes + 1]

    def find(self, values: np.ndarray) -> np.ndarray:
        """Leaf (data index) for each prefix-sum value, vectorised over a batch."""
        values = np.asarray(values, dtype=np.float64).copy()
        nodes = np.ones(len(values), dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * nodes
            go_left = values < self.tree[left]
            values = np.where(go_left, values, values - self.tree[left])
            nodes = np.where(go_left, left, left + 1)
        return nodes - self.n_leaves


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized experience replay."""

    def __init__(self, capacity: int, obs_size: int, alpha: float = 0.6,
                 eps: float = 1e-3, seed: Optional[int] = None):
        super().__init__(capacity, obs_size, seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # raw |td| + eps; new data gets the largest seen

    def add(self, obs, action, reward, next_obs, terminated,
            discount: Optional[float] = None) -> int:
        i = super().add(obs, action, reward, next_obs, terminated, discount)
        self.tree.set(np.array([i]), np.array([self.max_priority ** self.alpha]))
        return i

    def sample(self, batch_size: int, beta: float = 0.4) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        total = self.tree.total
        # Stratified sampling: one draw from each of batch_size equal segments.
        bounds = np.linspace(0.0, total, batch_size + 1)
        values = self.rng.uniform(bounds[:-1], bounds[1:])
        idx = self.tree.find(np.minimum(values, np.nextafter(total, 0)))
        idx = np.minimum(idx, self.size - 1)  # guards float round-off at the edge
        probs = self.tree.get(idx) / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()
        return self._batch(idx, weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        raw = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        self.max_priority = max(self.max_priority, float(raw.max()))
        self.tree.set(indices, raw ** self.alpha)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(alpha=self.alpha, eps=self.eps, max_priority=self.max_priority,
                     priorities=self.tree.get(np.arange(self.size)).copy())
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.alpha, self.eps = state["alpha"], state["eps"]
        self.max_priority = state["max_priority"]
        self.tree = SumTree(self.capacity)
        if self.size:
            self.tree.set(np.arange(self.size), state["priorities"])


class MistakeReplayBuffer:
    """A main buffer plus a reserve that remembers mistake transitions.

    Every transition goes into the main buffer exactly as without this
    wrapper. Transitions flagged as mistakes are also copied into a separate
    circular reserve, which keeps them after the main buffer has overwritten
    them. Each batch takes

        k = min(round(fraction * batch_size), batch_size - 1, len(reserve))

    transitions uniformly from the reserve and the rest from the main buffer
    (uniform or prioritized). This deliberately over-weights mistakes and is
    not corrected by importance weights: reserve samples get weight 1, the
    largest weight a prioritized batch can contain.

    Reserve samples are returned with indices offset by the main buffer's
    capacity, so update_priorities can pass only the main buffer's indices on.

    Transitions are numbered 0, 1, 2, ... as they are added, and each reserve
    slot remembers its transition's number, so diagnostics() can show how far
    back each buffer remembers (bookkeeping only; it never affects sampling).
    """

    def __init__(self, main: ReplayBuffer, capacity: int, fraction: float = 0.25,
                 seed: Optional[int] = None):
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction must be in [0, 1)")
        self.main = main
        self.fraction = fraction
        self.reserve = ReplayBuffer(capacity, main.obs_size)
        # Its own generator, so the main buffer's random stream is untouched.
        self.reserve.rng = np.random.default_rng(None if seed is None else [seed, 1])
        self.mistakes_seen = 0
        self.added = 0                                       # transitions added so far
        self.reserve_added_at = np.zeros(capacity, dtype=np.int64)

    @property
    def rng(self) -> np.random.Generator:
        """The main buffer's generator (the trainer saves and restores it)."""
        return self.main.rng

    @property
    def capacity(self) -> int:
        return self.main.capacity

    @property
    def obs_size(self) -> int:
        return self.main.obs_size

    def __len__(self) -> int:
        return len(self.main)

    def add(self, obs, action, reward, next_obs, terminated, mistake: bool = False) -> int:
        i = self.main.add(obs, action, reward, next_obs, terminated)
        if mistake:
            j = self.reserve.add(obs, action, reward, next_obs, terminated)
            self.reserve_added_at[j] = self.added
            self.mistakes_seen += 1
        self.added += 1
        return i

    def diagnostics(self) -> dict:
        """How far back each buffer remembers.

        Transition number t is still in the main buffer iff
        t >= added - len(main). `reserve_only` counts reserved mistakes the
        main buffer has already overwritten: the ones only the reserve keeps.
        """
        n = len(self.reserve)
        first_in_main = self.added - len(self.main)
        stamps = self.reserve_added_at[:n]
        return {"transitions_added": self.added, "main_capacity": self.main.capacity,
                "main_size": len(self.main), "main_overwritten": first_in_main,
                "reserve_capacity": self.reserve.capacity, "reserve_size": n,
                "mistakes_seen": self.mistakes_seen,
                "reserve_overwritten": self.mistakes_seen - n,
                "reserve_span": int(self.added - stamps.min()) if n else 0,
                "reserve_only": int((stamps < first_in_main).sum())}

    def n_from_reserve(self, batch_size: int) -> int:
        return max(0, min(int(round(self.fraction * batch_size)), batch_size - 1,
                          len(self.reserve)))

    def sample(self, batch_size: int, beta: Optional[float] = None) -> Dict[str, np.ndarray]:
        k = self.n_from_reserve(batch_size)
        main = (self.main.sample(batch_size - k) if beta is None
                else self.main.sample(batch_size - k, beta=beta))
        main["from_reserve"] = np.zeros(batch_size - k, dtype=bool)
        if k == 0:
            return main
        idx = self.reserve.rng.integers(0, len(self.reserve), size=k)
        extra = self.reserve._batch(idx, np.ones(k))
        extra["indices"] = idx + self.main.capacity
        extra["from_reserve"] = np.ones(k, dtype=bool)
        return {key: np.concatenate([main[key], extra[key]]) for key in main}

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        indices = np.asarray(indices)
        keep = indices < self.main.capacity
        if keep.any():
            self.main.update_priorities(indices[keep], np.asarray(td_errors)[keep])

    def state_dict(self) -> dict:
        return {"main": self.main.state_dict(), "reserve": self.reserve.state_dict(),
                "reserve_rng": self.reserve.rng.bit_generator.state,
                "fraction": self.fraction, "mistakes_seen": self.mistakes_seen,
                "added": self.added,
                "reserve_added_at": self.reserve_added_at[:len(self.reserve)].copy()}

    def load_state_dict(self, state: dict) -> None:
        self.main.load_state_dict(state["main"])
        self.reserve.load_state_dict(state["reserve"])
        self.reserve.rng.bit_generator.state = state["reserve_rng"]
        self.fraction = float(state["fraction"])
        self.mistakes_seen = int(state["mistakes_seen"])
        # Checkpoints from before the stamps existed: number the stored
        # transitions as if the main buffer were all that was ever added.
        self.added = int(state.get("added", len(self.main)))
        self.reserve_added_at[:] = 0
        if "reserve_added_at" in state:
            self.reserve_added_at[:len(self.reserve)] = state["reserve_added_at"]
