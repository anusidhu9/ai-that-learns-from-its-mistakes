"""Grid-world environments for MIND.

Every environment is a `GridWorld` configured by an `EnvConfig`. The API mirrors
Gymnasium without depending on it:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

Observations are always flat float32 vectors of length `env.obs_size`, so tabular
Q-learning (which hashes the vector) and DQN (which feeds it to a network) see
exactly the same information.

Mistakes (collision, hazard, revisit, timeout) are reported per step in
`info["events"]`, logged in `env.mistake_log`, and summarised in
`info["episode"]` when an episode ends.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

EMPTY, WALL, HAZARD, GOAL = 0, 1, 2, 3
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
ACTION_NAMES = ("up", "right", "down", "left")
MISTAKE_EVENTS = ("collision", "hazard", "revisit", "timeout")
LAYOUTS = ("maze", "open")
OBS_MODES = ("grid", "local")

Pos = Tuple[int, int]


@dataclass(frozen=True)
class RewardConfig:
    """Reward for each event. Penalties are negative and add to the step cost."""

    goal: float = 1.0
    step: float = -0.01
    collision: float = -0.05
    hazard: float = -0.5
    revisit: float = -0.02
    timeout: float = -0.1


@dataclass
class EnvConfig:
    """Configuration for a GridWorld.

    Layout selection on reset (first match wins):
      1. `layout_seed` passed to `reset()` (used for unseen-layout evaluation),
      2. a seed drawn from `layout_pool` (training on a fixed set of layouts),
      3. a fresh random seed if `randomize_layout`,
      4. the fixed `layout_seed` from this config.
    """

    height: int = 9
    width: int = 9
    layout: str = "maze"            # "maze" (DFS maze) or "open" (scattered walls)
    loop_prob: float = 0.1          # maze: chance of removing each removable wall
    wall_density: float = 0.15      # open: fraction of interior cells made walls
    n_hazards: int = 0              # hazards never block every start->goal path
    hazard_terminal: bool = True    # stepping on a hazard ends the episode
    random_start_goal: bool = False  # False: start top-left, goal bottom-right
    max_steps: int = 100
    obs_mode: str = "grid"          # "grid": full map; "local": egocentric window
    view_radius: int = 2            # local mode: window is (2r+1) x (2r+1)
    layout_seed: int = 0
    layout_pool: Optional[Tuple[int, ...]] = None
    randomize_layout: bool = False
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        if self.obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {self.obs_mode!r}")
        if self.height < 3 or self.width < 3:
            raise ValueError("height and width must be at least 3")
        if self.layout == "maze" and (self.height % 2 == 0 or self.width % 2 == 0):
            raise ValueError("maze layouts need odd height and width")
        if not 0.0 <= self.loop_prob <= 1.0 or not 0.0 <= self.wall_density < 1.0:
            raise ValueError("loop_prob must be in [0, 1] and wall_density in [0, 1)")
        if self.n_hazards < 0 or self.max_steps < 1 or self.view_radius < 1:
            raise ValueError("n_hazards >= 0, max_steps >= 1 and view_radius >= 1 required")
        if self.layout_pool is not None:
            if len(self.layout_pool) == 0:
                raise ValueError("layout_pool must not be empty")
            self.layout_pool = tuple(int(s) for s in self.layout_pool)


# ---------------------------------------------------------------------------
# Layout generation
# ---------------------------------------------------------------------------

def shortest_path_length(grid: np.ndarray, start: Pos, goal: Pos,
                         avoid_hazards: bool = True) -> Optional[int]:
    """BFS distance from start to goal, or None if unreachable."""
    h, w = grid.shape
    blocked = (WALL, HAZARD) if avoid_hazards else (WALL,)
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist and grid[nr, nc] not in blocked:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return None


def _bfs_distances(grid: np.ndarray, start: Pos) -> Dict[Pos, int]:
    h, w = grid.shape
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist and grid[nr, nc] != WALL:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return dist


def _carve_maze(h: int, w: int, rng: np.random.Generator, loop_prob: float) -> np.ndarray:
    """Recursive-backtracker maze on odd coordinates, then optional loops."""
    grid = np.full((h, w), WALL, dtype=np.int8)
    grid[1, 1] = EMPTY
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        options = [(r + 2 * dr, c + 2 * dc) for dr, dc in ACTIONS
                   if 0 < r + 2 * dr < h - 1 and 0 < c + 2 * dc < w - 1
                   and grid[r + 2 * dr, c + 2 * dc] == WALL]
        if options:
            nr, nc = options[rng.integers(len(options))]
            grid[(r + nr) // 2, (c + nc) // 2] = EMPTY
            grid[nr, nc] = EMPTY
            stack.append((nr, nc))
        else:
            stack.pop()
    if loop_prob > 0:
        # Remove interior walls that separate two open cells along one axis only,
        # turning the perfect maze into one with multiple routes.
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if grid[r, c] != WALL:
                    continue
                horizontal = grid[r, c - 1] == EMPTY and grid[r, c + 1] == EMPTY
                vertical = grid[r - 1, c] == EMPTY and grid[r + 1, c] == EMPTY
                if horizontal != vertical and rng.random() < loop_prob:
                    grid[r, c] = EMPTY
    return grid


def _scatter_walls(h: int, w: int, rng: np.random.Generator, density: float) -> np.ndarray:
    grid = np.full((h, w), WALL, dtype=np.int8)
    interior = np.where(rng.random((h - 2, w - 2)) < density, WALL, EMPTY).astype(np.int8)
    grid[1:-1, 1:-1] = interior
    return grid


def _pick_start_goal(grid: np.ndarray, rng: np.random.Generator) -> Tuple[Pos, Pos]:
    """Random start, and a reachable goal at least a quarter of the perimeter away."""
    h, w = grid.shape
    empties = [tuple(map(int, p)) for p in np.argwhere(grid == EMPTY)]
    start = empties[rng.integers(len(empties))]
    dist = _bfs_distances(grid, start)
    min_dist = max(2, (h + w) // 4)
    far = sorted(p for p, d in dist.items() if d >= min_dist)
    if far:
        goal = far[rng.integers(len(far))]
    else:
        goal = max(dist, key=lambda p: (dist[p], p))
    return start, goal


def _place_hazards(grid: np.ndarray, start: Pos, goal: Pos, n: int,
                   rng: np.random.Generator) -> None:
    """Place up to n hazards, never cutting off every hazard-free path to the goal."""
    candidates = [tuple(map(int, p)) for p in np.argwhere(grid == EMPTY)
                  if tuple(p) not in (start, goal)]
    order = rng.permutation(len(candidates))
    placed = 0
    for idx in order:
        if placed == n:
            break
        cell = candidates[idx]
        grid[cell] = HAZARD
        if shortest_path_length(grid, start, goal, avoid_hazards=True) is None:
            grid[cell] = EMPTY
        else:
            placed += 1


def generate_layout(config: EnvConfig, layout_seed: int) -> Tuple[np.ndarray, Pos, Pos]:
    """Deterministically build (grid, start, goal) from a config and a layout seed."""
    rng = np.random.default_rng(layout_seed)
    h, w = config.height, config.width
    default_start, default_goal = (1, 1), (h - 2, w - 2)

    if config.layout == "maze":
        grid = _carve_maze(h, w, rng, config.loop_prob)
    else:
        grid = None
        for _ in range(100):
            candidate = _scatter_walls(h, w, rng, config.wall_density)
            if not config.random_start_goal:
                candidate[default_start] = EMPTY
                candidate[default_goal] = EMPTY
                if shortest_path_length(candidate, default_start, default_goal) is None:
                    continue
            elif np.count_nonzero(candidate == EMPTY) < 2:
                continue
            grid = candidate
            break
        if grid is None:  # extremely dense configs: fall back to an empty arena
            grid = _scatter_walls(h, w, rng, 0.0)

    if config.random_start_goal:
        start, goal = _pick_start_goal(grid, rng)
    else:
        start, goal = default_start, default_goal

    _place_hazards(grid, start, goal, config.n_hazards, rng)
    grid[goal] = GOAL
    return grid, start, goal


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GridWorld:
    """Grid navigation task with rewards, penalties, and mistake tracking."""

    n_actions = len(ACTIONS)
    action_names = ACTION_NAMES

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.config = config if config is not None else EnvConfig()
        self.rng = np.random.default_rng(seed)
        self._custom_layout: Optional[Tuple[np.ndarray, Pos, Pos]] = None
        self._layout_cache: Dict[int, Tuple[np.ndarray, Pos, Pos]] = {}
        self.grid: Optional[np.ndarray] = None
        self.start: Pos = (1, 1)
        self.goal: Pos = (1, 1)
        self.pos: Pos = (1, 1)
        self.layout_seed: Optional[int] = None
        self.optimal_length: Optional[int] = None
        self.steps = 0
        self.done = True
        self.visited: set = set()
        self.mistake_log: List[dict] = []
        self.stats: dict = {}

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_ascii(cls, text: str, config: Optional[EnvConfig] = None,
                   seed: Optional[int] = None) -> "GridWorld":
        """Build a fixed layout from text: '#' wall, '.' empty, 'H' hazard,
        'S' start, 'G' goal. Rows must have equal length."""
        rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len({len(r) for r in rows}) != 1:
            raise ValueError("all rows must have the same length")
        h, w = len(rows), len(rows[0])
        codes = {"#": WALL, ".": EMPTY, "H": HAZARD, "S": EMPTY, "G": GOAL}
        grid = np.zeros((h, w), dtype=np.int8)
        start = goal = None
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch not in codes:
                    raise ValueError(f"unknown character {ch!r} at row {r}, col {c}")
                grid[r, c] = codes[ch]
                if ch == "S":
                    start = (r, c)
                elif ch == "G":
                    goal = (r, c)
        if start is None or goal is None:
            raise ValueError("layout needs exactly one 'S' and one 'G'")
        base = config if config is not None else EnvConfig()
        # "open" skips the odd-size check; the layout itself is fixed.
        env = cls(replace(base, height=h, width=w, layout="open"), seed=seed)
        env._custom_layout = (grid, start, goal)
        return env

    # -- properties ----------------------------------------------------------

    @property
    def obs_size(self) -> int:
        cfg = self.config
        if cfg.obs_mode == "grid":
            return 4 * cfg.height * cfg.width
        side = 2 * cfg.view_radius + 1
        return 3 * side * side + 2

    # -- core API ------------------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              layout_seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self._custom_layout is not None:
            grid, start, goal = self._custom_layout
            self.layout_seed = None
        else:
            self.layout_seed = self._choose_layout_seed(layout_seed)
            grid, start, goal = self._get_layout(self.layout_seed)

        self.grid = grid.copy()
        self.start, self.goal, self.pos = start, goal, start
        self.optimal_length = shortest_path_length(self.grid, start, goal, avoid_hazards=True)
        r = self.config.view_radius
        self._padded = np.pad(self.grid, r, constant_values=WALL)
        self.steps = 0
        self.done = False
        self.visited = {start}
        self.mistake_log = []
        self.stats = {"total_reward": 0.0, "collision": 0, "hazard": 0,
                      "revisit": 0, "timeout": 0, "success": False}
        return self._observe(), {"pos": self.pos, "events": [], "layout_seed": self.layout_seed}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("episode has finished; call reset() first")
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")

        cfg, rw = self.config, self.config.rewards
        self.steps += 1
        events: List[str] = []
        reward = rw.step
        terminated = False

        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        h, w = self.grid.shape
        if not (0 <= nr < h and 0 <= nc < w) or self.grid[nr, nc] == WALL:
            events.append("collision")
            reward += rw.collision
        else:
            self.pos = (nr, nc)
            cell = self.grid[nr, nc]
            if cell == GOAL:
                events.append("goal")
                reward += rw.goal
                terminated = True
                self.stats["success"] = True
            elif cell == HAZARD:
                events.append("hazard")
                reward += rw.hazard
                terminated = cfg.hazard_terminal
            elif self.pos in self.visited:
                events.append("revisit")
                reward += rw.revisit
            self.visited.add(self.pos)

        truncated = not terminated and self.steps >= cfg.max_steps
        if truncated:
            events.append("timeout")
            reward += rw.timeout

        self.done = terminated or truncated
        self.stats["total_reward"] += reward
        for event in events:
            if event in MISTAKE_EVENTS:
                self.stats[event] += 1
                self.mistake_log.append({"step": self.steps, "event": event,
                                         "pos": self.pos, "action": ACTION_NAMES[action]})

        info = {"pos": self.pos, "events": events, "layout_seed": self.layout_seed}
        if self.done:
            info["episode"] = self.episode_summary()
        return self._observe(), float(reward), terminated, truncated, info

    def episode_summary(self) -> dict:
        """Metrics for the current (or just-finished) episode."""
        s = self.stats
        return {
            "layout_seed": self.layout_seed,
            "steps": self.steps,
            "total_reward": round(s["total_reward"], 6),
            "success": s["success"],
            "collisions": s["collision"],
            "hazards": s["hazard"],
            "revisits": s["revisit"],
            "timeout": s["timeout"],
            "mistakes": s["collision"] + s["hazard"] + s["revisit"] + s["timeout"],
            "optimal_length": self.optimal_length,
        }

    def render_text(self) -> str:
        """ASCII picture: '#' wall, '.' empty, 'H' hazard, 'G' goal, 'A' agent."""
        chars = {EMPTY: ".", WALL: "#", HAZARD: "H", GOAL: "G"}
        lines = []
        for r in range(self.grid.shape[0]):
            row = [chars[int(v)] for v in self.grid[r]]
            if r == self.pos[0]:
                row[self.pos[1]] = "A"
            lines.append("".join(row))
        return "\n".join(lines)

    # -- internals -----------------------------------------------------------

    def _choose_layout_seed(self, layout_seed: Optional[int]) -> int:
        cfg = self.config
        if layout_seed is not None:
            return int(layout_seed)
        if cfg.layout_pool is not None:
            return int(cfg.layout_pool[self.rng.integers(len(cfg.layout_pool))])
        if cfg.randomize_layout:
            return int(self.rng.integers(2**31 - 1))
        return int(cfg.layout_seed)

    def _get_layout(self, layout_seed: int) -> Tuple[np.ndarray, Pos, Pos]:
        if layout_seed in self._layout_cache:
            return self._layout_cache[layout_seed]
        layout = generate_layout(self.config, layout_seed)
        if not self.config.randomize_layout or self.config.layout_pool is not None:
            self._layout_cache[layout_seed] = layout  # bounded: fixed seed or pool
        return layout

    def _observe(self) -> np.ndarray:
        cfg = self.config
        if cfg.obs_mode == "grid":
            obs = np.zeros((4, cfg.height, cfg.width), dtype=np.float32)
            obs[0] = self.grid == WALL
            obs[1] = self.grid == HAZARD
            obs[2][self.goal] = 1.0
            obs[3][self.pos] = 1.0
            return obs.ravel()

        r = cfg.view_radius
        pr, pc = self.pos  # padded offset r cancels the -r window start
        window = self._padded[pr:pr + 2 * r + 1, pc:pc + 2 * r + 1]
        local = np.stack([window == WALL, window == HAZARD, window == GOAL]).astype(np.float32)
        h, w = self.grid.shape
        direction = np.array([(self.goal[0] - self.pos[0]) / (h - 1),
                              (self.goal[1] - self.pos[1]) / (w - 1)], dtype=np.float32)
        return np.concatenate([local.ravel(), direction])


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, dict] = {
    "maze": dict(layout="maze", height=9, width=9, loop_prob=0.1),
    "maze_hazards": dict(layout="maze", height=9, width=9, loop_prob=0.3, n_hazards=4,
                         hazard_terminal=False),
    "open": dict(layout="open", height=8, width=8, wall_density=0.1, random_start_goal=True),
    "open_hazards": dict(layout="open", height=8, width=8, wall_density=0.1,
                         random_start_goal=True, n_hazards=5, hazard_terminal=False),
}


def make_env(name: str = "maze", seed: Optional[int] = None, **overrides) -> GridWorld:
    """Create a preset environment; keyword overrides replace EnvConfig fields.

    Example: make_env("maze", layout_pool=tuple(range(50)), obs_mode="local")
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return GridWorld(EnvConfig(**{**PRESETS[name], **overrides}), seed=seed)


if __name__ == "__main__":
    for preset in PRESETS:
        env = make_env(preset)
        env.reset(seed=0)
        print(f"== {preset} (layout_seed={env.layout_seed}, "
              f"optimal_length={env.optimal_length}, obs_size={env.obs_size})")
        print(env.render_text(), end="\n\n")
