"""Training loops, persistent logs, checkpoints, and resumption for MIND.

A training run lives in one directory:

    runs/<name>/
        config.json        TrainConfig (environment, agent, seed, episode budget)
        episodes.csv       one row per training episode (metrics + reward breakdown)
        mistakes.jsonl     one line per mistake (episode, step, event, position, action)
        checkpoint.pt      agent + replay buffer + every RNG state, for exact resumption

Resuming from checkpoint.pt continues the run as if it had never stopped: with
the same seed, an interrupted-and-resumed run produces the same episode log as
an uninterrupted one. Log rows written after the last checkpoint are discarded
on resume, because the restored agent never lived through them.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from src.agents import DQNAgent, EpsilonSchedule, make_agent
from src.envs import EnvConfig, GridWorld, PRESETS, RewardConfig

PathLike = Union[str, Path]

LOG_FIELDS = [
    "episode", "layout_seed", "success", "steps", "optimal_length", "path_efficiency",
    "total_reward", "mistakes", "collisions", "hazards", "revisits", "timeout",
    "epsilon", "td_error", "loss",
    "r_goal", "r_step", "r_collision", "r_hazard", "r_revisit", "r_timeout",
]
_INT_FIELDS = {"episode", "steps", "mistakes", "collisions", "hazards", "revisits", "timeout"}
_OPTIONAL_INT_FIELDS = {"layout_seed", "optimal_length"}
_BOOL_FIELDS = {"success"}

CONFIG_FILE = "config.json"
EPISODES_FILE = "episodes.csv"
MISTAKES_FILE = "mistakes.jsonl"
CHECKPOINT_FILE = "checkpoint.pt"


# ---------------------------------------------------------------------------
# Single episodes
# ---------------------------------------------------------------------------

def run_episode(env: GridWorld, agent, train: bool = True, greedy: bool = False,
                seed: Optional[int] = None, layout_seed: Optional[int] = None) -> dict:
    """Play one episode and return the environment's episode summary.

    train=True calls agent.update after every step and agent.end_episode at the
    end. greedy=True disables exploration (use with train=False for evaluation).
    Adds: epsilon used, mean |TD error| and loss of the updates made (NaN if
    none), path efficiency, the reward split by event, and the mistake log.
    """
    obs, _ = env.reset(seed=seed, layout_seed=layout_seed)
    epsilon = 0.0 if greedy else agent.epsilon
    td_errors, losses = [], []
    info: dict = {}
    done = False
    while not done:
        action = agent.act(obs, greedy=greedy)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if train:
            stats = agent.update(obs, action, reward, next_obs, terminated, truncated)
            if stats:
                td_errors.append(abs(stats["td_error"]))
                if "loss" in stats:
                    losses.append(stats["loss"])
        obs = next_obs
        done = terminated or truncated
    if train:
        agent.end_episode()

    summary = dict(info["episode"])
    summary["epsilon"] = epsilon
    summary["td_error"] = float(np.mean(td_errors)) if td_errors else math.nan
    summary["loss"] = float(np.mean(losses)) if losses else math.nan
    # Shortest hazard-free path / steps taken, for successful episodes that
    # touched no hazard; 0 otherwise. Crossing a hazard can be shorter than the
    # safe optimum, so counting those episodes would push efficiency above 1
    # and reward the shortcut.
    opt = summary["optimal_length"]
    safe_success = summary["success"] and summary["hazards"] == 0 and opt
    summary["path_efficiency"] = (opt / summary["steps"]) if safe_success else 0.0
    summary.update(reward_breakdown(summary, env.config.rewards))
    summary["mistake_log"] = list(env.mistake_log)
    return summary


def reward_breakdown(summary: dict, rewards: RewardConfig) -> Dict[str, float]:
    """Split an episode's total reward into the contribution of each event."""
    return {
        "r_goal": rewards.goal * int(summary["success"]),
        "r_step": rewards.step * summary["steps"],
        "r_collision": rewards.collision * summary["collisions"],
        "r_hazard": rewards.hazard * summary["hazards"],
        "r_revisit": rewards.revisit * summary["revisits"],
        "r_timeout": rewards.timeout * summary["timeout"],
    }


def train_agent(env: GridWorld, agent, episodes: int,
                seed: Optional[int] = None) -> List[dict]:
    """Train in memory (no files); returns one summary per episode.

    `seed` seeds the environment's RNG once at the start, so a run is
    reproducible given the same agent seed.
    """
    history = []
    for i in range(episodes):
        summary = run_episode(env, agent, train=True, seed=seed if i == 0 else None)
        summary["episode"] = agent.episodes - 1
        history.append(summary)
    return history


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Everything needed to rebuild a run. JSON-serialisable.

    env_overrides replace EnvConfig fields of the preset (a "rewards" dict becomes
    a RewardConfig). agent_kwargs go to make_agent (an "epsilon" dict becomes an
    EpsilonSchedule).
    """

    env_preset: str = "maze"
    env_overrides: dict = field(default_factory=dict)
    agent: str = "tabular_q"
    agent_kwargs: dict = field(default_factory=dict)
    episodes: int = 400
    seed: int = 0
    checkpoint_every: int = 50
    save_buffer: bool = True        # DQN: store replay buffer for exact resumption

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TrainConfig":
        return cls(**json.loads(text))


def build_env(config: TrainConfig) -> GridWorld:
    if config.env_preset not in PRESETS:
        raise ValueError(f"unknown preset {config.env_preset!r}; choose from {sorted(PRESETS)}")
    overrides = dict(config.env_overrides)
    if isinstance(overrides.get("rewards"), dict):
        overrides["rewards"] = RewardConfig(**overrides["rewards"])
    if isinstance(overrides.get("layout_pool"), list):
        overrides["layout_pool"] = tuple(overrides["layout_pool"])
    env_config = EnvConfig(**{**PRESETS[config.env_preset], **overrides})
    return GridWorld(env_config, seed=config.seed)


def build_agent(config: TrainConfig, env: GridWorld):
    kwargs = dict(config.agent_kwargs)
    if isinstance(kwargs.get("epsilon"), dict):
        kwargs["epsilon"] = EpsilonSchedule(**kwargs["epsilon"])
    if isinstance(kwargs.get("hidden"), list):
        kwargs["hidden"] = tuple(kwargs["hidden"])
    return make_agent(config.agent, env.obs_size, env.n_actions, seed=config.seed, **kwargs)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Runs training episodes, writes logs, and saves/restores checkpoints.

    trainer = Trainer(TrainConfig(agent="dqn", episodes=300), run_dir="runs/dqn_s0")
    trainer.train()                         # stops at config.episodes
    trainer = Trainer.resume("runs/dqn_s0") # later, after a disconnect
    trainer.train()

    With run_dir=None everything stays in memory (useful for quick demos).
    """

    def __init__(self, config: TrainConfig, run_dir: Optional[PathLike] = None):
        self.config = config
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.env = build_env(config)
        self.agent = build_agent(config, self.env)
        self.history: List[dict] = []
        self.mistakes: List[dict] = []
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.run_dir / CONFIG_FILE
            if config_path.exists() and config_path.read_text() != config.to_json():
                raise FileExistsError(f"{self.run_dir} holds a different run; "
                                      "use Trainer.resume() or a new run_dir")
            config_path.write_text(config.to_json())

    @classmethod
    def resume(cls, run_dir: PathLike, episodes: Optional[int] = None) -> "Trainer":
        """Rebuild a run from its directory and continue from its checkpoint.

        `episodes` raises (or lowers) the run's total episode budget, e.g. to
        extend a finished run. Without a checkpoint the run restarts from
        episode 0 with fresh logs.
        """
        run_dir = Path(run_dir)
        config = TrainConfig.from_json((run_dir / CONFIG_FILE).read_text())
        if episodes is not None and episodes != config.episodes:
            config.episodes = episodes
            (run_dir / CONFIG_FILE).write_text(config.to_json())
        trainer = cls(config, run_dir)
        checkpoint = run_dir / CHECKPOINT_FILE
        if checkpoint.exists():
            trainer.load_checkpoint(checkpoint)
        n = trainer.episodes_done
        trainer.history = [r for r in load_history(run_dir) if r["episode"] < n]
        trainer.mistakes = [m for m in load_mistakes(run_dir) if m["episode"] < n]
        trainer._rewrite_logs()
        return trainer

    @property
    def episodes_done(self) -> int:
        return self.agent.episodes

    def train(self, episodes: Optional[int] = None,
              callback: Optional[Callable[[dict], Optional[bool]]] = None) -> List[dict]:
        """Train until config.episodes in total (or for `episodes` more).

        callback(row) is called after every episode with the logged row plus
        its "mistake_log"; returning True stops training early (used by the
        dashboard's stop button). A checkpoint is saved every
        config.checkpoint_every episodes and when training stops.
        Returns the rows produced by this call.
        """
        target = self.config.episodes if episodes is None else self.episodes_done + episodes
        new_rows = []
        while self.episodes_done < target:
            first = self.episodes_done == 0
            summary = run_episode(self.env, self.agent, train=True,
                                  seed=self.config.seed if first else None)
            row = self._record(summary)
            new_rows.append(row)
            if self.run_dir is not None and self.episodes_done % self.config.checkpoint_every == 0:
                self.save_checkpoint()
            if callback is not None and callback({**row, "mistake_log": summary["mistake_log"]}):
                break
        if self.run_dir is not None and new_rows:
            self.save_checkpoint()
        return new_rows

    # -- checkpoints ----------------------------------------------------------

    def checkpoint_state(self, include_buffer: Optional[bool] = None) -> dict:
        include_buffer = self.config.save_buffer if include_buffer is None else include_buffer
        if isinstance(self.agent, DQNAgent):
            agent_state = self.agent.state_dict(include_buffer=include_buffer)
        else:
            agent_state = self.agent.state_dict()
        rng = {"env": self.env.rng.bit_generator.state,
               "agent": self.agent.rng.bit_generator.state,
               "torch": torch.get_rng_state()}
        if isinstance(self.agent, DQNAgent):
            rng["buffer"] = self.agent.buffer.rng.bit_generator.state
        return {"format": 1, "config": asdict(self.config), "agent": agent_state,
                "rng": rng, "episodes_done": self.episodes_done}

    def save_checkpoint(self, path: Optional[PathLike] = None,
                        include_buffer: Optional[bool] = None) -> Path:
        """Save atomically (write a temp file, then rename) so a disconnect
        mid-save never leaves a corrupt checkpoint."""
        if path is None:
            if self.run_dir is None:
                raise ValueError("no run_dir: pass an explicit checkpoint path")
            path = self.run_dir / CHECKPOINT_FILE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.checkpoint_state(include_buffer), tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, path: PathLike) -> None:
        state = _read_checkpoint(path)
        if state["config"]["agent"] != self.config.agent:
            raise ValueError(f"checkpoint is for agent {state['config']['agent']!r}, "
                             f"not {self.config.agent!r}")
        self.agent.load_state_dict(state["agent"])
        rng = state["rng"]
        self.env.rng.bit_generator.state = rng["env"]
        self.agent.rng.bit_generator.state = rng["agent"]
        torch.set_rng_state(rng["torch"])
        if "buffer" in rng and isinstance(self.agent, DQNAgent):
            self.agent.buffer.rng.bit_generator.state = rng["buffer"]

    # -- logging --------------------------------------------------------------

    def _record(self, summary: dict) -> dict:
        row = {k: summary[k] for k in LOG_FIELDS if k != "episode"}
        row["episode"] = self.agent.episodes - 1
        row = {k: row[k] for k in LOG_FIELDS}
        mistakes = [{"episode": row["episode"], "layout_seed": row["layout_seed"],
                     "step": m["step"], "event": m["event"], "pos": list(m["pos"]),
                     "action": m["action"]} for m in summary["mistake_log"]]
        self.history.append(row)
        self.mistakes.extend(mistakes)
        if self.run_dir is not None:
            self._append_logs([row], mistakes)
        return row

    def _append_logs(self, rows: List[dict], mistakes: List[dict]) -> None:
        assert self.run_dir is not None, "logs are only written for runs with a run_dir"
        episodes_path = self.run_dir / EPISODES_FILE
        new_file = not episodes_path.exists()
        with open(episodes_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)
        with open(self.run_dir / MISTAKES_FILE, "a") as f:
            for m in mistakes:
                f.write(json.dumps(m) + "\n")

    def _rewrite_logs(self) -> None:
        assert self.run_dir is not None, "logs are only written for runs with a run_dir"
        for name in (EPISODES_FILE, MISTAKES_FILE):
            (self.run_dir / name).unlink(missing_ok=True)
        self._append_logs(self.history, self.mistakes)


def _safe_checkpoint_types() -> list:
    """The only non-PyTorch types a MIND checkpoint contains (numpy arrays)."""
    try:
        from numpy._core.multiarray import _reconstruct  # numpy 2.x
    except ImportError:  # pragma: no cover - numpy 1.x
        from numpy.core.multiarray import _reconstruct  # type: ignore
    dtypes = [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32", "int8",
                                          "uint32", "uint64", "bool")]
    return [np.ndarray, np.dtype, _reconstruct] + dtypes


def _read_checkpoint(path: PathLike) -> dict:
    # weights_only=True refuses to unpickle arbitrary objects, so a malicious
    # uploaded file cannot run code; only plain containers, tensors and the
    # numpy array types listed above are accepted.
    with torch.serialization.safe_globals(_safe_checkpoint_types()):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_agent(path: PathLike):
    """Load a checkpoint for evaluation; returns (agent, env, TrainConfig)."""
    state = _read_checkpoint(path)
    config = TrainConfig(**state["config"])
    env = build_env(config)
    agent = build_agent(config, env)
    agent.load_state_dict(state["agent"])
    return agent, env, config


# ---------------------------------------------------------------------------
# Reading logs and analysing mistakes
# ---------------------------------------------------------------------------

def _parse_row(raw: dict) -> dict:
    row = {}
    for key in LOG_FIELDS:
        value = raw[key]
        if key in _INT_FIELDS:
            row[key] = int(value)
        elif key in _OPTIONAL_INT_FIELDS:
            row[key] = int(value) if value != "" else None
        elif key in _BOOL_FIELDS:
            row[key] = value == "True"
        else:
            row[key] = float(value)
    return row


def load_history(run_dir: PathLike) -> List[dict]:
    """Episode rows from a run's episodes.csv (empty list if none yet)."""
    path = Path(run_dir) / EPISODES_FILE
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [_parse_row(raw) for raw in csv.DictReader(f)]


def load_mistakes(run_dir: PathLike) -> List[dict]:
    """Mistake records from a run's mistakes.jsonl (empty list if none yet)."""
    path = Path(run_dir) / MISTAKES_FILE
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


PHASE_METRICS = ["success", "steps", "total_reward", "path_efficiency", "mistakes",
                 "collisions", "hazards", "revisits", "timeout"]


def compare_phases(history: List[dict], window: int = 50) -> dict:
    """Mean of each metric over the first and last `window` episodes.

    Returns {"early": {...}, "late": {...}, "change": {late - early}}. This is a
    description of one training run, not evidence of improvement on its own;
    stage 5 evaluates greedily across seeds before any claim is made.
    """
    if not history:
        raise ValueError("history is empty")
    window = min(window, len(history))
    early, late = history[:window], history[-window:]

    def means(rows):
        return {m: float(np.mean([float(r[m]) for r in rows])) for m in PHASE_METRICS}

    e, l = means(early), means(late)
    return {"early": e, "late": l, "change": {m: l[m] - e[m] for m in PHASE_METRICS},
            "window": window}


def mistake_hotspots(mistakes: List[dict], top: int = 10,
                     episodes: Optional[range] = None) -> List[dict]:
    """Most frequent (event, position) pairs: the agent's recurring failures.

    Restrict to a range of episodes (e.g. range(0, 50) vs the last 50) to see
    which failure patterns disappear and which persist.
    """
    selected = mistakes if episodes is None else [m for m in mistakes if m["episode"] in episodes]
    counts = Counter((m["event"], tuple(m["pos"])) for m in selected)
    return [{"event": event, "pos": pos, "count": count}
            for (event, pos), count in counts.most_common(top)]


def format_phase_table(comparison: dict) -> str:
    """Plain-text table of compare_phases output."""
    w = comparison["window"]
    lines = [f"{'metric':<16}{'first ' + str(w):>12}{'last ' + str(w):>12}{'change':>12}"]
    for m in PHASE_METRICS:
        e, l, c = comparison["early"][m], comparison["late"][m], comparison["change"][m]
        lines.append(f"{m:<16}{e:>12.3f}{l:>12.3f}{c:>+12.3f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """Command line: train a new run, or resume one after a disconnect.

    python -m src.training --agent dqn --preset maze --episodes 400 --run-dir runs/dqn_s0
    python -m src.training --resume runs/dqn_s0
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train a MIND agent with logs and checkpoints.")
    parser.add_argument("--resume", metavar="RUN_DIR", help="continue an existing run")
    parser.add_argument("--run-dir", help="directory for a new run's logs and checkpoint")
    parser.add_argument("--agent", default="tabular_q",
                        choices=["random", "tabular_q", "dqn", "dqn_per"])
    parser.add_argument("--preset", default="maze", choices=sorted(PRESETS))
    parser.add_argument("--episodes", type=int, default=None,
                        help="total episodes (default 400; with --resume, extends the run)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decay-episodes", type=int, default=300,
                        help="episodes over which epsilon decays from 1.0 to 0.05")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args(argv)

    if args.resume:
        trainer = Trainer.resume(args.resume, episodes=args.episodes)
        print(f"Resuming {args.resume} at episode {trainer.episodes_done} "
              f"of {trainer.config.episodes}")
    else:
        if not args.run_dir:
            parser.error("--run-dir is required for a new run")
        kwargs = {} if args.agent == "random" else {"epsilon": {"decay_episodes": args.decay_episodes}}
        config = TrainConfig(env_preset=args.preset, agent=args.agent, agent_kwargs=kwargs,
                             episodes=args.episodes or 400, seed=args.seed,
                             checkpoint_every=args.checkpoint_every)
        trainer = Trainer(config, args.run_dir)

    from src import fun  # imported here: src.fun depends on this module

    print(f"Agent: {trainer.agent.nickname} ({trainer.config.agent}). "
          f"{fun.TAGLINES[trainer.config.agent]}")

    def progress(row: dict) -> None:
        if (row["episode"] + 1) % 50 == 0:
            recent = trainer.history[-50:]
            success = float(np.mean([r["success"] for r in recent]))
            mistakes = float(np.mean([r["mistakes"] for r in recent]))
            print(f"episode {row['episode'] + 1:>5}  success {success:.2f}  "
                  f"mistakes {mistakes:6.1f}  epsilon {row['epsilon']:.2f}  "
                  f"| {fun.progress_quip(success, mistakes, row['episode'])}", flush=True)

    trainer.train(callback=progress)
    if trainer.history:
        print()
        print(format_phase_table(compare_phases(trainer.history)))
        print("\nMost repeated mistakes in the first 50 episodes:")
        for h in mistake_hotspots(trainer.mistakes, top=5, episodes=range(0, 50)):
            print(f"  {h['event']:<10} at {h['pos']}: {h['count']}")
        last = range(max(0, trainer.episodes_done - 50), trainer.episodes_done)
        print("Most repeated mistakes in the last 50 episodes:")
        for h in mistake_hotspots(trainer.mistakes, top=5, episodes=last) or [{"event": "none"}]:
            print(f"  {h['event']:<10} at {h.get('pos', '-')}: {h.get('count', 0)}")
        print("\n" + fun.format_hall_of_shame(fun.hall_of_shame(trainer.mistakes, top=3)))
        earned = fun.achievements(trainer.history)
        print(f"\nAchievements unlocked: {len(earned)}/{len(fun.ACHIEVEMENTS)}")
        for a in earned:
            print(f"  [episode {a['episode']:>4}] {a['title']}")
        print(f"\nLast episode: {fun.episode_line(trainer.history[-1])}")


if __name__ == "__main__":
    main()
