"""Evaluation and experiments for MIND.

Every claim in the report should come from this module:

* `evaluate_agent` plays greedy episodes (exploration off) without changing the
  agent, on the training layout(s) or on layouts the agent never saw.
* `run_experiment` trains several agents over several seeds, evaluates each
  greedily every `eval_every` episodes (the greedy learning curve) and at the
  end on training and unseen layouts, and writes everything to CSV/JSON.
  It is resumable: finished runs are skipped and interrupted runs continue
  from their last checkpoint.
* `summarize` reports each metric as mean and 95% confidence interval across
  seeds; `permutation_test` checks whether two agents really differ.

    python -m src.evaluation --experiment fixed_maze
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.envs import GridWorld
from src.training import (CHECKPOINT_FILE, PHASE_METRICS, TrainConfig, Trainer,
                          _read_checkpoint, build_env, run_episode)

PathLike = Union[str, Path]
# success, steps, total_reward, path_efficiency, mistakes, ..., then steps_to_goal:
# mean steps over successful episodes only, None when a run had no success.
# Unlike `steps`, it does not count failed episodes as max_steps.
# hazard_rate: fraction of episodes that stepped on at least one hazard.
METRICS = PHASE_METRICS + ["steps_to_goal", "hazard_rate"]
CURVE_FILE = "eval_curve.csv"
FINAL_FILE = "final_eval.json"

# Two-sided 95% Student-t critical values for df = 1..30 (beyond 30, 1.96 is used).
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and half-width of the 95% t confidence interval (NaN if n < 2).

    The interval is mean ± t_{0.975, n-1} * s / sqrt(n), with s the sample
    standard deviation. Each value should be one seed's result, so the
    interval describes run-to-run variation.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return math.nan, math.nan
    if n == 1:
        return float(x[0]), math.nan
    t = _T95[n - 2] if n - 1 <= len(_T95) else 1.96
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(n))


def format_p(p_value: float) -> str:
    """'p = 0.008', or 'p < 0.001' for tiny values (never the impossible 'p = 0.000')."""
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def permutation_test(a: Sequence[float], b: Sequence[float],
                     n_resamples: int = 20_000, seed: int = 0) -> float:
    """Two-sided p-value for "a and b have the same mean".

    Exact (all relabellings) when that is at most n_resamples; otherwise a
    Monte Carlo estimate. With 5 seeds per agent the smallest possible p-value
    is 2/252 ≈ 0.008, so five seeds can show a clear difference but not a tiny one.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    n_a, n = len(a), len(a) + len(b)
    observed = abs(a.mean() - b.mean())
    if observed == 0:
        return 1.0
    total = math.comb(n, n_a)
    if total <= n_resamples:
        diffs = []
        for idx in itertools.combinations(range(n), n_a):
            mask = np.zeros(n, dtype=bool)
            mask[list(idx)] = True
            diffs.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
        diffs = np.array(diffs)
        return float(np.mean(diffs >= observed - 1e-12))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(pooled)
        count += abs(perm[:n_a].mean() - perm[n_a:].mean()) >= observed - 1e-12
    return float((count + 1) / (n_resamples + 1))


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env: GridWorld, layout_seeds: Optional[Sequence[int]] = None,
                   episodes: int = 10, seed: int = 0) -> dict:
    """Greedy (epsilon = 0) evaluation that never changes the agent.

    With layout_seeds, plays one episode on each listed layout; otherwise plays
    `episodes` episodes on the environment's own layout(s). The agent's random
    generator (used only to break ties between equal Q-values) is swapped for a
    seeded one during evaluation and restored afterwards, so evaluating in the
    middle of training does not change how training continues.

    Returns the mean of every metric plus the per-episode summaries.
    """
    original_rng = agent.rng
    agent.rng = np.random.default_rng(seed)
    try:
        runs = []
        if layout_seeds is not None:
            for i, layout in enumerate(layout_seeds):
                runs.append(run_episode(env, agent, train=False, greedy=True,
                                        seed=seed if i == 0 else None, layout_seed=int(layout)))
        else:
            for i in range(episodes):
                runs.append(run_episode(env, agent, train=False, greedy=True,
                                        seed=seed if i == 0 else None))
    finally:
        agent.rng = original_rng
    for r in runs:
        r.pop("mistake_log", None)
    result = {m: float(np.mean([float(r[m]) for r in runs])) for m in PHASE_METRICS}
    goal_steps = [r["steps"] for r in runs if r["success"]]
    result["steps_to_goal"] = float(np.mean(goal_steps)) if goal_steps else None
    result["hazard_rate"] = float(np.mean([r["hazards"] > 0 for r in runs]))
    result["n_episodes"] = len(runs)
    result["episodes"] = runs
    return result


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """A comparison of several agents over several seeds on one environment.

    unseen_layouts: layout seeds held out from training (requires env_overrides
    with a layout_pool); each final evaluation also runs on these.
    agent_kwargs: per-agent overrides, e.g. {"dqn": {"lr": 5e-4}}.
    agent_types: arm label -> agent name, so one algorithm can appear as several
    arms with different settings, e.g. {"dqn_5k": "dqn", "dqn_10k": "dqn"} with
    the buffer sizes in agent_kwargs. Labels not listed are agent names.
    arm_env_overrides: per-arm environment overrides added to env_overrides,
    e.g. {"dqn_shaped": {"rewards": {"shaping": 1.0}}}.
    """

    name: str
    env_preset: str = "maze"
    env_overrides: dict = field(default_factory=dict)
    agents: Tuple[str, ...] = ("random", "tabular_q", "dqn", "dqn_per")
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    episodes: int = 400
    decay_episodes: int = 300
    eval_every: int = 50
    eval_episodes: int = 10
    unseen_layouts: Optional[Tuple[int, ...]] = None
    agent_kwargs: Dict[str, dict] = field(default_factory=dict)
    description: str = ""
    agent_types: Dict[str, str] = field(default_factory=dict)
    arm_env_overrides: Dict[str, dict] = field(default_factory=dict)

    def train_config(self, agent: str, seed: int) -> TrainConfig:
        kwargs = {} if agent == "random" else {"epsilon": {"decay_episodes": self.decay_episodes}}
        kwargs.update(self.agent_kwargs.get(agent, {}))
        overrides = {**self.env_overrides, **self.arm_env_overrides.get(agent, {})}
        return TrainConfig(env_preset=self.env_preset, env_overrides=overrides,
                           agent=self.agent_types.get(agent, agent), agent_kwargs=kwargs,
                           episodes=self.episodes, seed=seed, checkpoint_every=self.eval_every)

    def train_layouts(self) -> Optional[List[int]]:
        pool = self.env_overrides.get("layout_pool")
        return list(pool) if pool is not None else None


POOL = tuple(range(50))
UNSEEN = tuple(range(1000, 1050))
# Fresh layouts and seeds for the long-memory study (none used by any earlier
# experiment): train on LM_POOL, test once at the end on LM_HELDOUT.
LM_POOL = tuple(range(3000, 3050))
LM_HELDOUT = tuple(range(4000, 4050))
LM_SEEDS = tuple(range(100, 110))
LM_ARMS = {"dqn_5k": "dqn", "dqn_mistake_5k": "dqn_mistake", "dqn_10k": "dqn"}
LM_KWARGS = {"dqn_5k": {"buffer_size": 5000},
             "dqn_mistake_5k": {"buffer_size": 5000, "mistake_capacity": 5000,
                                "mistake_fraction": 0.25},
             "dqn_10k": {"buffer_size": 10_000}}
LM_COMMON = dict(env_preset="maze_hazards", episodes=3000, decay_episodes=2000,
                 eval_every=500, agent_types=LM_ARMS, agent_kwargs=LM_KWARGS)
# Reward shaping study (pre-registered in results/shaping_*/PROTOCOL.md): fresh
# seeds and layouts, shaping strength fixed at 1.0 in advance (never tuned).
SHAPING = {"rewards": {"shaping": 1.0}}
SH_SEEDS = tuple(range(200, 210))
SH_LAYOUT = 7000
SH_POOL = tuple(range(7100, 7150))
SH_HELDOUT = tuple(range(7200, 7250))

EXPERIMENTS: Dict[str, Experiment] = {
    "fixed_maze": Experiment(
        "fixed_maze", "maze",
        description="One fixed 9x9 maze: can each agent learn to solve it without mistakes?"),
    "hazard_maze": Experiment(
        "hazard_maze", "maze_hazards", seeds=tuple(range(10)),
        description="Fixed 9x9 maze with 4 non-terminal hazards (-0.5 each); 10 seeds."),
    "terminal_hazards": Experiment(
        "terminal_hazards", "maze_hazards", env_overrides={"hazard_terminal": True},
        agents=("random", "tabular_q", "dqn"),
        description="Same maze, but stepping on a hazard ends the episode."),
    "open_hazards": Experiment(
        "open_hazards", "open_hazards",
        description="Fixed 8x8 open grid with 5 hazards and a random start/goal."),
    "mistakes_hazard_maze": Experiment(
        "mistakes_hazard_maze", "maze_hazards", seeds=tuple(range(10)),
        agents=("dqn", "dqn_per", "dqn_mistake", "dqn_per_mistake"),
        description="hazard_maze (same maze, seeds and schedule) comparing DQN, prioritized "
                    "replay and mistake-aware replay (25% of each batch from a reserve of "
                    "collisions and hazards)."),
    "long_memory_mistakes": Experiment(
        "long_memory_mistakes", **LM_COMMON,
        env_overrides={"layout_pool": list(LM_POOL), "obs_mode": "local"},
        agents=tuple(LM_ARMS), seeds=LM_SEEDS, unseen_layouts=LM_HELDOUT,
        description="Hazard mazes with a 5,000-transition replay buffer, so old experience is "
                    "overwritten: does a 5,000-mistake reserve (25% of each batch) help? "
                    "dqn_10k matches the variant's total memory. Train on layouts 3000-3049, "
                    "test on held-out 4000-4049; seeds 100-109."),
    "long_memory_pilot": Experiment(
        "long_memory_pilot", **LM_COMMON,
        env_overrides={"layout_pool": list(range(5000, 5050)), "obs_mode": "local"},
        agents=("dqn_5k",), seeds=(900,), unseen_layouts=tuple(range(6000, 6050)),
        description="Feasibility pilot for long_memory_mistakes: baseline arm only, on its "
                    "own seed and layouts, to measure run time and check that the buffer "
                    "overwrites old transitions. Never used to choose settings for the variant."),
    "shaping_speed": Experiment(
        "shaping_speed", "maze", env_overrides={"layout_seed": SH_LAYOUT},
        agents=("tabular_q", "tabular_q_shaped", "dqn", "dqn_shaped"), seeds=SH_SEEDS,
        episodes=400, decay_episodes=300, eval_every=25,
        agent_types={"tabular_q_shaped": "tabular_q", "dqn_shaped": "dqn"},
        arm_env_overrides={"tabular_q_shaped": SHAPING, "dqn_shaped": SHAPING},
        description="One fresh 9x9 maze (layout 7000): does potential-based reward shaping "
                    "(strength 1.0) make tabular Q and DQN learn faster? Seeds 200-209."),
    "shaping_generalization": Experiment(
        "shaping_generalization", "maze",
        env_overrides={"layout_pool": list(SH_POOL), "obs_mode": "local"},
        agents=("dqn", "dqn_shaped"), seeds=SH_SEEDS,
        episodes=3000, decay_episodes=2000, eval_every=500, unseen_layouts=SH_HELDOUT,
        agent_types={"dqn_shaped": "dqn"}, arm_env_overrides={"dqn_shaped": SHAPING},
        description="Train DQN with a local 5x5 view on 50 fresh mazes (7100-7149), with and "
                    "without reward shaping (strength 1.0); test on 50 held-out mazes "
                    "(7200-7249). Seeds 200-209."),
    "generalization": Experiment(
        "generalization", "maze",
        env_overrides={"layout_pool": list(POOL), "obs_mode": "local"},
        agents=("random", "tabular_q", "dqn"), seeds=tuple(range(10)),
        episodes=3000, decay_episodes=2000, eval_every=500, unseen_layouts=UNSEEN,
        description="Train on 50 maze layouts with a local 5x5 view, test on 50 unseen layouts; 10 seeds."),
}


def _write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("agent", "split", "experiment"):
                continue
            if v == "":  # None when written (steps_to_goal of a run with no success)
                r[k] = None
                continue
            r[k] = float(v)
            if k in ("episode", "seed"):
                r[k] = int(r[k])
    return rows


EVAL_SEED_OFFSET = 10_000


def _evaluate_splits(exp: Experiment, trainer: Trainer, env: GridWorld,
                     final: bool) -> Dict[str, dict]:
    # Each run gets its own evaluation seed. A shared one would make every
    # seed's random baseline (and every tie-break) identical, hiding the
    # run-to-run variation the confidence intervals are meant to measure.
    eval_seed = EVAL_SEED_OFFSET + trainer.config.seed
    splits = {}
    layouts = exp.train_layouts()
    splits["train"] = evaluate_agent(trainer.agent, env, layout_seeds=layouts,
                                     episodes=exp.eval_episodes, seed=eval_seed)
    if final and exp.unseen_layouts is not None:
        splits["unseen"] = evaluate_agent(trainer.agent, env, layout_seeds=exp.unseen_layouts,
                                          seed=eval_seed)
    return splits


def run_single(exp: Experiment, agent: str, seed: int, exp_dir: Path,
               log: Callable[[str], None] = print) -> dict:
    """Train one (agent, seed) run with periodic greedy evaluation; resumable."""
    run_dir = exp_dir / f"{agent}_s{seed}"
    final_path = run_dir / FINAL_FILE
    config = exp.train_config(agent, seed)
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved_config = TrainConfig.from_json(config_path.read_text())
        if asdict(saved_config) | {"episodes": 0} != asdict(config) | {"episodes": 0}:
            raise ValueError(f"{run_dir} was trained with different settings; "
                             "delete it or use another --root")

    if final_path.exists():
        saved = json.loads(final_path.read_text())
        if saved.pop("episodes_trained", None) == exp.episodes:
            return saved
        final_path.unlink()  # experiment was extended: train further, re-evaluate

    if config_path.exists():
        # resume() also restarts cleanly if the run crashed before its first checkpoint
        trainer = Trainer.resume(run_dir, episodes=exp.episodes)
    else:
        trainer = Trainer(config, run_dir)
    eval_env = build_env(config)

    curve_fields = ["episode"] + METRICS
    curve = [r for r in _read_csv(run_dir / CURVE_FILE) if r["episode"] <= trainer.episodes_done]
    if not curve or curve[-1]["episode"] != trainer.episodes_done:
        point = _evaluate_splits(exp, trainer, eval_env, final=False)["train"]
        curve.append({"episode": trainer.episodes_done, **{m: point[m] for m in METRICS}})
    _write_csv(run_dir / CURVE_FILE, curve, curve_fields)

    while trainer.episodes_done < exp.episodes:
        chunk = min(exp.eval_every, exp.episodes - trainer.episodes_done)
        trainer.train(episodes=chunk)
        point = _evaluate_splits(exp, trainer, eval_env, final=False)["train"]
        curve.append({"episode": trainer.episodes_done, **{m: point[m] for m in METRICS}})
        _write_csv(run_dir / CURVE_FILE, curve, curve_fields)
        log(f"  {agent:<10} seed {seed}  episode {trainer.episodes_done:>5}  "
            f"greedy success {point['success']:.2f}  mistakes {point['mistakes']:.1f}")

    splits = _evaluate_splits(exp, trainer, eval_env, final=True)
    final = {split: {m: res[m] for m in METRICS} for split, res in splits.items()}
    final_path.write_text(json.dumps({**final, "episodes_trained": trainer.episodes_done},
                                     indent=2))
    return final


def _quiet(_msg: str) -> None:
    pass


def run_experiment(exp: Experiment, root: PathLike = "runs/experiments",
                   log: Callable[[str], None] = print, workers: int = 1) -> dict:
    """Run (or resume) every agent x seed, then aggregate.

    Writes to <root>/<exp.name>/: experiment.json, one run directory per
    agent and seed, results.csv (final greedy metrics per run and split),
    curves.csv (greedy learning curves), summary.csv (mean and 95% CI) and,
    for DQN agents, buffer_diagnostics.csv.

    workers > 1 trains runs in that many processes first; each run is
    independent and seeded, so the results are the same as with workers=1.
    """
    exp_dir = Path(root) / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.json").write_text(json.dumps(asdict(exp), indent=2))
    log(f"Experiment {exp.name}: {exp.description}")
    if workers > 1:
        # Fresh worker processes ("spawn"), not fork: forking after PyTorch has
        # started its thread pool can deadlock on Linux. Windows always spawns.
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
            jobs = {pool.submit(run_single, exp, agent, seed, exp_dir, _quiet): (agent, seed)
                    for agent in exp.agents for seed in exp.seeds}
            for job, (agent, seed) in jobs.items():
                final = job.result()
                log(f"  {agent:<15} seed {seed}  finished: train success "
                    f"{final['train']['success']:.2f}")

    results, curves = [], []
    for agent in exp.agents:
        for seed in exp.seeds:
            final = run_single(exp, agent, seed, exp_dir, log)
            for split, metrics in final.items():
                results.append({"experiment": exp.name, "agent": agent, "seed": seed,
                                "split": split, **metrics})
            for row in _read_csv(exp_dir / f"{agent}_s{seed}" / CURVE_FILE):
                curves.append({"agent": agent, "seed": seed, **row})

    _write_csv(exp_dir / "results.csv", results, ["experiment", "agent", "seed", "split"] + METRICS)
    _write_csv(exp_dir / "curves.csv", curves, ["agent", "seed", "episode"] + METRICS)
    summary = summarize(results)
    summary_fields = ["agent", "split", "n_seeds"] + [f"{m}_{s}" for m in METRICS
                                                      for s in ("mean", "ci")] + \
        ["steps_to_goal_n"]
    _write_csv(exp_dir / "summary.csv", summary, summary_fields)
    buffers = buffer_diagnostics(exp, exp_dir)
    if buffers:
        _write_csv(exp_dir / "buffer_diagnostics.csv", buffers, list(buffers[0]))
    return {"experiment": exp, "results": results, "curves": curves, "summary": summary,
            "buffers": buffers, "dir": exp_dir}


def buffer_diagnostics(exp: Experiment, exp_dir: Path) -> List[dict]:
    """Replay-buffer use at the end of each DQN run, read from its checkpoint.

    Shows whether the main buffer overwrote old transitions and, for
    mistake-aware replay, how many reserved mistakes only the reserve still
    holds. Runs without a saved buffer (or without replay) are skipped.
    """
    rows = []
    for agent in exp.agents:
        for seed in exp.seeds:
            path = exp_dir / f"{agent}_s{seed}" / CHECKPOINT_FILE
            if not path.exists():
                continue
            state = _read_checkpoint(path)["agent"]
            buf = state.get("buffer")
            if buf is None:
                continue
            row = {"agent": agent, "seed": seed}
            if "main" in buf:
                stamps = np.asarray(buf["reserve_added_at"])
                added, main_size = int(buf["added"]), int(buf["main"]["size"])
                first_in_main = added - main_size
                row.update(transitions_added=added, main_capacity=buf["main"]["capacity"],
                           main_size=main_size, main_overwritten=first_in_main,
                           reserve_capacity=buf["reserve"]["capacity"],
                           reserve_size=len(stamps), mistakes_seen=buf["mistakes_seen"],
                           reserve_overwritten=buf["mistakes_seen"] - len(stamps),
                           reserve_span=int(added - stamps.min()) if len(stamps) else 0,
                           reserve_only=int((stamps < first_in_main).sum()))
            else:
                added, main_size = int(state["env_steps"]), int(buf["size"])
                row.update(transitions_added=added, main_capacity=buf["capacity"],
                           main_size=main_size, main_overwritten=added - main_size)
            rows.append(row)
    fields = ["agent", "seed", "transitions_added", "main_capacity", "main_size",
              "main_overwritten", "reserve_capacity", "reserve_size", "mistakes_seen",
              "reserve_overwritten", "reserve_span", "reserve_only"]
    return [{f: r.get(f, "") for f in fields} for r in rows]


def load_experiment(exp_dir: PathLike) -> dict:
    """Read a finished experiment's results, curves and summary from disk."""
    exp_dir = Path(exp_dir)
    results = _read_csv(exp_dir / "results.csv")
    return {"experiment": Experiment(**json.loads((exp_dir / "experiment.json").read_text())),
            "results": results, "curves": _read_csv(exp_dir / "curves.csv"),
            "summary": summarize(results), "dir": exp_dir}


def _values(runs: List[dict], metric: str) -> List[float]:
    """One value per run, skipping runs where the metric is undefined (None)
    or absent (results saved before the metric existed)."""
    return [float(r[metric]) for r in runs if r.get(metric) is not None]


def summarize(results: List[dict]) -> List[dict]:
    """One row per (agent, split): each metric's mean and 95% CI over seeds.

    steps_to_goal averages only the seeds that reached the goal at least once;
    steps_to_goal_n says how many seeds that is.
    """
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in results:
        groups.setdefault((r["agent"], r["split"]), []).append(r)
    rows = []
    for (agent, split), runs in groups.items():
        row = {"agent": agent, "split": split, "n_seeds": len(runs)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_ci"] = mean_ci(_values(runs, m))
        row["steps_to_goal_n"] = len(_values(runs, "steps_to_goal"))
        rows.append(row)
    return rows


def curve_means(curves: List[dict], metric: str = "success") -> List[dict]:
    """Per (agent, seed): the metric averaged over every greedy evaluation
    during training except the one before training starts. For success this
    is the area under the learning curve: higher means it learned sooner."""
    groups: Dict[tuple, List[float]] = {}
    for row in curves:
        if row["episode"] > 0 and row.get(metric) is not None:
            groups.setdefault((row["agent"], int(row["seed"])), []).append(float(row[metric]))
    return [{"agent": agent, "seed": seed, "split": "curve", metric: float(np.mean(vals))}
            for (agent, seed), vals in sorted(groups.items())]


def compare_agents(results: List[dict], a: str, b: str, metric: str = "success",
                   split: str = "train") -> dict:
    """Difference in a metric between two agents with a permutation p-value."""
    va = _values([r for r in results if r["agent"] == a and r["split"] == split], metric)
    vb = _values([r for r in results if r["agent"] == b and r["split"] == split], metric)
    if not va or not vb:
        raise ValueError(f"no results for {a!r} and/or {b!r} on split {split!r}")
    return {"a": a, "b": b, "metric": metric, "split": split,
            "mean_a": float(np.mean(va)), "mean_b": float(np.mean(vb)),
            "difference": float(np.mean(va) - np.mean(vb)),
            "p_value": permutation_test(va, vb)}


def format_summary(summary: List[dict],
                   metrics: Sequence[str] = ("success", "path_efficiency", "steps",
                                             "mistakes", "hazards")) -> str:
    """Plain-text table: mean +/- 95% CI per agent and split."""
    header = f"{'agent':<16}{'split':<8}{'seeds':>6}" + "".join(f"{m:>20}" for m in metrics)
    lines = [header, "-" * len(header)]
    for row in summary:
        cells = []
        for m in metrics:
            mean, ci = row[f"{m}_mean"], row[f"{m}_ci"]
            if math.isnan(mean):
                cells.append("-")
            else:
                cells.append(f"{mean:.2f} +/- {ci:.2f}" if not math.isnan(ci) else f"{mean:.2f}")
        lines.append(f"{row['agent']:<16}{row['split']:<8}{row['n_seeds']:>6}"
                     + "".join(f"{c:>20}" for c in cells))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    from src import fun  # imported here: src.fun depends on src.training

    parser = argparse.ArgumentParser(description="Run a MIND experiment (resumable).")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--root", default="runs/experiments")
    parser.add_argument("--seeds", type=int, default=None,
                        help="use seeds 0..N-1: fewer is quicker, more gives narrower "
                             "intervals and smaller attainable p-values")
    parser.add_argument("--agents", nargs="+", default=None, help="subset of agents to run")
    parser.add_argument("--workers", type=int, default=1,
                        help="train runs in this many processes (same results as 1)")
    args = parser.parse_args(argv)

    exp = EXPERIMENTS[args.experiment]
    if args.seeds is not None:
        exp = Experiment(**{**asdict(exp), "seeds": tuple(range(args.seeds))})
    if args.agents is not None:
        exp = Experiment(**{**asdict(exp), "agents": tuple(args.agents)})
    out = run_experiment(exp, args.root, workers=args.workers)

    print("\nGreedy evaluation, mean +/- 95% CI across seeds:")
    print(format_summary(out["summary"]))
    print("\nEpisode reward, steps to goal (successful episodes only) and mistake types:")
    print(format_summary(out["summary"], ("total_reward", "steps_to_goal", "hazard_rate",
                                          "collisions", "revisits")))
    n_goal = {(r["agent"], r["split"]): r["steps_to_goal_n"] for r in out["summary"]}
    missing = {k: v for k, v in n_goal.items() if v < len(exp.seeds)}
    if missing:
        print("steps_to_goal leaves out seeds that never reached the goal:",
              ", ".join(f"{a}/{s}: {n} of {len(exp.seeds)} seeds" for (a, s), n in missing.items()))
    agents = list(exp.agents)
    splits = ["train"] + (["unseen"] if exp.unseen_layouts is not None else [])
    print("\nSuccess-rate differences (permutation test):")
    for split in splits:
        for a, b in itertools.combinations(agents, 2):
            if len(exp.seeds) < 2:
                continue
            c = compare_agents(out["results"], b, a, "success", split)
            print(f"  [{split}] {b} - {a}: {c['difference']:+.2f}  ({format_p(c['p_value'])})")
            print(f"           {fun.verdict(c['p_value'], c['difference'], b, a)}")
    speed = curve_means(out["curves"])
    if len(exp.seeds) >= 2:
        print("\nLearning speed: greedy success averaged over training checkpoints "
              "(permutation test):")
        for a, b in itertools.combinations(agents, 2):
            c = compare_agents(speed, b, a, "success", "curve")
            print(f"  {b} {c['mean_a']:.3f} vs {a} {c['mean_b']:.3f}: "
                  f"difference {c['difference']:+.3f}  ({format_p(c['p_value'])})")
    print("\nOther differences (permutation test):")
    for metric in ("total_reward", "steps_to_goal", "hazards", "hazard_rate", "collisions"):
        for split in splits:
            for a, b in itertools.combinations(agents, 2):
                if len(exp.seeds) < 2:
                    continue
                try:
                    c = compare_agents(out["results"], b, a, metric, split)
                except ValueError:  # no defined values (e.g. no seed reached the goal)
                    continue
                print(f"  [{split}] {metric:<13} {b} - {a}: {c['difference']:+.3f}  "
                      f"({format_p(c['p_value'])})")
    if out["buffers"]:
        print("\nReplay buffer at the end of training (transitions; mean over seeds):")
        for agent in exp.agents:
            rows = [r for r in out["buffers"] if r["agent"] == agent]
            if not rows:
                continue
            line = (f"  {agent:<15} added {np.mean([r['transitions_added'] for r in rows]):>9.0f}"
                    f"  main overwrote {np.mean([r['main_overwritten'] for r in rows]):>9.0f}")
            if rows[0]["reserve_size"] != "":
                line += (f"  reserve {np.mean([r['reserve_size'] for r in rows]):>6.0f}, of which "
                         f"only in reserve {np.mean([r['reserve_only'] for r in rows]):>6.0f}")
            print(line)
    print(f"\nFiles written to {out['dir']}")


if __name__ == "__main__":
    main()
