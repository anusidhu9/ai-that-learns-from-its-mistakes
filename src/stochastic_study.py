"""Stage K: re-evaluate saved networks with a little randomness (no training).

Every network saved by the CNN, memory and PPO studies
(results/<study>/models/<arm>_s<seed>.pt, 80 in all) plays its study's held-out
mazes in several modes:

* greedy       - the best move (eval seed 10000 + run seed, one pass). Must
                 reproduce the study's saved held-out results exactly; the run
                 stops if it does not.
* eps0.05/0.1/0.2 - the best move, replaced by a uniformly random move with
                 that probability (eval seed 20000 + run seed, 5 passes).
* sample       - PPO only: moves drawn from the policy (seed 20000 + run seed,
                 5 passes).

The networks are never changed. Pre-registered in
results/stochastic_evaluation/PROTOCOL.md.

    python -m src.stochastic_study --out runs/experiments_stochastic --workers 2

Resumable: each network's results go to <out>/raw/<study>__<arm>_s<seed>.json
and finished networks are skipped.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src.evaluation import EVAL_SEED_OFFSET, EXPERIMENTS, METRICS, evaluate_agent
from src.training import load_agent

STUDIES: Dict[str, Tuple[str, ...]] = {
    "cnn_generalization": ("dqn", "dqn_cnn", "dqn_wide"),
    "rnn_generalization": ("dqn", "dqn_rnn", "dqn_wide"),
    "ppo_generalization": ("dqn", "ppo"),
}
STOCHASTIC_SEED_OFFSET = 20_000
REPEATS = 5
# label -> (policy, epsilon, repeats)
MODES: Dict[str, Tuple[str, float, int]] = {
    "greedy": ("greedy", 0.0, 1),
    "eps0.05": ("epsilon", 0.05, REPEATS),
    "eps0.1": ("epsilon", 0.1, REPEATS),
    "eps0.2": ("epsilon", 0.2, REPEATS),
    "sample": ("sample", 0.0, REPEATS),
}
# greedy results must match the study's saved results.csv on these
CHECKED = ("success", "steps", "total_reward", "collisions", "revisits", "timeout")
EPISODE_FIELDS = ("layout_seed", "success", "steps", "total_reward", "collisions",
                  "revisits", "timeout", "optimal_length", "path_efficiency")


def networks(results_root: Path = Path("results")) -> List[Tuple[str, str, int, Path]]:
    """(study, arm, seed, model path) for every saved network, in a fixed order."""
    out = []
    for study, arms in STUDIES.items():
        for arm in arms:
            for seed in EXPERIMENTS[study].seeds:
                out.append((study, arm, seed, results_root / study / "models" / f"{arm}_s{seed}.pt"))
    return out


def modes_for(arm: str) -> List[str]:
    return [m for m in MODES if m != "sample" or arm == "ppo"]


def saved_heldout(study: str, arm: str, seed: int,
                  results_root: Path = Path("results")) -> dict:
    with open(results_root / study / "results.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["agent"] == arm and int(row["seed"]) == seed and row["split"] == "unseen":
                return row
    raise KeyError((study, arm, seed))


def evaluate_network(study: str, arm: str, seed: int, model: Path,
                     results_root: Path = Path("results"),
                     layouts: Optional[List[int]] = None) -> dict:
    """All modes for one network; checks greedy against the saved results."""
    torch.set_num_threads(1)
    agent, env, _ = load_agent(model)
    layouts = list(EXPERIMENTS[study].unseen_layouts) if layouts is None else layouts
    out = {"study": study, "agent": arm, "seed": seed, "modes": {}}
    for label in modes_for(arm):
        policy, epsilon, repeats = MODES[label]
        offset = EVAL_SEED_OFFSET if policy == "greedy" else STOCHASTIC_SEED_OFFSET
        res = evaluate_agent(agent, env, layout_seeds=layouts, seed=offset + seed,
                             policy=policy, epsilon=epsilon, repeats=repeats)
        episodes = [{k: float(e[k]) if k == "path_efficiency" else int(e[k])
                     for k in EPISODE_FIELDS} for e in res["episodes"]]
        for i, e in enumerate(episodes):
            e["repeat"] = i // len(layouts)
        out["modes"][label] = {"metrics": {m: res[m] for m in METRICS},
                               "episodes": episodes}
    if results_root is not None:
        saved = saved_heldout(study, arm, seed, results_root)
        greedy = out["modes"]["greedy"]["metrics"]
        for m in CHECKED:
            if abs(greedy[m] - float(saved[m])) > 1e-9:
                raise RuntimeError(f"{study} {arm} s{seed}: greedy {m} {greedy[m]} does not "
                                   f"reproduce the saved {saved[m]}")
    return out


def _job(args) -> str:
    study, arm, seed, model, raw_dir = args
    path = Path(raw_dir) / f"{study}__{arm}_s{seed}.json"
    if path.exists():
        return f"skip {path.name}"
    result = evaluate_network(study, arm, seed, Path(model))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    s = {k: round(v["metrics"]["success"], 3) for k, v in result["modes"].items()}
    return f"done {study} {arm} s{seed}: {s}"


def collect(out_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Per network x mode metrics, and every episode."""
    rows, episodes = [], []
    for study, arm, seed, _ in networks():
        data = json.loads((out_dir / "raw" / f"{study}__{arm}_s{seed}.json").read_text())
        for label, block in data["modes"].items():
            rows.append({"study": study, "agent": arm, "seed": seed, "mode": label,
                         **block["metrics"]})
            for e in block["episodes"]:
                episodes.append({"study": study, "agent": arm, "seed": seed, "mode": label, **e})
    return rows, episodes


def write_tables(out_dir: Path) -> None:
    rows, episodes = collect(out_dir)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["study", "agent", "seed", "mode"] + METRICS)
        w.writeheader()
        w.writerows(rows)
    fields = ["study", "agent", "seed", "mode", "repeat"] + list(EPISODE_FIELDS)
    with gzip.open(out_dir / "episodes.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(episodes)


def run(out_dir: Path, workers: int = 1, log=print) -> None:
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = [(st, arm, seed, str(model), str(raw)) for st, arm, seed, model in networks()]
    missing = [j for j in jobs if not Path(j[3]).exists()]
    if missing:
        raise FileNotFoundError(f"missing saved networks: {[j[3] for j in missing]}")
    log(f"{len(jobs)} networks, modes {list(MODES)}, workers {workers}")
    if workers <= 1:
        for j in jobs:
            log(_job(j))
    else:
        # spawn: forked workers can deadlock with torch's thread pools
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for msg in pool.map(_job, jobs):
                log(msg)
    write_tables(out_dir)
    log(f"wrote {out_dir / 'results.csv'} and {out_dir / 'episodes.csv.gz'}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="runs/experiments_stochastic")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)
    run(Path(args.out), workers=args.workers, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
