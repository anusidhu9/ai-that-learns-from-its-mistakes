"""Pre-planned analysis for memory_vs_replay (see PROTOCOL.md). Written before the run.

    PYTHONPATH=. python results/memory_vs_replay/analyze.py [experiment dir]

The directory (default runs/experiments_memory_replay/memory_vs_replay) is the
experiment's output from `python -m src.evaluation --experiment memory_vs_replay`,
including each run's checkpoint.pt (used for the epsilon = 0.05 evaluation).
"""
import json
import sys
from pathlib import Path

import numpy as np

from src.evaluation import (EVAL_SEED_OFFSET, EXPERIMENTS, compare_agents, curve_means,
                            evaluate_agent, load_experiment, mean_ci, permutation_test)
from src.stochastic_study import STOCHASTIC_SEED_OFFSET
from src.training import load_agent, load_history

EXP = EXPERIMENTS["memory_vs_replay"]
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("runs/experiments_memory_replay/memory_vs_replay")
ARMS = EXP.agents
N_RESAMPLES = 200_000   # exact: C(20, 10) = 184,756 relabellings

out = load_experiment(DIR)
res, curves = out["results"], out["curves"]
for r in res:
    r["revisit_rate"] = r["revisits"] / r["steps"]
speed = curve_means(curves)


def per_seed(rows, agent, split, metric="success"):
    d = {r["seed"]: r[metric] for r in rows if r["agent"] == agent and r["split"] == split}
    return np.array([d[s] for s in EXP.seeds])


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def line(rows, a, b, metric, split, label):
    va, vb = per_seed(rows, a, split, metric), per_seed(rows, b, split, metric)
    p = permutation_test(va, vb, n_resamples=N_RESAMPLES)
    print(f"{label}: {a} {va.mean():.3f} vs {b} {vb.mean():.3f}  diff {va.mean() - vb.mean():+.3f}"
          f"  p {p:.4f}  higher/tied/lower {(va > vb).sum()}/{(va == vb).sum()}/{(va < vb).sum()}")
    return p


PAIRS = [("dqn_rnn", "dqn_rnn_nomem", "H1 memory itself"),
         ("dqn_batch512", "dqn", "H2 more steps per update (MLP)"),
         ("dqn_rnn", "dqn", "H3 Stage F replication")]

print("Primary (Holm over H1-H3), held-out greedy success:")
ps = [line(res, a, b, "success", "unseen", label) for a, b, label in PAIRS]
print("Holm-adjusted (H1, H2, H3):", [round(p, 4) for p in holm(ps)])

print("\nExploratory: other held-out success pairs:")
for a, b in [("dqn_rnn_nomem", "dqn"), ("dqn_rnn_nomem", "dqn_batch512"),
             ("dqn_rnn", "dqn_batch512")]:
    line(res, a, b, "success", "unseen", "  held-out success")

print("\nExploratory: primary pairs on other measures:")
for a, b, label in PAIRS:
    line(speed, a, b, "success", "curve", f"  learning speed ({label})")
    line(res, a, b, "success", "train", f"  final train success ({label})")
    for m in ("revisits", "revisit_rate", "steps", "timeout", "collisions", "path_efficiency"):
        line(res, a, b, m, "unseen", f"  held-out {m} ({label})")

print("\nSummary (mean +/- CI over seeds):")
METRICS = ("success", "path_efficiency", "steps", "timeout", "collisions", "revisits",
           "revisit_rate", "total_reward")
for split in ("unseen", "train"):
    for agent in ARMS:
        parts = []
        for m in METRICS:
            mean, ci = mean_ci(per_seed(res, agent, split, m))
            parts.append(f"{m} {mean:.3f}+/-{ci:.3f}")
        print(f"  [{split}] {agent:<13} " + "  ".join(parts))
for agent in ARMS:
    print(f"  [unseen] {agent:<13} per seed {per_seed(res, agent, 'unseen').tolist()}")
    mean, ci = mean_ci(per_seed(speed, agent, "curve"))
    print(f"  [speed]  {agent:<13} {mean:.3f}+/-{ci:.3f}")

print("\nCurve (mean greedy train success per eval point):")
for agent in ARMS:
    pts = {}
    for r in curves:
        if r["agent"] == agent:
            pts.setdefault(int(r["episode"]), []).append(r["success"])
    print(f"  {agent:<13}", "  ".join(f"{e}:{np.mean(v):.2f}" for e, v in sorted(pts.items())))

print("\nCompute: env steps, updates and wall-clock per run (config.json to final_eval.json):")
for agent in ARMS:
    steps, minutes = [], []
    for s in EXP.seeds:
        run = DIR / f"{agent}_s{s}"
        steps.append(sum(r["steps"] for r in load_history(run)))
        minutes.append(((run / "final_eval.json").stat().st_mtime
                        - (run / "config.json").stat().st_mtime) / 60)
    print(f"  {agent:<13} mean env steps {np.mean(steps):,.0f}, minutes per run "
          f"{np.mean(minutes):.1f} (min {min(minutes):.1f}, max {max(minutes):.1f})")

print("\nSecondary (pre-specified, exploratory): held-out success with epsilon = 0.05")
print("(5 passes, evaluation seed 20,000 + run seed; greedy re-checked against saved results)")
eps = {a: [] for a in ARMS}
for agent in ARMS:
    for seed in EXP.seeds:
        model, env, _ = load_agent(DIR / f"{agent}_s{seed}" / "checkpoint.pt")
        g = evaluate_agent(model, env, layout_seeds=EXP.unseen_layouts,
                           seed=EVAL_SEED_OFFSET + seed)
        saved = json.loads((DIR / f"{agent}_s{seed}" / "final_eval.json").read_text())
        assert abs(g["success"] - saved["unseen"]["success"]) < 1e-9, (agent, seed)
        e = evaluate_agent(model, env, layout_seeds=EXP.unseen_layouts,
                           seed=STOCHASTIC_SEED_OFFSET + seed, policy="epsilon",
                           epsilon=0.05, repeats=5)
        eps[agent].append(e["success"])
eps_rows = [{"agent": a, "seed": s, "split": "eps", "success": v}
            for a in ARMS for s, v in zip(EXP.seeds, eps[a])]
for agent in ARMS:
    mean, ci = mean_ci(eps[agent])
    print(f"  {agent:<13} {mean:.3f}+/-{ci:.3f}  per seed {[round(v, 3) for v in eps[agent]]}")
for a, b, label in PAIRS:
    line(eps_rows, a, b, "success", "eps", f"  eps 0.05 {label}")
line(eps_rows, "dqn_rnn_nomem", "dqn", "success", "eps", "  eps 0.05 nomem vs dqn")
with open(DIR / "eps005.json", "w") as f:
    json.dump(eps, f)
