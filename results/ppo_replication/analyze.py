"""Pre-planned analysis for ppo_replication (see PROTOCOL.md). Written before the run.

    PYTHONPATH=. python results/ppo_replication/analyze.py [experiment dir]

The directory (default runs/experiments_ppo_replication/ppo_replication) is the
experiment's output from `python -m src.evaluation --experiment ppo_replication`,
including each run's checkpoint.pt (used for the re-evaluations). Stage I's saved
results (results/ppo_generalization/results.csv) are read for the pooled summary.
"""
import json
import sys
from pathlib import Path

import numpy as np

from src.evaluation import (EVAL_SEED_OFFSET, EXPERIMENTS, curve_means, evaluate_agent,
                            load_experiment, mean_ci, permutation_test)
from src.stochastic_study import STOCHASTIC_SEED_OFFSET
from src.training import load_agent, load_history

EXP = EXPERIMENTS["ppo_replication"]
ORIG = EXPERIMENTS["ppo_generalization"]
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("runs/experiments_ppo_replication/ppo_replication")
STAGE_I = Path("results/ppo_generalization")
ARMS = EXP.agents
N_RESAMPLES = 200_000   # exact: C(20, 10) = 184,756 relabellings

out = load_experiment(DIR)
res, curves = out["results"], out["curves"]
speed = curve_means(curves)


def per_seed(rows, agent, split, metric="success", seeds=EXP.seeds):
    d = {r["seed"]: r[metric] for r in rows if r["agent"] == agent and r["split"] == split}
    return np.array([d[s] for s in seeds])


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def line(rows, metric, split, label, a="ppo", b="dqn"):
    va, vb = per_seed(rows, a, split, metric), per_seed(rows, b, split, metric)
    p = permutation_test(va, vb, n_resamples=N_RESAMPLES)
    print(f"{label}: {a} {va.mean():.3f} vs {b} {vb.mean():.3f}  diff {va.mean() - vb.mean():+.3f}"
          f"  p {p:.4f}  higher/tied/lower {(va > vb).sum()}/{(va == vb).sum()}/{(va < vb).sum()}")
    return p, va.mean() - vb.mean()


print("Primary (Holm over H1-H2):")
p1, d1 = line(res, "success", "unseen", "H1 held-out success")
p2, d2 = line(speed, "success", "curve", "H2 learning speed")
adj = holm([p1, p2])
print("Holm-adjusted (H1, H2):", [round(p, 4) for p in adj])
for name, p, d, want in (("H1", adj[0], d1, +1), ("H2", adj[1], d2, -1)):
    if p < 0.05 and np.sign(d) == want:
        verdict = "REPLICATED (significant, same direction as Stage I)"
    elif p < 0.05:
        verdict = "REVERSAL (significant, opposite direction to Stage I)"
    else:
        verdict = "NOT REPLICATED (no evidence of a difference in this replication)"
    print(f"  {name}: {verdict}")

print("\nPooled with Stage I (pre-specified descriptive summary, not a test):")
orig_rows = load_experiment(STAGE_I)
orig_speed = curve_means(orig_rows["curves"])
for label, new_rows, old_rows, split in (("held-out success", res, orig_rows["results"], "unseen"),
                                         ("learning speed", speed, orig_speed, "curve")):
    diffs = np.concatenate([
        per_seed(new_rows, "ppo", split) - per_seed(new_rows, "dqn", split),
        per_seed(old_rows, "ppo", split, seeds=ORIG.seeds)
        - per_seed(old_rows, "dqn", split, seeds=ORIG.seeds)])
    mean, ci = mean_ci(diffs)
    print(f"  {label}: PPO minus DQN over 20 seed pairs {mean:+.3f} +/- {ci:.3f} "
          f"(replication {diffs[:10].mean():+.3f}, Stage I {diffs[10:].mean():+.3f})")

print("\nExploratory: other measures (no correction):")
line(res, "success", "train", "  final train success")
for m in ("path_efficiency", "steps", "timeout", "collisions", "revisits"):
    line(res, m, "unseen", f"  held-out {m}")
for ep in sorted({int(r["episode"]) for r in curves if int(r["episode"]) > 0}):
    rows = [r for r in curves if int(r["episode"]) == ep]
    line([{**r, "split": "at"} for r in rows], "success", "at", f"  train success at {ep}")

print("\nSummary (mean +/- CI over seeds):")
for split in ("unseen", "train"):
    for agent in ARMS:
        parts = []
        for m in ("success", "path_efficiency", "steps", "timeout", "collisions", "revisits",
                  "total_reward"):
            mean, ci = mean_ci(per_seed(res, agent, split, m))
            parts.append(f"{m} {mean:.3f}+/-{ci:.3f}")
        print(f"  [{split}] {agent:<4} " + "  ".join(parts))
for agent in ARMS:
    print(f"  [unseen] {agent:<4} per seed {per_seed(res, agent, 'unseen').tolist()}")
    mean, ci = mean_ci(per_seed(speed, agent, "curve"))
    print(f"  [speed]  {agent:<4} {mean:.3f}+/-{ci:.3f}  per seed "
          f"{[round(float(v), 3) for v in per_seed(speed, agent, 'curve')]}")
print("\nCurve (mean greedy train success per eval point):")
for agent in ARMS:
    pts = {}
    for r in curves:
        if r["agent"] == agent:
            pts.setdefault(int(r["episode"]), []).append(r["success"])
    print(f"  {agent:<4}", "  ".join(f"{e}:{np.mean(v):.2f}" for e, v in sorted(pts.items())))

print("\nCompute: env steps and wall-clock per run (config.json to final_eval.json):")
for agent in ARMS:
    steps, minutes = [], []
    for s in EXP.seeds:
        run = DIR / f"{agent}_s{s}"
        steps.append(sum(r["steps"] for r in load_history(run)))
        minutes.append(((run / "final_eval.json").stat().st_mtime
                        - (run / "config.json").stat().st_mtime) / 60)
    print(f"  {agent:<4} mean env steps {np.mean(steps):,.0f}, minutes per run "
          f"{np.mean(minutes):.1f} (min {min(minutes):.1f}, max {max(minutes):.1f})")

print("\nSecondary (pre-specified, exploratory): re-evaluation of the final networks")
print("(greedy re-checked against saved results; 5 passes, evaluation seed 20,000 + run seed)")
modes = {"dqn": [("eps0.05", "epsilon")], "ppo": [("eps0.05", "epsilon"), ("sample", "sample")]}
split_vals = {a: {"short": [], "long": []} for a in ARMS}
stoch, shortest = [], None
for agent in ARMS:
    for seed in EXP.seeds:
        run = DIR / f"{agent}_s{seed}"
        model, env, _ = load_agent(run / "checkpoint.pt")
        g = evaluate_agent(model, env, layout_seeds=EXP.unseen_layouts,
                           seed=EVAL_SEED_OFFSET + seed)
        saved = json.loads((run / "final_eval.json").read_text())["unseen"]["success"]
        assert abs(g["success"] - saved) < 1e-9, (agent, seed, g["success"], saved)
        eps = g["episodes"]
        shortest = min(e["optimal_length"] for e in eps)
        split_vals[agent]["short"].append(
            np.mean([e["success"] for e in eps if e["optimal_length"] == shortest]))
        split_vals[agent]["long"].append(
            np.mean([e["success"] for e in eps if e["optimal_length"] > shortest]))
        for name, policy in modes[agent]:
            e = evaluate_agent(model, env, layout_seeds=EXP.unseen_layouts,
                               seed=STOCHASTIC_SEED_OFFSET + seed, policy=policy,
                               epsilon=0.05 if policy == "epsilon" else 0.0, repeats=5)
            stoch.append({"agent": f"{agent}_{name}", "seed": seed, "split": "stoch",
                          "success": e["success"]})
n_short = sum(e["optimal_length"] == shortest for e in eps)
for arm in ("dqn_eps0.05", "ppo_eps0.05", "ppo_sample"):
    mean, ci = mean_ci(per_seed(stoch, arm, "stoch"))
    print(f"  {arm:<12} {mean:.3f}+/-{ci:.3f}  per seed "
          f"{[round(float(v), 3) for v in per_seed(stoch, arm, 'stoch')]}")
line(stoch, "success", "stoch", "  S1 PPO sampled vs DQN eps 0.05", "ppo_sample", "dqn_eps0.05")
line(stoch, "success", "stoch", "  S2 PPO eps 0.05 vs DQN eps 0.05", "ppo_eps0.05", "dqn_eps0.05")
print(f"\nLong vs short held-out mazes ({n_short} with the minimum path length {shortest}, "
      f"{len(eps) - n_short} longer):")
for part in ("short", "long"):
    rows = [{"agent": ag, "seed": s, "split": part, "success": v}
            for ag in ARMS for s, v in zip(EXP.seeds, split_vals[ag][part])]
    line(rows, "success", part, f"  {part}")
with open(DIR / "secondary.json", "w") as f:
    json.dump({"stochastic": stoch, "split": split_vals, "shortest": shortest}, f)
