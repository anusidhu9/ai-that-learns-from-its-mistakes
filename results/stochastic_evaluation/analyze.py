"""Pre-planned analysis for Stage K (see PROTOCOL.md). Written before the run.

    PYTHONPATH=. python results/stochastic_evaluation/analyze.py [results dir]

The directory (default runs/experiments_stochastic) must hold results.csv and
episodes.csv.gz written by `python -m src.stochastic_study`.
"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation import mean_ci, paired_permutation_test, permutation_test
from src.stochastic_study import STUDIES

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/experiments_stochastic")
N_RESAMPLES = 200_000
EPS_MODES = ("greedy", "eps0.05", "eps0.1", "eps0.2")
METRICS = ("success", "steps", "timeout", "collisions", "revisits", "path_efficiency")
SHORT = {"cnn_generalization": "E", "rnn_generalization": "F", "ppo_generalization": "I"}

with open(DIR / "results.csv", newline="") as f:
    ROWS = list(csv.DictReader(f))
V = {}  # (study, agent, mode, metric) -> {seed: value}
for r in ROWS:
    for m in METRICS:
        V.setdefault((r["study"], r["agent"], r["mode"], m), {})[int(r["seed"])] = float(r[m])


def vals(study, agent, mode, metric="success"):
    d = V[(study, agent, mode, metric)]
    return np.array([d[s] for s in sorted(d)])


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def counts(a, b):
    return f"higher/tied/lower {(a > b).sum()}/{(a == b).sum()}/{(a < b).sum()}"


def paired(label, a, b):
    p = paired_permutation_test(a, b, n_resamples=N_RESAMPLES)
    print(f"{label}: {a.mean():.3f} vs {b.mean():.3f}  diff {(a - b).mean():+.3f}  "
          f"(CI of diff +/-{mean_ci(a - b)[1]:.3f})  p {p:.4f}  {counts(a, b)}")
    return p


def unpaired(label, a, b):
    p = permutation_test(a, b, n_resamples=N_RESAMPLES)
    print(f"{label}: {a.mean():.3f} vs {b.mean():.3f}  diff {a.mean() - b.mean():+.3f}  "
          f"p {p:.4f}")
    return p


# ---------------------------------------------------------------------------
print("Primary (Holm over H1-H3), held-out success:")
dqn_g = np.concatenate([vals(s, "dqn", "greedy") for s in STUDIES])
dqn_e = np.concatenate([vals(s, "dqn", "eps0.05") for s in STUDIES])
p1 = paired("H1 dqn eps0.05 vs greedy (30 networks, paired)", dqn_e, dqn_g)
p2 = unpaired("H2 Stage I ppo sample vs dqn eps0.05",
              vals("ppo_generalization", "ppo", "sample"),
              vals("ppo_generalization", "dqn", "eps0.05"))
p3 = unpaired("H3 Stage F dqn_rnn eps0.05 vs dqn eps0.05",
              vals("rnn_generalization", "dqn_rnn", "eps0.05"),
              vals("rnn_generalization", "dqn", "eps0.05"))
print("Holm-adjusted (H1, H2, H3):", [round(p, 4) for p in holm([p1, p2, p3])])
for s in STUDIES:
    d = vals(s, "dqn", "eps0.05") - vals(s, "dqn", "greedy")
    print(f"  H1 by study {SHORT[s]}: mean diff {d.mean():+.3f}  {counts(d, 0 * d)}")

# ---------------------------------------------------------------------------
print("\nExploratory: success by mode (mean +/- 95% CI over seeds):")
for s, arms in STUDIES.items():
    for a in arms:
        modes = EPS_MODES + (("sample",) if a == "ppo" else ())
        parts = []
        for mode in modes:
            m, ci = mean_ci(vals(s, a, mode))
            parts.append(f"{mode} {m:.3f}+/-{ci:.3f}")
        print(f"  {SHORT[s]} {a:<9} " + "  ".join(parts))

print("\nExploratory: each arm, stochastic vs greedy (paired, uncorrected):")
for s, arms in STUDIES.items():
    for a in arms:
        for mode in ("eps0.05", "eps0.1", "eps0.2") + (("sample",) if a == "ppo" else ()):
            paired(f"  {SHORT[s]} {a:<9} {mode:<7} vs greedy", vals(s, a, mode), vals(s, a, "greedy"))

print("\nExploratory: earlier between-arm comparisons, greedy vs eps0.05 (unpaired, uncorrected):")
for s, a, b in [("cnn_generalization", "dqn_cnn", "dqn"), ("cnn_generalization", "dqn_wide", "dqn"),
                ("rnn_generalization", "dqn_rnn", "dqn"), ("rnn_generalization", "dqn_wide", "dqn"),
                ("ppo_generalization", "ppo", "dqn")]:
    for mode in ("greedy", "eps0.05"):
        unpaired(f"  {SHORT[s]} {a} vs {b} [{mode}]", vals(s, a, mode), vals(s, b, mode))

print("\nExploratory: other held-out metrics by mode (mean over seeds):")
for s, arms in STUDIES.items():
    for a in arms:
        modes = EPS_MODES + (("sample",) if a == "ppo" else ())
        for m in METRICS[1:]:
            print(f"  {SHORT[s]} {a:<9} {m:<15} " +
                  "  ".join(f"{mode} {vals(s, a, mode, m).mean():.2f}" for mode in modes))

# ---------------------------------------------------------------------------
# Episode-level breakdowns
eps_by = defaultdict(list)   # (study, agent, seed, mode) -> episodes
with gzip.open(DIR / "episodes.csv.gz", "rt", newline="") as f:
    for e in csv.DictReader(f):
        eps_by[(e["study"], e["agent"], int(e["seed"]), e["mode"])].append(e)

print("\nExploratory: success by shortest path (12 vs longer), mean over seeds:")
for s, arms in STUDIES.items():
    for a in arms:
        modes = ("greedy", "eps0.05") + (("sample",) if a == "ppo" else ())
        parts = []
        for mode in modes:
            for part, test in (("short", lambda L: L == 12), ("long", lambda L: L > 12)):
                per_seed = []
                for seed in sorted(V[(s, a, mode, "success")]):
                    sel = [int(e["success"]) for e in eps_by[(s, a, seed, mode)]
                           if test(int(e["optimal_length"]))]
                    per_seed.append(np.mean(sel))
                parts.append(f"{mode}/{part} {np.mean(per_seed):.3f}")
        print(f"  {SHORT[s]} {a:<9} " + "  ".join(parts))

print("\nExploratory: eps0.05 success on mazes the greedy policy solved vs timed out on:")
for s, arms in STUDIES.items():
    for a in arms:
        solved, failed = [], []
        for seed in sorted(V[(s, a, "greedy", "success")]):
            greedy_ok = {e["layout_seed"]: int(e["success"]) for e in eps_by[(s, a, seed, "greedy")]}
            st = eps_by[(s, a, seed, "eps0.05")]
            ok = [int(e["success"]) for e in st if greedy_ok[e["layout_seed"]]]
            bad = [int(e["success"]) for e in st if not greedy_ok[e["layout_seed"]]]
            if ok:
                solved.append(np.mean(ok))
            if bad:
                failed.append(np.mean(bad))
        print(f"  {SHORT[s]} {a:<9} greedy-solved mazes kept {np.mean(solved):.3f}   "
              f"greedy-failed mazes now solved {np.mean(failed):.3f}")

print("\nExploratory: spread of success between the 5 passes (mean over seeds of the SD):")
for s, arms in STUDIES.items():
    for a in arms:
        modes = ("eps0.05", "eps0.1", "eps0.2") + (("sample",) if a == "ppo" else ())
        parts = []
        for mode in modes:
            sds = []
            for seed in sorted(V[(s, a, mode, "success")]):
                by_rep = defaultdict(list)
                for e in eps_by[(s, a, seed, mode)]:
                    by_rep[e["repeat"]].append(int(e["success"]))
                sds.append(np.std([np.mean(v) for v in by_rep.values()], ddof=1))
            parts.append(f"{mode} {np.mean(sds):.3f}")
        print(f"  {SHORT[s]} {a:<9} " + "  ".join(parts))
