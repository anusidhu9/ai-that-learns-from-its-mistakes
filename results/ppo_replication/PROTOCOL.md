# PPO against DQN, replication: pre-registered protocol

Covers one experiment, `ppo_replication`. Written 2026-11-10, **before** any run of it.
This file is not edited after results exist; anything that changes is recorded under
"Deviations" in `ANALYSIS.md`.

## Question

In Stage I ([`ppo_generalization`](../ppo_generalization/ANALYSIS.md), seeds 700–709), PPO
solved slightly more held-out mazes than DQN (0.69 vs 0.64, Holm-adjusted p = 0.037, ahead
in 8 of 10 seeds) and learned its training mazes more slowly (learning speed 0.826 vs
0.896, Holm-adjusted p = 0.0003, behind in 10 of 10). The held-out result was small and
only just significant, so the report says it "needs replicating". This study repeats
Stage I exactly with fresh seeds and fresh mazes:

1. Does PPO again solve more held-out mazes than DQN?
2. Does PPO again learn its training mazes more slowly?

## Arms

Exactly the two arms and settings of Stage I; nothing is changed or tuned.

| Arm | Algorithm | Network | Settings |
|---|---|---|---|
| `dqn` | Double DQN, uniform replay | MLP 77 → 128 → 128 → 4 | as in Stages E, F, I and L (batch 64, buffer 20,000, Adam 1e-3, train every 2 steps, target every 100 updates, ε 1.0 → 0.05 over 2,000 episodes) |
| `ppo` | PPO, clipped objective, GAE | actor 77 → 128 → 128 → 4 (tanh) and separate critic | as in Stage I (rollout 2,048, 4 epochs, minibatch 256, clip 0.2, λ 0.95, Adam 3e-4, entropy 0.01, value 0.5, grad clip 0.5; no ε) |

## Task, layouts and seeds

The same task as Stage I: the `maze` preset (9×9, loop_prob 0.1, no hazards, max 100
steps) with a 5×5 local view plus the direction to the goal, 3,000 training episodes.

- **Training layouts**: 10100–10149 (50 distinct mazes).
- **Held-out layouts**: 10200–10250 except 10218 (50 distinct mazes). Layout 10218
  generates the same maze as training layout 10149, so it is left out; no held-out maze
  repeats a training maze or another held-out maze (checked before writing this protocol
  and by a test). Because 9×9 mazes are limited, 6 of the training mazes and 11 of the
  held-out mazes also appeared in earlier studies; this does not matter here, since every
  network in this study is trained from scratch.
- **Seeds**: 1100–1109 (10 per arm), never used before. Evaluation seed = 10,000 + run seed.
- Greedy evaluation on all 50 training layouts every 500 episodes; the held-out layouts are
  played once, greedily, at the end.

## Primary outcomes and tests

Each test compares 10 vs 10 per-seed values with a two-sided permutation test (200,000
resamples, exact: C(20,10) = 184,756). H1 and H2 are corrected together with
Holm-Bonferroni at α = 0.05.

- **H1** (replication of Stage I H1): final greedy success on the 50 held-out layouts,
  `ppo` vs `dqn`.
- **H2** (replication of Stage I H2): learning speed, the mean of greedy success on the 50
  training layouts at episodes 500, 1,000, …, 3,000, `ppo` vs `dqn`.

**Reading the results.**
- H1 is reported as **replicated** only if it is significant after correction *and* PPO
  is ahead (the same direction as Stage I). If PPO is significantly *behind*, it is
  reported as a reversal. Otherwise it is reported as **not replicated**: "no evidence of
  a difference in this replication", never as "PPO is not better".
- H2 is read the same way (replicated only if PPO is significantly slower).
- In every case the Stage I result stays as it was; the README and report then describe
  the held-out difference using both studies. As a pre-specified descriptive summary (not a
  test), the per-seed differences of the two studies are pooled (20 seeds per arm) and
  reported as a mean with a 95% CI.

## Secondary (pre-specified, exploratory, no correction)

- **Play with randomness.** After the run, every final network plays the held-out mazes 5
  times (evaluation seed 20,000 + run seed, `evaluate_agent`, after its greedy result is
  re-checked against the saved one): DQN with ε = 0.05, PPO with ε = 0.05, and PPO sampling
  its policy. S1: PPO sampled vs DQN ε 0.05. S2: PPO ε 0.05 vs DQN ε 0.05. These repeat
  Stage K's comparisons for these networks.
- **Long vs short mazes.** In Stage I PPO's held-out advantage came entirely from the mazes
  longer than the minimum path length. H1's comparison is repeated separately on the
  held-out mazes with the minimum shortest-path length and on the longer ones.
- Success at each evaluation point; final training success; held-out path efficiency,
  steps, timeouts, collisions and revisits.
- Compute: environment steps and time per run.

## Compute and execution

CPU only (Windows laptop), runs in `runs/experiments_ppo_replication/`:
`python -m src.evaluation --experiment ppo_replication --root
runs/experiments_ppo_replication --workers 2`, started in its own PowerShell window
(outside Claude Code). Stage I took 46 minutes with the same arms and settings, so about
45–60 minutes are expected. Runs are resumable from their last 500-episode checkpoint with
the same command; any stop is recorded as a deviation. Results are copied to
`results/ppo_replication/` with an `ANALYSIS.md` written after the run, and every run's
final network is exported to `results/ppo_replication/models/`. The code in `src/` is not
changed while the experiment runs. The analysis script `results/ppo_replication/analyze.py`
is written before the run (and smoke-tested on a short copy of the experiment); it is not
changed afterwards except to fix errors, which are recorded as deviations.

## Code state

To be implemented after commit 06ef71d: the `ppo_replication` experiment (two arms, the
seeds and layouts above) and a test of its settings and duplicate-free layouts. No agent,
environment or training code changes. Before the run, 300 episodes of Stage I's DQN seed
700 and PPO seed 700 are re-run and must reproduce their saved logs exactly.
