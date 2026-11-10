# Memory vs sequence replay: pre-registered protocol

Covers one experiment, `memory_vs_replay`. Written 2026-11-09, **before** any run of
it. The project owner approved the plan on 2026-11-09 ("approved"): four arms, fresh
seeds and layouts, greedy held-out success as the primary measure and ε = 0.05 as a
pre-specified secondary one. This file is not edited after results exist; anything
that changes is recorded under "Deviations" in `ANALYSIS.md`.

## Question

In Stage F (`rnn_generalization`), a DQN whose network has a GRU memory solved 0.73 of
held-out mazes against 0.29 for the standard network, and Stage K confirmed the gap
with a little randomness in play (0.80 vs 0.42). But the recurrent agent was also
*trained* differently: each update learns from 32 replayed runs of 16 consecutive
steps (512 trained steps, after up to 16 burn-in steps each), while the standard
agent learns from 64 single steps. Stage F could not say whether the gain came from

1. the memory itself (carrying information from one step to the next), or
2. how the recurrent agent is trained (8× as many steps per update, replayed as runs).

## Arms

All arms use the same task, training mazes, episode budget, ε schedule and the Stage F
settings for everything not listed. Nothing was tuned.

| Arm | Network | Memory | Replay | Trained steps per update | Weights |
|---|---|---|---|---|---|
| `dqn` (control) | MLP 77 → 128 → 128 → 4 | no | uniform, batch 64 | 64 | 27,012 |
| `dqn_batch512` | the same MLP | no | uniform, batch 512 | 512 | 27,012 |
| `dqn_rnn` | 77 → 128 → GRU 128 → 4 (exactly Stage F) | **yes** | 32 runs × 16 steps, burn-in ≤ 16 | 512 | 109,572 |
| `dqn_rnn_nomem` | the same network as `dqn_rnn` | **no** (reset every step) | the same as `dqn_rnn` | 512 | 109,572 |

**`dqn_rnn_nomem`** (`DQNConfig.reset_memory_every_step=True`) is `dqn_rnn` with one
change: the GRU starts from a zero memory at every step, h_t = GRU(layers(o_t), 0), when
acting, during burn-in, in training and in the targets. Its layers, initial weights
(same seed), replayed runs, burn-in, batch, learning rate and update schedule are
identical to `dqn_rnn`'s (tested). It can only see the current 5×5 window and the goal
direction, as the MLPs can. The only difference from `dqn_rnn` is the memory.

**`dqn_batch512`** gives the standard MLP as many trained steps per update as the
recurrent agents (512), with everything else unchanged (learning rate 1e-3, train_freq
2, target update every 100 updates).

## Task, layouts and seeds

The same task as Stages E, F and I: the `maze` preset (9×9, loop_prob 0.1, no hazards,
max 100 steps) with a 5×5 local view plus the direction to the goal, 3,000 training
episodes, ε 1.0 → 0.05 over 2,000 episodes.

- **Training layouts**: 9900–9949 (50 distinct mazes).
- **Held-out layouts**: 10001–10050 (50 distinct mazes). Layout 10000 generates the same
  maze as training layout 9918, so the held-out set starts at 10001; no held-out maze
  repeats a training maze or another held-out maze (checked before writing this
  protocol and by a test). Because 9×9 mazes are limited, at least 14 of these 100 layouts
  generate mazes that also appeared in earlier studies; this does not matter here,
  since every network in this study is trained from scratch.
- **Seeds**: 800–809 (10 per arm), never used before. Evaluation seed = 10,000 + run seed.
- Greedy evaluation on all 50 training layouts every 500 episodes; the held-out layouts
  are played once, greedily, at the end.

## Primary outcomes and tests

Outcome: final greedy success on the 50 held-out layouts. Each test compares 10 vs 10
per-seed values with a two-sided permutation test (200,000 resamples, which makes it
exact: C(20,10) = 184,756).

- **H1** (the memory itself): `dqn_rnn` vs `dqn_rnn_nomem`.
- **H2** (more trained steps per update for a plain MLP): `dqn_batch512` vs `dqn`.
- **H3** (replication of Stage F with fresh seeds and mazes): `dqn_rnn` vs `dqn`.
- H1–H3 are corrected together with Holm-Bonferroni at α = 0.05.

**Reading the results.** A significant H1 in favour of `dqn_rnn` is reported as "the
memory itself improves held-out success, with this network and training". A non-
significant H1 is reported as "no evidence that the memory itself matters beyond how
the recurrent agent is trained", never as "memory doesn't matter". H2 and H3 are read
the same way. If H1 and H3 are both significant, the share of the Stage F gain
attributable to the memory is described with `dqn_rnn_nomem`'s held-out success
relative to `dqn` and `dqn_rnn`, as an exploratory, descriptive comparison.

## Secondary (exploratory, no correction, labelled as exploratory)

- **ε = 0.05 evaluation** (pre-specified secondary measure). After the run, every
  final network plays the held-out mazes 5 times with ε = 0.05 (evaluation seed
  20,000 + run seed, `evaluate_agent(policy="epsilon")`), after its greedy result is
  re-checked against the saved one; H1–H3 are repeated on these values.
- The other held-out pairs: `dqn_rnn_nomem` vs `dqn`, `dqn_rnn_nomem` vs `dqn_batch512`,
  `dqn_rnn` vs `dqn_batch512`.
- For the primary pairs: learning speed (greedy training-maze success averaged over the
  evaluations at 500, …, 3,000 episodes), final training success, and held-out
  revisits, revisits per step, steps, timeouts, collisions and path efficiency.
- Compute: environment steps and time per run.

## Compute and execution

CPU only (Windows laptop), runs in `runs/experiments_memory_replay/`:
`python -m src.evaluation --experiment memory_vs_replay --root runs/experiments_memory_replay
--workers 2`, started in its own PowerShell window (outside Claude Code). A timing
check on unrelated layouts (0–49, local view, seed 999, 150 episodes, one process)
measured 1.68 ms per environment step for `dqn`, 2.59 for `dqn_batch512`, 24.6 for
`dqn_rnn` and 24.4 for `dqn_rnn_nomem`; it printed no success values. Expected total:
about 10–16 hours with two workers, dominated by the two recurrent arms. Runs are
resumable from their last 500-episode checkpoint with the same command (same or fewer
workers); the run may be split over several sessions, and any stop is recorded as a
deviation. Results are copied to `results/memory_vs_replay/` with an `ANALYSIS.md`
written after the run, and every run's final network is exported to
`results/memory_vs_replay/models/`. The code in `src/` is not changed while the
experiment runs. The analysis script `results/memory_vs_replay/analyze.py` was
written before the run (and smoke-tested on a two-episode copy of the experiment); it
is not changed afterwards except to fix errors, which are recorded as deviations.

## Code state

Implemented and tested after commit c36e2aa (uncommitted at the time of writing):
`DQNConfig.reset_memory_every_step` and `RecurrentQNetwork(reset_every_step=True)`,
agent `dqn_rnn_nomem`, the `memory_vs_replay` experiment (with `dqn_batch512` as an
arm of `dqn` with `batch_size=512`), chart styles, a nickname and tagline. Existing
agents are unchanged: 300 episodes of `rnn_generalization` DQN seed 600 and 150
episodes of `rnn_generalization` `dqn_rnn` seed 600 reproduce their saved logs exactly.
