# Convolutional Q-network: pre-registered protocol

Covers one experiment, `cnn_generalization`. Written 2026-10-31 (ADT), **before** any
run of it. The project owner approved the plan (CNN vs the standard MLP, fresh seeds
and layouts, network sizes fixed in advance) first. The wide-MLP arm was added before
any run, as an exploratory check on network size (see "Arms"). This file is not edited
after results exist; anything that changes is recorded under "Deviations" in
`ANALYSIS.md`.

## Question

Shaping, 3-step returns and an easiest-first curriculum all failed to help DQN on
mazes it had never seen (REPORT 6.8–6.10). Those changes affected how values are
learned or which experience is seen. This experiment changes what the network can
represent. A convolutional network applies the same small filters at every position
of the 5×5 local view, so a corridor, a corner or a dead end is recognised wherever it
appears. Does such a network

1. solve more held-out mazes, and
2. learn its training mazes sooner,

than the standard fully connected network?

## Arms

All three arms are the same DQN on the same task with the same settings. Only the
network differs.

| Arm | Network | Weights |
|---|---|---|
| `dqn` (control) | MLP: 77 inputs → 128 → 128 → 4 | 27,012 |
| `dqn_cnn` | The 3×5×5 window (wall, hazard, goal channels) → 3×3 conv, 16 channels → 3×3 conv, 32 channels (padding 1, ReLU, so maps stay 5×5) → flatten (800) + the 2 goal-direction values → 128 → 4 | 108,388 |
| `dqn_wide` (exploratory) | MLP: 77 → 288 → 288 → 4, about the CNN's number of weights | 106,852 |

The CNN has four times as many weights as the standard MLP, so a CNN advantage could
come from size alone. `dqn_wide` is there to show whether size alone helps. It is
**exploratory** and not part of the corrected primary tests. The CNN's channel counts
(16, 32), its single hidden layer (128) and the wide MLP's width (288, chosen only to
match the CNN's weight count) were fixed before any run and are not tuned.

Settings shared by all arms: Double DQN target, Adam lr 1e-3, γ 0.99, batch 64,
replay 20,000, learning_starts 500, train_freq 2, target_update 100, Huber loss,
gradient clip 10. 3,000 episodes, ε 1.0 → 0.05 over 2,000.

## Task, layouts and seeds

The same task as the earlier generalization experiments: the `maze` preset (9×9,
loop_prob 0.1, no hazards, max 100 steps) with a 5×5 local view plus the direction to
the goal.

- **Training layouts**: 9300–9349, 50 distinct mazes.
- **Held-out layouts**: 9400–9449. None is identical to a training maze or to another
  held-out maze (checked before writing this, and by a test).
- Path lengths: training 12 (25 mazes), 14 (2), 16 (12), 18 (2), 22 (3), 24 (2), 26 (1),
  28 (3); held-out 12 (29), 14 (1), 16 (8), 18 (2), 20 (7), 22 (1), 28 (2).
- **Seeds**: 500–509 (10 per arm). Evaluation seed = 10,000 + run seed, as before.
- Greedy evaluation on all 50 training layouts every 500 episodes; the held-out layouts
  are played once, greedily, at the end.

None of these seeds or layouts were used by any earlier experiment.

## Primary outcomes and tests

Each test compares 10 per-seed values with a two-sided permutation test (20,000
resamples).

- **H1**: final greedy success on the 50 held-out layouts, `dqn_cnn` vs `dqn`.
- **H2**: learning speed, i.e. greedy success on the 50 training layouts averaged over the
  6 evaluations during training (episodes 500, 1000, …, 3000), `dqn_cnn` vs `dqn`.
- H1 and H2 are corrected together with Holm-Bonferroni at α = 0.05.

**Reading the results.** A significant difference is reported as the CNN helping (or
hurting). Anything else is reported as "no evidence of a difference", never as "no
effect" and never as a trend. If H1 or H2 is significant, the result is described as
an effect of the CNN *network*, not of convolutions as such, unless the exploratory
`dqn_wide` comparison also points that way. Even then, that comparison is exploratory.

## Secondary (exploratory, no correction, labelled as exploratory)

- `dqn_wide` vs `dqn` and `dqn_cnn` vs `dqn_wide`, on held-out success and learning
  speed.
- Final greedy success and path efficiency on the training layouts.
- Held-out steps, collisions, revisits and path efficiency.
- Held-out success split by path length (12 vs longer), computed afterwards by
  re-evaluating each run's final checkpoint with the same evaluation seed. The split is
  checked to reproduce each run's saved held-out success first.
- Compute cost: environment steps per run and time per step.

## Compute and execution

CPU only (Windows laptop), runs in `runs/experiments_cnn/`:
`python -m src.evaluation --experiment cnn_generalization --root runs/experiments_cnn
--workers 3`. A short timing check on unrelated layouts (0–49, seed 999) measured about
5.1 ms per environment step for the CNN against 1.4 ms for the MLP; it was not used to
choose any setting. Runs are resumable; a crash (for example, the machine running out
of memory) is resumed with the same command, with the same or fewer workers. Results
are copied to `results/cnn_generalization/` with an `ANALYSIS.md` written after the
run. The code in `src/` is not changed while the experiment runs.

## Code state

Implemented and tested after commit cfc0cd8 (uncommitted at the time of writing):
`DQNConfig.conv` and `conv_input`, `QNetwork`'s convolutional encoder, the `dqn_cnn`
agent (`CNN_DEFAULT`: conv (16, 32), hidden (128,)), `GridWorld.image_shape` (CNN
agents read the image shape from the environment), and the `cnn_generalization`
experiment. With `conv` empty (the default), networks are built exactly as before (same
initial weights, tested), and the first 500 episodes of `dqn_variants_generalization`
DQN seed 300 reproduce the saved log exactly.
