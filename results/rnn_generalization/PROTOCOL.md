# Recurrent Q-network (memory): pre-registered protocol

Covers one experiment, `rnn_generalization`. Written 2026-11-01, **before** any run of
it. The project owner approved the plan (a GRU memory against the standard MLP, a
capacity-matched wide MLP that also retests Stage E's exploratory result, fresh seeds
and layouts, all settings fixed in advance) first. This file is not edited after
results exist; anything that changes is recorded under "Deviations" in `ANALYSIS.md`.

## Question

On mazes it has never seen, DQN with a 5×5 local view mostly fails by going round in
circles: in Stage E it revisited a cell 47 times per held-out episode and timed out in
two thirds of them. Without memory it cannot know it has been somewhere before (the
revisit penalty itself depends on history the observation does not contain).
Shaping, 3-step returns, a curriculum and a convolutional network did not help it on
new mazes (REPORT 6.8–6.11). A recurrent network carries a memory through the
episode. Does a DQN with memory

1. solve more held-out mazes, and
2. revisit cells less often on them,

than the standard DQN? And, as a retest of Stage E's exploratory result with fresh
seeds and mazes:

3. does a wide MLP of the same size solve more held-out mazes than the standard DQN?

## Arms

All three arms are the same DQN on the same task with the same training settings
(listed below). Only the network, and for the recurrent arm how replay is sampled,
differs.

| Arm | Network | Weights |
|---|---|---|
| `dqn` (control) | MLP: 77 inputs → 128 → 128 → 4 | 27,012 |
| `dqn_rnn` | 77 → 128 (ReLU) → GRU with 128 memory units → 4. The memory is cleared at the start of every episode. | 109,572 |
| `dqn_wide` | MLP: 77 → 288 → 288 → 4 | 106,852 |

`dqn_rnn` and `dqn_wide` have nearly the same number of weights, so a difference
between them is not a matter of size. `dqn_wide` is the same arm as in Stage E.

**How the recurrent arm learns.** It replays runs of 16 consecutive steps from one
episode instead of single steps: each update draws 32 runs, each starting at a step
chosen uniformly from the buffer. Before each run, up to 16 earlier steps of the same
episode (as many as exist, and none before the oldest stored step) are fed through the
network with its memory starting at zero, to rebuild the memory. They are not trained
on (burn-in, as in R2D2). A run never crosses into another episode; steps past the
episode's end are masked out. The loss is the Huber loss averaged over the trained
steps. Each step's target feeds the next observation to the target network after the
target network's memory at that step (Double DQN: the online network chooses the
action the same way), so timeouts and a run's last step are handled exactly.

Sequence length 16, burn-in 16, 32 runs per update and the network sizes were fixed
before any run and are not tuned. With these, one update trains on up to 512 steps
(64 for the MLPs).

Shared settings: Double DQN target, Adam lr 1e-3, γ 0.99, replay 20,000 transitions,
learning_starts 500, train_freq 2, target_update 100, gradient clip 10. MLPs: batch 64
transitions. 3,000 episodes, ε 1.0 → 0.05 over 2,000.

## Task, layouts and seeds

The same task as Stages E and H: the `maze` preset (9×9, loop_prob 0.1, no hazards,
max 100 steps) with a 5×5 local view plus the direction to the goal.

- **Training layouts**: 9500–9549, 50 distinct mazes.
- **Held-out layouts**: 9600–9651 except 9619 and 9632. Layout 9619 generates the
  same maze as training layout 9544, and 9632 the same as 9623; both were found before
  any run and replaced by 9650 and 9651. None of the 50 is identical to a training maze
  or to another held-out maze (checked by a test).
- Path lengths: training 12 (19 mazes), 14 (2), 16 (12), 18 (2), 20 (9), 22 (1),
  24 (2), 26 (1), 28 (2); held-out 12 (19), 14 (1), 16 (14), 18 (1), 20 (5), 22 (3),
  24 (5), 26 (1), 28 (1).
- **Seeds**: 600–609 (10 per arm). Evaluation seed = 10,000 + run seed, as before.
- Greedy evaluation on all 50 training layouts every 500 episodes; the held-out layouts
  are played once, greedily, at the end. Greedy play uses the memory as in training
  (cleared at the start of each episode).

None of these seeds or layouts were used by any earlier experiment.

## Primary outcomes and tests

Each test compares 10 per-seed values with a two-sided permutation test (20,000
resamples).

- **H1**: final greedy success on the 50 held-out layouts, `dqn_rnn` vs `dqn`.
- **H2**: revisits per episode on the 50 held-out layouts (the `revisits` metric as
  recorded), `dqn_rnn` vs `dqn`.
- **H3**: final greedy success on the 50 held-out layouts, `dqn_wide` vs `dqn`.
- H1, H2 and H3 are corrected together with Holm-Bonferroni at α = 0.05.

**Reading the results.** A significant difference is reported as the network helping
(or hurting). Anything else is reported as "no evidence of a difference", never as "no
effect" and never as a trend. Revisits per episode also depend on how long episodes
last (a solved maze ends early), so a significant H2 is read together with H1 and the
exploratory revisits-per-step rate below; it is not on its own evidence that memory
prevents loops. If H1 is significant, the effect is attributed to memory only if the
exploratory `dqn_rnn` vs `dqn_wide` comparison points the same way; otherwise to the
recurrent network as a whole. If H3 is significant, Stage E's exploratory wide-MLP
result is reported as replicated on this task; if not, as not replicated.

## Secondary (exploratory, no correction, labelled as exploratory)

- `dqn_rnn` vs `dqn_wide` (same size) on held-out success and revisits.
- Learning speed (greedy success on the 50 training layouts averaged over the 6
  evaluations at episodes 500, …, 3000) for each arm against `dqn`.
- Final greedy success and path efficiency on the training layouts.
- Held-out steps, timeouts, collisions, path efficiency, and revisits per step.
- Held-out success split by path length (12 vs longer), computed afterwards by
  re-evaluating each run's final network with the same evaluation seed. The split is
  checked to reproduce each run's saved held-out success first.
- Compute cost: environment steps per run and time per run.

## Compute and execution

CPU only (Windows laptop), runs in `runs/experiments_rnn/`:
`python -m src.evaluation --experiment rnn_generalization --root runs/experiments_rnn
--workers 2`, started in its own PowerShell window (outside Claude Code, whose
low-memory handling stopped two earlier experiments). A short timing check on unrelated
layouts (0–49, seed 999, 200 episodes, one process) measured about 19 ms per
environment step for `dqn_rnn`, 2.3 ms for `dqn_wide` and 1.6 ms for `dqn`; nearly all
of the recurrent arm's time is its sequence updates (about 35 ms each). The check also
printed training success; it was not used to choose any setting. Expected total: about
6 hours. Runs are resumable; a crash is resumed with the same command, with the same or
fewer workers. Results are copied to `results/rnn_generalization/` with an
`ANALYSIS.md` written after the run, and every run's final network is exported to
`results/rnn_generalization/models/` as in Stage E. The code in `src/` is not changed
while the experiment runs.

## Code state

Implemented and tested after commit 8e3cb4d (uncommitted at the time of writing):
`RecurrentQNetwork`, `DQNConfig.recurrent`, `seq_length` and `burn_in`, the `dqn_rnn`
agent (`RNN_DEFAULT`: recurrent 128, hidden (128,), batch 32),
`SequenceReplayBuffer`, the optional `begin_episode` hook (called by `run_episode` and
episode recording), and the `rnn_generalization` experiment. A recurrent agent refuses
PER, mistake replay, n-step returns, dueling heads and convolutions. Plain DQN is
unchanged: the first 300 episodes of `cnn_generalization` DQN seed 500 reproduce the
saved log exactly, and a recurrent run resumed from a checkpoint reproduces an
uninterrupted one exactly (tested).
