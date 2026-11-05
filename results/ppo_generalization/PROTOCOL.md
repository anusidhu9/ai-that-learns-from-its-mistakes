# PPO against DQN: pre-registered protocol

Covers one experiment, `ppo_generalization`. Written 2026-11-04, **before** any run of
it. The project owner approved the plan (PPO with standard settings fixed in advance
against the standard DQN, fresh seeds and layouts, a greedy primary test and a sampled
exploratory one) first. This file is not edited after results exist; anything that
changes is recorded under "Deviations" in `ANALYSIS.md`.

## Question

Every learning agent so far estimates the value of each move (tabular Q-learning and
DQN with its variants). PPO (Schulman et al., 2017) learns the policy directly, a
probability for each move, together with a critic that estimates how good a state is.
It learns only from the steps it has just played, with no replay buffer. On the local
view task, where DQN solves about a third of new mazes, does PPO

1. solve more held-out mazes, and
2. learn its training mazes faster,

than DQN?

## Arms

Both arms use the same task, the same training mazes, the same episode budget and
networks of the same shape (two hidden layers of 128). Everything else is each
algorithm's own standard setting.

| Arm | Learner | Networks | Weights |
|---|---|---|---|
| `dqn` (control) | DQN exactly as in Stages E and F (Double DQN, replay 20,000, batch 64, Adam 1e-3, ε 1.0 → 0.05 over 2,000 episodes, train_freq 2, target_update 100, ReLU) | Q-network 77 → 128 → 128 → 4 | 27,012 |
| `ppo` | PPO with a clipped objective and GAE | actor 77 → 128 → 128 → 4 (tanh), separate critic 77 → 128 → 128 → 1 (tanh), orthogonal initialisation | 27,012 + 26,625 |

**PPO settings** (`PPOConfig` defaults; the common published values for small discrete
tasks, as in CleanRL and Stable-Baselines3; fixed before any run, not tuned): rollout of
2,048 environment steps per update (episodes end and start inside a rollout), 4 epochs
over each rollout in shuffled minibatches of 256, clip 0.2, GAE λ 0.95, γ 0.99, Adam lr
3e-4 for both networks together, entropy bonus 0.01, critic loss weight 0.5, gradient
norm clip 0.5, advantages normalised per minibatch. A timeout is not a terminal state: it
bootstraps from the critic, as for DQN. PPO has no ε: it explores by sampling its own
policy. Greedy play takes the most likely move (ties broken at random).

## Task, layouts and seeds

The same task as Stages E, F and H: the `maze` preset (9×9, loop_prob 0.1, no hazards,
max 100 steps) with a 5×5 local view plus the direction to the goal, 3,000 training
episodes.

- **Training layouts**: 9700–9749, 50 distinct mazes.
- **Held-out layouts**: 9800–9849, 50 distinct mazes. None is identical to a training
  maze or to another held-out maze (checked before writing this protocol and by a test).
- **Seeds**: 700–709 (10 per arm). Evaluation seed = 10,000 + run seed, as before.
- Greedy evaluation on all 50 training layouts every 500 episodes; the held-out layouts
  are played once, greedily, at the end.

None of these seeds or layouts were used by any earlier experiment.

## Primary outcomes and tests

Each test compares 10 per-seed values with a two-sided permutation test (20,000
resamples).

- **H1**: final greedy success on the 50 held-out layouts, `ppo` vs `dqn`.
- **H2**: learning speed: greedy success on the 50 training layouts averaged over the 6
  evaluations at episodes 500, 1000, …, 3000, `ppo` vs `dqn`.
- H1 and H2 are corrected together with Holm-Bonferroni at α = 0.05.

**Reading the results.** A significant difference is reported as PPO doing better (or
worse) than DQN *with these settings*. Anything else is reported as "no evidence of a
difference", never as "no effect" and never as a trend. The comparison is of two
algorithms each with its own standard settings, not of tuned versions of either: a
result is not evidence about PPO or DQN in general.

## Secondary (exploratory, no correction, labelled as exploratory)

- **Sampled play on held-out mazes.** PPO's greedy policy may loop where sampling from
  its policy would break the loop. After the run, each final PPO network is also played
  once on every held-out maze sampling its moves (evaluation seed 10,000 + run seed),
  and compared with its greedy result and with DQN's greedy result.
- Held-out steps, timeouts, collisions, revisits, revisits per step and path efficiency.
- Final greedy success and path efficiency on the training layouts, and success at each
  evaluation point.
- Held-out success split by path length (12 vs longer), computed afterwards by
  re-evaluating each run's final network with the same evaluation seed. The split is
  checked to reproduce each run's saved held-out success first.
- Compute cost: environment steps per run and time per run.

## Compute and execution

CPU only (Windows laptop), runs in `runs/experiments_ppo/`:
`python -m src.evaluation --experiment ppo_generalization --root runs/experiments_ppo
--workers 2`, started in its own PowerShell window (outside Claude Code). A timing check
on unrelated layouts (0–49, local view, seed 999, 200 episodes, one process) measured
1.40 ms per environment step for `dqn` and 0.30 ms for `ppo`. The check also printed
training success (DQN 0.18, PPO 0.56 over those 200 episodes); it was not used to choose
any setting. Expected total: about 1–1.5 hours (DQN runs took about 9 minutes each with
two workers in Stage F). A first timing attempt measured 19.8 ms per DQN step while the
laptop was busy after a test run and was discarded. Runs are
resumable; a crash is resumed with the same command, with the same or fewer workers. A
PPO checkpoint includes the half-collected rollout, so a resumed run reproduces an
uninterrupted one exactly (tested). Results are copied to `results/ppo_generalization/`
with an `ANALYSIS.md` written after the run, and every run's final network is exported
to `results/ppo_generalization/models/`. The code in `src/` is not changed while the
experiment runs.

## Code state

Implemented and tested after commit 4e75d00 (uncommitted at the time of writing):
`PPOConfig`, `PolicyNetwork`, `PPOAgent` (`make_agent("ppo")`; an ε schedule passed to it
is ignored), PPO checkpoints with the rollout (`save_buffer`), weights-only export, the
policy map (most likely move, shaded by the critic) and replays (move probabilities) for
policy agents, the dashboard entry, and the `ppo_generalization` experiment.
`Experiment.train_config` gives PPO no ε schedule. Plain DQN is unchanged: 300 episodes
of `rnn_generalization` DQN seed 600 reproduce its saved log exactly. A sanity check before this protocol
(fixed `maze` preset, full-map view, seed 999, unrelated to this experiment) confirmed
that PPO with these settings solves a single maze: greedy success 1.00 with the
shortest path from episode 500 of 1,500. It was not used to choose any setting.
