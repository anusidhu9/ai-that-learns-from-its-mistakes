# Reward shaping: pre-registered protocol

Covers two experiments, `shaping_speed` and `shaping_generalization`. Written 2026-10-26
21:18 (ADT), **before** any run of either experiment. The plan (two experiments, shaping
strength fixed at 1.0, fresh seeds and layouts) was approved by the project owner first.
This file is not edited after results exist; anything that changes is recorded under
"Deviations" in each experiment's `ANALYSIS.md`.

## Question

Does a hint toward the goal, added to the reward, help the agents?

1. **Speed**: do tabular Q and DQN learn a fixed maze sooner with shaping?
2. **Generalization**: does a DQN trained with shaping solve more mazes it has never seen?

## The shaping

Potential-based shaping (Ng, Harada & Russell, 1999). Each step the agent learns from the
task reward plus

    F = k * (gamma * phi(s') - phi(s)),   phi(s) = -d(s) / d_max

where d(s) is the maze distance from the agent's cell to the goal (walls block it, hazards
don't), d_max is the largest such distance in the layout, and phi(s') = 0 when the episode
terminates. k = 1.0 and gamma = 0.99, the agents' own discount. With matching gamma this
provably leaves the optimal policy unchanged, so the agent can't profit from walking in
circles (tested: the discounted sum of F over an episode telescopes to
gamma^T phi(s_T) - phi(s_0)).

k = 1.0 was chosen in advance, not tuned: phi lies in [-1, 0], so the most shaping an
episode can add is about the size of the goal reward (+1). No other value is tried. If the
results are null, that is the result for k = 1.0.

Shaping changes only the reward the agent **learns from**. Every reported metric (success,
path efficiency, steps, mistakes, and `total_reward`, which excludes shaping) is the task's.
Evaluation is greedy (no exploration), as in every earlier experiment.

## Arms

All agents use their standard settings from earlier experiments (tabular Q: alpha 0.5,
gamma 0.99; DQN: Double DQN target, MLP 128x128, Adam lr 1e-3, gamma 0.99, batch 64,
replay 20,000, learning_starts 500, train_freq 2, target_update 100). Each shaped arm is
identical to its baseline except for the shaping term.

| Experiment | Arms |
|---|---|
| `shaping_speed` | `tabular_q`, `tabular_q_shaped`, `dqn`, `dqn_shaped` |
| `shaping_generalization` | `dqn`, `dqn_shaped` |

## Task, layouts and seeds

None of these seeds or layouts were used by any earlier experiment.

- **`shaping_speed`**: `maze` preset (9x9, loop_prob 0.1, no hazards, max 100 steps,
  full-map observation), one fixed layout, seed 7000 (shortest path 16 steps). 400 episodes,
  epsilon 1.0 -> 0.05 over 300. Greedy evaluation (10 episodes) every 25 episodes.
- **`shaping_generalization`**: `maze` preset with a 5x5 local view plus the direction to
  the goal. Training layouts 7100-7149 (one drawn at random each episode); held-out
  layouts 7200-7249, used only in the final evaluation. 3,000 episodes, epsilon 1.0 -> 0.05
  over 2,000. Greedy evaluation on the training layouts every 500 episodes; at the end, one
  greedy episode on each of the 50 held-out layouts.
- **Seeds**: 200-209 (10 per arm). Evaluation seed = 10,000 + run seed, as before.

Note: with a local view the agent can't see the maze distance that phi uses, so the hint
is only partly predictable from its observation. That is part of what experiment 2 tests.

## Primary outcomes and tests

Each test compares 10 per-seed values per arm with a two-sided permutation test
(20,000 resamples, as in every earlier experiment).

- **H1** (`shaping_speed`): learning speed of tabular Q, measured as greedy success
  averaged over the 16 evaluations during training (episodes 25-400; the curve's area).
  `tabular_q_shaped` vs `tabular_q`.
- **H2** (`shaping_speed`): the same for DQN, `dqn_shaped` vs `dqn`.
- H1 and H2 are corrected together with Holm-Bonferroni at alpha = 0.05.
- **H3** (`shaping_generalization`): final greedy success on the 50 held-out layouts,
  `dqn_shaped` vs `dqn`, alpha = 0.05.

**Reading the results.** A significant difference is reported as shaping helping (or
hurting). Anything else is reported as "no evidence of a difference", never as "no effect"
and never as a trend.

## Secondary (exploratory, no correction, labelled as exploratory)

Final greedy success and path efficiency on the training layout(s); held-out path
efficiency; learning-curve area in `shaping_generalization`; steps, collisions and
revisits; the episode at which each run first reached greedy success 1.0.

## Compute and execution

CPU only (Windows laptop), `--workers 4`, runs in `runs/experiments_shaping/`. Runs are
resumable; a crash is resumed, not restarted with different settings. Results are copied
to `results/<experiment>/` with an `ANALYSIS.md` written after the run.

## Code state

Shaping implemented and tested after commit 56202bd (uncommitted at the time of writing).
Before writing this, shaping-off was checked to reproduce committed `hazard_maze` results
exactly (tabular Q and DQN, seeds 0-1).
