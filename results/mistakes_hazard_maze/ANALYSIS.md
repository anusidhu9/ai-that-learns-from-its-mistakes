# mistakes_hazard_maze: analysis (negative result)

**Question.** Does mistake-aware replay (25% of every batch drawn from a reserve of
collision and hazard transitions) improve DQN on the `hazard_maze` task?

**Answer from this experiment: no evidence that it does.** No difference between any
two agents was statistically significant on any metric (all permutation p ≥ 0.4).

## Setup

Identical to `hazard_maze`: one fixed 9×9 maze with 4 non-terminal hazards (−0.5 each),
seeds 0–9, 400 training episodes, ε decaying over 300, greedy evaluation every 50
episodes and at the end (10 episodes, evaluation seed 10,000 + run seed).
`mistake_fraction = 0.25` and `mistake_capacity = 5000` were fixed before the run and not
tuned. Main replay buffer: 20,000 transitions (the DQN default).

## Results (greedy, mean ± 95% CI over 10 seeds)

| Agent | Success | Episode reward | Steps to goal* | Collisions | Hazards | Seeds crossing a hazard |
|---|---|---|---|---|---|---|
| DQN | 0.90 ± 0.23 | 0.13 ± 1.46 | 17.3 ± 1.5 (9 seeds) | 9.1 ± 20.6 | 0.10 ± 0.23 | 1 / 10 |
| DQN + PER | 1.00 ± 0.00 | 0.69 ± 0.15 | 16.2 ± 2.1 | 0.0 | 0.30 ± 0.35 | 3 / 10 |
| DQN + mistakes | 0.80 ± 0.30 | −0.30 ± 1.69 | 17.3 ± 1.8 (8 seeds) | 10.0 ± 22.6 | 0.10 ± 0.23 | 1 / 10 |
| DQN + PER + mistakes | 1.00 ± 0.00 | 0.73 ± 0.13 | 16.8 ± 1.8 | 0.0 | 0.20 ± 0.30 | 2 / 10 |

\*Mean steps over successful episodes only; seeds that never reached the goal are left out.

Selected permutation tests: success, DQN + mistakes − DQN: −0.10 (p = 1.000); reward:
−0.43 (p = 0.665); success, DQN + PER + mistakes − DQN + PER: +0.00 (p = 1.000); reward:
+0.04 (p = 1.000). The full list is at the end of `run.log`.

## Observations

- **Reproduction.** The `dqn` and `dqn_per` runs reproduced `results/hazard_maze` exactly
  on every seed and metric, so turning the option off leaves the original agents unchanged.
- **Failed seeds.** DQN + mistakes never solved the maze at any evaluation point on seeds 5
  (ended pushing into a wall, 100 collisions) and 6 (ended walking in loops). Plain DQN's
  seed 5 solved the maze from episode 150 to 350 and then unlearned it. Two failures in ten
  runs are too few to say mistake replay makes DQN less stable.
- **The long-memory part of the idea was barely tested** (`buffer_diagnostics.csv`). Runs
  collected about 14,500–22,600 transitions, and the main buffer holds 20,000, so in 13 of the
  20 mistake-replay runs it never overwrote anything and in 5 more it overwrote at most 2,624.
  Only the two failed seeds overwrote a lot (15,578 and 16,245), because failing episodes are
  long. The reserve therefore mostly made mistakes appear more often in batches rather than
  remembering them for longer. Mistakes were already about 30% of all transitions, so batches
  were roughly 45–50% mistakes.

## What this does and does not show

- Shows: on this task and budget, over-weighting recent mistakes in replay gave no
  measurable benefit, with or without prioritized replay.
- Does not show: whether *remembering* mistakes after ordinary replay has forgotten them
  helps. That is tested separately, with a 5,000-transition main buffer, in
  `results/long_memory_mistakes/`.

## Reproduce

```bash
python -m src.evaluation --experiment mistakes_hazard_maze --root runs/experiments_mistakes
```

About 20 minutes on a laptop CPU; rerun the same command to resume after an interruption.
