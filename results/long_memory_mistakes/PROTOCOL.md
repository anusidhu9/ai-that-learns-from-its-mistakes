# long_memory_mistakes: pre-registered protocol

Written 2026-10-23 21:52 (ADT), **before** the pilot or any run of this experiment, and
approved by the project owner before anything was run. It is not edited after the
results exist; anything that changes later is recorded under "Deviations" in
`ANALYSIS.md`.

## Question

When the replay buffer is small enough to forget old experience, does keeping a separate
long-term memory of mistakes (collisions and hazards) improve DQN on mazes it has never seen?

## Why a new experiment

`mistakes_hazard_maze` (see `results/mistakes_hazard_maze/ANALYSIS.md`) found no benefit,
but its 20,000-transition main buffer barely overwrote anything, so the reserve only
over-sampled recent mistakes. It never tested *remembering* them. Its seeds (0–9) and maze
have been seen and are not reused here.

## Arms

All three arms are plain DQN (Double DQN target, MLP 128×128, Adam lr 1e-3, γ 0.99,
batch 64, learning_starts 500, train_freq 2, target_update 100, Huber loss, grad clip 10,
ε 1.0 → 0.05 over 2,000 episodes), identical except for replay memory:

| Arm | Main buffer | Mistake reserve | Share of each batch from the reserve |
|---|---|---|---|
| `dqn_5k` (baseline) | 5,000 | none | none |
| `dqn_mistake_5k` (variant) | 5,000 | 5,000 transitions with a collision or hazard event | 0.25 |
| `dqn_10k` (memory control) | 10,000 | none | none |

`dqn_10k` holds as many transitions as the variant in total, so a benefit that is only
"more memory" is not attributed to remembering mistakes. 0.25 and 5,000 were fixed before
the previous experiment and are not tuned. Mistakes are identified from the environment's
step events (`collision`, `hazard`), never from the reward.

## Task, layouts and seeds

- Environment: `maze_hazards` preset (9×9 maze, loop_prob 0.3, 4 non-terminal hazards at
  −0.5, max 100 steps), 5×5 local observation plus the direction to the goal.
- Training layouts: layout seeds 3000–3049. Held-out test layouts: 4000–4049. A test
  checks that no held-out maze (grid, start, goal) equals a training maze, and that none
  of these layout seeds was used by an earlier experiment.
- Run seeds 100–109 (fresh). Evaluation seed = 10,000 + run seed (10100–10109), the rule
  used by every experiment in this project.
- Training budget per run: 3,000 episodes, identical for every arm.

## Evaluation

- Every 500 episodes: greedy evaluation (ε = 0), one episode on each of the 50 training
  layouts. These points form the learning curves.
- After training: greedy evaluation on the 50 training layouts and, once, on the 50
  held-out layouts. Held-out layouts are never used during training or for the curves
  (enforced by a test).
- Metrics per run: success rate, episode reward (`total_reward`), `steps_to_goal` (mean
  steps of successful episodes only; undefined if none), hazard steps per episode
  (`hazards`), `hazard_rate` (fraction of episodes that touched a hazard), collisions,
  path efficiency.
- Summaries: mean and 95% t-interval over the 10 seeds; per-seed tables.

## Analysis plan

- **Primary endpoint:** held-out success rate, `dqn_mistake_5k` − `dqn_5k`, two-sided
  permutation test on the 10 per-seed values per arm, α = 0.05.
- **Secondary (exploratory):** the same comparison on every other metric and on the
  training layouts; `dqn_mistake_5k` vs `dqn_10k`; `dqn_10k` vs `dqn_5k`. Reported with
  p-values, not used for claims.
- **Claim rule.**
  - "Long-term mistake memory improved held-out success" only if the primary difference is
    positive with p < 0.05.
  - Attributed to *mistake* memory rather than more memory only if, in addition,
    `dqn_mistake_5k` is not lower than `dqn_10k` on held-out success.
  - Otherwise the result is reported as "no evidence" (or "worse" if the primary
    difference is negative with p < 0.05).
- **Power.** In `generalization` (same observation, 50 + 50 layouts) DQN's held-out success
  had a seed-to-seed standard deviation of about 0.05, so 10 seeds per arm can detect a
  difference of roughly 0.06–0.07. Smaller effects will appear as "not established".

## Checking that forgetting happens

`buffer_diagnostics.csv` records, per run at the end of training: transitions added, main
buffer size and how many transitions it overwrote, and for the variant the reserve size,
mistakes seen, how far back the reserve reaches, and how many reserved mistakes are no
longer in the main buffer (`reserve_only`). The experiment only tests long-term memory if
the main buffer overwrote transitions and `reserve_only` > 0 in the variant's runs; this
is checked and reported.

## Pilot

One run of `dqn_5k` only (experiment `long_memory_pilot`: seed 900, training layouts
5000–5049, held-out 6000–6049, none of which are used by the main experiment), to
measure run time and confirm that the buffer overwrites old transitions. The variant is
not run in the pilot and no setting is chosen from it. If the pilot's final training
success is below 0.2, the task is considered too hard for a meaningful comparison, and the
experiment is stopped and re-planned with the owner instead of adjusted.

## Compute and execution

```bash
python -m src.evaluation --experiment long_memory_pilot    --root runs/experiments_long_memory
python -m src.evaluation --experiment long_memory_mistakes --root runs/experiments_long_memory --workers 4
```

30 runs; about 9 minutes each, so about 1.25–1.5 hours with 4 parallel processes. Runs
are independent and seeded, so parallel and sequential execution give the same results
(tested). Interrupted runs resume from their last checkpoint.

## Code state

Uncommitted working tree on top of commit `52e3a5b`. SHA-256 prefixes of the source files
at the time this protocol was written:

```
969ec0ab2d411825  src/agents.py
fc010b1e7e364ac8  src/app.py
a8644b9d2289d8c5  src/envs.py
9668dd64b6bd33dc  src/evaluation.py
9b5a510fd4886e3c  src/fun.py
56b260bf21ed4e2b  src/replay.py
85e223cf995ca623  src/training.py
487bcab06f51f9f5  src/visualization.py
```
