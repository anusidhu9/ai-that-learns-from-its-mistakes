# long_memory_mistakes: analysis

Protocol: `PROTOCOL.md` (written and approved before any run). Raw outputs: `results.csv`
(per seed and split), `curves.csv`, `summary.csv`, `buffer_diagnostics.csv`, `run.log`.
Pilot: `pilot/`.

## Answer

**No evidence that long-term mistake memory helps.** On held-out mazes, DQN with a
5,000-mistake reserve reached the goal less often than the same DQN without it (0.30 vs
0.37), and was ahead on only 1 of the 10 seeds. That difference is not significant
(p = 0.082), so by the pre-registered rule the result is "no evidence", not "worse".

## Did the experiment test long-term memory? Yes

From `buffer_diagnostics.csv` (end of training, every run):

| Arm | Transitions added | Overwritten by the main buffer | Mistakes kept only in the reserve |
|---|---|---|---|
| `dqn_5k` | 103,963–111,959 | 98,963–106,959 | — |
| `dqn_mistake_5k` | 106,425–121,498 | 101,425–116,498 | 4,096–4,874 of 5,000 |
| `dqn_10k` | 99,746–105,794 | 89,746–95,794 | — |

In every variant run, 82–98% of the reserved mistakes were no longer in the main buffer, and
the reserve reached back 40,203–44,678 transitions against the main buffer's 5,000. Unlike
`mistakes_hazard_maze`, this experiment really did test remembering mistakes after
ordinary replay had forgotten them.

## Results (greedy, mean ± 95% CI over seeds 100–109)

**Held-out mazes (layouts 4000–4049, never seen in training)**

| Arm | Success | Episode reward | Steps to goal* | Hazard steps / episode | Hazard rate | Collisions | Revisits |
|---|---|---|---|---|---|---|---|
| DQN (5k) | 0.37 ± 0.04 | −2.26 ± 0.17 | 12.2 ± 0.1 | 0.04 ± 0.04 | 0.04 ± 0.04 | 23.2 ± 4.0 | 35.9 ± 6.3 |
| DQN + mistake memory (5k + 5k) | 0.30 ± 0.08 | −2.58 ± 0.58 | 12.2 ± 0.1 | 0.43 ± 0.48 | 0.05 ± 0.03 | 17.8 ± 5.5 | 48.1 ± 5.4 |
| DQN (10k) | 0.42 ± 0.05 | −2.10 ± 0.22 | 12.2 ± 0.2 | 0.05 ± 0.03 | 0.05 ± 0.03 | 24.2 ± 3.3 | 30.0 ± 5.9 |

**Training mazes (layouts 3000–3049)**

| Arm | Success | Episode reward | Steps to goal* | Hazard rate | Collisions |
|---|---|---|---|---|---|
| DQN (5k) | 0.91 ± 0.05 | 0.44 ± 0.22 | 12.9 ± 0.1 | 0.00 ± 0.01 | 3.0 ± 1.5 |
| DQN + mistake memory (5k + 5k) | 0.84 ± 0.10 | 0.19 ± 0.45 | 12.8 ± 0.1 | 0.01 ± 0.01 | 1.9 ± 1.7 |
| DQN (10k) | 0.94 ± 0.03 | 0.59 ± 0.12 | 12.9 ± 0.1 | 0.01 ± 0.01 | 0.8 ± 1.0 |

\*Mean steps of successful episodes only. Every seed of every arm reached the goal at
least once, so no seed is left out.

**Held-out success per seed (100–109)**

| Arm | Seeds 100–109 |
|---|---|
| DQN (5k) | 0.38, 0.28, 0.40, 0.40, 0.40, 0.48, 0.34, 0.36, 0.30, 0.40 |
| DQN + mistake memory | 0.26, 0.22, 0.34, 0.14, 0.38, 0.14, 0.30, 0.34, 0.50, 0.36 |
| DQN (10k) | 0.36, 0.32, 0.46, 0.42, 0.50, 0.52, 0.44, 0.42, 0.32, 0.48 |

![Success on training and held-out mazes](../../docs/figures/long_memory_success.png)

![Greedy success on the training mazes during training](../../docs/figures/long_memory_curves.png)

## Pre-registered comparisons

| Comparison | Metric (split) | Difference | p | Status |
|---|---|---|---|---|
| **Mistake memory − DQN (5k)** | **success (held-out)** | **−0.08** | **0.082** | **Primary: not significant → "no evidence"** |
| Mistake memory − DQN (5k) | success (training) | −0.07 | 0.183 | exploratory |
| Mistake memory − DQN (5k) | reward (held-out) | −0.32 | 0.261 | exploratory |
| Mistake memory − DQN (5k) | collisions (held-out) | −5.4 | 0.087 | exploratory |
| Mistake memory − DQN (5k) | hazard rate (held-out) | +0.01 | 0.702 | exploratory |
| DQN (10k) − mistake memory | success (held-out) | +0.13 | 0.009 | exploratory |
| DQN (10k) − mistake memory | success (training) | +0.10 | 0.020 | exploratory |
| DQN (10k) − DQN (5k) | success (held-out) | +0.05 | 0.118 | exploratory |

All other pairs and metrics are listed at the end of `run.log`; none of the steps-to-goal or
hazard-rate differences came close to significance (all p ≥ 0.41). The p-values come from
the project's permutation test, a Monte Carlo estimate (20,000 resamples) for 10 vs 10
seeds. A check of the primary comparison with 200,000 resamples gave p = 0.081.

**Claim rule applied.** The primary difference is negative and p ≥ 0.05, so: no evidence
that long-term mistake memory improves held-out success. The condition for attributing a
benefit to *mistake* memory (not being worse than `dqn_10k`) is moot, and it would have
failed: the memory-matched control was ahead on held-out success.

## Observations (exploratory, not claims)

- **Same total memory spent on ordinary experience did better.** DQN (10k) beat the variant
  on held-out and training success (p = 0.009 and 0.020). This comparison was secondary
  and is reported without a multiple-comparison correction, so it suggests, rather than
  establishes, that a reserve of mistakes is a worse use of memory than more recent
  ordinary experience.
- **The variant bumped into walls less but looped more.** On held-out mazes it averaged 5.4
  fewer collisions than the baseline (p = 0.087) but 12 more revisits per episode. This
  echoes `terminal_hazards`: avoiding the penalised event is not the same as reaching the
  goal.
- **Hazard steps versus hazard rate.** The variant's mean of 0.43 hazard steps per episode
  comes from three seeds (101, 105, 106) where 3–5 of the 50 episodes spent
  51–98 steps in total moving on and off hazards. The share of episodes that touched a hazard at all (0.05) was
  the same in every arm.
- **Slower and less stable learning.** The variant's mean training-layout success was the
  lowest of the three arms at every evaluation point after the start (0.40 vs 0.56 and 0.68 at episode 500),
  and its spread across seeds was about twice the baseline's (held-out SD 0.11 vs 0.06).
  Seed 105 reached only 0.46 on the training mazes.
- **Steps to goal is uninformative here.** Successful episodes took about 12 steps in every
  arm. The local view can only solve short routes, so the successes are the easy mazes.

## Limitations

- One task family (9×9 hazard mazes, local view), one budget (3,000 episodes) and one
  setting of the reserve (5,000 mistakes, 25% of each batch). A smaller sampling share, a
  reserve sampled only once ordinary experience is exhausted, or a task where mistakes are
  rare could behave differently. None of these were tested, and testing them on these
  seeds would mean tuning on the test set.
- With 10 seeds per arm, effects smaller than about 0.06–0.07 in success cannot be
  detected.
- Mistake memory over-weights mistakes: they were 27–29% of all transitions, so with a
  quarter of each batch from the reserve, about 46% of a batch was mistake transitions,
  without an importance-weight correction. That is part of the method as specified,
  and may explain its lower (but not significantly lower) success.

## Deviations from the protocol

None in design or analysis. Execution notes:

- The pilot ran in 137 s, not the estimated 9 minutes.
- With 4 parallel processes competing for the CPU, the main experiment took 2,699 s
  (45 min), longer than the estimated 15–25 min.

Neither affected any setting or result.

The experiment ran on the source files whose checksums are listed in `PROTOCOL.md`.
Afterwards only `src/visualization.py` was changed, to add this study's two figures to
`make_report_figures`. That code draws charts from saved results and plays no part in
training or evaluation.

## Reproduce

```bash
python -m src.evaluation --experiment long_memory_pilot    --root runs/experiments_long_memory
python -m src.evaluation --experiment long_memory_mistakes --root runs/experiments_long_memory --workers 4
```

The second command takes about 45 minutes with 4 processes on a laptop CPU, or an estimated 70
minutes with `--workers 1` (30 runs × the pilot's 137 s). Both give the same results (tested). Rerun the same command to
resume after an interruption.
