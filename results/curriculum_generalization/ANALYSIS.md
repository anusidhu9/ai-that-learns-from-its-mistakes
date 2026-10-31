# curriculum_generalization: analysis

Protocol: [`PROTOCOL.md`](PROTOCOL.md), hypotheses H1 and H2 (written before any run).
Raw outputs: `results.csv` (final greedy evaluation per seed and split), `curves.csv`
(greedy evaluation on all 50 training mazes every 500 episodes), `summary.csv`,
`buffer_diagnostics.csv`, `run.log`.

## Answer

**The easiest-first curriculum made DQN learn its training mazes more slowly, and there
is no evidence that it changed success on new mazes.** The curriculum arm was behind
plain DQN at every evaluation point, in all 10 seeds on learning speed (H2, supported in
the opposite direction to the one hoped for). On the 50 held-out mazes it was also
behind, 0.49 against 0.57, in 8 of 10 seeds, but that difference is not significant
(H1, Holm-adjusted p = 0.093). By the pre-registered rule, H1 is "no evidence of a
difference". It is not evidence that the curriculum is harmless on new mazes either.

| Hypothesis | DQN | DQN + curriculum | Seeds ahead / tied / behind | p | Holm-adjusted p | Verdict |
|---|---|---|---|---|---|---|
| H1: held-out success | 0.57 ± 0.06 | 0.49 ± 0.08 | 2 / 0 / 8 | 0.093 | 0.093 | No evidence of a difference |
| H2: learning speed | 0.851 | 0.722 | 0 / 0 / 10 | 0.0001 | 0.0003 | **Curriculum slower** |

*Learning speed = greedy success on the 50 training mazes averaged over the evaluations
at episodes 500, 1000, …, 3000. "Ahead" means the curriculum arm scored higher.*

## Results (greedy, mean ± 95% CI over seeds 400–409)

**Held-out mazes (layouts 9200–9250 except 9225, never seen in training)**

| Arm | Success | Path efficiency | Steps | Collisions | Revisits | Task reward |
|---|---|---|---|---|---|---|
| DQN | 0.57 ± 0.06 | 0.57 ± 0.06 | 50.7 ± 5.3 | 7.4 ± 4.6 | 32.6 ± 7.0 | −1.00 ± 0.30 |
| DQN + curriculum | 0.49 ± 0.08 | 0.49 ± 0.08 | 57.6 ± 6.8 | 16.1 ± 6.3 | 32.3 ± 9.2 | −1.59 ± 0.39 |

**Training mazes (layouts 9100–9149)**

| Arm | Success | Path efficiency | Steps | Collisions | Revisits | Task reward |
|---|---|---|---|---|---|---|
| DQN | 0.90 ± 0.05 | 0.89 ± 0.05 | 22.9 ± 4.0 | 1.7 ± 1.5 | 7.9 ± 4.9 | 0.42 ± 0.19 |
| DQN + curriculum | 0.84 ± 0.04 | 0.83 ± 0.04 | 27.6 ± 3.1 | 6.3 ± 3.6 | 8.7 ± 3.3 | 0.06 ± 0.23 |

**Held-out success per seed (400–409)**

| Arm | 400 | 401 | 402 | 403 | 404 | 405 | 406 | 407 | 408 | 409 |
|---|---|---|---|---|---|---|---|---|---|---|
| DQN | 0.54 | 0.64 | 0.56 | 0.48 | 0.66 | 0.72 | 0.60 | 0.52 | 0.58 | 0.42 |
| DQN + curriculum | 0.52 | 0.42 | 0.52 | 0.46 | 0.68 | 0.66 | 0.34 | 0.42 | 0.40 | 0.48 |

**Greedy success on the 50 training mazes during training (mean over seeds)**

| Episode | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| DQN | 0.66 | 0.81 | 0.89 | 0.92 | 0.93 | 0.90 |
| DQN + curriculum | 0.57 | 0.62 | 0.73 | 0.78 | 0.78 | 0.84 |

![Greedy success during training](../../docs/figures/curriculum_curves.png)

![Success on training and held-out mazes](../../docs/figures/curriculum_success.png)

## Exploratory results (not pre-registered as primary; no correction)

- **The gap outlasts the curriculum.** From episode 1,500 both arms draw from the same 50
  mazes, yet the curriculum arm was still 0.14–0.15 behind at episodes 2,000 and 2,500
  (p ≤ 0.0005 each). It narrowed to 0.06 by episode 3,000 (p = 0.058). At episode 500,
  when the curriculum arm had seen only its 10–23 easiest mazes, the gap was 0.08
  (p = 0.135).
- **Final training-maze performance**: success 0.84 vs 0.90 (p = 0.058), path efficiency
  0.83 vs 0.89 (p = 0.047), wall collisions 6.3 vs 1.7 per episode (p = 0.015).
- **Held-out mazes**: more wall collisions with the curriculum (16.1 vs 7.4 per episode,
  p = 0.023); revisits unchanged (p = 0.95); path efficiency 0.49 vs 0.57 (p = 0.090).
- **By maze difficulty** (held-out success re-evaluated from each run's final
  checkpoint, which reproduces the saved results exactly): on the 25 held-out mazes with
  the minimum path length (12), 0.65 with the curriculum vs 0.74 without (p = 0.120); on
  the 25 longer ones, 0.33 vs 0.40 (p = 0.188). The curriculum arm was behind on both
  the easy and the hard mazes.
- **How training time was spent** (mean episodes per run, mazes grouped by difficulty
  rank): the curriculum arm played its 10 easiest mazes 920 times (DQN 608), the middle
  30 1,740 times (1,806), and the 10 hardest only 339 times (586). The hardest maze was
  played 29 times on average, against 59. The hardest mazes also enter the curriculum
  last, when ε has fallen to about 0.29 (it is 0.05 from episode 2,000), so they were
  explored less when they were first met.

## Interpretation

Showing DQN its easy mazes first did not help it here. It learned the full training set
more slowly and did not do better on new mazes, and its point estimate there was lower.
The data cannot say why, but two features of this design are consistent with it, and
neither was tested separately:

1. **Less practice on hard mazes.** The total episode budget is fixed, so extra episodes
   on easy mazes come out of the hard ones (339 against 586 episodes on the hardest 10).
2. **Hard mazes arrive when exploration has mostly ended.** With ε tied to the episode
   count, the curriculum arm meets its hardest mazes at ε ≈ 0.3, while the control met
   them throughout the high-ε phase. The gap lasting well past episode 1,500 fits this:
   the early experience in the replay buffer and the network came mostly from easy mazes.

On these 9×9 mazes the "easy" mazes are not much easier: 29 of the 50 training mazes
have the minimum path length, and the agent sees only a 5×5 window, so a short maze
need not look different from a long one locally. A curriculum may matter more when
difficulty differs more, e.g. by maze size, which this study did not test.

Across Stages C, D and H, the held-out success of plain DQN has varied between studies
(0.48, 0.36 and now 0.57) with similar shares of easy held-out mazes (27, 26 and 25 of
50 with path length 12). Only comparisons within a study, on the same mazes and seeds,
are meaningful.

## Replay buffer

From `buffer_diagnostics.csv`: runs added about 112,000 (DQN) and 117,000 (curriculum)
transitions; both 20,000-transition buffers overwrote most of them.

## Deviations from the protocol

- **The run was interrupted once.** With `--workers 4`, Claude Code stopped the
  experiment because the laptop was low on memory. By then 8 DQN runs had finished and 4
  more runs were partway through. As the protocol requires, it was resumed with the same
  command, not restarted, using `--workers 2` to need less memory. The number of workers
  does not change results (each run is independently seeded; tested). Resumed runs
  continue from their last checkpoint and reproduce an uninterrupted run exactly
  (tested). `run.log` is the output of the resumed run; the first attempt's buffered
  output was lost when it was stopped.
- Otherwise none. Before any run, the protocol already recorded the dead-end tie-breaker
  in the difficulty ranking and the replacement of held-out layout 9225 by 9250.
