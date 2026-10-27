# shaping_speed: analysis

Protocol: `PROTOCOL.md` (written before any run; covers this experiment and
`shaping_generalization`). Raw outputs: `results.csv` (final greedy evaluation per seed),
`curves.csv` (greedy evaluation every 25 episodes), `summary.csv`, `buffer_diagnostics.csv`,
`run.log`.

## Answer

**Yes: with potential-based reward shaping, both agents learned this maze sooner.** Both
pre-registered tests are significant after Holm-Bonferroni correction. The effect was small
for tabular Q, which was already fast, and large for DQN.

| Hypothesis | Without shaping | With shaping | Seeds ahead / tied / behind | p | Holm-adjusted p |
|---|---|---|---|---|---|
| H1: tabular Q learning speed | 0.931 | 0.986 | 9 / 1 / 0 | 0.0008 | 0.0017 |
| H2: DQN learning speed | 0.388 | 0.794 | 7 / 2 / 1 | 0.0020 | 0.0020 |

Learning speed = greedy success averaged over the 16 evaluations during training
(episodes 25-400), one value per seed, compared with a two-sided permutation test. The
one DQN seed where shaping was behind is 208 (0.688 vs 0.750).

## Final performance (greedy, mean ± 95% CI over seeds 200-209)

| Arm | Success | Path efficiency | Steps | Collisions | Revisits |
|---|---|---|---|---|---|
| Tabular Q | 1.00 ± 0.00 | 1.00 ± 0.00 | 16.0 ± 0.0 | 0.0 | 0.0 |
| Tabular Q + shaping | 1.00 ± 0.00 | 1.00 ± 0.00 | 16.0 ± 0.0 | 0.0 | 0.0 |
| DQN | 0.60 ± 0.37 | 0.60 ± 0.37 | 49.6 ± 31.0 | 19.6 ± 29.6 | 19.7 ± 29.7 |
| DQN + shaping | 1.00 ± 0.00 | 1.00 ± 0.00 | 16.0 ± 0.0 | 0.0 | 0.0 |

The shortest path is 16 steps. Every successful final episode, shaped or not, took it, as
the theory predicts: shaping changed how fast the agents learned, not what they learned.

**Exploratory (not a primary test):** shaped DQN finished with success 1.00 against 0.60
for plain DQN, but with 10 seeds this is not significant (p = 0.093). Plain DQN failed at
the end in 4 seeds (204, 205, 207, 209); shaped DQN failed in none.

## When each run first reached greedy success 1.0

| Arm | Seeds 200-209 (training episode; "never" = not by episode 400) |
|---|---|
| Tabular Q | 50, 50, 50, 50, 75, 50, 50, 50, 50, 50 |
| Tabular Q + shaping | 25, 50, 50, 25, 50, 25, 50, 25, 25, 25 |
| DQN | 50, 75, 275, 150, never, never, 175, 375, 100, never |
| DQN + shaping | 25, 50, 50, 25, 100, 50, 25, 50, 25, 25 |

Final greedy success per seed, DQN: 1, 1, 1, 1, 0, 0, 1, 0, 1, 0. Seeds 204, 205 and 209
never solved the maze; seed 207 solved it at episode 375 and lost it again by episode 400.
Every shaped DQN run had solved it by episode 100 and still solved it at the end.

![Greedy success during training](../../docs/figures/shaping_speed_curves.png)

Shaped DQN dipped between episodes 75 and 125 (mean greedy success 0.2-0.4) before
settling at 1.0 from episode 175. Plain DQN never passed 0.6.

## What this does and doesn't show

- It shows faster learning on **one** 9x9 maze (layout 7000), with shaping strength fixed
  at 1.0. It does not show that shaping speeds up learning in general, or at other
  strengths.
- Shaping uses the maze distance to the goal, which the experimenter computes from the
  full map. The agent is given a hint it could not compute itself. That is fine for
  speeding up training, but it means a shaped agent had more help, not that it is a
  better learner.
- `total_reward` and every other metric are the task's own; the shaping term is logged
  separately as `r_shaping`.

## Deviations from the protocol

None. Both experiments ran once each with the protocol's settings
(`--workers 4`, 11 minutes for this one). The background job that ran them reported exit
code 1 because of a `tail` command after both runs had finished; the experiments
themselves completed normally (see `run.log`).
