# Stochastic evaluation of saved networks: pre-registered protocol

Stage K. Written 2026-11-09, **before** any stochastic evaluation of the saved
networks. The project owner approved the plan on 2026-11-09 ("start k"). This file is
not edited after results exist; anything that changes is recorded under "Deviations"
in `ANALYSIS.md`.

## Question

Every result so far was measured with greedy play: the agent always takes the move it
rates best. A greedy policy with a local view can get stuck: it sees the same 5×5 window
twice, picks the same move, and loops until the 100-step limit. Stage I found, as an
exploratory result, that PPO playing by *sampling* its policy solved 0.90 of the held-out
mazes against 0.69 greedily, but DQN was never evaluated with any randomness, so that
was not a fair comparison. This study asks, with no further training:

1. Does a small amount of randomness (ε = 0.05, the exploration rate DQN ended its
   training with) change how many held-out mazes plain DQN solves?
2. Once both are allowed their natural randomness, does PPO (sampled) still solve more
   held-out mazes than DQN (ε = 0.05)?
3. Does the memory result of Stage F (GRU 0.73 vs MLP 0.29 held-out, greedy) hold when
   both play with ε = 0.05?

## Networks

The 80 final networks saved by three earlier pre-registered studies (weights-only
exports in `results/<study>/models/`, each already checked to reproduce its run's saved
results):

| Study | Arms (10 seeds each) | Seeds | Held-out layouts |
|---|---|---|---|
| `cnn_generalization` (Stage E) | `dqn`, `dqn_cnn`, `dqn_wide` | 500–509 | 9400–9449 |
| `rnn_generalization` (Stage F) | `dqn`, `dqn_rnn`, `dqn_wide` | 600–609 | 9600–9651 except 9619, 9632 |
| `ppo_generalization` (Stage I) | `dqn`, `ppo` | 700–709 | 9800–9849 |

Each network is played only on its own study's 50 held-out mazes (local 5×5 view,
max 100 steps). Nothing is trained and no network is changed.

## Evaluation modes

| Mode | Moves | Passes over the 50 mazes | Evaluation seed |
|---|---|---|---|
| `greedy` | the best move (ties at random) | 1 | 10,000 + run seed (as in the studies) |
| `eps0.05` | the best move; with probability 0.05 a uniformly random one instead | 5 | 20,000 + run seed |
| `eps0.1` | same, probability 0.1 | 5 | 20,000 + run seed |
| `eps0.2` | same, probability 0.2 | 5 | 20,000 + run seed |
| `sample` (PPO only) | drawn from the policy's probabilities | 5 | 20,000 + run seed |

The agent is asked for its best move at every step even when a random move replaces it,
so the GRU's memory sees every observation, as during training. The ε modes apply to
all 80 networks, PPO included. A network's value in a mode is its mean over all
episodes of that mode (250 for the stochastic modes). The greedy mode must reproduce
each network's saved held-out success, steps, reward, collisions, revisits and timeouts
exactly; the run stops if any differs.

## Primary outcomes and tests

Outcome: held-out success (fraction of episodes reaching the goal).

- **H1** (randomness for DQN): plain `dqn`, `eps0.05` vs `greedy`, over the 30 plain DQN
  networks of all three studies. Paired: each network is compared with itself. Two-sided
  sign-flip permutation test on the 30 differences (`paired_permutation_test`, 200,000
  Monte Carlo resamples, seed 0).
- **H2** (PPO vs DQN, both with their own randomness): Stage I networks, `ppo` in
  `sample` mode vs `dqn` in `eps0.05` mode. Two-sided permutation test on the 10 vs 10
  per-seed values (`permutation_test`, 200,000 resamples, which makes it exact:
  C(20,10) = 184,756).
- **H3** (memory under randomness): Stage F networks, `dqn_rnn` vs `dqn`, both in
  `eps0.05` mode. Same unpaired test as H2.
- H1–H3 are corrected together with Holm-Bonferroni at α = 0.05.

**Reading the results.** A significant H1 is reported as "a little randomness raises
(or lowers) plain DQN's held-out success *with these networks*". A significant H2 or H3
is reported in the direction found. Anything else is reported as "no evidence of a
difference", never as "no effect" and never as a trend. The repeats measure the
randomness of play; the unit of every test is the network (seed), so run-to-run
variation is what the tests use.

**What is already known.** Before this protocol, PPO's sampled play had been measured
once (Stage I, one pass, evaluation seed 10,000 + run seed): 0.90 held-out. H2 uses new
seeds (20,000 + run seed) and 5 passes, but PPO's side of H2 is therefore not a blind
measurement; DQN's side, and everything in H1 and H3, was never measured.

## Secondary (exploratory, no correction, labelled as exploratory)

- `eps0.1` and `eps0.2` for every arm, and how success changes with ε.
- Paired `eps0.05` vs `greedy` for each other arm (`dqn_cnn`, `dqn_wide` in E and F,
  `dqn_rnn`, `ppo`) and `sample` vs `greedy` for PPO.
- Whether the earlier between-arm comparisons change under `eps0.05`: Stage E (`dqn_cnn`
  vs `dqn`, `dqn_wide` vs `dqn`), Stage F (`dqn_wide` vs `dqn`), Stage I (`ppo` vs `dqn`,
  both `eps0.05`).
- Steps, timeouts, collisions, revisits and path efficiency in each mode.
- Success split by shortest path length (12 vs longer), and how much of the change comes
  from mazes the greedy policy timed out on.
- How much success varies between the 5 passes of the same network.

## Compute and execution

CPU only, no training. `python -m src.stochastic_study --out runs/experiments_stochastic
--workers 2`, started in its own PowerShell window (outside Claude Code). A timing check
before this protocol (one network per type, one process) measured 0.8–1.5 ms per
environment step and 7–22 s per stochastic mode per network; it also confirmed that the
greedy mode reproduces the saved held-out success of `cnn` `dqn` s500 (0.24), `dqn_cnn`
s500 (0.38), `rnn` `dqn_rnn` s600 (0.70) and `ppo` `ppo` s700 (0.72). It printed no
stochastic success values. Expected total: under an hour. Resumable: each network's
results are saved when it finishes and finished networks are skipped. Results go to
`results/stochastic_evaluation/` (`results.csv`, `episodes.csv.gz`), with `ANALYSIS.md`
written after the run. The analysis script `results/stochastic_evaluation/analyze.py` is
written before the run and is not changed afterwards except to fix errors, which are
recorded as deviations.

## Code state

Implemented and tested after commit c8bf11f (uncommitted at the time of writing):
`evaluate_agent(policy="greedy" | "epsilon" | "sample", epsilon, repeats)` (greedy
remains the default and unchanged; the agent's random generator and a recurrent agent's
memory are restored afterwards), `paired_permutation_test`, and `src/stochastic_study.py`
(the runner, which checks the greedy reproduction).
