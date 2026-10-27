# Dueling DQN and n-step returns: pre-registered protocol

Covers two experiments, `dqn_variants_speed` and `dqn_variants_generalization`. Written
2026-10-26 22:56 (ADT), **before** any run of either experiment. The plan (two
experiments, n = 3 fixed, fresh seeds and layouts) was approved by the project owner
first. This file is not edited after results exist; anything that changes is recorded
under "Deviations" in each experiment's `ANALYSIS.md`.

## Question

Do two standard DQN improvements help on these mazes?

1. **Speed**: does a dueling network, or learning from 3-step returns, make DQN learn a
   fixed maze sooner?
2. **Generalization**: does either make DQN solve more mazes it has never seen?

## The variants

- **Dueling** (Wang et al., 2016): the network's last layer is replaced by two heads, a
  state value V(s) and per-action advantages A(s, a), combined as
  Q = V + A − mean(A). Same hidden layers (128×128) as plain DQN.
- **3-step returns** (as in Rainbow, Hessel et al., 2018): each stored transition carries
  the discounted sum of the next 3 rewards and bootstraps from the state 3 steps later
  with γ³. At the end of an episode the remaining shorter transitions are stored with
  γ^k (k < 3); after a timeout they still bootstrap. The returns come from the
  exploring policy and are not corrected for it.
- n = 3 is fixed in advance (Rainbow's value). No other n is tried.

Every other setting is the standard DQN used in all earlier experiments: Double DQN
target, MLP 128×128, Adam lr 1e-3, γ 0.99, batch 64, replay 20,000, learning_starts 500,
train_freq 2, target_update 100, Huber loss, gradient clip 10. With both options off the
agent reproduces earlier results exactly (checked on `hazard_maze`, DQN and DQN + PER,
seeds 0–1).

## Arms

| Experiment | Arms |
|---|---|
| `dqn_variants_speed` | `dqn`, `dqn_dueling`, `dqn_nstep`, `dqn_dueling_nstep` |
| `dqn_variants_generalization` | `dqn`, `dqn_dueling`, `dqn_nstep` |

The combined arm (`dqn_dueling_nstep`) runs only in the cheaper speed experiment and is
exploratory.

## Task, layouts and seeds

None of these seeds or layouts were used by any earlier experiment.

- **`dqn_variants_speed`**: `maze` preset (9×9, loop_prob 0.1, no hazards, max 100 steps,
  full-map observation), one fixed layout, seed 8000 (shortest path 16 steps). 400
  episodes, ε 1.0 → 0.05 over 300. Greedy evaluation (10 episodes) every 25 episodes.
- **`dqn_variants_generalization`**: `maze` preset with a 5×5 local view plus the
  direction to the goal. Training layouts 8100–8149 (one drawn at random each episode);
  held-out layouts 8200–8249, used only in the final evaluation. 3,000 episodes, ε 1.0 →
  0.05 over 2,000. Greedy evaluation on the training layouts every 500 episodes; at the
  end, one greedy episode on each of the 50 held-out layouts.
- **Seeds**: 300–309 (10 per arm). Evaluation seed = 10,000 + run seed, as before.

## Primary outcomes and tests

Each test compares 10 per-seed values with a two-sided permutation test (20,000
resamples).

- **H1** (`dqn_variants_speed`): learning speed (greedy success averaged over the 16
  evaluations during training, episodes 25–400), `dqn_dueling` vs `dqn`.
- **H2** (`dqn_variants_speed`): the same, `dqn_nstep` vs `dqn`.
- H1 and H2 are corrected together with Holm-Bonferroni at α = 0.05.
- **H3** (`dqn_variants_generalization`): final greedy success on the 50 held-out
  layouts, `dqn_dueling` vs `dqn`.
- **H4** (`dqn_variants_generalization`): the same, `dqn_nstep` vs `dqn`.
- H3 and H4 are corrected together with Holm-Bonferroni at α = 0.05.

**Reading the results.** A significant difference is reported as the variant helping (or
hurting). Anything else is reported as "no evidence of a difference", never as "no
effect" and never as a trend.

## Secondary (exploratory, no correction, labelled as exploratory)

The combined arm against each other arm; final greedy success and path efficiency on
the training layout(s); learning-curve area in `dqn_variants_generalization`; steps,
collisions and revisits; the episode at which each run first reached greedy success 1.0;
how many runs solved the fixed maze and then lost it again.

## Compute and execution

CPU only (Windows laptop), `--workers 4`, runs in `runs/experiments_dqn_variants/`. Runs
are resumable; a crash is resumed, not restarted with different settings. Results are
copied to `results/<experiment>/` with an `ANALYSIS.md` written after the run. The code
in `src/` is not changed while the experiments run.

## Code state

Dueling and n-step implemented and tested after commit 129af9f (uncommitted at the time
of writing).
