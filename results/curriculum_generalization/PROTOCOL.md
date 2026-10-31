# Training curriculum: pre-registered protocol

Covers one experiment, `curriculum_generalization`. Written 2026-10-30 20:20 (ADT),
**before** any run of it. The project owner approved the plan (two arms, a fixed
schedule, fresh seeds and layouts) first. This file is not edited after results exist;
anything that changes is recorded under "Deviations" in `ANALYSIS.md`.

## Question

Earlier studies (reward shaping, dueling, 3-step returns) made DQN learn one maze
faster, but none of them made it solve more mazes it had never seen. Does the **order**
in which the training mazes are shown matter? Specifically: if DQN meets its easiest
training mazes first, does it

1. solve more held-out mazes at the end, and
2. learn its training mazes sooner?

## The two arms

Both arms train on the **same 50 mazes**, for the same number of episodes, with the
same agent and ε schedule. Only the order in which mazes are drawn differs, so any
difference comes from ordering, not from more or different data.

- **`dqn`** (control): each episode draws one of the 50 training mazes uniformly at
  random, as in every earlier generalization experiment.
- **`dqn_curriculum`**: the 50 mazes are ranked from easiest to hardest, and episode
  *e* draws uniformly from the easiest *k(e)* of them:

  k(e) = 10 + ⌊40 · e / 1500⌋ for e < 1500, and 50 from episode 1500 on.

  So the arm starts on the 10 easiest mazes, adds one about every 37 episodes, and
  trains on all 50 (exactly like the control) for the last 1,500 episodes.

**Difficulty ranking** (`envs.rank_layouts`), fixed in advance:
1. shortest start→goal path (BFS), shorter is easier;
2. ties: number of dead ends (open cells, other than start and goal, with exactly one
   open neighbour), fewer is easier;
3. remaining ties: layout seed.

The second key was added before any run, after checking the layouts. 29 of the 50
training mazes share the minimum path length (12), so without it the first ~700
episodes would have used an arbitrary subset rather than easy mazes. Dead-end counts
range 0–4 among those 29 and are nearly uncorrelated with path length (r = −0.06). The
training mazes' path lengths are 12 (29 mazes), 14 (5), 16 (8), 18 (1), 20 (3), 24 (3)
and 28 (1).

The start size (10), the full-pool episode (1,500, before ε reaches its floor at
2,000) and the ranking were chosen once and are not tuned. No other schedule is tried.

## Task, layouts and seeds

Same task as `generalization`, `shaping_generalization` and
`dqn_variants_generalization`: the `maze` preset (9×9, loop_prob 0.1, no hazards, max 100
steps) with a 5×5 local view plus the direction to the goal. Standard DQN (Double DQN
target, MLP 128×128, Adam lr 1e-3, γ 0.99, batch 64, replay 20,000, learning_starts 500,
train_freq 2, target_update 100, Huber loss, gradient clip 10). 3,000 episodes, ε 1.0 →
0.05 over 2,000.

- **Training layouts**: 9100–9149.
- **Held-out layouts**: 9200–9250 except 9225 (50 layouts). Layout 9225 generates the
  same maze as training layout 9139, so it was replaced by 9250 before any run. A
  test now checks that no held-out maze is identical to a training maze. (The same
  check showed that `shaping_generalization` and `dqn_variants_generalization` each
  had one such duplicate. Both arms in each saw the same mazes, so those comparisons
  stay fair, but their "held-out" set was 49/50 unseen.)
- **Seeds**: 400–409 (10 per arm). Evaluation seed = 10,000 + run seed, as before.
- Greedy evaluation on all 50 training layouts (one episode each) every 500 episodes,
  for **both** arms. The held-out layouts are played once, greedily, at the end.

None of these seeds or layouts were used by any earlier experiment.

## Primary outcomes and tests

Each test compares the 10 per-seed values of the two arms with a two-sided permutation
test (20,000 resamples).

- **H1**: final greedy success on the 50 held-out layouts, `dqn_curriculum` vs `dqn`.
- **H2**: learning speed, i.e. greedy success on the 50 training layouts averaged over the
  6 evaluations during training (episodes 500, 1000, …, 3000), `dqn_curriculum` vs
  `dqn`.
- H1 and H2 are corrected together with Holm-Bonferroni at α = 0.05.

**Reading the results.** A significant difference is reported as the curriculum
helping (or hurting). Anything else is reported as "no evidence of a difference", never
as "no effect" and never as a trend. A note for H2: during the first 1,500 episodes the
curriculum arm has trained on fewer of the 50 mazes that are being evaluated, so it may
score lower early on for that reason alone. This is part of what H2 measures, not a
flaw to correct afterwards.

## Secondary (exploratory, no correction, labelled as exploratory)

- Final greedy success and path efficiency on the training layouts.
- Held-out steps, collisions, revisits and path efficiency.
- Held-out success split by path length (12 vs longer), computed afterwards by
  re-evaluating each run's final checkpoint with the same evaluation seed. The split
  is checked to reproduce each run's saved held-out success first.
- The learning curve at each evaluation point, especially episode 500 vs 1,500.

## Compute and execution

CPU only (Windows laptop), `--workers 4`, runs in `runs/experiments_curriculum/`
(`python -m src.evaluation --experiment curriculum_generalization --root
runs/experiments_curriculum --workers 4`). Runs are resumable; a crash is resumed, not
restarted with different settings. Results are copied to
`results/curriculum_generalization/` with an `ANALYSIS.md` written after the run. The
code in `src/` is not changed while the experiment runs.

## Code state

Curriculum implemented and tested after commit 029a64a (uncommitted at the time of
writing): `envs.rank_layouts`, `envs.count_dead_ends`, `training.curriculum_size`,
`TrainConfig.curriculum`, `Experiment.arm_curriculum`. With no curriculum, training is
unchanged: the first 500 episodes of `dqn_variants_generalization` DQN seed 300
reproduce the saved log exactly.
