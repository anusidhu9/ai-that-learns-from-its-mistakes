# MIND: an AI that learns from its own mistakes — final report

## 1. Summary

This project trains reinforcement-learning agents in grid worlds where they are rewarded for reaching a goal and penalised for mistakes (wall collisions, hazards, revisiting cells, timeouts), then tests whether they really make fewer mistakes. Four agents (a random baseline, tabular Q-learning, a Deep Q-Network and DQN with prioritized experience replay) were compared in five experiments with 3–5 independent seeds each, using greedy evaluation (exploration off), 95% confidence intervals and exact permutation tests.

Main findings:

1. **Both learning agents learned from their mistakes on every fixed layout.** Greedy success rose from 0.00–0.40 before training to 1.00 for tabular Q-learning and DQN in all three fixed-layout experiments with non-terminal hazards (p = 0.008 against random, the smallest p-value possible with five seeds per agent).
2. **Fewer mistakes is not the same as better behaviour.** When stepping on a hazard ended the episode, DQN learned in 4 of 5 seeds to walk into the nearest hazard after 6 steps. Under the reward design, a quick death (−0.51) cost less than an unsuccessful 100-step search (about −3.6). Its mistake count fell to 0.8 per episode while its success rate was 0.20, against 1.00 for tabular Q-learning (p = 0.048).
3. **Generalization to unseen mazes is suggestive but not established.** On 50 held-out mazes DQN succeeded 40–52% of the time per seed against 26–30% for tabular Q-learning and 6–8% for random, and DQN's successes were near-shortest paths while tabular Q's were mostly long wanders. With three seeds, however, no difference can reach p < 0.10.
4. **Prioritized experience replay gave no measurable benefit** on these tasks; one of its five fixed-maze runs ended with a greedy policy stuck against a wall.

## 2. Question and approach

*Does an agent that is penalised for its mistakes measurably stop making them, and how should that be measured?*

The project was built in stages, each with tests: environments (Stage 1), a random baseline and tabular Q-learning (2), DQN with replay (3), persistent logs, checkpoints and exact resumption (4), multi-seed evaluation (5), visualization and an interactive dashboard (6–7). The project rule throughout was that no improvement is claimed unless the experiments support it.

## 3. Environments

A `GridWorld` holds walls, optional hazards, a start and a goal. Actions are up, right, down and left; moving into a wall leaves the agent in place. Episodes end at the goal, on a hazard if hazards are terminal, or after 100 steps (a truncation, not a true terminal state).

| Event | Reward |
|---|---|
| each step | −0.01 |
| reach the goal | +1.0 |
| walk into a wall | −0.05 |
| step on a hazard | −0.5 |
| revisit a cell in the same episode | −0.02 |
| timeout after 100 steps | −0.1 |

Penalties are added to the step cost. An episode's mistakes are its collisions + hazards + revisits + timeouts.

| Preset | Layout | Hazards |
|---|---|---|
| `maze` | 9×9 maze from a randomized depth-first search, 10% of removable walls removed to create loops; start top-left, goal bottom-right | none |
| `maze_hazards` | 9×9 maze, 30% loops | 4, non-terminal |
| `open` | 8×8 grid, 10% random walls, random start and goal | none |
| `open_hazards` | as `open` | 5, non-terminal |

Every layout is generated deterministically from a seed and is guaranteed to have a hazard-free path to the goal. The length of that path (by breadth-first search) is the **optimal length** used for path efficiency.

**Observations.** All agents receive the same flat float32 vector. In `grid` mode it is four one-hot planes (walls, hazards, goal, agent) over the whole map; on a fixed layout this makes the state equivalent to the agent's position. In `local` mode it is a 5×5 window around the agent (walls, hazards, goal; out of bounds counts as wall) plus the normalised direction to the goal, which does not depend on absolute position and so can transfer between layouts. The revisit penalty depends on the agent's history, which neither mode contains, so that component of the reward is not Markovian; including the visited set would have made the tabular state space intractable.

## 4. Agents

**Random.** Uniform random actions; never learns.

**Tabular Q-learning.** A dictionary maps each observation to four action values, updated after every step:

```
target  = r                               if s' is terminal
        = r + γ · max_a' Q(s', a')        otherwise (including timeouts)
Q(s,a) ← Q(s,a) + α · (target − Q(s,a))
```

The quantity `target − Q(s,a)` is the temporal-difference (TD) error: how wrong the agent's prediction was. A penalty produces a negative TD error that lowers the value of the action that caused it, which is the precise sense in which the agent learns from a mistake. Unvisited states start at Q = 0. Because every reward except the goal is negative, zero is optimistic, which drives systematic exploration.

**DQN** (Mnih et al., 2015). A multilayer perceptron (two hidden layers of 128 ReLU units) replaces the table. Each gradient step samples a batch of 64 past transitions and minimises

```
L(θ) = mean_i  w_i · Huber( y_i − Q(s_i, a_i; θ) )
y_i  = r_i + γ · Q(s'_i, argmax_a Q(s'_i, a; θ); θ⁻)     (Double DQN; y_i = r_i if terminal)
```

where θ⁻ is a target network copied from θ every 100 gradient steps. Experience replay (a 20,000-transition buffer) breaks the correlation between consecutive steps and lets each mistake be learned from repeatedly.

**DQN + PER** (Schaul et al., 2016). Transition *i* is replayed with probability `P(i) = p_i^α / Σ p_k^α`, where `p_i = |TD error_i| + 0.001` and α = 0.6, so the most surprising experiences are replayed most. The resulting bias is corrected with importance weights `w_i = (N·P(i))^(−β) / max w`, with β annealed from 0.4 to 1. Sampling uses a sum tree (O(log N)).

**Exploration.** ε-greedy, decaying linearly from 1.0 to 0.05 (over 300 episodes in the fixed-layout experiments, 2,000 in the generalization experiment). Greedy actions break ties randomly.

| Hyperparameter | Tabular Q | DQN / DQN + PER |
|---|---|---|
| discount γ | 0.99 | 0.99 |
| learning rate | α = 0.5 | Adam, 1e-3 |
| batch size / buffer | — | 64 / 20,000 |
| learning starts after | — | 500 steps |
| gradient step every | every step | 2 env steps |
| target network update | — | every 100 gradient steps |
| loss / gradient clip | — | Huber / norm 10 |
| PER α, β | — | 0.6, 0.4 → 1 over 20,000 updates |

DQN's `train_freq` and `target_update` were tuned on the fixed maze during development (Section 8). No other tuning was done.

## 5. Evaluation protocol

- **Greedy evaluation.** Every reported result is measured with exploration switched off, using a separate environment and a separately seeded tie-breaking generator that is restored afterwards, so evaluation never changes training (tested).
- **When.** Before training (episode 0), every 50 episodes (every 500 for generalization), and at the end.
- **What.** On a fixed layout, 10 greedy episodes; with a layout pool, one episode on each of the 50 training layouts; for generalization, also one episode on each of 50 held-out layouts (seeds 1000–1049) never used in training.
- **Metrics.** Success rate; path efficiency (optimal length ÷ steps, for successful episodes that touched no hazard, otherwise 0); steps; mistakes by type; total reward.
- **Statistics.** Each seed contributes one value. Results are the mean across seeds ± the half-width of a 95% Student-t confidence interval. Two agents are compared with an exact two-sided permutation test on the per-seed values. With 5 seeds per agent the smallest attainable p-value is 2/252 ≈ 0.008; with 3 seeds it is 0.10.
- **Seeds.** 5 per agent for the fixed-layout experiments, 3 for generalization (each DQN run there is 3,000 episodes).

## 6. Results

### 6.1 Fixed maze (`fixed_maze`, 5 seeds, 400 episodes)

| Agent | Success | Path efficiency | Steps | Mistakes |
|---|---|---|---|---|
| Random | 0.06 ± 0.07 | 0.01 ± 0.01 | 97.6 ± 3.1 | 86.8 ± 2.7 |
| Tabular Q | **1.00 ± 0.00** | **1.00 ± 0.00** | 12.0 ± 0.0 | **0.0 ± 0.0** |
| DQN | **1.00 ± 0.00** | **1.00 ± 0.00** | 12.0 ± 0.0 | **0.0 ± 0.0** |
| DQN + PER | 0.80 ± 0.56 | 0.80 ± 0.56 | 29.6 ± 48.9 | 20.2 ± 56.1 |

Tabular Q-learning and DQN found the 12-step shortest path with no mistakes in all five seeds (each p = 0.008 against random). Tabular Q-learning was greedy-perfect by episode 50 in every seed; DQN needed 150 episodes and was less stable, dipping to 0.6 mean greedy success at episode 250 before recovering. DQN + PER succeeded in four seeds; in the fifth its final greedy policy walked into the same wall on all 100 steps.

![Greedy success during training on the fixed maze](docs/figures/fixed_maze_success.png)

Before training, tabular Q-learning behaved exactly like the random agent (all Q-values equal, ties broken randomly), whereas the untrained DQN scored 0: its network has no ties, so its greedy policy repeats the same moves.

**Where the mistakes went.** In a single 400-episode tabular run (seed 0), mistakes fell from 74.6 per episode in episodes 0–49 to 0.6 in episodes 350–399. Early mistakes were concentrated in the dead-end corridor below the start; the most repeated were walking into the wall at the start cell and revisiting the corridor cells. Collision and revisit penalties accounted for most of the lost reward early on and almost none at the end:

![Reward breakdown, first vs last 50 episodes](docs/figures/reward_breakdown.png)

### 6.2 Maze with non-terminal hazards (`hazard_maze`, 5 seeds)

| Agent | Success | Path efficiency | Steps | Hazards stepped on |
|---|---|---|---|---|
| Random | 0.12 ± 0.06 | 0.00 ± 0.00 | 96.9 ± 1.8 | 3.52 ± 2.13 |
| Tabular Q | **1.00 ± 0.00** | 0.20 ± 0.56 | 13.2 ± 3.3 | 0.80 ± 0.56 |
| DQN | **1.00 ± 0.00** | 0.80 ± 0.56 | 16.8 ± 3.3 | 0.20 ± 0.56 |
| DQN + PER | **1.00 ± 0.00** | 0.80 ± 0.56 | 16.8 ± 3.3 | 0.20 ± 0.56 |

All learning agents always reached the goal. They differed in the route: the safe path is 18 steps, and a 12-step route crosses one hazard. Crossing it returns about 0.38 per episode against about 0.82 for the safe route, so the shortcut is a genuine mistake, not a rational trade. Tabular Q-learning took the shortcut in 4 of 5 seeds, both DQN variants in 1 of 5.

The difference is **not statistically established** (p = 0.21). A development analysis of one tabular run showed the mechanism behind the shortcut: the safe route was explored mostly during the random early phase, picked up collision and revisit penalties then, and was rarely walked again once ε was small. Its estimated value stayed at 0.33 against a true value of about 0.76, and longer training did not correct it.

![Hazards stepped on in greedy evaluation](docs/figures/hazard_maze_hazards.png)

### 6.3 Terminal hazards (`terminal_hazards`, 5 seeds)

| Agent | Success | Steps | Mistakes | Hazards |
|---|---|---|---|---|
| Random | 0.00 ± 0.00 | 72.6 ± 1.9 | 63.2 ± 2.4 | 0.70 ± 0.12 |
| Tabular Q | **1.00 ± 0.00** | 18.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 |
| DQN | 0.20 ± 0.56 | 8.4 ± 6.7 | 0.8 ± 0.6 | 0.80 ± 0.56 |

The same maze, but a hazard ends the episode. Tabular Q-learning took the safe 18-step path in every seed (p = 0.008 against random). In 4 of 5 seeds, DQN's final greedy policy walked into the nearest hazard after 6 steps; the fifth found the safe path. DQN versus tabular Q: −0.80 success, p = 0.048.

![Greedy mistakes with terminal hazards](docs/figures/terminal_hazards_mistakes.png)

By mistake count alone, DQN looks almost as good as tabular Q-learning (0.8 vs 0.0 per episode) and far better than random (63.2). It is in fact the worst learning agent. The behaviour is rational under the reward function as long as the goal has not been found reliably: an unsuccessful 100-step search costs about −3.6 (step costs, collisions, revisits, timeout), while a hazard costs −0.51. Tabular Q-learning escapes the trap through optimistic initialisation, trying untried actions until it finds the goal; DQN's network generalises its pessimism to states it has not visited. This result is why the hazard presets use non-terminal hazards, and why success rate is the primary metric throughout.

### 6.4 Open grid with hazards (`open_hazards`, 5 seeds)

| Agent | Success | Path efficiency | Mistakes | Hazards |
|---|---|---|---|---|
| Random | 0.40 ± 0.23 | 0.07 ± 0.05 | 66.6 ± 12.9 | 11.40 ± 5.24 |
| Tabular Q | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.0 ± 0.0** | 0.00 ± 0.00 |
| DQN | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.0 ± 0.0** | 0.00 ± 0.00 |
| DQN + PER | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.0 ± 0.0** | 0.00 ± 0.00 |

All learning agents found the 7-step optimal path with no mistakes in every seed (p = 0.008 against random). This is the easiest task: a short path with few walls.

### 6.5 Generalization to unseen mazes (`generalization`, 3 seeds, 3,000 episodes)

Agents trained on 50 maze layouts with the local 5×5 observation, then were evaluated on those 50 layouts and on 50 they had never seen.

| Agent | Success, training mazes | Success, unseen mazes | Path efficiency, unseen | Mistakes, unseen |
|---|---|---|---|---|
| Random | 0.10 ± 0.10 | 0.07 ± 0.03 | 0.01 ± 0.01 | 88.9 ± 1.4 |
| Tabular Q | 0.69 ± 0.43 | 0.29 ± 0.06 | 0.12 ± 0.04 | 74.1 ± 9.8 |
| DQN | 0.75 ± 0.21 | 0.45 ± 0.15 | 0.45 ± 0.15 | 51.9 ± 15.3 |

Per seed on unseen mazes: DQN 0.44, 0.52, 0.40; tabular Q 0.30, 0.26, 0.30; random 0.06, 0.08, 0.06.

![Success on training and unseen mazes](docs/figures/generalization_success.png)

- **Both learners transfer something** to unseen mazes, beating random in every seed, but neither comes close to its training-set performance.
- **DQN was ahead of tabular Q-learning in every seed** (difference +0.17), but with three seeds the smallest attainable p-value is 0.10, so this is **not statistically established**.
- **The quality of the successes differs.** DQN's path efficiency on unseen mazes equals its success rate: when it succeeds, it takes a near-shortest path. Tabular Q-learning's efficiency (0.12) is far below its success rate (0.29), so most of its unseen successes are long wanders that happen to reach the goal.
- **Learning speed differs.** DQN reached 0.67 greedy success on the training mazes by episode 500 and plateaued around 0.7. Tabular Q-learning was still climbing at episode 3,000 (0.17 → 0.37 → 0.49 → 0.51 → 0.63 → 0.69 at 500-episode intervals) and might match DQN on the training mazes with a longer budget.

A development check with the full-map observation showed tabular Q-learning near random on unseen layouts (6–10% success): every new layout produces observations the table has never seen.

## 7. Discussion

**Did the agents learn from their mistakes?** Yes, on every fixed layout, and the logs show how. Mistakes concentrated in specific places early in training (the corridor below the start), the corresponding penalties dominated the reward, and both disappeared as Q-values for the offending actions fell. The early-versus-late comparison, the mistake hotspots and the reward breakdown all point the same way, and greedy evaluation confirms it with exploration off.

**Measuring mistakes needs care.** Three measurement problems appeared, and each would have produced a misleading claim:
- A falling mistake count can mean the agent found a way to end episodes early (Section 6.3).
- Path efficiency computed against the safe optimum rewarded hazard shortcuts with values above 1, until it was restricted to hazard-free successes.
- Training-time metrics include exploration; the default DQN settings reached 80–90% success while exploring but 0% greedily (Section 8).

**Reward design shapes what counts as a mistake.** With terminal hazards and per-step penalties, dying is the reward-maximising choice for an agent that cannot yet find the goal. Whether that is a "mistake" depends on the designer's intent, not on the agent's objective.

**Tabular Q-learning versus DQN.** On fixed layouts both reach optimal behaviour; tabular Q-learning is faster (greedy-perfect within 50 episodes) and far cheaper per step. DQN's advantage, if any, appears where generalization matters, and that advantage is not yet statistically established here. Tabular Q-learning benefited from optimistic initialisation, which DQN lacks, and that difference explains the terminal-hazard result.

**Prioritized replay.** On tasks this small, replaying surprising transitions more often gave no benefit, and one PER run converged to a stuck policy. A single failure in five seeds is not evidence that PER hurts.

## 8. Problems found and fixed during development

These are reported because each would have changed the conclusions.

| Problem | Effect | Fix |
|---|---|---|
| DQN defaults (gradient step every env step, target update every 250) | 80–90% success while exploring, 0% under greedy evaluation (looping policies) | A four-setting sweep over 3 seeds; `train_freq=2`, `target_update=100` solved every preset greedily |
| Terminal hazards in the hazard presets | DQN learned to end episodes on a hazard | Hazard presets made non-terminal; the terminal case kept as its own experiment |
| One evaluation seed shared by all runs | The random baseline replayed identical moves for every seed; confidence intervals too narrow | Each run gets its own evaluation seed; all experiments rerun |
| Path efficiency against the safe optimum | Hazard shortcuts scored above 1 (tabular Q: 1.40) | Efficiency counts only hazard-free successes; hazard experiment rerun |
| An early claim, from 3 seeds, that DQN finds the safe route where tabular Q does not | Overstated | With 5 seeds the difference has p = 0.21 and is reported as not established |

## 9. Limitations

- **Small tasks.** 8×8 and 9×9 grids with 100-step episodes, chosen to fit a free CPU. The ranking of algorithms may change on larger problems.
- **Few seeds.** Five (three for generalization) can show large effects only. Several differences reported here as "not established" might become significant with 10–20 seeds.
- **Limited tuning.** One hyperparameter setting per algorithm. DQN's update frequency and target-update interval were chosen from four settings on the fixed maze; tabular Q-learning was not tuned at all.
- **Fixed training budgets.** Tabular Q-learning was still improving at the end of the generalization runs.
- **Non-Markovian revisit penalty**, as described in Section 3.
- **Statistics.** t-intervals on 3–5 bounded values are approximate; the charts clip them to the possible range.

## 10. Reproducibility

Everything in this report can be regenerated from the repository:

```bash
pip install -r requirements.txt
python -m src.evaluation --experiment fixed_maze        # also: hazard_maze, terminal_hazards,
                                                        #       open_hazards, generalization
python -m src.visualization                             # regenerates docs/figures from results/
python -m pytest -q                                     # 150 tests
```

The saved outputs in `results/<experiment>/` are `summary.csv` (mean and CI per agent), `results.csv` (per seed), `curves.csv` (greedy learning curves), `experiment.json` (exact settings) and `run.log`. Experiments are resumable. Two of them were interrupted when a working session ended and were completed by rerunning the same command, as their logs record.

Results were produced on a Windows 11 laptop CPU with Python 3.11.9, PyTorch 2.14.1 (CPU), NumPy 2.4.6, Matplotlib 3.11.2, pandas 3.0.6 and Gradio 6.30.0. Exact reproduction is guaranteed for the same software on the same machine. Other platforms should give statistically similar but not bit-identical numbers, because floating-point results can differ between builds.

## 11. Future work

- More seeds (10–20) for the open questions: hazard shortcuts, generalization, PER.
- A larger layout pool, convolutional networks for the full-map observation, and longer training for the generalization study.
- Optimistic initialisation or count-based exploration bonuses for DQN, to test the explanation of the terminal-hazard failure.
- A reward variant without per-step shaping, to separate "mistakes" from reward design.
- Hosting the dashboard permanently (for example, on Hugging Face Spaces).

## References

- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *ICLR*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
