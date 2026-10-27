# Changelog

What each stage of MIND added or changed, with the reason behind each change.

## Dueling DQN and n-step returns (Stage D)

**Added**
- DQN options `dueling` (value and advantage heads, `Q = V + A − mean(A)`) and `n_step` (learn from the discounted sum of the next n rewards, bootstrapping with γⁿ; shorter tails at the end of an episode, which still bootstrap after a timeout). Off by default (`dueling=False`, `n_step=1`); then the network is built and trained exactly as before (checked: identical initial weights and training trajectory, and `hazard_maze` reproduced for DQN and DQN + PER, seeds 0–1).
- Agents `dqn_dueling`, `dqn_nstep` (n = 3) and `dqn_dueling_nstep`, with nicknames, in the dashboard's Train tab and About page, and chart styles (dashed, dense-dot and dash-dot-dot lines; diamond, down-triangle and plus markers; their own hatches).
- Replay buffers store a per-transition bootstrap discount. Buffers and checkpoints saved before this still load. n-step returns combined with mistake replay are refused with a clear error, because it is undefined which of the n steps made the mistake.
- Experiments `dqn_variants_speed` and `dqn_variants_generalization`, pre-registered in `results/dqn_variants_speed/PROTOCOL.md` with fresh seeds (300–309) and layouts (8000–8249). Results, analyses, two figures, README and REPORT section 6.9.
- 16 tests (243 in total): the dueling combination, defaults reproducing the old agent, hand-computed 3-step transitions for episodes ending at the goal and by timeout, the n-step learning target, PER with n-step, checkpoints old and new, each variant solving a corridor, and the dashboard.

**Results**
- `dqn_variants_speed`: 3-step returns made DQN learn the maze much faster (learning speed 0.369 → 0.863, Holm-corrected p = 0.0003, ahead in all 10 seeds); dueling made no measurable difference (0.338, p = 0.82).
- `dqn_variants_generalization`: on 50 held-out mazes, dueling 0.39 and 3-step 0.40 against 0.36 for DQN (Holm-corrected p = 0.93 each): no evidence of a difference.

## Reward shaping (Stage C)

**Added**
- Potential-based reward shaping (`RewardConfig.shaping`, `shaping_gamma`): each step adds k · (γ·φ(next) − φ(now)), where φ is minus the maze distance to the goal scaled to [−1, 0] and is 0 once the episode terminates. Off by default; with it off every earlier result is reproduced exactly (checked on `hazard_maze`, tabular Q and DQN, seeds 0–1).
- The shaping term is logged separately (`shaping` in the episode summary, `r_shaping` in the episode log and the reward-breakdown chart). `total_reward` stays the task reward, so shaped and unshaped runs compare directly. Episode logs saved before this column existed still load and resume.
- `GridWorld.potential(pos)`; `Experiment.arm_env_overrides` (per-arm environment settings, e.g. a shaped arm); `curve_means()`, a learning-speed metric (greedy success averaged over the evaluations during training), printed by the experiment command line for every pair of arms.
- Experiments `shaping_speed` and `shaping_generalization`, pre-registered in `results/shaping_speed/PROTOCOL.md` with fresh seeds (200–209) and layouts (7000–7249). Results, analyses, two figures, README and REPORT section 6.8.
- Dashboard: a **Reward shaping** checkbox in the Train tab and an explanation in About. Shaped arms are drawn in their base algorithm's color with dotted lines, triangle markers and dotted bars.
- 11 tests (227 in total): the potential, the signs of the shaping term, that its discounted sum telescopes (so loops earn nothing), that `total_reward` excludes it, old logs, per-arm settings, the learning-speed metric and the dashboard checkbox.

**Results**
- `shaping_speed`: both agents learned the maze sooner with shaping (tabular Q 0.931 → 0.986, DQN 0.388 → 0.794; Holm-corrected p = 0.002 each). Shaped DQN solved it in 10 of 10 seeds, plain DQN in 6 (exploratory, p = 0.093). All final routes were the 16-step shortest path.
- `shaping_generalization`: held-out success 0.54 with shaping vs 0.48 without (p = 0.366): no evidence of a difference. Exploratory: better on the training mazes (0.94 vs 0.85, p = 0.005) and far fewer wall collisions on held-out mazes (3.0 vs 14.2, p = 0.004).

## Policy map and maze editor (Stage B)

**Added**
- Dashboard tab **🗺️ Policy map & editor**: paint walls, hazards, the start and the goal by clicking cells, or paste a maze as text, then see the move the agent would make on every cell with exploration off, shaded by its best Q-value. The editor checks that the goal can be reached without crossing a hazard, the same rule generated layouts follow.
- **Train on my maze** in the Train tab, and **My maze** in Watch it play. An agent that sees the whole map only fits mazes of its training size (one input per cell), so the editor says so instead of failing.
- `policy_map()` and `policy_map_figure()` (`src/visualization.py`). For tabular Q, states it never visited get a grey dot instead of an arrow, because its zeros there are not knowledge; ties are marked `?` because greedy play picks one at random. `maze_image()` and `cell_at()` turn clicks into cells. `docs/figures/policy_map.png` is regenerated by `python -m src.visualization` (other figures are unchanged byte for byte).
- `GridWorld.observation_at(pos)` (what the agent would see on any cell, without changing the episode), `GridWorld.to_ascii()`, and module functions `parse_layout`, `layout_to_ascii` and `check_layout` (`src/envs.py`).
- `TrainConfig.layout_text`: an optional hand-drawn maze used instead of the preset's layouts; the preset still sets the rules. Checkpoints keep the maze. Configs and checkpoints saved before this field load unchanged.
- 19 tests (216 in total): observations on any cell match walking there, text layouts round-trip, bad layouts are rejected, every arrow equals `agent.act(greedy=True)`, unvisited cells are marked, clicks map to the right cell, and training on a custom maze survives a checkpoint round trip.

**Changed**
- An agent trained on a custom maze is tested on its preset's generated layouts in Evaluate and Watch it play when the observation sizes match; otherwise the page says there are no unseen layouts to test it on.

No experiments were run in this stage, and no results changed.

## Parallel runs on Linux

**Fixed**
- `--workers N` now starts its worker processes with "spawn" instead of Linux's default "fork". The parallel-workers test hung on GitHub's Linux runners until the 20-minute job limit, most likely because forking after PyTorch has started its thread pool can deadlock; `--workers` could have hung the same way in Colab. Windows always spawns, so results are unchanged.

**Changed**
- The GitHub workflow runs pytest with a 5-minute per-test limit (`pytest-timeout`), so a hanging test fails quickly and names itself.

## Mistake-aware replay (Stage G)

**Added**
- `MistakeReplayBuffer` (`src/replay.py`): wraps the uniform or prioritized buffer and also keeps every collision and hazard transition in a circular reserve; `mistake_fraction` of each batch is drawn from it. Reserve samples get importance weight 1 and never change PER priorities. Insertion counters (`diagnostics()`) show how far back each buffer remembers; they use no random numbers.
- DQN options `mistake_replay`, `mistake_fraction` (0.25), `mistake_capacity` (5,000) and `mistake_events`, and the agents `dqn_mistake` and `dqn_per_mistake`. Off by default: DQN and DQN + PER train exactly as before (checked with training fingerprints and by reproducing `hazard_maze`).
- `agent.update(..., info=None)`: an optional keyword argument through which agents can read the step's events. A mistake is identified from `info["events"]`, never from the reward, because a revisit plus a timeout costs more than a wall bump and rewards are configurable. Positional calls are unchanged.
- Metrics `steps_to_goal` (mean steps of successful episodes only; empty when a run never succeeded) and `hazard_rate` (share of episodes touching a hazard). Results saved before these metrics existed still load.
- Experiments `mistakes_hazard_maze`, `long_memory_pilot` and `long_memory_mistakes`. `Experiment.agent_types` lets one algorithm appear as several arms (for example two buffer sizes). `--workers N` trains N runs at a time with the same results. `buffer_diagnostics.csv` records how much each DQN run's buffer forgot.
- `results/mistakes_hazard_maze/` and `results/long_memory_mistakes/` (protocol, pilot, raw results, analysis), two figures, README and REPORT sections 6.6–6.7, and 43 tests (197 in total). These cover reserve capacity, sampling proportions, that mistakes come only from collision and hazard events, compatibility with PER, exact resumption, no duplicated transitions, and no leakage of held-out mazes into training.

**Results**
- `mistakes_hazard_maze` (seeds 0–9): no significant difference on any metric. The 20,000-transition buffer barely forgot anything, so this tested over-weighting mistakes, not remembering them.
- `long_memory_mistakes` (pre-registered; fresh seeds 100–109; 50 training and 50 held-out hazard mazes; 5,000-transition buffer): held-out success 0.30 with mistake memory vs 0.37 without (p = 0.082, behind in 9 of 10 seeds). That is no evidence of benefit. A DQN with a 10,000-transition buffer, matching the total memory, did best at 0.42 (exploratory, p = 0.009 against mistake memory).

**Changed**
- Chart styles: mistake-replay variants share their base algorithm's color with dash-dot lines, square markers and hatched bars, because no palette color passed the colorblind checks next to orange and aqua. The memory control uses violet.
- The dashboard's Results tab and the experiment command line handle arm labels, `steps_to_goal` and `hazard_rate`.

## Continuous integration

- `.github/workflows/tests.yml` runs `pytest -m "not slow"` on Python 3.10 and 3.12 for every push to `main` and every pull request, so a broken change shows up on GitHub instead of in someone's Colab session. It installs the CPU-only PyTorch wheel (the default one bundles CUDA and is several GB). The slow full-training test runs only when the workflow is started by hand with "Also run the slow full-training test" ticked.
- The README shows a Tests badge with the latest result.

## generalization with 10 seeds

- `generalization` now uses 10 seeds (was 3). With 3 seeds no difference could reach p < 0.10; with 10, DQN generalizes better than tabular Q-learning: 45% vs 34% success on 50 unseen mazes (p = 0.003, ahead in 9 of 10 seeds), with path efficiency 0.45 vs 0.15 (p < 0.001). On the training mazes they are level (p = 0.56). Seeds 0–2 were retrained and reproduced the earlier results exactly. `results/generalization`, README, REPORT (Sections 1, 5, 6.5, 7, 9–11), notebook section 11 and the generalization figure are updated.
- The notebook's significance table uses the shared p-value formatter, so it shows "p < 0.001" instead of "0.000".

## hazard_maze with 10 seeds

- `hazard_maze` now uses 10 seeds (was 5). The hazard-shortcut difference that was not established with 5 seeds (p = 0.21) is established with 10: tabular Q-learning crossed a hazard in 9 of 10 runs, DQN in 1 (p = 0.002). One DQN run solved the maze and then unlearned it, so DQN's success is 0.90. Seeds 0–4 were retrained and reproduced the earlier results exactly. `results/hazard_maze`, the README, REPORT (Sections 1, 5, 6.2, 7–11), notebook section 11 and the hazard figure are updated.
- `python -m src.evaluation --seeds N` now means seeds 0..N-1, so it can extend an experiment beyond its defined seeds (before, it could only shrink it).
- Very small p-values are shown as "p < 0.001" in the dashboard, verdicts and command line, instead of rounding to an impossible "0.000".

## Documentation and chart fixes

**Added**
- `REPORT_TEMPLATE.md`: a fill-in template for writing up new experiments, with the file each number comes from.
- `docs/PUBLISHING.md`: how to publish on GitHub, open in Colab and host on Hugging Face Spaces, including keeping tokens out of the repository.
- `space/`: the Hugging Face Space files and `deploy.py`; `LICENSE` (MIT).

**Fixed**
- Chart headings overlapping chart titles in the dashboard: Gradio's plot labels (drawn over each chart's top-left corner) are hidden, and every legend now sits in a row below its chart instead of inside the plot, where it covered lines and bars. A test checks that no legend overlaps a plot or title and no two titles overlap.
- Long replay titles split over two lines; the significance-test table has column widths so verdicts are no longer cut off.
- The notebook's launch cell closes any dashboard started earlier in the Colab session and reloads the project code from disk before launching, so it always serves the latest pulled code (before, an already-used session kept serving the old charts after `git pull`). The setup cell prints the code version it pulled.
- The dashboard header shows the version (git commit and date) it is running, so an old copy still running somewhere (for example a Colab session started before an update) is easy to spot. `space/deploy.py` stamps the same version into the Space. All 342 charts the dashboard can draw were checked: none has overlapping or cut-off text.
- The Colab notebook now draws its charts with `src/visualization.py` (the dashboard's own code), so its legends also sit below the charts and its greedy-path panels have two-line titles. It enables inline charts explicitly (the first chart previously appeared only as text in some environments) and explains that an open session must be restarted after pulling new code.

## Stages 6 & 7: visualization, Gradio dashboard, and the lighter side

**Added**
- `src/visualization.py`: every chart as a matplotlib `Figure` (no pyplot, so it is safe inside a web server): learning curves, mistakes by type, reward breakdown, early/late mistake heatmaps, multi-seed bars and curves with 95% CIs, and a step-by-step episode replay with Q-values. Colours are fixed per agent and per mistake type and were checked for colour-blind separation.
- `src/app.py`: the MIND dashboard (`python -m src.app`, or `--share` for a public link). Tabs: Results (all saved experiments, significance tests), Train (live curves and commentary, stop button), Mistake history (heatmap, reward breakdown, Hall of Shame, mistake log), Watch it play (record and replay greedy episodes, including unseen layouts), Q-learning vs DQN race, Evaluate, Checkpoints (save/download and upload/load), About.
- `src/fun.py`: commentary for every mistake, agent nicknames, seven achievements, the Hall of Shame (built on the real mistake counts), training progress quips, and experiment verdicts that follow the p-values.
- Tests: `tests/test_fun.py`, `tests/test_visualization.py`, `tests/test_app.py`.

**The lighter side in every stage**
- Stage 1, `python -m src.envs`: each preset now takes its first step into a wall and comments on it.
- Stages 2–3: every agent has a `nickname` (Chaos Monkey, The Spreadsheet, Neural Ned, Neural Ned (Holds Grudges)).
- Stage 4, `python -m src.training`: progress lines carry a quip; the summary adds the Hall of Shame, achievements and a comment on the last episode.
- Stage 5, `python -m src.evaluation`: every permutation test gets a verdict that only declares a winner when p < 0.05.
- Notebook: section 12 (nicknames, Hall of Shame, achievements, verdict), section 13 (charts of saved results), section 14 (launch the dashboard).

**Changed**
- Checkpoints now load with PyTorch's safe `weights_only=True` mode, allowing only the numpy array types a checkpoint contains. Before, loading used full unpickling, which could run code from a malicious file; that mattered once the dashboard accepted uploads. A test checks that a booby-trapped file is refused.

The commentary only reads logs and results. A test confirms training produces identical results with and without it.

## Stage 5: multi-seed evaluation (`ac6f779`)

**Added**
- `src/evaluation.py`: greedy evaluation that never changes the agent; resumable experiments over agents × seeds with greedy learning curves and unseen-layout tests; 95% t confidence intervals; exact permutation tests; CLI `python -m src.evaluation --experiment <name>`.
- `results/`: summaries, per-run metrics, curves and logs of five experiments.
- Notebook section 11: result tables, significance tests, and what the evidence does and does not support.

**Fixed**
- Every run used the same evaluation seed, so the random baseline replayed identical moves for every seed and the intervals were too narrow. Each run now gets its own evaluation seed.
- Path efficiency could exceed 1 when the agent cut through a hazard (shorter than the safe optimum), rewarding the shortcut. It now counts only hazard-free successes.

**Findings:** both learning agents beat random on every fixed layout (p = 0.008). With terminal hazards DQN learned to end episodes on a hazard (success 0.20 vs 1.00, p = 0.048). Not established: the hazard-shortcut difference (p = 0.21) and DQN's lead on unseen layouts (3 seeds, p = 0.10).

## Stage 4: logs, checkpoints, resumption (`be7e9b3`)

**Added**
- `Trainer` with per-run directories: `config.json`, `episodes.csv`, `mistakes.jsonl`, `checkpoint.pt`.
- Exact resumption (replay buffer and every RNG state are saved), crash recovery (log rows after the last checkpoint are discarded), atomic saves, run extension, a per-episode callback, and the CLI `python -m src.training`.
- Reward breakdown per event, path efficiency, TD error and loss logging, early-vs-late comparison, mistake hotspots, `load_agent`.

## Starter notebook (`efb959f`)

- `MIND_Colab.ipynb`: setup, tests, environment tour, training of all three agents, learning curves, early-vs-late table, greedy paths, mistake heatmap.

## Stage 3: DQN and replay (`34f287c`)

**Added**
- `ReplayBuffer`, `SumTree`, `PrioritizedReplayBuffer` (importance-sampling weights, β annealing).
- `DQNAgent` with a target network, Double DQN, Huber loss and gradient clipping; `make_agent`.

**Changed**
- DQN defaults tuned to `train_freq=2`, `target_update=100`: the original settings learned while exploring but looped under greedy evaluation.
- Hazard presets use non-terminal hazards: with terminal hazards DQN learned to end episodes on a hazard, because that cost less than wandering.

## Stage 2: random baseline and tabular Q-learning (`45d9a28`)

**Added**
- The shared agent interface (`act`, `update`, `end_episode`, `state_dict`), linear ε schedule, random tie-breaking.
- `RandomAgent` and `TabularQAgent`, bootstrapping correctly through timeouts but not through true terminal states; a minimal episode loop.

**Changed**
- Minimum grid size lowered to 3×3 so small hand-drawn layouts work in tests.

## Stage 1: environments (`18af517`)

**Added**
- `GridWorld` with maze and open layouts, hazards, layout pools for generalization tests, and `from_ascii` for hand-drawn layouts.
- Shared float32 observations (`grid` and `local` modes), configurable rewards and penalties, per-step mistake events and episode summaries, four presets.
