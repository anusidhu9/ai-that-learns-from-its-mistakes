# Changelog

What each stage of MIND added or changed, with the reason behind each change.

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
