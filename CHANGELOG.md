# Changelog

What each stage of MIND added or changed, with the reason behind each change.

## Final documentation pass (Stage J)

After 17 studies the README's findings had grown into a 15-bullet list and parts of the report still described the project as it was several stages earlier. No code, experiment or result changed.

**Changed**
- README: a new intro, six **Main findings**, and an **Every study at a glance** table (question, result, verdict, link; pre-registered studies marked) before the baseline table; the long list of findings it replaces is now covered by the table and the per-study sections ("The studies in detail"). The agents list, limitations and the quick test's timing were updated.
- REPORT: the summary lists the 17 studies by group; Section 2 lists every stage and explains the pre-registration practice; Section 4 describes the two Stage L controls; Section 5 explains Holm correction and the paired test and has an index of every pre-registered study with its seeds and mazes.
- `requirements.txt`: `gradio>=6.0` (the dashboard passes theme, css and js to `launch()`, which earlier versions reject), and a note of the versions the results were produced with.
- Notebook: an updated overview, a one-paragraph summary of the later studies before their cells, and ids for the two cells that had none (the notebook now validates with `nbformat`). Every code cell except setup, tests and the dashboard was run locally without errors.
- `results/rnn_generalization`, `ppo_generalization` and `cnn_generalization` `ANALYSIS.md`: a dated "Later result" note at the end pointing to the study that answered each one's open question (memory vs replay, stochastic evaluation, the wide-network retest). The original text is unchanged.

**Checked**: every relative link and image in the 29 Markdown files, every in-page anchor, the headline numbers of the report's findings against each study's `ANALYSIS.md` and `summary.csv`, and the test count (330).

## Play it yourself, and replay controls (Stage M)

The studies showed that a local-view agent without memory loops because one 5 × 5 window can't tell it where it has been. This stage lets people feel that for themselves and look at the agent's choices more closely. No experiment, agent or result changes.

**Added**
- **🕹️ You play** tab: play a training, unseen or hand-drawn maze with the arrow keys (or on-screen buttons) under exactly the environment's rules and rewards, with live steps, reward and mistakes. **Agent's view** shows only the 5 × 5 window and the goal's direction, with no trail, as a local-view agent without memory sees it. **Compare with the agent** lets the current agent play the same maze greedily and shows both paths side by side with a table and a commentary line (`fun.human_vs_agent_line`). Every key press is queued and moves run one at a time.
- **🎬 Watch it play**: ⏸ Pause (Play resumes from the current step), ◀ Step / Step ▶ buttons, a seconds-per-step slider instead of three speeds, and a clickable map that records the same maze again with the agent starting on the clicked cell.
- `GridWorld.reset(start=...)` (optional; refuses walls, hazards, the goal, cells outside the maze and cells the goal can't be reached from) and `record_episode(start=...)`. Without `start` both behave exactly as before, including their random draws (tested).
- `visualization.play_image` (fast pixel rendering with a fog mode) and `compare_paths_figure`.
- 15 tests (330 in total) in `tests/test_play.py`: the start option, recording from another start, both renderings pixel by pixel, moves matching the environment exactly, timeouts, the agent's view, restarts, the editor's maze without an agent, comparing with a trained agent and with a saved Stage L network, refusing a comparison on a different maze, the commentary lines, the step buttons and resumed playback, recording from a clicked cell, and the new controls and key bindings in the built app. Checked in Edge with Playwright: arrow keys, buttons, the agent's view, the comparison, step buttons, pause and the start map.

**Changed**
- The dashboard is launched with a small script (`launch(js=...)`) that turns arrow keys into the You-play buttons, only while that tab is open and never while typing in a field.

## Memory vs sequence replay (Stage L)

Stage F's memory network was also trained differently (runs of 16 consecutive steps, 8× as many trained steps per update), and Stage F could not say which of the two explained its gain. This stage separates them.

**Added**
- `DQNConfig.reset_memory_every_step` and `RecurrentQNetwork(reset_every_step=True)`: the GRU starts from a zero memory at every step (when acting, during burn-in, in training and in the targets), so each answer depends on the current observation alone. Layers, initial weights, sequence replay, burn-in, batch and update schedule are those of the network with memory. It needs a recurrent agent; checkpoints with and without it refuse to load into each other, and older recurrent checkpoints still load.
- Agent `dqn_rnn_nomem` (109,572 weights, the same as `dqn_rnn`) with a nickname, tagline and chart style (aqua, the GRU's line pattern).
- Experiment `memory_vs_replay`, pre-registered in `results/memory_vs_replay/PROTOCOL.md` with its analysis script `analyze.py` written before the run: arms `dqn`, `dqn_batch512` (the same MLP, batch 512), `dqn_rnn` and `dqn_rnn_nomem`; fresh seeds 800–809 and layouts (9900–9949 training, 10001–10050 held-out; 10000 duplicated a training maze). Results, `analysis_output.txt`, `eps005.json` (pre-specified ε = 0.05 secondary evaluation), `ANALYSIS.md`, all 40 trained networks in `models/` (11.1 MB, each checked to reproduce its saved results), two figures (`memory_replay_curves.png`, `memory_replay_success.png`), notebook cells, README section and REPORT section 6.15 and finding 14.
- 6 tests (315 in total): the reset network answering each step from its observation alone (no gradient to earlier steps, the given memory ignored) and matching the memory network on the first step, `dqn_rnn_nomem` being `dqn_rnn` with only the option changed (same settings and initial weights; acting leaves no trace), refusing the option without a recurrent network, training and exact checkpoint round-trips (a memory network refuses the control's checkpoint), older recurrent checkpoints loading, and the protocol's settings and duplicate-free layouts. Existing agents are unchanged: 300 episodes of `rnn_generalization` DQN seed 600 and 150 of `dqn_rnn` seed 600 reproduce their saved logs exactly.

**Result** (pre-registered, Holm-corrected): with its memory reset at every step, the GRU network solved 0.70 of held-out mazes against 0.88 with memory (p < 0.001, behind in all 10 seeds), the same as the standard MLP (0.70), so the Stage F gain is the memory, not the sequence training. The MLP with batch 512 was not better than batch 64 (0.47 vs 0.70, p = 0.061; three of its ten runs never learned). The GRU again beat the standard MLP with fresh seeds and mazes (0.88 vs 0.70, p < 0.001, 10/10). The same held with ε = 0.05 in play. The run was stopped twice at the owner's request and resumed with the same command; `run.log` was regenerated afterwards (outputs byte-identical).

## Stochastic evaluation of saved networks (Stage K)

Every earlier result was greedy, and Stage I found (exploratory) that PPO did far better when it sampled its moves, while DQN had never been evaluated with any randomness. This stage re-evaluates saved networks instead of training new ones.

**Added**
- `evaluate_agent(..., policy="greedy" | "epsilon" | "sample", epsilon, repeats)`. `"epsilon"` takes the agent's best move and replaces it with a uniformly random one with probability ε (any agent; the agent is always asked first, so a recurrent agent's memory sees every observation); `"sample"` draws moves from a policy agent's probabilities (PPO only; value-based agents are refused). The training ε schedule is never used. Greedy remains the default and is unchanged. The agent's random generator and a recurrent agent's memory are restored afterwards.
- `paired_permutation_test`: a two-sided sign-flip test for the same networks evaluated two ways (exact up to `n_resamples` sign patterns, Monte Carlo beyond).
- `src/stochastic_study.py` (`python -m src.stochastic_study --workers 2`): plays the 80 saved networks of `cnn_generalization`, `rnn_generalization` and `ppo_generalization` on their held-out mazes greedily and with ε 0.05 / 0.1 / 0.2 (and PPO sampled), 5 passes per stochastic mode, evaluation seed 20,000 + run seed. It stops unless greedy play reproduces each network's saved results exactly. Resumable per network.
- Study `stochastic_evaluation`, pre-registered in `results/stochastic_evaluation/PROTOCOL.md` with its analysis script `analyze.py` written before the run. Results (`results.csv`, `episodes.csv.gz`, 66,500 episodes), `ANALYSIS.md`, figure `stochastic_success.png` (`stochastic_success_figure`), notebook cells, README section and REPORT section 6.14.
- 9 tests (309 in total): the paired test (exact values, symmetry, pairing vs not, Monte Carlo path, length check), ε evaluation leaving the agent unchanged and taking random moves, reproducibility and repeated layouts, refusing sampling for value-based agents and bad settings, a recurrent agent's memory left untouched, the runner's coverage of all 80 networks, every mode on an exported network without changing it, the greedy reproduction check, and the new chart.

**Result** (pre-registered, Holm-corrected): ε 0.05 raised plain DQN's held-out success from 0.42 to 0.55 (all 30 networks improved, p < 0.001); PPO sampled beat DQN with ε 0.05 (0.90 vs 0.71, p < 0.001); the GRU network beat the plain one with ε 0.05 for both (0.80 vs 0.42, p < 0.001). Every one of the 80 networks did better with random moves than greedily. No earlier conclusion changed; greedy results are now described as a lower bound.

## PPO, a policy learner (Stage I)

**Added**
- `PPOAgent` and `PPOConfig` (`make_agent("ppo")`): Proximal Policy Optimization with a clipped objective and GAE. An actor (77 → 128 → 128 → 4, tanh, orthogonal initialisation) and a separate critic of the same shape; rollouts of 2,048 steps that may span episodes, 4 epochs in minibatches of 256, clip 0.2, λ 0.95, γ 0.99, Adam 3e-4, entropy bonus 0.01, critic weight 0.5, gradient clip 0.5. Standard published settings, fixed in advance and not tuned. Timeouts bootstrap from the critic. No ε: it samples its policy to explore, and an ε schedule passed to it (as the dashboard does) is ignored.
- PPO checkpoints save the half-collected rollout (`save_buffer`), so a resumed run continues exactly; `export_weights` leaves it out.
- The policy map and step-by-step replays work for policy agents: arrows show the most likely move, shaded by the critic's value, and replays show move probabilities instead of Q-values. PPO is in the dashboard's Train tab and About page, with a nickname, a tagline and its own chart style (purple, short dashes).
- Experiment `ppo_generalization`, pre-registered in `results/ppo_generalization/PROTOCOL.md` with fresh seeds (700–709) and layouts (9700–9749 training, 9800–9849 held-out, no duplicates). `Experiment.train_config` gives PPO no ε schedule. Results, analysis, all 20 trained networks in `models/` (3.3 MB, each checked to reproduce its saved results), two figures (`ppo_curves.png`, `ppo_success.png`), notebook cells, README and REPORT section 6.13.
- 16 tests (300 in total): GAE against its definition across episode ends, timeouts bootstrapping and terminals not, no leaking between episodes, the initial policy and greedy choice, network sizes, one update per rollout, the clipped update favouring rewarded moves, settings checks, PPO solving a corridor, exact resumption mid-rollout and determinism, checkpoints without the rollout, exported weights, the policy map and replays, the protocol settings and fresh layouts, and the dashboard with PPO. Plain DQN is unchanged: 300 episodes of `rnn_generalization` DQN seed 600 reproduce the saved log exactly.

**Changed**
- A test that used `"ppo"` as an example of an unknown agent name now uses `"a2c"`.

**Results**
- `ppo_generalization`: PPO solved slightly more held-out mazes (0.69 vs 0.64, Holm-corrected p = 0.037, ahead in 8 of 10 seeds), all on the longer mazes, and learned its training mazes more slowly (learning speed 0.826 vs 0.896, p = 0.0003, behind in all 10 seeds; level at the end). Exploratory: sampling its moves instead of playing greedily raised PPO's held-out success to 0.90; DQN was not evaluated that way. A PPO run cost about a quarter of a DQN run's time. The run was not interrupted; `run.log` was regenerated in UTF-8 afterwards (all CSVs byte-identical).

## Recurrent Q-network with memory (Stage F)

**Added**
- `RecurrentQNetwork`: one hidden layer of 128 (ReLU), then a GRU with 128 memory units, then the Q-values. DQN options `recurrent` (memory size; 0, the default, builds networks exactly as before), `seq_length` and `burn_in`.
- `SequenceReplayBuffer` (DRQN-style): records where each episode starts and ends and samples runs of consecutive steps that never cross into another episode, with up to `burn_in` earlier steps to rebuild the memory from zero (R2D2-style), never before the oldest stored step. Targets are Double DQN step by step with the target network's own memory; steps past an episode's end are masked.
- An optional `begin_episode` hook, called by `run_episode` and by episode recording, so an agent with memory starts each episode blank.
- Agent `dqn_rnn` (`RNN_DEFAULT`: memory 128, hidden (128,), 32 runs of 16 steps, burn-in 16; 109,572 weights) with a nickname and commentary line, in the dashboard's Train tab and About page, and a chart style. A recurrent agent refuses PER, mistake replay, n-step returns, dueling heads and convolutions. The policy map explains that it cannot show a memory agent's policy as one arrow per cell.
- Experiment `rnn_generalization`, pre-registered in `results/rnn_generalization/PROTOCOL.md` with fresh seeds (600–609) and layouts (9500–9549 training; 9600–9651 except 9619 and 9632 held-out, which duplicated other mazes), arms `dqn`, `dqn_rnn` and `dqn_wide` (a pre-registered retest of Stage E's exploratory result). Results, analysis, all 30 trained networks in `models/` (9.9 MB, each checked to reproduce its saved results), two figures (`rnn_curves.png`, `rnn_success.png`), notebook cells, README and REPORT section 6.12.
- 18 tests (284 in total): a recurrent step-by-step pass equals a whole-sequence pass, memory carries information forward and is cleared per episode, the default sizes, refused combinations, sequences staying inside one episode and after the oldest stored step, buffer and agent checkpoints, a learning step, the GRU agent solving a corridor, the protocol settings and fresh held-out mazes, and the dashboard with a recurrent agent. Plain DQN is unchanged: the first 300 episodes of `cnn_generalization` DQN seed 500 reproduce the saved log exactly.

**Results**
- `rnn_generalization`: memory helped. Held-out success 0.73 vs 0.29 for the standard MLP (Holm-corrected p = 0.0004, ahead in all 10 seeds); held-out revisits per episode 20.1 vs 44.1 (p = 0.0004). The same-size wide MLP reached 0.34 (exploratory, p = 0.0001 against the GRU), so the gain is not a matter of size; memory and the 16-step replay were not separated. The first change in the project to improve success on new mazes.
- Stage E's exploratory wide-MLP result did not replicate (0.34 vs 0.29, Holm-corrected p = 0.32).
- The run was stopped and resumed three times (a closed window, moving the project to another drive, an overnight pause) with the same command; resumption does not change results. It took about 11 hours against the 6 estimated. `run.log` was regenerated afterwards by rerunning the finished experiment (all CSVs byte-identical).

## Convolutional Q-network (Stage E)

**Added**
- DQN option `conv` (channel counts of 3×3 convolutions, padding 1, ReLU) and `conv_input` (the image's channels, rows and columns). The image is the first C·H·W values of the observation; anything after it (the goal direction in local view) is concatenated with the flattened maps before the hidden layers. Off by default (`conv=()`); then networks are built exactly as before (same initial weights, tested), and 500 episodes of `dqn_variants_generalization` DQN seed 300 reproduce the saved log exactly.
- `GridWorld.image_shape`; CNN agents read the image shape from the environment (4×H×W for the full map, 3×5×5 for the local view).
- Agent `dqn_cnn` (two convolutions with 16 and 32 channels, then one hidden layer of 128) with a nickname, in the dashboard's Train tab and About page, and chart styles for `dqn_cnn` and `dqn_wide`.
- Experiment `cnn_generalization`, pre-registered in `results/cnn_generalization/PROTOCOL.md` with fresh seeds (500–509) and layouts (9300–9349 training, 9400–9449 held-out), plus an exploratory capacity-matched arm `dqn_wide` (MLP 288 × 288). Results, analysis, two figures, notebook cells, README and REPORT section 6.11.
- `training.export_weights`: a small evaluation-only copy of a checkpoint (config and trained network, without the replay buffer, target network, optimizer or generator states). `load_agent` and the dashboard read it; `Trainer.load_checkpoint` refuses it with a clear message. All 30 networks of the study are in `results/cnn_generalization/models/` (9.8 MB), each checked to reproduce its run's saved results exactly.
- 12 tests (266 in total): the image layout and shared filters, building and checkpointing the CNN agent (including old checkpoints and mismatches), the environment's image shape, the CNN solving a corridor, the protocol settings and fresh held-out mazes, exported weights for an MLP and a CNN, and loading a saved study network in the dashboard. The dashboard test now covers the CNN agent too.

**Fixed**
- The notebook's curriculum cell had the same cell id as the DQN-variants cell; it now has its own.

**Results**
- `cnn_generalization`: no evidence that the CNN helps. Held-out success 0.27 vs 0.33 for the standard MLP, learning speed 0.617 vs 0.639 (Holm-corrected p = 0.55 for both). Exploratory: the wide MLP reached 0.47 on held-out mazes (p = 0.019 vs the MLP, uncorrected), which needs a pre-registered test of its own. The CNN cost about 3–4× the compute.
- The first attempt (`--workers 3`) was stopped for lack of memory after 6 runs and resumed with `--workers 2` in a separate window; resumption does not change results. `run.log` was regenerated afterwards by rerunning the finished experiment (all CSVs byte-identical).

## Training curriculum (Stage H)

**Added**
- `TrainConfig.curriculum` (`{"start": k0, "full_at": E}`): the Trainer draws each episode's maze from the easiest `curriculum_size(e)` mazes of `layout_pool`, growing linearly from k0 to the whole pool at episode E. One draw from the environment's generator per episode, as an ordinary pool draw makes, so curriculum runs are deterministic and resume exactly. Off by default; then training is unchanged (checked: 500 episodes of `dqn_variants_generalization` DQN seed 300 reproduce the saved log exactly).
- `envs.rank_layouts` (difficulty order: shortest path, then dead ends, then seed) and `envs.count_dead_ends`. `Experiment.arm_curriculum` (per-arm curriculum).
- Experiment `curriculum_generalization`, pre-registered in `results/curriculum_generalization/PROTOCOL.md` with fresh seeds (400–409) and layouts (9100–9149 training, 9200–9250 except 9225 held-out). Results, analysis, two figures, README and REPORT section 6.10. Chart style for `dqn_curriculum`.
- 11 tests (254 in total): dead-end counting, the difficulty order, the schedule, that a curriculum run only ever draws from the easiest mazes allowed so far, reproducibility and exact resumption, that a curriculum needs a pool, the protocol settings, and that held-out mazes are fresh and **never identical to a training maze**.

**Changed**
- A run directory's saved config is now compared with the requested one as a config, not as text, so runs saved before a setting existed (it takes its default) still resume.

**Found**
- A 9×9 maze has only a 4×4 grid of rooms, so different layout seeds can generate the same maze. `shaping_generalization` (7223 = 7147) and `dqn_variants_generalization` (8224 = 8106) each had one held-out maze that was also a training maze. Re-evaluated without it, every conclusion is unchanged; post-hoc notes were added to both analyses.

**Results**
- `curriculum_generalization`: the curriculum made DQN learn its training mazes more slowly (learning speed 0.851 → 0.722, Holm-corrected p = 0.0003, behind in all 10 seeds). Held-out success 0.49 vs 0.57 (Holm-corrected p = 0.093): no evidence of a difference.
- The first attempt was stopped for lack of memory with `--workers 4` and resumed with `--workers 2`; resumption does not change results.

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
