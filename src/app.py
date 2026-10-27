"""MIND dashboard: a Gradio web app over every part of the project.

    python -m src.app            # http://127.0.0.1:7860
    python -m src.app --share    # also prints a temporary public link (e.g. from Colab)

Tabs: saved multi-seed results, live training, mistake history, step-by-step
replay, policy map and maze editor, Q-learning vs DQN race, greedy evaluation,
checkpoints, and an About page. Each browser session has its own agent; nothing is shared between users.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no GUI backend inside the web server

import itertools
import tempfile
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import gradio as gr
import numpy as np
import pandas as pd

from src import fun
from src.envs import (EMPTY, GOAL, HAZARD, PRESETS, WALL, RewardConfig, check_layout,
                      layout_to_ascii, parse_layout, shortest_path_length)
from src.evaluation import (EXPERIMENTS, compare_agents, evaluate_agent, format_p,
                            load_experiment)
from src.training import (TrainConfig, Trainer, build_env, compare_phases, load_agent)
from src.visualization import (AGENT_COLORS, AGENT_LABELS, _agents_in, cell_at,
                               experiment_bars_figure, experiment_curves_figure, frame_figure,
                               learning_curves_figure, maze_figure, maze_image,
                               mistake_heatmap_figure, mistake_types_figure, policy_map,
                               policy_map_figure, record_episode, reward_breakdown_figure)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def code_version() -> str:
    """Short git commit of the running code, shown in the header so anyone can
    tell whether a page runs the latest version (compare with GitHub).

    Deployed copies without git history (the Hugging Face Space) read the
    VERSION file that space/deploy.py writes."""
    import subprocess
    stamp = RESULTS_DIR.parent / "VERSION"
    if stamp.exists():
        return stamp.read_text(encoding="utf-8").strip()
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h (%cd)", "--date=short"],
                             cwd=RESULTS_DIR.parent, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"

AGENT_CHOICES = [(f"{AGENT_LABELS[a]} ({fun.NICKNAMES[a]})", a)
                 for a in ("tabular_q", "dqn", "dqn_dueling", "dqn_nstep", "dqn_dueling_nstep",
                           "dqn_per", "dqn_mistake", "dqn_per_mistake", "random")]
PRESET_CHOICES = list(PRESETS)
METRIC_CHOICES = [("Success rate", "success"), ("Path efficiency", "path_efficiency"),
                  ("Mistakes per episode", "mistakes"), ("Steps", "steps"),
                  ("Total reward", "total_reward")]
UNSEEN_BASE = 1000
MY_MAZE = "My maze"
SHAPING_STRENGTH = 1.0  # the value the shaping experiments used
CELL_PX = 36
BRUSHES = [("⬛ Wall", "wall"), ("⬜ Empty", "empty"), ("🟥 Hazard", "hazard"),
           ("🔵 Start", "start"), ("🟩 Goal", "goal")]
MAX_EPISODES = {"random": 1000, "tabular_q": 2000, "dqn": 800, "dqn_per": 800,
                "dqn_mistake": 800, "dqn_per_mistake": 800, "dqn_dueling": 800,
                "dqn_nstep": 800, "dqn_dueling_nstep": 800}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def new_session() -> dict:
    return {"trainer": None, "frames": [], "replay_env": None, "editor": None,
            "commentary": deque(maxlen=14), "announced": set()}


def _need_trainer(state: dict) -> Optional[str]:
    if state.get("trainer") is None:
        return "No agent yet. Train one in the **Train** tab or load a checkpoint."
    return None


# ---------------------------------------------------------------------------
# Results tab (saved multi-seed experiments)
# ---------------------------------------------------------------------------

def available_experiments() -> List[str]:
    return [name for name in EXPERIMENTS if (RESULTS_DIR / name / "results.csv").exists()]


def _summary_frame(summary: List[dict]) -> pd.DataFrame:
    rows = []
    for r in summary:
        row = {"agent": AGENT_LABELS.get(r["agent"], r["agent"]), "split": r["split"],
               "seeds": r["n_seeds"]}
        for label, m in [("success", "success"), ("path efficiency", "path_efficiency"),
                         ("steps", "steps"), ("mistakes", "mistakes"), ("hazards", "hazards")]:
            ci = r[f"{m}_ci"]
            row[label] = (f"{r[m + '_mean']:.2f} ± {ci:.2f}" if not np.isnan(ci)
                          else f"{r[m + '_mean']:.2f}")
        rows.append(row)
    return pd.DataFrame(rows)


def _tests_frame(results: List[dict]) -> pd.DataFrame:
    agents = _agents_in(results)
    splits = [s for s in ("train", "unseen") if any(r["split"] == s for r in results)]
    rows = []
    for split in splits:
        for a, b in itertools.combinations(agents, 2):
            c = compare_agents(results, b, a, "success", split)
            rows.append({"split": split,
                         "comparison": f"{AGENT_LABELS.get(b, b)} − {AGENT_LABELS.get(a, a)}",
                         "difference": round(c["difference"], 2),
                         "p-value": format_p(c["p_value"]).removeprefix("p = ").removeprefix("p "),
                         "verdict": fun.verdict(c["p_value"], c["difference"], b, a)})
    return pd.DataFrame(rows)


def show_results(name: str, metric: str):
    if not name:
        empty = pd.DataFrame()
        return "No saved results found in `results/`.", None, None, empty, empty
    exp = load_experiment(RESULTS_DIR / name)
    e = exp["experiment"]
    unseen = (f" Tested on {len(e.unseen_layouts)} unseen layouts."
              if e.unseen_layouts else "")
    header = (f"### {name}\n{e.description}\n\n"
              f"**{len(e.seeds)} seeds** × {', '.join(AGENT_LABELS.get(a, a) for a in e.agents)}; "
              f"{e.episodes} training episodes each; greedy evaluation every "
              f"{e.eval_every} episodes.{unseen} Error bars and bands are 95% confidence "
              f"intervals across seeds.")
    return (header, experiment_bars_figure(exp["summary"], metric),
            experiment_curves_figure(exp["curves"], metric),
            _summary_frame(exp["summary"]), _tests_frame(exp["results"]))


# ---------------------------------------------------------------------------
# Train tab
# ---------------------------------------------------------------------------

def _status(trainer: Trainer, started: float, window: int = 25) -> str:
    h = trainer.history
    if not h:
        return "Starting..."
    recent = h[-window:]
    success = float(np.mean([r["success"] for r in recent]))
    mistakes = float(np.mean([r["mistakes"] for r in recent]))
    name = trainer.config.agent
    return (f"**{fun.NICKNAMES[name]}** · episode **{trainer.episodes_done} / "
            f"{trainer.config.episodes}** · last {len(recent)}: success **{success:.2f}**, "
            f"mistakes **{mistakes:.1f}**, ε {h[-1]['epsilon']:.2f} · "
            f"{time.time() - started:.0f}s\n\n> {fun.progress_quip(success, mistakes, len(h))}")


def _achievements_md(history: List[dict]) -> str:
    earned = fun.achievements(history)
    if not earned:
        return "**Achievements:** none yet. Everyone starts somewhere."
    items = "\n".join(f"- 🏅 **{a['title'].split(':')[0]}** (episode {a['episode']}): "
                      f"{a['title'].split(':', 1)[1].strip()}" for a in earned)
    return f"**Achievements unlocked ({len(earned)}/{len(fun.ACHIEVEMENTS)}):**\n{items}"


def _commentary_text(state: dict) -> str:
    return "\n".join(reversed(state["commentary"])) or "The commentator is warming up."


def train_agent_ui(preset, agent, episodes, seed, decay_pct, state, use_maze=False,
                   shaping=False):
    state = state or new_session()
    episodes = int(min(int(episodes), MAX_EPISODES.get(agent, 1000)))
    kwargs = {} if agent == "random" else {
        "epsilon": {"decay_episodes": max(1, int(episodes * decay_pct / 100))}}
    layout_text = None
    if use_maze:
        ed = _editor(state, preset)
        problem = check_layout(ed["grid"], ed["start"], ed["goal"])
        if problem:
            yield (f"Can't train on your maze: {problem}. Fix it in the **🗺️ Policy map & "
                   "editor** tab.", None, None, _commentary_text(state), _achievements_md([]),
                   state)
            return
        layout_text = layout_to_ascii(ed["grid"], ed["start"], ed["goal"])
    overrides = {"rewards": {"shaping": SHAPING_STRENGTH}} if shaping else {}
    config = TrainConfig(env_preset=preset, env_overrides=overrides, agent=agent,
                         agent_kwargs=kwargs, episodes=episodes, seed=int(seed),
                         layout_text=layout_text)
    trainer = Trainer(config)  # in memory; checkpoints are saved on request
    state.update(trainer=trainer, frames=[], replay_env=None)
    state["commentary"] = deque(maxlen=14)
    state["announced"] = set()
    started = time.time()
    chunk = max(5, episodes // 30)

    def on_episode(row):
        line = fun.episode_line(row)
        state["commentary"].append(f"ep {row['episode']:>4}  [{fun.mood(row)}] {line}")
        for a in fun.achievements(trainer.history):
            if a["key"] not in state["announced"]:
                state["announced"].add(a["key"])
                state["commentary"].append(f"ep {a['episode']:>4}  ACHIEVEMENT: {a['title']}")

    yield ("Starting...", learning_curves_figure({agent: []}), None,
           _commentary_text(state), _achievements_md([]), state)
    while trainer.episodes_done < episodes:
        trainer.train(episodes=min(chunk, episodes - trainer.episodes_done), callback=on_episode)
        yield (_status(trainer, started), learning_curves_figure({agent: trainer.history}),
               mistake_types_figure(trainer.history), _commentary_text(state),
               _achievements_md(trainer.history), state)
    done = _status(trainer, started) + "\n\n✅ **Training finished.** Explore the other tabs."
    yield (done, learning_curves_figure({agent: trainer.history}),
           mistake_types_figure(trainer.history), _commentary_text(state),
           _achievements_md(trainer.history), state)


# ---------------------------------------------------------------------------
# Mistake history tab
# ---------------------------------------------------------------------------

def mistake_history(state, window):
    state = state or new_session()
    message = _need_trainer(state)
    if message:
        return message, None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    trainer: Trainer = state["trainer"]
    h, n = trainer.history, len(trainer.history)
    if not h:
        return ("This agent was loaded from a checkpoint, which holds no episode logs. "
                "Train it further to build a history.", None, None, pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame())
    window = int(min(window, max(1, n // 2)))
    cmp = compare_phases(h, window)
    phases = pd.DataFrame([{"metric": m, f"first {window}": round(cmp["early"][m], 3),
                            f"last {window}": round(cmp["late"][m], 3),
                            "change": round(cmp["change"][m], 3)} for m in cmp["early"]])
    early, late = range(0, window), range(n - window, n)
    heat = mistake_heatmap_figure(trainer.env.grid, trainer.mistakes, early, late,
                                  start=trainer.env.start)
    shame = pd.DataFrame([{"#": r["rank"], "award": r["award"], "mistake": r["event"],
                           "cell": str(r["pos"]), "times": r["count"]}
                          for r in fun.hall_of_shame(trainer.mistakes, top=8)])
    recent = pd.DataFrame([{"episode": m["episode"], "step": m["step"], "mistake": m["event"],
                            "cell": str(tuple(m["pos"])), "action": m["action"],
                            "commentary": fun.event_line(m["event"], m["pos"],
                                                         (m["episode"], m["step"]))}
                           for m in trainer.mistakes[-150:][::-1]])
    pool = trainer.env.config.layout_pool
    note = ("Heatmap cells combine several layouts (this agent trains on a layout pool)."
            if pool else "")
    summary = (f"**{fun.NICKNAMES[trainer.config.agent]}** made **{len(trainer.mistakes)}** "
               f"mistakes over {n} episodes. {note}")
    return summary, heat, reward_breakdown_figure(h, window), phases, shame, recent


# ---------------------------------------------------------------------------
# Watch tab (step-by-step replay)
# ---------------------------------------------------------------------------

def _frame_outputs(state, index):
    frames, env = state["frames"], state["replay_env"]
    index = int(max(0, min(index, len(frames) - 1)))
    f = frames[index]
    color = AGENT_COLORS.get(state["trainer"].config.agent, "#2a78d6")
    info = (f"**Step {f['step']} / {len(frames) - 1}** · action: {f['action'] or 'start'} · "
            f"reward {f['reward']:+.2f} · total {f['total']:+.2f}\n\n> {f['line']}")
    return frame_figure(env, frames, index, color), info


def record_replay(state, layout_mode, layout_seed):
    state = state or new_session()
    message = _need_trainer(state)
    if message:
        return None, message, gr.update(maximum=1, value=0), state
    trainer: Trainer = state["trainer"]
    env, ls = build_env(trainer.config), None
    if layout_mode == MY_MAZE:
        env, problem = _maze_env(state)
        if problem:
            return None, problem, gr.update(maximum=1, value=0), state
    elif layout_mode == "Unseen layout":
        env, ls = _unseen_env(trainer), UNSEEN_BASE + int(layout_seed)
        if env is None:
            return None, _NO_UNSEEN, gr.update(maximum=1, value=0), state
    frames = record_episode(env, trainer.agent, layout_seed=ls)
    state.update(frames=frames, replay_env=env)
    ep = frames[-1].get("episode") or {}
    fig, info = _frame_outputs(state, 0)
    verdict = (f"\n\n**Episode result:** {'reached the goal' if ep.get('success') else 'failed'} "
               f"in {ep.get('steps')} steps (shortest safe path: {ep.get('optimal_length')}), "
               f"{ep.get('mistakes')} mistakes. {fun.episode_line(ep) if ep else ''}")
    return fig, info + verdict, gr.update(maximum=max(1, len(frames) - 1), value=0), state


def scrub_replay(state, index):
    if not state or not state.get("frames"):
        return None, "Record an episode first."
    return _frame_outputs(state, index)


def play_replay(state, speed):
    if not state or not state.get("frames"):
        yield None, "Record an episode first.", gr.update()
        return
    delay = {"Slow": 0.6, "Normal": 0.3, "Fast": 0.1}.get(speed, 0.3)
    for i in range(len(state["frames"])):
        fig, info = _frame_outputs(state, i)
        yield fig, info, gr.update(value=i)
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Policy map and maze editor tab
# ---------------------------------------------------------------------------

_NO_UNSEEN = ("This agent was trained on your own maze, and its observations don't fit the "
              "preset's generated layouts, so there are no unseen layouts to test it on.")


def _unseen_env(trainer: Trainer):
    """Environment that generates fresh layouts for this agent, or None.

    An agent trained on a hand-drawn maze is tested on its preset's generated
    layouts, which only works when the observation sizes match."""
    if not trainer.config.layout_text:
        return build_env(trainer.config)
    env = build_env(replace(trainer.config, layout_text=None))
    return env if env.obs_size == trainer.env.obs_size else None


def _locked_shape(state: dict):
    """Maze size the current agent can handle, or None if any size works.

    Full-map observations have one input per cell, so the network only fits
    mazes of its training size; local views work on any size."""
    trainer = state.get("trainer")
    if trainer is None or trainer.env.config.obs_mode != "grid":
        return None
    return trainer.env.grid.shape


def _layout_of(env) -> dict:
    return {"grid": env.grid.copy(), "start": env.start, "goal": env.goal}


def _training_layout(state: dict, preset: str) -> dict:
    trainer = state.get("trainer")
    if trainer is None:
        return _layout_of(_preview_env(preset))
    env = build_env(trainer.config)
    env.reset(seed=0)
    return _layout_of(env)


def _editor(state: dict, preset: str = "maze") -> dict:
    """The session's hand-drawn maze; it starts as a copy of the training layout."""
    if state.get("editor") is None:
        state["editor"] = _training_layout(state, preset)
    return state["editor"]


def _editor_outputs(state: dict, note: str = ""):
    ed = state["editor"]
    grid, start, goal = ed["grid"], ed["start"], ed["goal"]
    problem = check_layout(grid, start, goal)
    lock = _locked_shape(state)
    h, w = grid.shape
    if problem:
        status = f"⚠️ **Can't use this maze yet:** {problem}."
    elif lock is not None and lock != grid.shape:
        status = (f"⚠️ This maze is {h} × {w}, but the current agent sees the whole map and "
                  f"only fits {lock[0]} × {lock[1]}. Press **Start from training layout** or "
                  "**Blank maze**, or train a new agent on it.")
    else:
        status = (f"✅ **Ready.** {h} × {w}, shortest safe path "
                  f"{shortest_path_length(grid, start, goal)} steps.")
    if note:
        status += f"\n\n{note}"
    return maze_image(grid, start, CELL_PX), layout_to_ascii(grid, start, goal), status, state


def editor_show(state, preset):
    state = state or new_session()
    _editor(state, preset)
    return _editor_outputs(state)


def editor_reset(state, preset, source):
    state = state or new_session()
    if source == "training":
        state["editor"] = _training_layout(state, preset)
        return _editor_outputs(state, "Copied the training layout.")
    h, w = _locked_shape(state) or _preview_env(preset).grid.shape
    grid = np.full((h, w), WALL, dtype=np.int8)
    grid[1:-1, 1:-1] = EMPTY
    goal = (h - 2, w - 2)
    grid[goal] = GOAL
    state["editor"] = {"grid": grid, "start": (1, 1), "goal": goal}
    return _editor_outputs(state, "Blank maze: walls around the edge, start top-left, goal "
                                  "bottom-right.")


def editor_click(state, brush, evt: gr.SelectData):
    state = state or new_session()
    ed = _editor(state)
    cell = cell_at(evt.index[0], evt.index[1], ed["grid"].shape, CELL_PX)
    if cell is None:
        return _editor_outputs(state)
    grid, note = ed["grid"], ""
    if brush in ("start", "goal"):
        other = ed["goal"] if brush == "start" else ed["start"]
        if cell == other:
            note = "Start and goal can't share a cell."
        elif brush == "start":
            grid[cell] = EMPTY
            ed["start"] = cell
        else:
            grid[ed["goal"]] = EMPTY
            grid[cell] = GOAL
            ed["goal"] = cell
    elif cell in (ed["start"], ed["goal"]):
        note = "To move the start or goal, paint it somewhere else."
    else:
        grid[cell] = {"wall": WALL, "empty": EMPTY, "hazard": HAZARD}[brush]
    return _editor_outputs(state, note)


def editor_import(state, text):
    state = state or new_session()
    try:
        grid, start, goal = parse_layout(text or "")
    except ValueError as exc:
        _editor(state)
        return _editor_outputs(state)[0], text, f"⚠️ **Could not read that text:** {exc}.", state
    state["editor"] = {"grid": grid, "start": start, "goal": goal}
    return _editor_outputs(state, "Loaded the maze from text.")


def _maze_env(state: dict):
    """(environment on the editor's maze for the current agent, problem message)."""
    trainer: Trainer = state["trainer"]
    ed = _editor(state, trainer.config.env_preset)
    problem = check_layout(ed["grid"], ed["start"], ed["goal"])
    if problem:
        return None, f"Your maze can't be used yet: {problem}."
    text = layout_to_ascii(ed["grid"], ed["start"], ed["goal"])
    env = build_env(replace(trainer.config, layout_text=text))
    if env.obs_size != trainer.env.obs_size:
        lock = _locked_shape(state)
        return None, (f"This agent sees the whole map and was trained on "
                      f"{lock[0]} × {lock[1]} mazes, so it can't read a "
                      f"{ed['grid'].shape[0]} × {ed['grid'].shape[1]} one. Resize your maze or "
                      "train an agent on it (tick **Train on my maze** in the Train tab).")
    env.reset(seed=0)
    return env, None


def policy_map_ui(state, target):
    state = state or new_session()
    message = _need_trainer(state)
    if message:
        return None, message
    trainer: Trainer = state["trainer"]
    if target == MY_MAZE:
        env, problem = _maze_env(state)
        if problem:
            return None, problem
        where = "your maze"
    else:
        env = build_env(trainer.config)
        env.reset(seed=0)
        where = "its training layout"
    agent_name = trainer.config.agent
    title = f"{AGENT_LABELS[agent_name]} on {where} after {trainer.episodes_done} episodes"
    fig = policy_map_figure(env, trainer.agent, title=title)
    pm = policy_map(env, trainer.agent)
    if pm is None:
        return fig, (f"**{fun.NICKNAMES[agent_name]}** picks moves at random: there is no "
                     "policy to draw.")
    n_open = int(np.count_nonzero((env.grid != WALL) & (env.grid != GOAL)))
    n_seen = int(pm["seen"].sum())
    if agent_name == "tabular_q":
        note = (f"It has a learned value for **{n_seen} of {n_open}** open cells. Grey dots are "
                "states it never visited: there it has nothing to go on and moves at random.")
        if n_seen == 0 and env.config.obs_mode == "grid":
            note += (" Its state is the **whole map**, so changing even one cell makes every "
                     "state new to it: a table can't transfer anything to a different maze.")
    else:
        note = ("A network answers for every cell, including ones it never visited, so "
                "arrows far from where it trained are guesses, not knowledge.")
    if target == MY_MAZE and not trainer.config.layout_text:
        note += (" It never trained on your maze: follow the arrows from the start ring "
                 "to see whether its habits transfer, or watch it in **🎬 Watch it play**.")
    return fig, (f"**{fun.NICKNAMES[agent_name]}**: the move it would make on each cell with "
                 f"exploration off. {note}")


# ---------------------------------------------------------------------------
# Q-learning vs DQN race
# ---------------------------------------------------------------------------

def race(preset, episodes, seed):
    episodes = int(min(int(episodes), 600))
    trainers = {}
    for name in ("tabular_q", "dqn"):
        trainers[name] = Trainer(TrainConfig(
            env_preset=preset, agent=name, episodes=episodes, seed=int(seed),
            agent_kwargs={"epsilon": {"decay_episodes": max(1, int(episodes * 0.75))}}))
    chunk = max(10, episodes // 20)
    started = time.time()
    done = 0
    while done < episodes:
        step = min(chunk, episodes - done)
        for t in trainers.values():
            t.train(episodes=step)
        done += step
        status = (f"Episode **{done} / {episodes}** for both agents · "
                  f"{time.time() - started:.0f}s")
        yield status, learning_curves_figure({k: t.history for k, t in trainers.items()}), None
    rows = []
    for name, t in trainers.items():
        res = evaluate_agent(t.agent, build_env(t.config), episodes=10, seed=10_000 + int(seed))
        rows.append({"agent": f"{AGENT_LABELS[name]} ({fun.NICKNAMES[name]})",
                     "greedy success": round(res["success"], 2),
                     "path efficiency": round(res["path_efficiency"], 2),
                     "steps": round(res["steps"], 1), "mistakes": round(res["mistakes"], 1),
                     "episodes": episodes})
    status = (f"**Race finished** in {time.time() - started:.0f}s. This is **one seed**: "
              "a demo, not evidence. The **Results** tab has the multi-seed comparison "
              "with confidence intervals and significance tests.")
    yield (status, learning_curves_figure({k: t.history for k, t in trainers.items()}),
           pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Evaluate tab
# ---------------------------------------------------------------------------

def evaluate_ui(state, train_episodes, n_unseen):
    state = state or new_session()
    message = _need_trainer(state)
    if message:
        return message, pd.DataFrame()
    trainer: Trainer = state["trainer"]
    env = build_env(trainer.config)
    seed = 10_000 + trainer.config.seed
    pool = trainer.config.env_overrides.get("layout_pool")
    rows = []
    train_res = evaluate_agent(trainer.agent, env, layout_seeds=pool,
                               episodes=int(train_episodes), seed=seed)
    splits = [("training layout(s)", train_res)]
    unseen_env = _unseen_env(trainer)
    if int(n_unseen) > 0 and unseen_env is not None:
        unseen = list(range(UNSEEN_BASE, UNSEEN_BASE + int(n_unseen)))
        splits.append((f"{len(unseen)} unseen layouts",
                       evaluate_agent(trainer.agent, unseen_env, layout_seeds=unseen,
                                      seed=seed)))
    for label, res in splits:
        rows.append({"evaluated on": label, "episodes": res["n_episodes"],
                     "success": round(res["success"], 3),
                     "path efficiency": round(res["path_efficiency"], 3),
                     "steps": round(res["steps"], 1), "mistakes": round(res["mistakes"], 1),
                     "collisions": round(res["collisions"], 1),
                     "revisits": round(res["revisits"], 1), "hazards": round(res["hazards"], 2)})
    note = ("Greedy evaluation (exploration off). The agent was trained on one fixed layout, "
            "so poor results on unseen layouts are expected: that is what the "
            "`generalization` experiment measures."
            if not pool else "Greedy evaluation (exploration off).")
    if trainer.config.layout_text:
        note = ("Greedy evaluation (exploration off) on your own maze; unseen layouts are "
                f"the `{trainer.config.env_preset}` preset's generated ones. "
                + ("" if unseen_env is not None else _NO_UNSEEN))
    return (f"**{fun.NICKNAMES[trainer.config.agent]}** after {trainer.episodes_done} "
            f"episodes. {note}"), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Checkpoints tab
# ---------------------------------------------------------------------------

def save_checkpoint_ui(state, model_only):
    state = state or new_session()
    message = _need_trainer(state)
    if message:
        return message, None
    trainer: Trainer = state["trainer"]
    folder = Path(tempfile.mkdtemp(prefix="mind_ckpt_"))
    name = f"mind_{trainer.config.agent}_{trainer.config.env_preset}_ep{trainer.episodes_done}.pt"
    path = trainer.save_checkpoint(folder / name, include_buffer=not model_only)
    size = path.stat().st_size / 1e6
    return f"Saved **{name}** ({size:.1f} MB). Download it below.", str(path)


def load_checkpoint_ui(file, state):
    state = state or new_session()
    if file is None:
        return "Choose a `.pt` checkpoint file first.", state
    path = file if isinstance(file, str) else getattr(file, "name", file)
    try:
        agent, env, config = load_agent(path)
    except Exception as exc:  # malformed or unsafe file: report, don't crash
        return f"Could not load that file: {type(exc).__name__}: {exc}", state
    trainer = Trainer(config)
    trainer.load_checkpoint(path)
    state.update(trainer=trainer, frames=[], replay_env=None)
    where = " (your own maze)" if config.layout_text else ""
    return (f"Loaded **{AGENT_LABELS[config.agent]} ({fun.NICKNAMES[config.agent]})** on "
            f"`{config.env_preset}`{where} after **{trainer.episodes_done}** episodes. Training can "
            f"continue from here; the episode log starts empty."), state


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#hero h1 { margin-bottom: 0.2rem; }
#hero p { color: var(--body-text-color-subdued); margin-top: 0; }
#maze-text textarea { font-family: ui-monospace, Consolas, monospace; line-height: 1.25; }
"""

ABOUT = f"""
## How it works

An agent explores a grid. Each step it picks an action (up, right, down, left) and the
environment answers with a reward:

| Event | Reward |
|---|---|
| each step | {RewardConfig().step:+} |
| reach the goal | {RewardConfig().goal:+} (episode ends) |
| walk into a wall (agent stays put) | {RewardConfig().collision:+} |
| step on a hazard | {RewardConfig().hazard:+} |
| revisit a cell this episode | {RewardConfig().revisit:+} |
| run out of steps | {RewardConfig().timeout:+} (episode ends) |
| *reward shaping (optional)* | {SHAPING_STRENGTH:g} × (0.99 · φ(next cell) − φ(this cell)) |

**Learning from mistakes** means turning those penalties into better decisions. Tabular
Q-learning keeps a table of action values and nudges each one toward
`reward + γ · best next value`; DQN replaces the table with a neural network trained on
replayed past experience. **Greedy evaluation** turns exploration off, so the agent has to
rely on what it learned.

**Reward shaping** (optional, Train tab) adds a hint toward the goal: φ is minus the maze
distance to the goal, scaled to [−1, 0], so moving closer earns a little and moving away
costs the same. This form (potential-based) provably doesn't change the best policy, and
going in circles earns nothing. The hint is only for learning: every reported reward and
result is the task's. Whether it actually helps is tested in the `shaping_speed` and
`shaping_generalization` experiments.

## Meet the agents

| Agent | Nickname | In one line |
|---|---|---|
""" + "\n".join(f"| {AGENT_LABELS[a]} | {fun.NICKNAMES[a]} | {fun.TAGLINES[a]} |"
                for a in ("random", "tabular_q", "dqn", "dqn_dueling", "dqn_nstep",
                          "dqn_dueling_nstep", "dqn_per", "dqn_mistake", "dqn_per_mistake")) + """

## Achievements

""" + "\n".join(f"- **{t.split(':')[0]}**: {t.split(':', 1)[1].strip()}"
                for t in fun.ACHIEVEMENTS.values()) + """

The jokes only read the logs. They never change a reward, an agent or a result, and the
verdicts follow the statistics: no celebrating a p-value that doesn't support it.
"""


def build_app() -> gr.Blocks:
    experiments = available_experiments()
    with gr.Blocks(title="MIND: an AI that learns from its mistakes") as demo:
        state = gr.State(new_session())
        gr.Markdown("# 🧠 MIND: an AI that learns from its own mistakes\n"
                    "Reinforcement-learning agents get rewarded, penalised, and (eventually) "
                    "wiser. Train one, watch it bonk into walls, and see whether it improves.\n\n"
                    f"<small>Version {code_version()}</small>",
                    elem_id="hero")

        with gr.Tab("📊 Results"):
            gr.Markdown("Multi-seed experiments saved in `results/`. Every number is a "
                        "**greedy** evaluation (exploration off).")
            with gr.Row():
                exp_dd = gr.Dropdown(experiments, value=experiments[0] if experiments else None,
                                     label="Experiment")
                metric_dd = gr.Dropdown(METRIC_CHOICES, value="success", label="Metric")
            exp_md = gr.Markdown()
            bars = gr.Plot(show_label=False)
            curves = gr.Plot(show_label=False)
            gr.Markdown("#### Mean ± 95% confidence interval across seeds")
            summary_df = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("#### Are the differences real? (permutation tests on success rate)")
            tests_df = gr.Dataframe(interactive=False, wrap=True,
                                    column_widths=["8%", "18%", "11%", "9%", "54%"])
            outputs = [exp_md, bars, curves, summary_df, tests_df]
            exp_dd.change(show_results, [exp_dd, metric_dd], outputs)
            metric_dd.change(show_results, [exp_dd, metric_dd], outputs)
            demo.load(show_results, [exp_dd, metric_dd], outputs)

        with gr.Tab("🏋️ Train"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    preset = gr.Dropdown(PRESET_CHOICES, value="maze", label="Environment")
                    agent = gr.Dropdown(AGENT_CHOICES, value="tabular_q", label="Agent")
                    episodes = gr.Slider(50, 2000, value=300, step=50, label="Episodes",
                                         info="DQN: about 1 minute per 300 episodes on CPU "
                                              "(capped at 800)")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    decay = gr.Slider(10, 100, value=75, step=5,
                                      label="Exploration decays over (% of episodes)")
                    shaping = gr.Checkbox(value=False, label="Reward shaping",
                                          info="Adds a hint toward the goal to the reward "
                                               "it learns from (see About).")
                    use_maze = gr.Checkbox(value=False, label="Train on my maze",
                                           info="Draw it in the 🗺️ tab. The environment "
                                                "above still sets the rules.")
                    with gr.Row():
                        train_btn = gr.Button("Train", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    preview = gr.Plot(show_label=False)
                with gr.Column(scale=3):
                    status = gr.Markdown("Pick an agent and press **Train**.")
                    curve_plot = gr.Plot(show_label=False)
                    types_plot = gr.Plot(show_label=False)
                    with gr.Row():
                        commentary = gr.Textbox(label="Live commentary", lines=14,
                                                max_lines=14, interactive=False)
                        achievements_md = gr.Markdown()
            preset.change(lambda p: maze_figure(_preview_env(p), title=f"{p} layout"),
                          preset, preview)
            demo.load(lambda p: maze_figure(_preview_env(p), title=f"{p} layout"),
                      preset, preview)
            train_event = train_btn.click(
                train_agent_ui, [preset, agent, episodes, seed, decay, state, use_maze,
                                 shaping],
                [status, curve_plot, types_plot, commentary, achievements_md, state])
            stop_btn.click(None, None, None, cancels=[train_event])

        with gr.Tab("🔥 Mistake history"):
            with gr.Row():
                window = gr.Slider(10, 200, value=50, step=10,
                                   label="Compare first N vs last N episodes")
                refresh = gr.Button("Refresh", variant="primary")
            hist_md = gr.Markdown(_need_trainer(new_session()))
            heat_plot = gr.Plot(show_label=False)
            reward_plot = gr.Plot(show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Early vs late training")
                    phases_df = gr.Dataframe(interactive=False)
                with gr.Column():
                    gr.Markdown("#### 🏆 Hall of Shame (most repeated mistakes)")
                    shame_df = gr.Dataframe(interactive=False)
            gr.Markdown("#### Mistake log (most recent first, with commentary)")
            log_df = gr.Dataframe(interactive=False, wrap=True, max_height=420)
            hist_outputs = [hist_md, heat_plot, reward_plot, phases_df, shame_df, log_df]
            refresh.click(mistake_history, [state, window], hist_outputs)
            train_event.then(mistake_history, [state, window], hist_outputs)

        with gr.Tab("🎬 Watch it play"):
            gr.Markdown("Records one **greedy** episode, then replays it step by step with the "
                        "agent's Q-values for each move.")
            with gr.Row():
                layout_mode = gr.Radio(["Training layout", "Unseen layout", MY_MAZE],
                                       value="Training layout", label="Layout")
                unseen_seed = gr.Number(value=0, precision=0,
                                        label=f"Unseen layout number (seed {UNSEEN_BASE} + n)")
                speed = gr.Radio(["Slow", "Normal", "Fast"], value="Normal", label="Speed")
            with gr.Row():
                record_btn = gr.Button("Record episode", variant="primary")
                play_btn = gr.Button("▶ Play")
            frame_plot = gr.Plot(show_label=False)
            frame_md = gr.Markdown()
            step_slider = gr.Slider(0, 1, value=0, step=1, label="Step")
            record_btn.click(record_replay, [state, layout_mode, unseen_seed],
                             [frame_plot, frame_md, step_slider, state])
            step_slider.release(scrub_replay, [state, step_slider], [frame_plot, frame_md])
            play_btn.click(play_replay, [state, speed], [frame_plot, frame_md, step_slider])

        with gr.Tab("🗺️ Policy map & editor"):
            gr.Markdown("Draw a maze by clicking cells, then see which way the agent would go "
                        "on every cell. Test it on your maze in **🎬 Watch it play** or train "
                        "a new agent on it in **🏋️ Train**.")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### Draw a maze")
                    brush = gr.Radio(BRUSHES, value="wall", label="Brush: click a cell to paint")
                    maze_img = gr.Image(show_label=False, interactive=False, type="numpy",
                                        format="png", elem_id="maze-editor")
                    editor_md = gr.Markdown()
                    with gr.Row():
                        from_train_btn = gr.Button("Start from training layout")
                        blank_btn = gr.Button("Blank maze")
                    ascii_box = gr.Textbox(label="Maze as text: # wall · . empty · H hazard · "
                                                 "S start · G goal", lines=9, max_lines=25,
                                           elem_id="maze-text")
                    import_btn = gr.Button("Load text into the editor")
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### The agent's policy")
                    with gr.Row():
                        map_target = gr.Radio([MY_MAZE, "Training layout"], value=MY_MAZE,
                                              label="Show the policy on")
                        map_btn = gr.Button("Show policy map", variant="primary")
                    map_plot = gr.Plot(show_label=False)
                    map_md = gr.Markdown(_need_trainer(new_session()))
            editor_outputs = [maze_img, ascii_box, editor_md, state]
            maze_img.select(editor_click, [state, brush], editor_outputs)
            from_train_btn.click(lambda s, p: editor_reset(s, p, "training"), [state, preset],
                                 editor_outputs)
            blank_btn.click(lambda s, p: editor_reset(s, p, "blank"), [state, preset],
                            editor_outputs)
            import_btn.click(editor_import, [state, ascii_box], editor_outputs)
            map_btn.click(policy_map_ui, [state, map_target], [map_plot, map_md])
            demo.load(editor_show, [state, preset], editor_outputs)
            train_event.then(editor_show, [state, preset], editor_outputs)

        with gr.Tab("⚔️ Q-learning vs DQN"):
            gr.Markdown(f"A live race: **{fun.NICKNAMES['tabular_q']}** vs "
                        f"**{fun.NICKNAMES['dqn']}** on the same environment and seed.")
            with gr.Row():
                race_preset = gr.Dropdown(PRESET_CHOICES, value="maze", label="Environment")
                race_eps = gr.Slider(50, 600, value=200, step=50, label="Episodes each")
                race_seed = gr.Number(value=0, precision=0, label="Seed")
                race_btn = gr.Button("Start race", variant="primary")
            race_md = gr.Markdown()
            race_plot = gr.Plot(show_label=False)
            race_df = gr.Dataframe(interactive=False)
            race_btn.click(race, [race_preset, race_eps, race_seed], [race_md, race_plot, race_df])

        with gr.Tab("✅ Evaluate"):
            gr.Markdown("Greedy evaluation of the current agent on its training layout(s) and "
                        "on layouts it has never seen.")
            with gr.Row():
                eval_eps = gr.Slider(1, 50, value=10, step=1,
                                     label="Episodes on the training layout")
                eval_unseen = gr.Slider(0, 50, value=20, step=5, label="Unseen layouts")
                eval_btn = gr.Button("Evaluate", variant="primary")
            eval_md = gr.Markdown()
            eval_df = gr.Dataframe(interactive=False)
            eval_btn.click(evaluate_ui, [state, eval_eps, eval_unseen], [eval_md, eval_df])

        with gr.Tab("💾 Checkpoints"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Save the current agent")
                    model_only = gr.Checkbox(value=True, label="Model only (small file; "
                                             "untick to include the replay buffer for exact "
                                             "resumption)")
                    save_btn = gr.Button("Save checkpoint", variant="primary")
                    save_md = gr.Markdown()
                    save_file = gr.File(label="Download", interactive=False)
                with gr.Column():
                    gr.Markdown("### Load a checkpoint\nFiles are loaded in PyTorch's safe "
                                "mode, which refuses anything except tensors and plain data.")
                    load_file = gr.File(label="Upload a .pt checkpoint", file_types=[".pt"])
                    load_btn = gr.Button("Load", variant="primary")
                    load_md = gr.Markdown()
            save_btn.click(save_checkpoint_ui, [state, model_only], [save_md, save_file])
            load_btn.click(load_checkpoint_ui, [load_file, state], [load_md, state])

        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT)
    return demo


def _preview_env(preset: str):
    env = build_env(TrainConfig(env_preset=preset))
    env.reset(seed=0)
    return env


def launch_app(share: bool = False, port: Optional[int] = None, **kwargs):
    demo = build_app()
    demo.queue(default_concurrency_limit=2)
    return demo.launch(share=share, server_port=port, theme=gr.themes.Soft(),
                       css=CSS, max_file_size="200mb", **kwargs)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the MIND Gradio dashboard.")
    parser.add_argument("--share", action="store_true",
                        help="create a temporary public link (useful in Colab)")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    launch_app(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
