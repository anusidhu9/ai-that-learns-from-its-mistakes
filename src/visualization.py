"""Charts and episode replays for MIND.

Every function returns a matplotlib `Figure` built without pyplot, so the
same code works in the notebook, in scripts and inside the Gradio app's
web server threads. Colours are fixed per entity (an agent or a mistake type
always has the same colour) and were checked for colour-blind separation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.envs import ACTION_NAMES, EMPTY, GOAL, HAZARD, WALL, GridWorld
from src.evaluation import mean_ci
from src.fun import event_line

AGENT_ORDER = ("random", "tabular_q", "dqn", "dqn_per", "dqn_mistake", "dqn_per_mistake",
               "dqn_5k", "dqn_mistake_5k", "dqn_10k")
AGENT_LABELS = {"random": "Random", "tabular_q": "Tabular Q", "dqn": "DQN", "dqn_per": "DQN + PER",
                "dqn_mistake": "DQN + mistakes", "dqn_per_mistake": "DQN + PER + mistakes",
                # long_memory_mistakes arms (buffer size in brackets)
                "dqn_5k": "DQN (5k)", "dqn_mistake_5k": "DQN + mistake memory (5k + 5k)",
                "dqn_10k": "DQN (10k)"}
# Mistake-aware variants share their base algorithm's hue (no palette hue left
# passes the colorblind checks beside orange and aqua) and add a second
# encoding instead: dash-dot lines, square markers and hatched bars.
AGENT_COLORS = {"random": "#8a8a85", "tabular_q": "#2a78d6", "dqn": "#eb6834", "dqn_per": "#1baf7a",
                "dqn_mistake": "#eb6834", "dqn_per_mistake": "#1baf7a",
                "dqn_5k": "#eb6834", "dqn_mistake_5k": "#eb6834", "dqn_10k": "#6250d6"}
AGENT_LINESTYLES = {"random": "--", "dqn_mistake": "-.", "dqn_per_mistake": "-.",
                    "dqn_mistake_5k": "-."}
AGENT_MARKERS = {"dqn_mistake": "s", "dqn_per_mistake": "s", "dqn_mistake_5k": "s"}
AGENT_HATCHES = {"dqn_mistake": "///", "dqn_per_mistake": "///", "dqn_mistake_5k": "///"}
MISTAKE_TYPES = ("collision", "revisit", "hazard", "timeout")
MISTAKE_COLORS = {"collision": "#e87ba4", "revisit": "#6250d6", "hazard": "#e34948",
                  "timeout": "#eda100"}
REWARD_PARTS = ("r_goal", "r_step", "r_collision", "r_hazard", "r_revisit", "r_timeout")

INK, MUTED, GRIDLINE, SURFACE = "#2b2b29", "#6b6b66", "#e6e5e0", "#ffffff"
EARLY_COLOR, LATE_COLOR = "#86b6ef", "#1c5cab"   # one hue, light = early, dark = late
CELL_COLORS = {EMPTY: "#fcfcfb", WALL: "#3d3d3a", HAZARD: "#e34948", GOAL: "#008300"}
CELL_CMAP = ListedColormap([CELL_COLORS[k] for k in (EMPTY, WALL, HAZARD, GOAL)])
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "mind_blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _figure(width: float, height: float, ncols: int = 1, nrows: int = 1, **kw):
    fig = Figure(figsize=(width, height), layout="constrained", facecolor=SURFACE)
    axes = fig.subplots(nrows, ncols, squeeze=False, **kw)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _legend_below(fig: Figure, ax) -> None:
    """One legend row underneath the chart, outside every plot area.

    A legend inside the axes can cover lines, bars or bands (and, near the
    top, crowd the title), so legends always go below the figure instead.
    """
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncols=len(handles),
                   frameon=False, fontsize=9)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=11, fontweight="bold")


def rolling_mean(values: Sequence[float], window: int) -> tuple:
    """(x, y) of a trailing rolling mean; shorter windows at the start."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return np.array([]), np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    return np.arange(len(x)), (c[idx] - c[lo]) / (idx - lo)


def _empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=10,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# ---------------------------------------------------------------------------
# Maze drawings
# ---------------------------------------------------------------------------

def draw_grid(ax, grid: np.ndarray, start=None, goal=None, pos=None, path=None,
              heat: Optional[np.ndarray] = None, heat_max: Optional[float] = None,
              path_color: str = "#2a78d6", title: str = ""):
    """Draw a maze on an axes: walls, hazards, goal, optional heatmap, path and agent."""
    ax.imshow(grid, cmap=CELL_CMAP, vmin=0, vmax=3, interpolation="nearest")
    image = None
    if heat is not None:
        masked = np.ma.masked_where((grid == WALL) | (heat <= 0), heat)
        image = ax.imshow(masked, cmap=HEAT_CMAP, vmin=0, vmax=heat_max or max(heat.max(), 1e-9),
                          interpolation="nearest")
        # keep hazards visible on top of the heat colours
        for r, c in np.argwhere(grid == HAZARD):
            ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, fill=False,
                                   edgecolor=CELL_COLORS[HAZARD], linewidth=2))
    if path is not None and len(path) > 1:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color=path_color, linewidth=2.5, alpha=0.9, solid_capstyle="round")
    if start is not None:
        ax.scatter([start[1]], [start[0]], s=90, facecolor="none", edgecolor=INK, linewidth=1.5,
                   zorder=3)
    if pos is not None:
        ax.scatter([pos[1]], [pos[0]], s=140, color=path_color, edgecolor="white", linewidth=2,
                   zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        _title(ax, title)
    return image


def maze_figure(env: GridWorld, path=None, title: str = "", color: str = "#2a78d6") -> Figure:
    """The environment's current layout, with an optional path."""
    fig, axes = _figure(4.2, 4.4)
    draw_grid(axes[0, 0], env.grid, start=env.start, pos=path[-1] if path else env.pos,
              path=path, path_color=color, title=title)
    return fig


def mistake_heatmap_figure(grid: np.ndarray, mistakes: List[dict], early: range,
                           late: range, start=None) -> Figure:
    """Mistakes per episode at each cell, early vs late training, on one shared scale."""
    def counts(episodes: range) -> np.ndarray:
        c = np.zeros(grid.shape)
        for m in mistakes:
            if m["episode"] in episodes:
                c[tuple(m["pos"])] += 1
        return c / max(1, len(episodes))

    a, b = counts(early), counts(late)
    vmax = max(a.max(), b.max(), 1e-9)
    fig, axes = _figure(8.4, 4.2, ncols=2)
    image = None
    for ax, data, label, rng in ((axes[0, 0], a, "Early", early), (axes[0, 1], b, "Late", late)):
        image = draw_grid(ax, grid, start=start, heat=data, heat_max=vmax,
                          title=f"{label}: episodes {rng.start}-{rng.stop - 1}\n"
                                f"{data.sum():.1f} mistakes per episode")
    if image is not None:
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
        bar.set_label("mistakes per episode at this cell", color=MUTED)
        bar.ax.tick_params(colors=MUTED, labelsize=8)
        bar.outline.set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# Training histories
# ---------------------------------------------------------------------------

def learning_curves_figure(histories: Dict[str, List[dict]], window: int = 25) -> Figure:
    """Rolling success rate and mistakes per episode, one line per agent."""
    fig, axes = _figure(10, 3.6, ncols=2)
    ax_s, ax_m = axes[0, 0], axes[0, 1]
    if not any(histories.values()):
        _empty(ax_s, "Train an agent to see its learning curve")
        _empty(ax_m, "")
        return fig
    for name, history in histories.items():
        if not history:
            continue
        color = AGENT_COLORS.get(name, INK)
        style = dict(color=color, linewidth=2, label=AGENT_LABELS.get(name, name),
                     linestyle=AGENT_LINESTYLES.get(name, "-"))
        x, y = rolling_mean([r["success"] for r in history], window)
        ax_s.plot(x, y, **style)
        x, y = rolling_mean([r["mistakes"] for r in history], window)
        ax_m.plot(x, y, **style)
    _title(ax_s, f"Success rate ({window}-episode rolling mean)")
    ax_s.set_ylim(-0.02, 1.02)
    _title(ax_m, "Mistakes per episode")
    ax_m.set_ylim(bottom=0)
    for ax in (ax_s, ax_m):
        ax.set_xlabel("Training episode")
    _legend_below(fig, ax_s)  # both panels share the same agents
    return fig


def mistake_types_figure(history: List[dict], window: int = 25) -> Figure:
    """Rolling count of each mistake type per episode."""
    fig, axes = _figure(10, 3.4)
    ax = axes[0, 0]
    if not history:
        _empty(ax, "No episodes yet")
        return fig
    columns = {"collision": "collisions", "revisit": "revisits", "hazard": "hazards",
               "timeout": "timeout"}
    for kind in MISTAKE_TYPES:
        x, y = rolling_mean([r[columns[kind]] for r in history], window)
        ax.plot(x, y, color=MISTAKE_COLORS[kind], linewidth=2, label=kind)
    _title(ax, f"Mistakes by type ({window}-episode rolling mean)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("per episode")
    ax.set_ylim(bottom=0)
    _legend_below(fig, ax)
    return fig


def reward_breakdown_figure(history: List[dict], window: int = 50) -> Figure:
    """Mean reward from each event, first vs last `window` episodes."""
    fig, axes = _figure(10, 3.6)
    ax = axes[0, 0]
    if not history:
        _empty(ax, "No episodes yet")
        return fig
    window = min(window, len(history))
    early = [np.mean([r[p] for r in history[:window]]) for p in REWARD_PARTS]
    late = [np.mean([r[p] for r in history[-window:]]) for p in REWARD_PARTS]
    y = np.arange(len(REWARD_PARTS))
    h = 0.38
    ax.barh(y - h / 2, early, height=h - 0.04, color=EARLY_COLOR, label=f"first {window}")
    ax.barh(y + h / 2, late, height=h - 0.04, color=LATE_COLOR, label=f"last {window}")
    for yi, (e, l) in enumerate(zip(early, late)):
        for value, offset in ((e, -h / 2), (l, h / 2)):
            if abs(value) > 1e-9:
                ax.text(value, yi + offset, f" {value:+.2f} ", va="center", fontsize=8,
                        color=INK, ha="left" if value > 0 else "right")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, [p[2:] for p in REWARD_PARTS])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _title(ax, "Where the reward comes from (mean per episode)")
    ax.set_xlabel("reward")
    _legend_below(fig, ax)
    lo = min(min(early), min(late), 0)
    hi = max(max(early), max(late), 0)
    pad = 0.25 * (hi - lo or 1)
    ax.set_xlim(lo - pad, hi + pad)
    return fig


# ---------------------------------------------------------------------------
# Experiments (multi-seed results)
# ---------------------------------------------------------------------------

BOUNDED_METRICS = ("success", "path_efficiency")


def _clip_interval(metric: str, low: float, high: float) -> tuple:
    """Clip a confidence interval to the values the metric can take.

    A t-interval on a few seeds can extend past 0 or 1 for a rate; the
    statistic itself is unchanged, only the drawn band is limited.
    """
    if metric in BOUNDED_METRICS:
        return max(low, 0.0), min(high, 1.0)
    if metric in ("steps", "mistakes", "collisions", "hazards", "revisits", "timeout"):
        return max(low, 0.0), high
    return low, high


def _agents_in(rows: List[dict]) -> List[str]:
    present = {r["agent"] for r in rows}
    return [a for a in AGENT_ORDER if a in present] + sorted(present - set(AGENT_ORDER))


def experiment_curves_figure(curves: List[dict], metric: str = "success",
                             ylabel: Optional[str] = None) -> Figure:
    """Greedy learning curve per agent: mean across seeds with a 95% CI band."""
    fig, axes = _figure(10, 3.8)
    ax = axes[0, 0]
    if not curves:
        _empty(ax, "No curve data")
        return fig
    for agent in _agents_in(curves):
        rows = [r for r in curves if r["agent"] == agent]
        episodes = sorted({int(r["episode"]) for r in rows})
        means, lows, highs = [], [], []
        for ep in episodes:
            m, ci = mean_ci([r[metric] for r in rows if int(r["episode"]) == ep])
            ci = 0.0 if np.isnan(ci) else ci
            low, high = _clip_interval(metric, m - ci, m + ci)
            means.append(m)
            lows.append(low)
            highs.append(high)
        color = AGENT_COLORS.get(agent, INK)
        ax.fill_between(episodes, lows, highs, color=color, alpha=0.10, linewidth=0)
        ax.plot(episodes, means, color=color, linewidth=2,
                marker=AGENT_MARKERS.get(agent, "o"), markersize=4,
                label=AGENT_LABELS.get(agent, agent),
                linestyle=AGENT_LINESTYLES.get(agent, "-"))
    _title(ax, f"Greedy {metric.replace('_', ' ')} during training (mean, 95% CI band)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    if metric in ("success", "path_efficiency"):
        ax.set_ylim(-0.03, 1.05)
    else:
        ax.set_ylim(bottom=0)
    _legend_below(fig, ax)
    return fig


def experiment_bars_figure(summary: List[dict], metric: str = "success") -> Figure:
    """Final greedy result per agent with 95% CI error bars, one panel per split."""
    splits = [s for s in ("train", "unseen") if any(r["split"] == s for r in summary)]
    # Always 10 inches wide, like the other charts, so a single panel fills
    # the page instead of sitting in the middle of empty space.
    fig, axes = _figure(10, 3.4, ncols=max(1, len(splits)), sharex=True)
    if not summary:
        _empty(axes[0, 0], "No results")
        return fig
    for ax, split in zip(axes[0], splits):
        rows = {r["agent"]: r for r in summary if r["split"] == split}
        agents = [a for a in _agents_in(list(rows.values()))]
        y = np.arange(len(agents))
        means = [rows[a][f"{metric}_mean"] for a in agents]
        cis = [0.0 if np.isnan(rows[a][f"{metric}_ci"]) else rows[a][f"{metric}_ci"]
               for a in agents]
        bounds = [_clip_interval(metric, m - ci, m + ci) for m, ci in zip(means, cis)]
        xerr = np.array([[m - lo for m, (lo, _) in zip(means, bounds)],
                         [hi - m for m, (_, hi) in zip(means, bounds)]])
        bars = ax.barh(y, means, height=0.6, color=[AGENT_COLORS.get(a, INK) for a in agents],
                       xerr=xerr, error_kw={"ecolor": INK, "elinewidth": 1.2, "capsize": 4})
        for bar, a in zip(bars, agents):
            if a in AGENT_HATCHES:  # surface-colored hatch over the shared hue
                bar.set_hatch(AGENT_HATCHES[a])
                bar.set_edgecolor(SURFACE)
                bar.set_linewidth(0)
        for yi, (m, (_, hi)) in enumerate(zip(means, bounds)):
            ax.text(hi, yi, f"  {m:.2f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [AGENT_LABELS.get(a, a) for a in agents])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        label = {"train": "training layouts", "unseen": "unseen layouts"}[split]
        _title(ax, f"{metric.replace('_', ' ').capitalize()}: {label}")
        if metric in ("success", "path_efficiency"):
            ax.set_xlim(0, 1.25)
    return fig


# ---------------------------------------------------------------------------
# Step-by-step replays
# ---------------------------------------------------------------------------

def record_episode(env: GridWorld, agent, layout_seed: Optional[int] = None,
                   seed: int = 0) -> List[dict]:
    """Play one greedy episode and record every step for replay.

    Like evaluation, this swaps in a seeded tie-breaking generator and
    restores the agent's own afterwards, so recording never changes training.
    Frame 0 is the start state; each later frame holds the action taken, the
    reward, the events, the agent's Q-values for the state it acted in, and a
    commentary line.
    """
    original_rng = agent.rng
    agent.rng = np.random.default_rng(seed)
    try:
        obs, _ = env.reset(seed=seed, layout_seed=layout_seed)
        frames = [{"step": 0, "pos": env.pos, "action": None, "reward": 0.0, "total": 0.0,
                   "events": [], "q_values": agent.q_values(obs).tolist()
                   if hasattr(agent, "q_values") else None,
                   "line": "Ready. The maze awaits."}]
        total, done, info = 0.0, False, {}
        while not done:
            q = agent.q_values(obs).tolist() if hasattr(agent, "q_values") else None
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
            line = " ".join(event_line(e, env.pos, env.steps) for e in info["events"]) \
                or f"Moved {ACTION_NAMES[action]}."
            frames.append({"step": env.steps, "pos": env.pos, "action": ACTION_NAMES[action],
                           "reward": reward, "total": total, "events": list(info["events"]),
                           "q_values": q, "line": line})
        frames[-1]["episode"] = info.get("episode")
    finally:
        agent.rng = original_rng
    return frames


def frame_figure(env: GridWorld, frames: List[dict], index: int,
                 color: str = "#2a78d6") -> Figure:
    """The maze with the path up to frame `index`, beside the agent's Q-values."""
    index = max(0, min(index, len(frames) - 1))
    frame = frames[index]
    path = [f["pos"] for f in frames[: index + 1]]
    fig = Figure(figsize=(8.6, 4.2), layout="constrained", facecolor=SURFACE)
    ax_maze, ax_q = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    _style(ax_q)
    events = ", ".join(frame["events"]) or "no event"
    # Two short lines: one long line can run into the Q-value panel's title.
    draw_grid(ax_maze, env.grid, start=env.start, pos=frame["pos"], path=path, path_color=color,
              title=f"Step {frame['step']}: {frame['action'] or 'start'}\n{events}")
    # Q-values of the state shown are stored with the *next* frame (the step
    # that acted from this state); frame 0 also carries them for the start.
    if index + 1 < len(frames):
        q = frames[index + 1]["q_values"]
    else:
        q = None
    if index + 1 >= len(frames):
        _empty(ax_q, "Episode finished")
    elif q is None:
        _empty(ax_q, "This agent has no Q-values")
    else:
        q = np.asarray(q, dtype=float)
        best = int(np.argmax(q))
        colors = [LATE_COLOR if i == best else EARLY_COLOR for i in range(len(q))]
        y = np.arange(len(q))
        ax_q.barh(y, q, color=colors, height=0.6)
        for yi, value in enumerate(q):
            ax_q.text(value, yi, f" {value:+.3f} ", va="center", fontsize=9, color=INK,
                      ha="left" if value >= 0 else "right")
        ax_q.axvline(0, color=MUTED, linewidth=1)
        ax_q.set_yticks(y, ACTION_NAMES)
        ax_q.invert_yaxis()
        ax_q.grid(axis="y", visible=False)
        span = max(abs(q).max(), 1e-6)
        ax_q.set_xlim(min(q.min(), 0) - 0.35 * span, max(q.max(), 0) + 0.35 * span)
        _title(ax_q, "Q-values at this cell\n(dark = chosen action)")
    return fig


def make_report_figures(results_dir="results", out_dir="docs/figures", seed: int = 0) -> list:
    """Regenerate every figure used in README.md and REPORT.md.

    Multi-seed charts come from the saved results; the single-run charts
    (heatmap, reward breakdown, replay) come from a fresh, seeded 400-episode
    tabular Q run on the fixed maze, so they are reproducible.
    """
    from pathlib import Path

    from src.evaluation import load_experiment
    from src.training import TrainConfig, Trainer

    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig: Figure, name: str) -> None:
        path = out_dir / name
        fig.savefig(path, dpi=110)
        saved.append(path)

    fixed = load_experiment(results_dir / "fixed_maze")
    save(experiment_curves_figure(fixed["curves"], "success"), "fixed_maze_success.png")
    save(experiment_curves_figure(fixed["curves"], "mistakes"), "fixed_maze_mistakes.png")
    terminal = load_experiment(results_dir / "terminal_hazards")
    save(experiment_curves_figure(terminal["curves"], "success"), "terminal_hazards_success.png")
    save(experiment_curves_figure(terminal["curves"], "mistakes"), "terminal_hazards_mistakes.png")
    save(experiment_bars_figure(load_experiment(results_dir / "hazard_maze")["summary"],
                                "hazards"), "hazard_maze_hazards.png")
    save(experiment_bars_figure(load_experiment(results_dir / "generalization")["summary"],
                                "success"), "generalization_success.png")
    long_memory = load_experiment(results_dir / "long_memory_mistakes")
    save(experiment_bars_figure(long_memory["summary"], "success"), "long_memory_success.png")
    save(experiment_curves_figure(long_memory["curves"], "success"), "long_memory_curves.png")

    trainer = Trainer(TrainConfig(env_preset="maze", agent="tabular_q", episodes=400, seed=seed,
                                  agent_kwargs={"epsilon": {"decay_episodes": 300}}))
    trainer.train()
    h, env = trainer.history, trainer.env
    save(learning_curves_figure({"tabular_q": h}), "single_run_curves.png")
    save(mistake_heatmap_figure(env.grid, trainer.mistakes, range(0, 50), range(350, 400),
                                start=env.start), "mistake_heatmap.png")
    save(reward_breakdown_figure(h), "reward_breakdown.png")
    frames = record_episode(env, trainer.agent)
    save(frame_figure(env, frames, 5, AGENT_COLORS["tabular_q"]), "replay_frame.png")
    return saved


if __name__ == "__main__":
    for p in make_report_figures():
        print("wrote", p)
