"""You-play tab, Watch-it-play controls and the start-cell option (Stage M)."""
from pathlib import Path

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

from src import app, fun  # noqa: E402
from src.agents import make_agent  # noqa: E402
from src.envs import ACTION_NAMES, EMPTY, GOAL, WALL, make_env, shortest_path_length  # noqa: E402
from src.visualization import (FOG_COLOR, PLAYER_COLOR, TRAIL_COLOR, play_image,  # noqa: E402
                               record_episode)

MODEL = Path(__file__).resolve().parent.parent / "results" / "memory_vs_replay" / "models" \
    / "dqn_rnn_s800.pt"


def rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def centre(img, cell, px=36):
    return tuple(img[cell[0] * px + px // 2, cell[1] * px + px // 2])


@pytest.fixture(scope="module")
def trained():
    state = app.new_session()
    for _ in app.train_agent_ui("maze", "tabular_q", 60, 0, 75, state):
        pass
    return state


# -- environment and recording ------------------------------------------------

def test_reset_start_changes_only_the_start():
    env = make_env("maze", layout_pool=tuple(range(20)))
    obs, _ = env.reset(seed=3)
    grid, goal, ls = env.grid.copy(), env.goal, env.layout_seed
    after = env.rng.random()
    empty = [tuple(c) for c in np.argwhere(env.grid == EMPTY) if tuple(c) != env.start]
    cell = empty[len(empty) // 2]
    env.reset(seed=3, start=cell)
    assert np.array_equal(env.grid, grid) and env.goal == goal and env.layout_seed == ls
    assert env.rng.random() == after                       # same random draws
    assert env.pos == env.start == cell and env.visited == {cell}
    assert env.optimal_length == shortest_path_length(grid, cell, goal)
    np.testing.assert_array_equal(env.observation_at(cell), env.reset(seed=3, start=cell)[0])
    obs2, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs, obs2)               # no start: exactly as before


def test_reset_refuses_bad_starts():
    env = make_env("maze")
    env.reset(seed=0)
    wall = tuple(np.argwhere(env.grid == WALL)[0])
    for bad in (wall, env.goal, (-1, 0), (0, 99)):
        with pytest.raises(ValueError, match="empty cell"):
            env.reset(seed=0, start=bad)
    open_env = make_env("open")
    open_env.reset(seed=0)
    grid = open_env.grid
    for r, c in np.argwhere(grid == EMPTY):    # an empty cell walled off from the goal
        if shortest_path_length(grid, (r, c), open_env.goal, avoid_hazards=False) is None:
            with pytest.raises(ValueError, match="reached"):
                open_env.reset(seed=0, start=(int(r), int(c)))
            break


def test_record_episode_from_another_start():
    env = make_env("maze", obs_mode="local")
    agent = make_agent("dqn", env.obs_size, env.n_actions, seed=0)
    env.reset(seed=0)
    cell = next(tuple(c) for c in np.argwhere(env.grid == EMPTY) if tuple(c) != env.start)
    rng = agent.rng
    frames = record_episode(env, agent, start=cell)
    assert frames[0]["pos"] == cell and agent.rng is rng
    assert frames[-1]["episode"]["optimal_length"] == shortest_path_length(env.grid, cell,
                                                                           env.goal)


# -- drawing -------------------------------------------------------------------

def test_play_image_full_map_and_agent_view():
    env = make_env("maze")
    env.reset(seed=0)
    path = [env.start]
    for move in ("down", "down", "right", "right"):
        env.step(ACTION_NAMES.index(move))
        path.append(env.pos)
    full = play_image(env.grid, env.start, env.pos, path)
    h, w = env.grid.shape
    assert full.shape == (h * 36, w * 36, 3)
    assert centre(full, env.pos) == rgb(PLAYER_COLOR)
    trail = [p for p in path[1:-1] if env.grid[p] == EMPTY and p != env.pos]
    assert trail and all(centre(full, p) == rgb(TRAIL_COLOR) for p in trail)
    assert centre(full, env.goal) != rgb(FOG_COLOR)
    view = play_image(env.grid, env.start, env.pos, path, view_radius=2)
    assert centre(view, env.pos) == rgb(PLAYER_COLOR)
    for r in range(h):
        for c in range(w):
            inside = abs(r - env.pos[0]) <= 2 and abs(c - env.pos[1]) <= 2
            assert (centre(view, (r, c)) == rgb(FOG_COLOR)) == (not inside)
    assert all(centre(view, p) != rgb(TRAIL_COLOR) for p in trail)   # no memory shown


# -- You play tab --------------------------------------------------------------

def test_moves_follow_the_environment_exactly():
    state = app.new_session()
    img, md, state = app.play_new(state, "Training layout", 0, app.FULL_VIEW)
    assert img is not None and "Step 0 / 100" in md
    twin = state["play"]["env"].__class__(state["play"]["env"].config)
    twin.reset(seed=0)
    moves = ["right", "down", "down", "up", "left", "right", "right", "down"]
    for move in moves:
        img, md, state = app.play_move(state, move, app.FULL_VIEW)
        twin.step(ACTION_NAMES.index(move))
    env = state["play"]["env"]
    assert env.pos == twin.pos and env.steps == twin.steps == len(moves)
    assert env.stats == twin.stats
    assert state["play"]["path"][-1] == env.pos and len(state["play"]["path"]) == len(moves) + 1
    assert f"Step {len(moves)} / 100" in md


def test_game_ends_by_timeout_and_then_ignores_moves():
    state = app.new_session()
    _, _, state = app.play_new(state, "Unseen layout", 3, app.FULL_VIEW)
    assert state["play"]["env"].layout_seed == app.UNSEEN_BASE + 3
    for _ in range(100):
        _, md, state = app.play_move(state, "up", app.FULL_VIEW)
    summary = state["play"]["summary"]
    assert summary["timeout"] and summary["steps"] == 100 and "Finished" in md
    _, md2, state = app.play_move(state, "down", app.FULL_VIEW)
    assert state["play"]["env"].steps == 100 and md2 == md


def test_agent_view_and_switching_views():
    state = app.new_session()
    img, md, state = app.play_new(state, "Training layout", 0, app.AGENT_VIEW)
    env = state["play"]["env"]
    dr, dc = env.goal[0] - env.pos[0], env.goal[1] - env.pos[1]
    assert f"{dr} down" in md and f"{dc} right" in md
    assert centre(img, env.goal) == rgb(FOG_COLOR)
    full, md_full, _ = app.play_view(state, app.FULL_VIEW)
    assert "Goal:" not in md_full and centre(full, env.goal) != rgb(FOG_COLOR)


def test_restart_and_messages_without_a_game():
    empty = app.new_session()
    assert "New game" in app.play_move(empty, "up", app.FULL_VIEW)[1]
    assert "New game" in app.play_view(empty, app.FULL_VIEW)[1]
    assert "Finish a game" in app.play_compare(empty)[1]
    _, _, state = app.play_new(empty, "Training layout", 0, app.FULL_VIEW)
    app.play_move(state, "right", app.FULL_VIEW)
    _, md, state = app.play_new(state, "Training layout", 0, app.FULL_VIEW)
    assert state["play"]["env"].steps == 0 and len(state["play"]["path"]) == 1


def test_my_maze_without_an_agent():
    state = app.new_session()
    _, md, state = app.play_new(state, app.MY_MAZE, 0, app.FULL_VIEW)
    ed = state["editor"]
    assert np.array_equal(state["play"]["env"].grid, ed["grid"])
    g = ed["grid"]
    g[1:-1, 1:-1] = WALL                           # wall the goal off from the start
    g[ed["start"]], g[ed["goal"]] = EMPTY, GOAL
    assert "can't be used" in app.play_new(state, app.MY_MAZE, 0, app.FULL_VIEW)[1]


def _finish(state):
    while not state["play"]["env"].done:
        app.play_move(state, "up", app.FULL_VIEW)


def test_compare_needs_an_agent_and_uses_the_same_maze(trained):
    state = app.new_session()
    _, _, state = app.play_new(state, "Training layout", 0, app.FULL_VIEW)
    _finish(state)
    fig, md = app.play_compare(state)
    assert fig is None and "No agent to compare" in md
    _, _, trained = app.play_new(trained, "Training layout", 0, app.FULL_VIEW)
    _finish(trained)
    fig, md = app.play_compare(trained)
    assert fig is not None and "| You |" in md and "Reached the goal" in md
    env = app.build_env(trained["trainer"].config)
    frames = record_episode(env, trained["trainer"].agent)
    assert np.array_equal(env.grid, trained["play"]["env"].grid)
    assert f"| {frames[-1]['episode']['steps']} |" in md
    trained["play"]["mode"] = "Unseen layout"     # the agent's maze would now differ
    assert "not the one you played" in app.play_compare(trained)[1]


@pytest.mark.skipif(not MODEL.exists(), reason="saved Stage L network not present")
def test_compare_with_a_saved_memory_network():
    state = app.new_session()
    message, state = app.load_checkpoint_ui(str(MODEL), state)
    assert "Loaded" in message
    _, _, state = app.play_new(state, "Unseen layout", 1, app.FULL_VIEW)
    _finish(state)
    fig, md = app.play_compare(state)
    assert fig is not None and "GRU" in md


def test_human_vs_agent_lines():
    def ep(success, steps):
        return {"success": success, "steps": steps, "mistakes": 0}

    cases = [(ep(True, 12), ep(False, 100)), (ep(False, 100), ep(True, 12)),
             (ep(False, 100), ep(False, 100)), (ep(True, 12), ep(True, 20)),
             (ep(True, 20), ep(True, 12)), (ep(True, 12), ep(True, 12))]
    lines = [fun.human_vs_agent_line(a, b) for a, b in cases]
    assert all(lines) and len(set(lines)) == len(lines)
    assert lines == [fun.human_vs_agent_line(a, b) for a, b in cases]   # deterministic
    assert all(line.isascii() for line in lines)


# -- Watch it play controls ----------------------------------------------------

def test_step_buttons_and_playback_from_a_step(trained):
    app.record_replay(trained, "Training layout", 0)
    n = len(trained["frames"])
    assert app.step_replay(trained, 0, -1)[2]["value"] == 0
    assert app.step_replay(trained, 0, 1)[2]["value"] == 1
    assert app.step_replay(trained, n - 1, 1)[2]["value"] == n - 1
    assert app.step_replay(app.new_session(), 0, 1)[1] == "Record an episode first."
    assert len(list(app.play_replay(trained, 0.05, 1))) == n - 1      # resumes at step 1
    assert len(list(app.play_replay(trained, 0.05, n - 1))) == n      # at the end: restart
    assert app._replay_delay(5) == 1.0 and app._replay_delay(0) == 0.05
    assert app._replay_delay("Fast") == 0.1


def test_record_from_a_clicked_cell(trained):
    assert app.record_from_cell(app.new_session(), (1, 1))[1] == "Record an episode first."
    app.record_replay(trained, "Training layout", 0)
    env = trained["replay_env"]
    cell = next(tuple(int(v) for v in c) for c in np.argwhere(env.grid == EMPTY)
                if tuple(c) != env.start)
    fig, md, update, state = app.record_from_cell(trained, cell)
    assert state["frames"][0]["pos"] == cell and f"from {cell}" in md
    start_map = app.replay_start_map(state)
    assert start_map.shape == (env.grid.shape[0] * 36, env.grid.shape[1] * 36, 3)
    wall = tuple(int(v) for v in np.argwhere(env.grid == WALL)[0])
    fig, md, update, state = app.record_from_cell(trained, wall)
    assert md.startswith("Can't start there") and state["frames"][0]["pos"] == cell
    assert app.replay_start_map(app.new_session()) is None
    assert env.grid[env.goal] == GOAL


def test_app_has_the_new_controls_and_key_bindings():
    demo = app.build_app()
    ids = {getattr(b, "elem_id", None) for b in demo.blocks.values()}
    for elem in ("play-up", "play-right", "play-down", "play-left", "play-new", "play-compare",
                 "play-board", "replay-back", "replay-forward", "replay-pause",
                 "replay-start-map"):
        assert elem in ids
        if elem.startswith("play-") and elem not in ("play-new", "play-compare", "play-board"):
            assert f'"{elem}"' in app.PLAY_KEYS_JS
