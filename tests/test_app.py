"""Smoke tests for the Gradio app's handlers (no browser or server needed)."""
import pandas as pd
import pytest

gr = pytest.importorskip("gradio")

from src import app  # noqa: E402


@pytest.fixture(scope="module")
def session():
    state = app.new_session()
    last = None
    for last in app.train_agent_ui("maze", "tabular_q", 60, 0, 75, state):
        pass
    assert "Training finished" in last[0]
    return state


def test_app_builds():
    assert isinstance(app.build_app(), gr.Blocks)


def test_results_tab_reads_saved_experiments():
    names = app.available_experiments()
    assert "fixed_maze" in names
    header, bars, curves, summary, tests = app.show_results("fixed_maze", "success")
    assert "fixed_maze" in header
    assert len(summary) == 4 and len(tests) == 6
    assert tests["verdict"].str.len().min() > 0


def test_tabs_explain_when_no_agent_exists():
    empty = app.new_session()
    assert "No agent yet" in app.mistake_history(empty, 50)[0]
    assert "No agent yet" in app.record_replay(empty, "Training layout", 0)[1]
    assert "No agent yet" in app.evaluate_ui(empty, 5, 5)[0]
    assert "No agent yet" in app.save_checkpoint_ui(empty, True)[0]


def test_mistake_history_tab(session):
    summary, heat, reward, phases, shame, log = app.mistake_history(session, 20)
    assert "mistakes over 60 episodes" in summary
    assert isinstance(phases, pd.DataFrame) and len(phases) > 0
    assert list(shame.columns) == ["#", "award", "mistake", "cell", "times"]
    assert "commentary" in log.columns and len(log) > 0


def test_replay_tab(session):
    fig, info, update, state = app.record_replay(session, "Training layout", 0)
    n = len(state["frames"])
    assert n >= 2 and "Episode result" in info
    assert len(list(app.play_replay(state, "Fast"))) == n
    assert app.scrub_replay(state, 1)[1].startswith("**Step 1")
    unseen = app.record_replay(session, "Unseen layout", 2)[1]
    assert "Episode result" in unseen


def test_evaluate_tab(session):
    message, table = app.evaluate_ui(session, 3, 5)
    assert list(table["evaluated on"]) == ["training layout(s)", "5 unseen layouts"]


def test_checkpoint_round_trip_through_the_ui(session, tmp_path):
    message, path = app.save_checkpoint_ui(session, True)
    assert path.endswith(".pt")
    fresh = app.new_session()
    loaded_message, fresh = app.load_checkpoint_ui(path, fresh)
    assert "Loaded" in loaded_message and fresh["trainer"].episodes_done == 60
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    assert "Could not load" in app.load_checkpoint_ui(str(bad), app.new_session())[0]


def test_race_tab():
    last = None
    for last in app.race("open", 50, 1):
        pass
    status, fig, table = last
    assert "one seed" in status
    assert len(table) == 2


def test_header_shows_code_version(tmp_path, monkeypatch):
    assert app.code_version() != ""
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "VERSION").write_text("abc1234 (2026-10-14)\n")
    assert app.code_version() == "abc1234 (2026-10-14)"


# -- policy map and maze editor ------------------------------------------------------

class Click:
    """Stand-in for gr.SelectData: only .index ([x, y] in pixels) is used."""

    def __init__(self, row, col):
        self.index = [col * app.CELL_PX + 5, row * app.CELL_PX + 5]


def test_editor_and_policy_map_on_my_maze(session):
    session["editor"] = None
    image, text, status, state = app.editor_show(session, "maze")
    assert image.shape[:2] == (9 * app.CELL_PX, 9 * app.CELL_PX)
    assert "Ready" in status and text.count("S") == 1 and text.count("G") == 1
    start = state["editor"]["start"]
    assert "somewhere else" in app.editor_click(state, "wall", Click(*start))[2]
    _, text, status, state = app.editor_click(state, "goal", Click(1, 3))
    assert state["editor"]["goal"] == (1, 3) and text.count("G") == 1
    fig, message = app.policy_map_ui(state, app.MY_MAZE)
    assert fig is not None and "learned value" in message and "never trained" in message
    assert "Episode result" in app.record_replay(state, app.MY_MAZE, 0)[1]
    assert app.policy_map_ui(state, "Training layout")[0] is not None


def test_editor_rejects_bad_mazes(session):
    _, _, status, state = app.editor_import(session, "#S.\n#.")
    assert "Could not read" in status
    _, _, status, state = app.editor_import(state, "#####\n#S#G#\n#####")
    assert "Can't use this maze" in status
    assert "can't be used" in app.policy_map_ui(state, app.MY_MAZE)[1]
    _, _, status, state = app.editor_import(state, "#####\n#S.G#\n#####")
    assert "only fits 9 × 9" in status
    assert "can't read" in app.policy_map_ui(state, app.MY_MAZE)[1]
    assert "Blank maze" in app.editor_reset(state, "maze", "blank")[2]
    assert state["editor"]["grid"].shape == (9, 9)


def test_train_on_my_maze():
    state = app.new_session()
    app.editor_import(state, "#####\n#S#G#\n#####")
    last = list(app.train_agent_ui("maze", "tabular_q", 20, 0, 75, state, True))[-1]
    assert "Can't train on your maze" in last[0] and state["trainer"] is None
    app.editor_import(state, "#######\n#S....#\n#.###.#\n#....G#\n#######")
    for last in app.train_agent_ui("maze", "tabular_q", 50, 0, 75, state, True):
        pass
    assert "Training finished" in last[0]
    assert state["trainer"].env.grid.shape == (5, 7)
    message, table = app.evaluate_ui(state, 3, 5)
    assert list(table["evaluated on"]) == ["training layout(s)"]
    assert "no unseen layouts" in message
    assert "trained on your own maze" in app.record_replay(state, "Unseen layout", 0)[1]


def test_train_with_reward_shaping():
    state = app.new_session()
    for last in app.train_agent_ui("maze", "tabular_q", 30, 0, 75, state, False, True):
        pass
    trainer = state["trainer"]
    assert trainer.env.config.rewards.shaping == app.SHAPING_STRENGTH
    assert any(r["r_shaping"] != 0.0 for r in trainer.history)
    assert "shaping" in app.ABOUT
