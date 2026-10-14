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
