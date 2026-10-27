from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.agents import EpsilonSchedule, TabularQAgent, make_agent
from src.envs import GOAL, WALL, make_env
from src.evaluation import load_experiment
from src.training import TrainConfig, Trainer, train_agent
from src.visualization import (_clip_interval, cell_at, experiment_bars_figure,
                               experiment_curves_figure, frame_figure, learning_curves_figure,
                               maze_figure, maze_image, mistake_heatmap_figure,
                               mistake_types_figure, policy_map, policy_map_figure,
                               record_episode, reward_breakdown_figure, rolling_mean)

RESULTS = Path(__file__).resolve().parent.parent / "results"


@pytest.fixture(scope="module")
def trained():
    trainer = Trainer(TrainConfig(env_preset="maze_hazards", episodes=60,
                                  agent_kwargs={"epsilon": {"decay_episodes": 40}}))
    trainer.train()
    return trainer


def test_rolling_mean():
    x, y = rolling_mean([1, 2, 3, 4], window=2)
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1.0, 1.5, 2.5, 3.5]
    assert rolling_mean([], 5)[1].size == 0


def test_clip_interval_limits_rates_only():
    assert _clip_interval("success", -0.2, 1.3) == (0.0, 1.0)
    assert _clip_interval("mistakes", -4.0, 9.0) == (0.0, 9.0)
    assert _clip_interval("total_reward", -4.0, 9.0) == (-4.0, 9.0)


def test_training_figures(trained):
    for fig in (learning_curves_figure({"tabular_q": trained.history}),
                mistake_types_figure(trained.history),
                reward_breakdown_figure(trained.history),
                mistake_heatmap_figure(trained.env.grid, trained.mistakes, range(0, 20),
                                       range(40, 60), start=trained.env.start),
                maze_figure(trained.env, title="x")):
        assert isinstance(fig, Figure)
        assert fig.axes


def test_figures_handle_empty_input():
    for fig in (learning_curves_figure({"dqn": []}), mistake_types_figure([]),
                reward_breakdown_figure([]), experiment_curves_figure([]),
                experiment_bars_figure([])):
        assert isinstance(fig, Figure)


@pytest.mark.skipif(not (RESULTS / "generalization").exists(), reason="saved results missing")
def test_experiment_figures_from_saved_results():
    exp = load_experiment(RESULTS / "generalization")
    bars = experiment_bars_figure(exp["summary"], "success")
    assert len(bars.axes) == 2                                # train and unseen panels
    curves = experiment_curves_figure(exp["curves"], "mistakes")
    assert curves.axes[0].get_legend() is None          # legend sits below the chart
    labels = {t.get_text() for t in curves.legends[0].get_texts()}
    assert labels == {"Random", "Tabular Q", "DQN"}


def test_record_episode_and_frames_do_not_change_agent():
    env = make_env("maze")
    agent = TabularQAgent(4, epsilon=EpsilonSchedule(decay_episodes=30), seed=0)
    train_agent(env, agent, 40, seed=0)
    before = agent.rng.bit_generator.state
    q_before = {k: v.copy() for k, v in agent.q.items()}
    frames = record_episode(make_env("maze"), agent)
    assert agent.rng.bit_generator.state == before
    assert all(np.array_equal(agent.q[k], q_before[k]) for k in q_before)
    assert frames[0]["step"] == 0 and frames[0]["action"] is None
    assert [f["step"] for f in frames] == list(range(len(frames)))
    assert frames[-1]["episode"]["steps"] == len(frames) - 1
    assert all(f["line"] for f in frames)
    for i in (0, len(frames) // 2, len(frames) - 1, 999):     # out of range is clamped
        assert isinstance(frame_figure(env, frames, i), Figure)


def test_recorded_episode_is_the_greedy_evaluation_episode():
    from src.evaluation import evaluate_agent
    env = make_env("maze")
    agent = TabularQAgent(4, epsilon=EpsilonSchedule(decay_episodes=30), seed=0)
    train_agent(env, agent, 60, seed=0)
    frames = record_episode(make_env("maze"), agent, seed=3)
    result = evaluate_agent(agent, make_env("maze"), episodes=1, seed=3)
    assert frames[-1]["episode"]["steps"] == result["episodes"][0]["steps"]


def _boxes_overlap(a, b) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _all_figures(trainer):
    env = make_env("maze")
    agent = TabularQAgent(4, epsilon=EpsilonSchedule(decay_episodes=30), seed=0)
    train_agent(env, agent, 40, seed=0)
    replay_env = make_env("maze")
    frames = record_episode(replay_env, agent)
    figs = [learning_curves_figure({"tabular_q": trainer.history, "random": trainer.history}),
            mistake_types_figure(trainer.history),
            reward_breakdown_figure(trainer.history),
            mistake_heatmap_figure(trainer.env.grid, trainer.mistakes, range(0, 20),
                                   range(40, 60), start=trainer.env.start),
            maze_figure(trainer.env, title="maze layout"),
            frame_figure(replay_env, frames, len(frames) // 2)]
    for name in ("fixed_maze", "generalization"):
        if (RESULTS / name).exists():
            exp = load_experiment(RESULTS / name)
            figs += [experiment_curves_figure(exp["curves"], "success"),
                     experiment_bars_figure(exp["summary"], "success")]
    return figs


def test_no_legend_or_title_overlaps(trained):
    for fig in _all_figures(trained):
        canvas = FigureCanvasAgg(fig)   # pyplot-free figures need a canvas to measure on
        canvas.draw()
        renderer = canvas.get_renderer()
        legends = [lg.get_window_extent(renderer) for lg in fig.legends]
        assert not any(ax.get_legend() for ax in fig.axes), "legends must sit outside the plots"
        plots = [ax.get_window_extent(renderer) for ax in fig.axes if ax.get_label() != "<colorbar>"]
        titles = [ax.title.get_window_extent(renderer) for ax in fig.axes if ax.get_title(loc="left")]
        for lg in legends:
            assert not any(_boxes_overlap(lg, p) for p in plots), "legend covers a plot"
            assert not any(_boxes_overlap(lg, t) for t in titles), "legend covers a title"
        for i, t in enumerate(titles):
            for other in titles[i + 1:]:
                assert not _boxes_overlap(t, other), "two titles overlap"


# -- policy map and maze editor image ------------------------------------------------

def test_policy_map_arrows_are_the_greedy_actions(trained):
    env = make_env("maze_hazards")
    env.reset(seed=0)
    pm = policy_map(env, trained.agent)
    assert pm["seen"].any()
    for r, c in np.argwhere(pm["seen"] & ~pm["tie"]):
        assert trained.agent.act(env.observation_at((r, c)), greedy=True) == pm["action"][r, c]
    assert (pm["action"][~pm["seen"]] == -1).all()
    assert np.isnan(pm["value"][~pm["seen"]]).all()
    assert isinstance(policy_map_figure(env, trained.agent), Figure)


def test_policy_map_marks_unvisited_cells_and_handles_random_agents():
    env = make_env("maze")
    env.reset(seed=0)
    fresh = TabularQAgent(env.n_actions)
    pm = policy_map(env, fresh)
    assert not pm["seen"].any()
    assert isinstance(policy_map_figure(env, fresh), Figure)
    random_agent = make_agent("random", env.obs_size, env.n_actions)
    assert policy_map(env, random_agent) is None
    assert isinstance(policy_map_figure(env, random_agent), Figure)


def test_policy_map_covers_every_open_cell_for_dqn():
    env = make_env("maze")
    env.reset(seed=0)
    agent = make_agent("dqn", env.obs_size, env.n_actions, hidden=(16,))
    pm = policy_map(env, agent)
    assert pm["seen"].sum() == np.count_nonzero((env.grid != WALL) & (env.grid != GOAL))


def test_maze_image_clicks_map_to_cells():
    env = make_env("maze_hazards")
    env.reset(seed=0)
    img = maze_image(env.grid, env.start, cell_px=20)
    h, w = env.grid.shape
    assert img.shape == (h * 20, w * 20, 3) and img.dtype == np.uint8
    for r, c in [(0, 0), env.goal, (h - 1, w - 1)]:
        assert cell_at(c * 20 + 10, r * 20 + 10, env.grid.shape, 20) == (r, c)
    assert tuple(img[env.goal[0] * 20 + 10, env.goal[1] * 20 + 10]) == (0, 0x83, 0)
    assert cell_at(w * 20 + 1, 5, env.grid.shape, 20) is None
