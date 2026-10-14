from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.agents import EpsilonSchedule, TabularQAgent
from src.envs import make_env
from src.evaluation import load_experiment
from src.training import TrainConfig, Trainer, train_agent
from src.visualization import (_clip_interval, experiment_bars_figure, experiment_curves_figure,
                               frame_figure, learning_curves_figure, maze_figure,
                               mistake_heatmap_figure, mistake_types_figure, record_episode,
                               reward_breakdown_figure, rolling_mean)

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
    labels = {t.get_text() for t in curves.axes[0].get_legend().get_texts()}
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
