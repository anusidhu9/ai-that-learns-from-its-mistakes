import pickle

import numpy as np
import pytest

from src.agents import EpsilonSchedule, RandomAgent, TabularQAgent, greedy_action
from src.envs import EnvConfig, GridWorld, make_env
from src.training import run_episode, train_agent

CORRIDOR = """
#####
#S.G#
#####
"""


def obs_of(x):
    return np.array([x], dtype=np.float32)


# -- exploration schedule ----------------------------------------------------------

def test_epsilon_schedule_decays_linearly_then_stays():
    s = EpsilonSchedule(start=1.0, end=0.1, decay_episodes=10)
    assert s.value(0) == pytest.approx(1.0)
    assert s.value(5) == pytest.approx(0.55)
    assert s.value(10) == pytest.approx(0.1)
    assert s.value(1000) == pytest.approx(0.1)


def test_greedy_action_breaks_ties_randomly():
    rng = np.random.default_rng(0)
    picks = {greedy_action(np.zeros(4), rng) for _ in range(100)}
    assert picks == {0, 1, 2, 3}
    assert greedy_action(np.array([0.0, 2.0, 1.0, 2.0]), rng) in (1, 3)


# -- Q-learning update rule ----------------------------------------------------------

def test_q_update_non_terminal_matches_formula():
    agent = TabularQAgent(2, alpha=0.5, gamma=0.9)
    agent.q[agent.key(obs_of(1))] = np.array([0.0, 4.0])
    agent.update(obs_of(0), 1, reward=1.0, next_obs=obs_of(1), terminated=False, truncated=False)
    # target = 1 + 0.9 * 4 = 4.6 ; Q = 0 + 0.5 * (4.6 - 0) = 2.3
    assert agent.q_values(obs_of(0))[1] == pytest.approx(2.3)


def test_q_update_terminal_ignores_next_state():
    agent = TabularQAgent(2, alpha=0.5, gamma=0.9)
    agent.q[agent.key(obs_of(1))] = np.array([0.0, 4.0])
    stats = agent.update(obs_of(0), 0, reward=1.0, next_obs=obs_of(1), terminated=True, truncated=False)
    assert agent.q_values(obs_of(0))[0] == pytest.approx(0.5)
    assert stats["td_error"] == pytest.approx(1.0)


def test_q_update_truncated_still_bootstraps():
    agent = TabularQAgent(2, alpha=1.0, gamma=0.9)
    agent.q[agent.key(obs_of(1))] = np.array([0.0, 4.0])
    agent.update(obs_of(0), 0, reward=0.0, next_obs=obs_of(1), terminated=False, truncated=True)
    assert agent.q_values(obs_of(0))[0] == pytest.approx(3.6)


def test_greedy_act_uses_q_values_and_epsilon_explores():
    agent = TabularQAgent(4, epsilon=EpsilonSchedule(1.0, 1.0, 1), seed=0)
    agent.q[agent.key(obs_of(0))] = np.array([0.0, 0.0, 5.0, 0.0])
    assert all(agent.act(obs_of(0), greedy=True) == 2 for _ in range(20))
    explored = {agent.act(obs_of(0)) for _ in range(200)}
    assert explored == {0, 1, 2, 3}


def test_invalid_hyperparameters_rejected():
    with pytest.raises(ValueError):
        TabularQAgent(4, alpha=0.0)
    with pytest.raises(ValueError):
        TabularQAgent(4, gamma=1.5)


# -- checkpointing ------------------------------------------------------------------

def test_state_dict_round_trip_through_pickle():
    env = make_env("maze")
    agent = TabularQAgent(env.n_actions, seed=0)
    train_agent(env, agent, 20, seed=0)
    blob = pickle.dumps(agent.state_dict())
    restored = TabularQAgent(env.n_actions)
    restored.load_state_dict(pickle.loads(blob))
    assert restored.episodes == 20 and restored.updates == agent.updates
    assert restored.epsilon == pytest.approx(agent.epsilon)
    obs, _ = env.reset()
    np.testing.assert_array_equal(restored.q_values(obs), agent.q_values(obs))


def test_load_rejects_wrong_action_count():
    state = TabularQAgent(4).state_dict()
    with pytest.raises(ValueError):
        TabularQAgent(3).load_state_dict(state)


# -- episode loop --------------------------------------------------------------------

def test_run_episode_eval_does_not_change_agent():
    env = make_env("maze")
    agent = TabularQAgent(env.n_actions, seed=0)
    summary = run_episode(env, agent, train=False, greedy=True)
    assert agent.episodes == 0 and agent.updates == 0 and agent.n_states == 0
    assert summary["epsilon"] == 0.0
    assert {"success", "steps", "mistakes", "mistake_log"} <= set(summary)


def test_random_agent_never_learns():
    env = make_env("maze")
    agent = RandomAgent(env.n_actions, seed=0)
    history = train_agent(env, agent, 5, seed=0)
    assert len(history) == 5 and agent.episodes == 5


# -- learning (the claims we make in the report) ------------------------------------

def test_q_learning_solves_corridor_optimally():
    env = GridWorld.from_ascii(CORRIDOR, EnvConfig(max_steps=20))
    agent = TabularQAgent(env.n_actions, epsilon=EpsilonSchedule(decay_episodes=30), seed=0)
    train_agent(env, agent, 50, seed=0)
    result = run_episode(env, agent, train=False, greedy=True)
    assert result["success"] and result["steps"] == 2 and result["mistakes"] == 0


@pytest.mark.parametrize("preset", ["maze", "maze_hazards", "open", "open_hazards"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q_learning_beats_random_and_reduces_mistakes(preset, seed):
    env = make_env(preset)
    agent = TabularQAgent(env.n_actions, epsilon=EpsilonSchedule(decay_episodes=300), seed=seed)
    history = train_agent(env, agent, 400, seed=seed)

    early = np.mean([h["mistakes"] for h in history[:50]])
    late = np.mean([h["mistakes"] for h in history[-50:]])
    assert late < 0.2 * early, f"mistakes early={early:.1f} late={late:.1f}"

    greedy = run_episode(env, agent, train=False, greedy=True)
    assert greedy["success"]
    assert greedy["steps"] == greedy["optimal_length"]

    rnd = RandomAgent(env.n_actions, seed=seed)
    random_success = np.mean([run_episode(env, rnd, train=False)["success"] for _ in range(50)])
    assert random_success < 1.0
