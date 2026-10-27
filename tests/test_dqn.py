import pickle

import numpy as np
import pytest
import torch

from src.agents import (DQNAgent, DQNConfig, EpsilonSchedule, QNetwork, RandomAgent,
                        TabularQAgent, is_mistake, make_agent)
from src.envs import EnvConfig, GridWorld, make_env
from src.replay import MistakeReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer
from src.training import run_episode, train_agent

CORRIDOR = """
#######
#S...G#
#######
"""


def tiny_config(**overrides):
    base = dict(hidden=(32,), batch_size=8, buffer_size=200, learning_starts=8,
                train_freq=1, target_update=10)
    return DQNConfig(**{**base, **overrides})


def test_network_output_shape():
    net = QNetwork(10, 4, hidden=(16, 16))
    assert net(torch.zeros(5, 10)).shape == (5, 4)


def test_make_agent_builds_every_variant():
    assert isinstance(make_agent("random", 8, 4), RandomAgent)
    assert isinstance(make_agent("tabular_q", 8, 4, alpha=0.3), TabularQAgent)
    dqn = make_agent("dqn", 8, 4, hidden=(16,))
    per = make_agent("dqn_per", 8, 4, hidden=(16,))
    assert dqn.name == "dqn" and isinstance(dqn.buffer, ReplayBuffer)
    assert not isinstance(dqn.buffer, PrioritizedReplayBuffer)
    assert per.name == "dqn_per" and isinstance(per.buffer, PrioritizedReplayBuffer)
    mis = make_agent("dqn_mistake", 8, 4, hidden=(16,))
    both = make_agent("dqn_per_mistake", 8, 4, hidden=(16,))
    assert mis.name == "dqn_mistake" and isinstance(mis.buffer, MistakeReplayBuffer)
    assert not isinstance(mis.buffer.main, PrioritizedReplayBuffer)
    assert both.name == "dqn_per_mistake" and isinstance(both.buffer.main, PrioritizedReplayBuffer)
    assert not isinstance(dqn.buffer, MistakeReplayBuffer)  # off by default
    with pytest.raises(ValueError):
        make_agent("ppo", 8, 4)


def test_no_learning_before_learning_starts():
    agent = DQNAgent(3, 2, tiny_config(learning_starts=5), seed=0)
    obs = np.zeros(3, dtype=np.float32)
    results = [agent.update(obs, 0, 0.0, obs, False, False) for _ in range(5)]
    assert results[:4] == [None] * 4
    assert results[4] is not None and {"loss", "td_error"} <= set(results[4])
    assert agent.updates == 1 and agent.env_steps == 5


def test_target_network_syncs_on_schedule():
    agent = DQNAgent(3, 2, tiny_config(learning_starts=1, target_update=3, lr=0.1), seed=0)
    obs = np.ones(3, dtype=np.float32)

    def same():
        return all(torch.equal(a, b) for a, b in
                   zip(agent.online.parameters(), agent.target.parameters()))

    agent.update(obs, 0, 1.0, obs, True, False)
    agent.update(obs, 0, 1.0, obs, True, False)
    assert not same()  # online moved, target still frozen
    agent.update(obs, 0, 1.0, obs, True, False)
    assert same()      # third gradient step copies online -> target


@pytest.mark.parametrize("prioritized", [False, True])
def test_terminal_transition_regresses_to_reward(prioritized):
    agent = DQNAgent(2, 2, tiny_config(lr=0.01, prioritized=prioritized), seed=0)
    s = np.array([1.0, 0.0], dtype=np.float32)
    for _ in range(400):
        agent.update(s, 1, 0.7, s, True, False)
    assert agent.q_values(s)[1] == pytest.approx(0.7, abs=0.05)


def test_non_terminal_target_bootstraps():
    # s0 -(a0, r=0)-> s1 (terminal reward 1). Expected Q(s0, a0) = gamma * 1.
    agent = DQNAgent(2, 2, tiny_config(lr=0.01, gamma=0.9, target_update=5), seed=0)
    s0 = np.array([1.0, 0.0], dtype=np.float32)
    s1 = np.array([0.0, 1.0], dtype=np.float32)
    for _ in range(1500):
        agent.update(s1, 0, 1.0, s0, True, False)
        agent.update(s1, 1, 1.0, s0, True, False)
        agent.update(s0, 0, 0.0, s1, False, False)
    assert agent.q_values(s0)[0] == pytest.approx(0.9, abs=0.08)


def test_epsilon_and_beta_schedules():
    agent = DQNAgent(2, 2, tiny_config(epsilon=EpsilonSchedule(1.0, 0.0, 10),
                                       per_beta_start=0.4, per_beta_steps=100))
    agent.episodes = 5
    assert agent.epsilon == pytest.approx(0.5)
    agent.updates = 50
    assert agent.beta == pytest.approx(0.7)
    agent.updates = 1000
    assert agent.beta == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["dqn", "dqn_per"])
def test_checkpoint_round_trip(name):
    env = make_env("maze")
    agent = make_agent(name, env.obs_size, env.n_actions, seed=0, hidden=(32,), learning_starts=20)
    train_agent(env, agent, 3, seed=0)
    blob = pickle.dumps(agent.state_dict(include_buffer=True))

    restored = make_agent(name, env.obs_size, env.n_actions, seed=1, hidden=(32,))
    restored.load_state_dict(pickle.loads(blob))
    obs, _ = env.reset()
    np.testing.assert_allclose(restored.q_values(obs), agent.q_values(obs))
    assert (restored.episodes, restored.env_steps, restored.updates) == \
        (agent.episodes, agent.env_steps, agent.updates)
    assert len(restored.buffer) == len(agent.buffer)
    assert restored.config.learning_starts == 20


def test_checkpoint_rejects_mismatched_agent():
    state = make_agent("dqn", 8, 4, hidden=(16,)).state_dict()
    with pytest.raises(ValueError):
        make_agent("dqn", 9, 4, hidden=(16,)).load_state_dict(state)
    with pytest.raises(ValueError):
        make_agent("dqn", 8, 4, hidden=(32,)).load_state_dict(state)
    with pytest.raises(ValueError):
        make_agent("dqn_per", 8, 4, hidden=(16,)).load_state_dict(state)


@pytest.mark.parametrize("name", ["dqn", "dqn_per"])
def test_dqn_solves_corridor(name):
    env = GridWorld.from_ascii(CORRIDOR, EnvConfig(max_steps=30))
    agent = make_agent(name, env.obs_size, env.n_actions, seed=0, hidden=(64,),
                       learning_starts=64, epsilon=EpsilonSchedule(decay_episodes=40))
    train_agent(env, agent, 80, seed=0)
    result = run_episode(env, agent, train=False, greedy=True)
    assert result["success"] and result["steps"] == 4


@pytest.mark.slow
def test_dqn_learns_fixed_maze_and_reduces_mistakes():
    env = make_env("maze")
    agent = make_agent("dqn", env.obs_size, env.n_actions, seed=0,
                       epsilon=EpsilonSchedule(decay_episodes=300))
    history = train_agent(env, agent, 400, seed=0)
    early = np.mean([h["mistakes"] for h in history[:50]])
    late = np.mean([h["mistakes"] for h in history[-50:]])
    assert late < 0.2 * early, f"mistakes early={early:.1f} late={late:.1f}"
    result = run_episode(env, agent, train=False, greedy=True)
    assert result["success"] and result["steps"] == result["optimal_length"]


# ---------------------------------------------------------------------------
# Mistake-aware replay
# ---------------------------------------------------------------------------

HAZARD_ROOM = """
#####
#SH.#
#..G#
#####
"""


def test_is_mistake_reads_events_not_rewards():
    assert is_mistake({"events": ["collision"]})
    assert is_mistake({"events": ["hazard"]})
    assert is_mistake({"events": ["collision", "timeout"]})
    for events in (["revisit"], ["timeout"], ["goal"], [], ["revisit", "timeout"]):
        assert not is_mistake({"events": events})
    assert not is_mistake(None) and not is_mistake({})
    assert is_mistake({"events": ["revisit"]}, events=("revisit",))  # configurable


def test_classification_of_real_environment_steps():
    env = GridWorld.from_ascii(HAZARD_ROOM, EnvConfig(hazard_terminal=False, max_steps=4))
    up, right, down, left = range(4)
    env.reset(seed=0)
    steps = [env.step(a) for a in (up, right, left, right)]  # wall, hazard, back, hazard again
    flags = [is_mistake(info) for *_, info in steps]
    assert flags == [True, True, False, True]
    env.reset(seed=0)
    steps = [env.step(a) for a in (down, up, down, up)]  # move, revisit, revisit, revisit+timeout
    rewards = [r for _, r, *_ in steps]
    assert [is_mistake(info) for *_, info in steps] == [False] * 4
    # revisit + timeout costs more than a wall bump, yet it is not a mistake:
    # classification never looks at the reward
    assert rewards[3] < env.config.rewards.step + env.config.rewards.collision


def test_agent_reserves_only_collisions_and_hazards():
    env = GridWorld.from_ascii(HAZARD_ROOM, EnvConfig(hazard_terminal=False, max_steps=30))
    agent = make_agent("dqn_mistake", env.obs_size, env.n_actions, seed=0, hidden=(16,),
                       learning_starts=10_000)
    history = train_agent(env, agent, 20, seed=0)
    expected = sum(h["collisions"] + h["hazards"] for h in history)
    assert expected > 0
    assert agent.buffer.mistakes_seen == len(agent.buffer.reserve) == expected
    assert len(agent.buffer) == sum(h["steps"] for h in history)  # main keeps everything
    # every reserved transition is a wall bump (obs unchanged) or a hazard step
    r, n = agent.buffer.reserve, len(agent.buffer.reserve)
    bumped = (r.obs[:n] == r.next_obs[:n]).all(axis=1)
    rw = env.config.rewards
    hazard_rewards = np.array([rw.step + rw.hazard, rw.step + rw.hazard + rw.timeout])
    hazard = np.isclose(r.rewards[:n, None], hazard_rewards, atol=1e-6).any(axis=1)
    assert (bumped | hazard).all() and hazard.any() and bumped.any()


def test_update_without_info_never_reserves():
    agent = DQNAgent(3, 2, tiny_config(mistake_replay=True), seed=0)
    obs = np.zeros(3, dtype=np.float32)
    for _ in range(20):
        agent.update(obs, 0, -5.0, obs, False, False)  # positional call, as before
    assert len(agent.buffer.reserve) == 0 and len(agent.buffer) == 20


@pytest.mark.parametrize("name", ["dqn_mistake", "dqn_per_mistake"])
def test_mistake_checkpoint_round_trip(name):
    env = make_env("maze")
    agent = make_agent(name, env.obs_size, env.n_actions, seed=0, hidden=(32,),
                       learning_starts=20, mistake_capacity=50)
    train_agent(env, agent, 3, seed=0)
    blob = pickle.dumps(agent.state_dict(include_buffer=True))
    restored = make_agent(name, env.obs_size, env.n_actions, seed=1, hidden=(32,),
                          mistake_capacity=50)
    restored.load_state_dict(pickle.loads(blob))
    assert len(restored.buffer.reserve) == len(agent.buffer.reserve) > 0
    assert restored.buffer.mistakes_seen == agent.buffer.mistakes_seen
    assert restored.config.mistake_events == ("collision", "hazard")
    obs, _ = env.reset()
    np.testing.assert_allclose(restored.q_values(obs), agent.q_values(obs))


def test_checkpoint_rejects_mismatched_mistake_replay():
    plain = make_agent("dqn", 8, 4, hidden=(16,)).state_dict()
    with pytest.raises(ValueError):
        make_agent("dqn_mistake", 8, 4, hidden=(16,)).load_state_dict(plain)
    with pytest.raises(ValueError):
        make_agent("dqn", 8, 4, hidden=(16,)).load_state_dict(
            make_agent("dqn_mistake", 8, 4, hidden=(16,)).state_dict())


def test_old_checkpoints_without_mistake_fields_still_load():
    state = make_agent("dqn", 8, 4, hidden=(16,)).state_dict()
    for key in ("mistake_replay", "mistake_fraction", "mistake_capacity", "mistake_events"):
        del state["config"][key]
    agent = make_agent("dqn", 8, 4, hidden=(16,))
    agent.load_state_dict(state)
    assert not agent.config.mistake_replay


@pytest.mark.parametrize("name", ["dqn_mistake", "dqn_per_mistake"])
def test_mistake_dqn_solves_corridor(name):
    env = GridWorld.from_ascii(CORRIDOR, EnvConfig(max_steps=30))
    agent = make_agent(name, env.obs_size, env.n_actions, seed=0, hidden=(64,),
                       learning_starts=64, epsilon=EpsilonSchedule(decay_episodes=40))
    train_agent(env, agent, 80, seed=0)
    assert len(agent.buffer.reserve) > 0
    result = run_episode(env, agent, train=False, greedy=True)
    assert result["success"] and result["steps"] == 4


def test_long_memory_reserve_outlives_a_small_main_buffer():
    env = make_env("maze_hazards", obs_mode="local", layout_pool=(0, 1, 2))
    agent = make_agent("dqn_mistake", env.obs_size, env.n_actions, seed=0, hidden=(16,),
                       buffer_size=300, mistake_capacity=300, learning_starts=100)
    history = train_agent(env, agent, 25, seed=0)
    d = agent.buffer.diagnostics()
    assert d["transitions_added"] == agent.env_steps == sum(h["steps"] for h in history)
    assert d["main_size"] == 300 and d["main_overwritten"] > 0   # old experience evicted
    assert d["reserve_size"] <= 300
    assert d["mistakes_seen"] == sum(h["collisions"] + h["hazards"] for h in history)
    assert d["reserve_only"] > 0          # some mistakes survive only in the reserve
    assert d["reserve_span"] > d["main_size"]


# -- dueling heads and n-step returns ------------------------------------------------

def test_dueling_network_combines_value_and_advantage():
    net = QNetwork(6, 4, hidden=(16,), dueling=True)
    x = torch.randn(5, 6)
    h = net.net(x)
    v, a = net.value(h), net.advantage(h)
    torch.testing.assert_close(net(x), v + a - a.mean(dim=1, keepdim=True))
    # the advantages are centred, so the mean Q over actions is the state value
    torch.testing.assert_close(net(x).mean(dim=1, keepdim=True), v)


def test_defaults_build_the_original_network_and_training():
    """dueling=False, n_step=1 must be the old agent: same weights, same trajectory."""
    env = make_env("maze")
    a = make_agent("dqn", env.obs_size, env.n_actions, seed=3, hidden=(32,), learning_starts=20)
    b = make_agent("dqn", env.obs_size, env.n_actions, seed=3, hidden=(32,), learning_starts=20,
                   dueling=False, n_step=1)
    torch.manual_seed(3)
    old_style = QNetwork(env.obs_size, env.n_actions, (32,))
    for p, q in zip(a.online.parameters(), old_style.parameters()):
        torch.testing.assert_close(p, q)
    ha = train_agent(make_env("maze"), a, 4, seed=0)
    hb = train_agent(make_env("maze"), b, 4, seed=0)
    assert [h["total_reward"] for h in ha] == [h["total_reward"] for h in hb]
    assert np.isnan(a.buffer.discounts[:len(a.buffer)]).all()  # n = 1 never stores one


def test_make_agent_builds_dueling_and_n_step_variants():
    duel = make_agent("dqn_dueling", 8, 4, hidden=(16,))
    nstep = make_agent("dqn_nstep", 8, 4, hidden=(16,))
    both = make_agent("dqn_dueling_nstep", 8, 4, hidden=(16,), n_step=5)
    assert duel.name == "dqn_dueling" and duel.config.dueling and duel.config.n_step == 1
    assert nstep.name == "dqn_nstep" and nstep.config.n_step == 3 and not nstep.config.dueling
    assert both.name == "dqn_dueling_nstep" and both.config.n_step == 5
    assert nstep.nickname == "Neural Ned (Looks Ahead)"
    with pytest.raises(ValueError, match="mistake replay"):
        DQNAgent(8, 4, tiny_config(n_step=3, mistake_replay=True))
    with pytest.raises(ValueError):
        DQNAgent(8, 4, tiny_config(n_step=0))


def _stored(agent):
    n = len(agent.buffer)
    b = agent.buffer
    return [(int(b.obs[i, 0]), float(b.rewards[i]), int(b.next_obs[i, 0]),
             bool(b.terminated[i]), float(b.discounts[i])) for i in range(n)]


def _feed(agent, rewards, end):
    """Episode with states 0, 1, 2, ... (obs[0] = state); end is 'terminated' or 'truncated'."""
    for t, r in enumerate(rewards):
        last = t == len(rewards) - 1
        obs = np.array([t, 0], dtype=np.float32)
        nxt = np.array([t + 1, 0], dtype=np.float32)
        agent.update(obs, 0, r, nxt, last and end == "terminated", last and end == "truncated")


@pytest.mark.parametrize("end", ["terminated", "truncated"])
def test_n_step_transitions_by_hand(end):
    g = 0.5
    agent = DQNAgent(2, 2, tiny_config(gamma=g, n_step=3, learning_starts=10_000), seed=0)
    _feed(agent, [1.0, 2.0, 4.0, 8.0, 16.0], end)
    term = end == "terminated"
    expected = [
        (0, 1 + g * 2 + g * g * 4, 3, False, g ** 3),     # full 3-step windows
        (1, 2 + g * 4 + g * g * 8, 4, False, g ** 3),
        (2, 4 + g * 8 + g * g * 16, 5, term, g ** 3),     # the episode's end flushes the rest
        (3, 8 + g * 16, 5, term, g ** 2),
        (4, 16.0, 5, term, g),
    ]
    got = _stored(agent)
    assert [(s, n, t) for s, _, n, t, _ in got] == [(s, n, t) for s, _, n, t, _ in expected]
    np.testing.assert_allclose([r for _, r, *_ in got], [r for _, r, *_ in expected])
    np.testing.assert_allclose([d for *_, d in got], [d for *_, d in expected])
    assert len(agent._recent) == 0  # nothing carries over into the next episode


def test_short_episode_and_n_step_target():
    g = 0.9
    agent = DQNAgent(2, 2, tiny_config(gamma=g, n_step=3, learning_starts=10_000), seed=0)
    _feed(agent, [1.0, 1.0], "truncated")   # shorter than n
    got = _stored(agent)
    assert [(s, n) for s, _, n, _, _ in got] == [(0, 2), (1, 2)]
    np.testing.assert_allclose([d for *_, d in got], [g ** 2, g])
    # the learning target bootstraps with the stored discount, not gamma
    agent.config.learning_starts = 1
    agent.config.lr = 0.0
    batch = agent.buffer._batch(np.array([0]), np.ones(1))
    agent.buffer.sample = lambda *a, **k: batch
    with torch.no_grad():
        next_q = agent.target(torch.as_tensor(batch["next_obs"]))
        best = agent.online(torch.as_tensor(batch["next_obs"])).argmax(dim=1)
        y = float(batch["rewards"][0]) + g ** 2 * float(next_q[0, best[0]])
        q = float(agent.online(torch.as_tensor(batch["obs"]))[0, 0])
    stats = agent._learn()
    assert stats["td_error"] == pytest.approx(abs(y - q), rel=1e-5)


@pytest.mark.parametrize("name", ["dqn_dueling", "dqn_nstep", "dqn_dueling_nstep"])
def test_new_variants_solve_corridor(name):
    # A smoke test of learning, not a benchmark: single seeds are noisy (the
    # dueling agent misses this corridor on seed 0 after 80 episodes), so
    # require most of three seeds to find the 4-step path.
    solved = 0
    for seed in range(3):
        env = GridWorld.from_ascii(CORRIDOR, EnvConfig(max_steps=30))
        agent = make_agent(name, env.obs_size, env.n_actions, seed=seed, hidden=(64,),
                           learning_starts=64, epsilon=EpsilonSchedule(decay_episodes=40))
        train_agent(env, agent, 100, seed=seed)
        result = run_episode(env, agent, train=False, greedy=True)
        solved += result["success"] and result["steps"] == 4
    assert solved >= 2


def test_prioritized_replay_with_n_step():
    env = make_env("maze")
    agent = DQNAgent(env.obs_size, env.n_actions,
                     tiny_config(prioritized=True, n_step=3, learning_starts=20), seed=0)
    train_agent(env, agent, 3, seed=0)
    assert agent.name == "dqn_per_nstep" and agent.updates > 0
    assert not np.isnan(agent.buffer.discounts[:len(agent.buffer)]).any()


@pytest.mark.parametrize("name", ["dqn_dueling", "dqn_nstep"])
def test_dueling_and_n_step_checkpoint_round_trip(name):
    env = make_env("maze")
    agent = make_agent(name, env.obs_size, env.n_actions, seed=0, hidden=(32,), learning_starts=20)
    train_agent(env, agent, 3, seed=0)
    blob = pickle.dumps(agent.state_dict(include_buffer=True))
    restored = make_agent(name, env.obs_size, env.n_actions, seed=1, hidden=(32,))
    restored.load_state_dict(pickle.loads(blob))
    obs, _ = env.reset()
    np.testing.assert_allclose(restored.q_values(obs), agent.q_values(obs))
    n = len(agent.buffer)
    np.testing.assert_array_equal(restored.buffer.discounts[:n], agent.buffer.discounts[:n])
    with pytest.raises(ValueError):
        make_agent("dqn", env.obs_size, env.n_actions, hidden=(32,)).load_state_dict(
            pickle.loads(blob))


def test_old_checkpoints_without_dueling_or_discounts_still_load():
    env = make_env("maze")
    agent = make_agent("dqn", env.obs_size, env.n_actions, seed=0, hidden=(16,),
                       learning_starts=20)
    train_agent(env, agent, 2, seed=0)
    state = agent.state_dict(include_buffer=True)
    for key in ("dueling", "n_step"):
        del state["config"][key]
    del state["buffer"]["discounts"]
    restored = make_agent("dqn", env.obs_size, env.n_actions, hidden=(16,))
    restored.load_state_dict(state)
    assert not restored.config.dueling and restored.config.n_step == 1
    assert len(restored.buffer) == len(agent.buffer)
