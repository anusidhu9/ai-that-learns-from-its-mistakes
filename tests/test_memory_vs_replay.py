"""Memory vs sequence replay: the memory-reset GRU control and the protocol."""
import pickle

import numpy as np
import pytest
import torch

from src import fun
from src.agents import DQNAgent, DQNConfig, RecurrentQNetwork, make_agent
from src.envs import make_env
from src.training import build_env, train_agent


def test_reset_network_answers_each_step_from_its_observation_alone():
    torch.manual_seed(0)
    with_memory = RecurrentQNetwork(6, 4, hidden=(8,), memory=5)
    torch.manual_seed(0)
    no_memory = RecurrentQNetwork(6, 4, hidden=(8,), memory=5, reset_every_step=True)
    for a, b in zip(with_memory.parameters(), no_memory.parameters()):
        assert torch.equal(a, b)                      # same layers, same initial weights
    x = torch.randn(3, 7, 6)
    q, _ = no_memory(x)
    alone, _ = no_memory(x.reshape(21, 1, 6))
    torch.testing.assert_close(q, alone.reshape(3, 7, 4))       # step t sees only o_t
    q_mem, _ = with_memory(x)
    torch.testing.assert_close(q_mem[:, 0], q[:, 0])            # first step: both from h = 0
    assert not torch.allclose(q_mem[:, 1:], q[:, 1:])          # later steps differ
    # no gradient flows from a later step's Q to an earlier observation
    x = torch.randn(1, 5, 6, requires_grad=True)
    no_memory(x)[0][0, -1].sum().backward()
    assert x.grad[0, :-1].abs().sum() == 0
    # step() ignores the memory it is given
    h = torch.randn(3, 5)
    torch.testing.assert_close(no_memory.step(x[0, :3].detach(), h),
                               no_memory.step(x[0, :3].detach(), torch.zeros(3, 5)))


def test_nomem_agent_is_the_recurrent_agent_without_memory():
    env = make_env("maze", obs_mode="local")
    rnn = make_agent("dqn_rnn", env.obs_size, env.n_actions, seed=3)
    nomem = make_agent("dqn_rnn_nomem", env.obs_size, env.n_actions, seed=3)
    assert nomem.name == "dqn_rnn_nomem" and nomem.recurrent
    assert nomem.config.reset_memory_every_step and not rnn.config.reset_memory_every_step
    assert DQNConfig(**{**vars(rnn.config), "reset_memory_every_step": True}) == nomem.config
    assert sum(p.numel() for p in nomem.online.parameters()) == 109_572
    for a, b in zip(rnn.online.parameters(), nomem.online.parameters()):
        assert torch.equal(a, b)                      # same seed, same initial network
    assert nomem.nickname == fun.NICKNAMES["dqn_rnn_nomem"] and fun.TAGLINES["dqn_rnn_nomem"]
    obs, _ = env.reset(seed=0)
    nomem.begin_episode()
    first = nomem.q_values(obs)
    for _ in range(3):
        obs2, *_ = env.step(nomem.act(obs, greedy=True))
    np.testing.assert_allclose(nomem.q_values(obs), first)     # acting leaves no trace


def test_reset_needs_a_recurrent_agent():
    with pytest.raises(ValueError, match="recurrent"):
        DQNAgent(75, 4, DQNConfig(reset_memory_every_step=True))


def test_nomem_trains_on_the_same_sequences_and_checkpoints():
    env = make_env("maze", obs_mode="local")
    kw = dict(seed=0, learning_starts=20, batch_size=4)
    nomem = make_agent("dqn_rnn_nomem", env.obs_size, env.n_actions, **kw)
    train_agent(env, nomem, 2, seed=0)
    assert nomem.updates > 0
    blob = pickle.loads(pickle.dumps(nomem.state_dict(include_buffer=True)))
    restored = make_agent("dqn_rnn_nomem", env.obs_size, env.n_actions, seed=1)
    restored.load_state_dict(blob)
    obs, _ = env.reset(seed=5)
    np.testing.assert_allclose(restored.q_values(obs), nomem.q_values(obs))
    with pytest.raises(ValueError):    # a memory network can't load the control's weights
        make_agent("dqn_rnn", env.obs_size, env.n_actions).load_state_dict(blob)


def test_old_recurrent_checkpoints_still_load():
    env = make_env("maze", obs_mode="local")
    agent = make_agent("dqn_rnn", env.obs_size, env.n_actions, seed=0)
    state = agent.state_dict()
    del state["config"]["reset_memory_every_step"]     # saved before the option existed
    restored = make_agent("dqn_rnn", env.obs_size, env.n_actions, seed=1)
    restored.load_state_dict(state)
    assert not restored.config.reset_memory_every_step


def test_memory_vs_replay_protocol_settings_and_fresh_layouts():
    from src.evaluation import EXPERIMENTS

    exp = EXPERIMENTS["memory_vs_replay"]
    assert exp.agents == ("dqn", "dqn_batch512", "dqn_rnn", "dqn_rnn_nomem")
    assert exp.seeds == tuple(range(800, 810))
    assert (exp.episodes, exp.decay_episodes, exp.eval_every) == (3000, 2000, 500)
    assert exp.unseen_layouts == tuple(range(10001, 10051))
    configs = {a: exp.train_config(a, 800) for a in exp.agents}
    assert configs["dqn_batch512"].agent == "dqn"
    assert configs["dqn_batch512"].agent_kwargs == {"epsilon": {"decay_episodes": 2000},
                                                    "batch_size": 512}
    for a in ("dqn", "dqn_rnn", "dqn_rnn_nomem"):
        assert configs[a].agent == a
        assert configs[a].agent_kwargs == {"epsilon": {"decay_episodes": 2000}}
    assert len({(c.env_preset, str(c.env_overrides), c.episodes) for c in configs.values()}) == 1
    # trained steps per update: 32 runs x 16 steps for the GRUs, 512 for the wide batch
    env = build_env(configs["dqn"])
    rnn = make_agent("dqn_rnn", env.obs_size, env.n_actions)
    assert rnn.config.batch_size * rnn.config.seq_length == 512

    keys = {}
    for s in exp.train_layouts():
        env.reset(layout_seed=s)
        keys[(env.grid.tobytes(), env.start, env.goal)] = s
    assert len(keys) == 50
    held = set()
    for s in exp.unseen_layouts:
        env.reset(layout_seed=s)
        k = (env.grid.tobytes(), env.start, env.goal)
        assert k not in keys, f"held-out {s} repeats training maze {keys.get(k)}"
        assert k not in held, f"held-out {s} repeats another held-out maze"
        held.add(k)
