"""PPO agent (Stage I): GAE, the clipped update, checkpoints and the protocol."""
import json

import numpy as np
import pytest
import torch

from src.agents import PPOAgent, PPOConfig, make_agent
from src.envs import EnvConfig, GridWorld
from src.training import (CHECKPOINT_FILE, Trainer, TrainConfig, export_weights, load_agent,
                          run_episode, train_agent)

CORRIDOR = """
#######
#S...G#
#######
"""
PPO = dict(agent="ppo", episodes=30, seed=2, checkpoint_every=4,
           env_overrides={"max_steps": 40},
           agent_kwargs={"hidden": [16], "n_steps": 64, "minibatch_size": 16})


def small_agent(**kwargs):
    cfg = {"hidden": (8,), "n_steps": 8, "minibatch_size": 4, **kwargs}
    return PPOAgent(3, 4, PPOConfig(**cfg), seed=0)


def fill(agent, steps):
    """steps: (reward, terminated, truncated) per step, observations random."""
    rng = np.random.default_rng(1)
    for reward, term, trunc in steps:
        i = agent.rollout_size
        r = agent.rollout
        r["obs"][i], r["next_obs"][i] = rng.normal(size=3), rng.normal(size=3)
        r["actions"][i], r["rewards"][i] = 0, reward
        r["terminated"][i], r["truncated"][i] = term, trunc
        agent.rollout_size += 1


def expected_gae(agent):
    cfg, r, n = agent.config, agent.rollout, agent.rollout_size
    with torch.no_grad():
        v = agent.critic(torch.as_tensor(r["obs"][:n])).squeeze(1).numpy().astype(np.float64)
        nv = agent.critic(torch.as_tensor(r["next_obs"][:n])).squeeze(1).numpy().astype(np.float64)
    adv = np.zeros(n)
    for t in range(n):  # forward sum: A_t = sum_k (gamma lambda)^k delta_{t+k}, stop after done
        total, factor = 0.0, 1.0
        for k in range(t, n):
            delta = r["rewards"][k] + cfg.gamma * nv[k] * (1 - r["terminated"][k]) - v[k]
            total += factor * delta
            if r["terminated"][k] or r["truncated"][k]:
                break
            factor *= cfg.gamma * cfg.gae_lambda
        adv[t] = total
    return adv, adv + v


def test_gae_matches_the_definition_across_episode_ends():
    agent = small_agent()
    # an episode ending at the goal (step 1), one cut by the time limit (step 4),
    # and an unfinished one at the end of the rollout
    fill(agent, [(-0.01, 0, 0), (1.0, 1, 0), (-0.01, 0, 0), (-0.01, 0, 0), (-0.1, 0, 1),
                 (-0.01, 0, 0), (-0.05, 0, 0)])
    adv, returns = agent.advantages()
    want_adv, want_returns = expected_gae(agent)
    np.testing.assert_allclose(adv, want_adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(returns, want_returns, rtol=1e-5, atol=1e-6)


def test_timeout_bootstraps_but_terminal_does_not():
    for term, trunc, should_change in ((1, 0, False), (0, 1, True)):
        agent = small_agent()
        fill(agent, [(-0.01, 0, 0), (-0.1, term, trunc)])
        before = agent.advantages()[0][1]
        agent.rollout["next_obs"][1] += 5.0  # a different state after the last step
        after = agent.advantages()[0][1]
        assert (abs(after - before) > 1e-6) == should_change


def test_advantages_do_not_leak_across_episodes():
    a, b = small_agent(), small_agent()
    fill(a, [(-0.01, 0, 0), (1.0, 1, 0), (-0.01, 0, 0)])
    fill(b, [(-0.01, 0, 0), (1.0, 1, 0), (-0.01, 0, 0)])
    b.rollout["rewards"][2] = 50.0  # only the next episode differs
    np.testing.assert_allclose(a.advantages()[0][:2], b.advantages()[0][:2])


def test_initial_policy_is_near_uniform_and_greedy_takes_the_mode():
    agent = PPOAgent(10, 4, seed=0)
    obs = np.random.default_rng(0).normal(size=10)
    probs = agent.policy_probs(obs)
    assert probs.shape == (4,) and abs(probs.sum() - 1) < 1e-9 and probs.min() > 0.2
    assert agent.act(obs, greedy=True) == int(np.argmax(probs))
    assert agent.epsilon == 0.0
    counts = np.bincount([agent.act(obs) for _ in range(400)], minlength=4)
    assert counts.min() > 50  # sampling explores every move


def test_network_sizes():
    agent = make_agent("ppo", 77, 4)
    n_actor = sum(p.numel() for p in agent.actor.parameters())
    n_critic = sum(p.numel() for p in agent.critic.parameters())
    assert (n_actor, n_critic) == (27_012, 26_625)  # the actor matches DQN's 128x128 MLP


def test_update_learns_once_per_rollout_and_clears_it():
    agent = small_agent()
    rng = np.random.default_rng(0)
    stats = []
    for t in range(20):
        stats.append(agent.update(rng.normal(size=3), int(rng.integers(4)), -0.01,
                                  rng.normal(size=3), False, t == 9))
    learned = [s for s in stats if s is not None]
    assert len(learned) == 2 and agent.updates == 2 and agent.rollout_size == 4
    assert all(np.isfinite(s["loss"]) and s["td_error"] >= 0 for s in learned)


def test_clipped_update_moves_the_policy_toward_good_actions():
    agent = small_agent(n_steps=64, minibatch_size=16, entropy_coef=0.0, lr=1e-2)
    obs = np.ones(3)
    before = agent.policy_probs(obs)[2]
    for _ in range(4):  # action 2 always pays, the others always cost
        for t in range(64):
            a = t % 4
            agent.update(obs, a, 1.0 if a == 2 else -1.0, obs, True, False)
    assert agent.policy_probs(obs)[2] > before + 0.2


def test_invalid_settings_and_ignored_epsilon():
    with pytest.raises(ValueError, match="multiple"):
        PPOAgent(3, 4, PPOConfig(n_steps=100, minibatch_size=64))
    agent = make_agent("ppo", 3, 4, epsilon={"decay_episodes": 5})  # the dashboard passes one
    assert isinstance(agent, PPOAgent)


def test_ppo_solves_corridor():
    solved = 0
    for seed in range(3):
        env = GridWorld.from_ascii(CORRIDOR, EnvConfig(max_steps=30, obs_mode="local"))
        agent = make_agent("ppo", env.obs_size, env.n_actions, seed=seed, hidden=(32,),
                           n_steps=128, minibatch_size=32, lr=3e-3)
        train_agent(env, agent, 150, seed=seed)
        result = run_episode(env, agent, train=False, greedy=True)
        solved += result["success"] and result["steps"] == 4
    assert solved >= 2


def test_training_is_deterministic_and_resumes_exactly_mid_rollout(tmp_path):
    full = Trainer(TrainConfig(**PPO))
    full.train()
    first = Trainer(TrainConfig(**PPO), tmp_path / "run")
    first.train(episodes=11)
    assert 0 < first.agent.rollout_size < 64  # stopped in the middle of a rollout
    resumed = Trainer.resume(tmp_path / "run")
    assert resumed.agent.rollout_size == first.agent.rollout_size
    resumed.train()
    assert [r["total_reward"] for r in resumed.history] == \
        [r["total_reward"] for r in full.history]
    again = Trainer(TrainConfig(**PPO))
    again.train()
    assert [r["total_reward"] for r in again.history] == [r["total_reward"] for r in full.history]


def test_checkpoint_without_rollout_and_mismatches(tmp_path):
    trainer = Trainer(TrainConfig(**{**PPO, "save_buffer": False}), tmp_path / "run")
    trainer.train(episodes=5)
    resumed = Trainer.resume(tmp_path / "run")
    assert resumed.agent.rollout_size == 0 and resumed.episodes_done == 5
    other = PPOAgent(resumed.agent.obs_size, 4, PPOConfig(hidden=(8,), n_steps=64,
                                                          minibatch_size=16))
    with pytest.raises(ValueError, match="differs"):
        other.load_state_dict(trainer.agent.state_dict())


def test_exported_ppo_weights_load_for_evaluation_only(tmp_path):
    trainer = Trainer(TrainConfig(**PPO), tmp_path / "run")
    trainer.train(episodes=6)
    out = export_weights(tmp_path / "run" / CHECKPOINT_FILE, tmp_path / "w" / "model.pt")
    agent, env, config = load_agent(out)
    assert config == trainer.config and agent.episodes == 6 and agent.rollout_size == 0
    obs, _ = env.reset()
    np.testing.assert_allclose(agent.policy_probs(obs), trainer.agent.policy_probs(obs))
    assert agent.state_value(obs) == pytest.approx(trainer.agent.state_value(obs))
    with pytest.raises(ValueError, match="weights-only"):
        Trainer(TrainConfig(**PPO)).load_checkpoint(out)


def test_policy_map_and_replay_show_probabilities():
    from src.visualization import frame_figure, policy_map, policy_map_figure, record_episode
    trainer = Trainer(TrainConfig(**PPO))
    trainer.train(episodes=3)
    env = trainer.env
    env.reset(seed=0)
    pm = policy_map(env, trainer.agent)
    assert pm is not None and pm["seen"].any() and (pm["action"] >= 0).any()
    assert np.isfinite(pm["value"][pm["seen"]]).all()
    policy_map_figure(env, trainer.agent)
    frames = record_episode(env, trainer.agent, seed=0)
    assert frames[1]["scores"] == "probs" and abs(sum(frames[1]["q_values"]) - 1) < 1e-6
    frame_figure(env, frames, 0)


def test_ppo_protocol_settings_and_fresh_layouts():
    from src.evaluation import EXPERIMENTS
    from src.training import build_env

    def _layout_keys(exp, layout_seeds):
        env = build_env(exp.train_config(exp.agents[0], exp.seeds[0]))
        keys = set()
        for s in layout_seeds:
            env.reset(layout_seed=s)
            keys.add((env.grid.tobytes(), env.start, env.goal))
        return keys

    exp = EXPERIMENTS["ppo_generalization"]
    assert exp.agents == ("dqn", "ppo") and exp.seeds == tuple(range(700, 710))
    assert (exp.episodes, exp.decay_episodes, exp.eval_every) == (3000, 2000, 500)
    dqn, ppo = exp.train_config("dqn", 700), exp.train_config("ppo", 700)
    assert ppo.agent == "ppo" and ppo.agent_kwargs == {}  # PPOConfig defaults, no epsilon
    assert dqn.agent_kwargs == {"epsilon": {"decay_episodes": 2000}}
    assert (dqn.env_preset, dqn.env_overrides, dqn.episodes, dqn.curriculum) == \
        (ppo.env_preset, ppo.env_overrides, ppo.episodes, ppo.curriculum)
    assert PPOConfig() == PPOConfig(hidden=(128, 128), lr=3e-4, gamma=0.99, gae_lambda=0.95,
                                    n_steps=2048, epochs=4, minibatch_size=256, clip=0.2,
                                    entropy_coef=0.01, value_coef=0.5, grad_clip=0.5)
    pool = tuple(exp.env_overrides["layout_pool"])
    assert pool == tuple(range(9700, 9750)) and exp.unseen_layouts == tuple(range(9800, 9850))
    train, held = _layout_keys(exp, pool), _layout_keys(exp, exp.unseen_layouts)
    assert len(train) == 50 and len(held) == 50 and not train & held
    others = [e for n, e in EXPERIMENTS.items() if n != "ppo_generalization"]
    used = set()
    for e in others:
        used |= set(e.env_overrides.get("layout_pool") or ()) | set(e.unseen_layouts or ())
        used.add(e.env_overrides.get("layout_seed", 0))
    assert not (set(pool) | set(exp.unseen_layouts)) & used
    assert not set(exp.seeds) & set().union(*(e.seeds for e in others))
    assert json.loads(ppo.to_json())["agent"] == "ppo"


def test_ppo_replication_matches_stage_i_with_fresh_seeds_and_mazes():
    from src.evaluation import EXPERIMENTS
    from src.training import build_env

    rep, orig = EXPERIMENTS["ppo_replication"], EXPERIMENTS["ppo_generalization"]
    assert rep.agents == orig.agents == ("dqn", "ppo")
    assert rep.seeds == tuple(range(1100, 1110))
    assert (rep.episodes, rep.decay_episodes, rep.eval_every) == \
        (orig.episodes, orig.decay_episodes, orig.eval_every)
    for agent in rep.agents:   # identical settings apart from seeds and mazes
        a, b = rep.train_config(agent, 1100), orig.train_config(agent, 700)
        assert (a.agent, a.agent_kwargs, a.env_preset, a.episodes, a.curriculum) == \
            (b.agent, b.agent_kwargs, b.env_preset, b.episodes, b.curriculum)
        assert {k: v for k, v in a.env_overrides.items() if k != "layout_pool"} == \
            {k: v for k, v in b.env_overrides.items() if k != "layout_pool"}
    pool = tuple(rep.env_overrides["layout_pool"])
    assert pool == tuple(range(10100, 10150))
    assert rep.unseen_layouts == tuple(s for s in range(10200, 10251) if s != 10218)
    env = build_env(rep.train_config("dqn", 1100))

    def keys(seeds):
        out = []
        for s in seeds:
            env.reset(layout_seed=s)
            out.append((env.grid.tobytes(), env.start, env.goal))
        return out

    train, held = keys(pool), keys(rep.unseen_layouts)
    assert len(set(train)) == 50 and len(set(held)) == 50 and not set(train) & set(held)
    assert keys([10218])[0] == keys([10149])[0]          # why 10218 was left out
    others = [e for n, e in EXPERIMENTS.items() if n != "ppo_replication"]
    assert not set(rep.seeds) & set().union(*(e.seeds for e in others))
