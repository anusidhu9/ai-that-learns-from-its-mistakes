import json
import math
import shutil

import numpy as np
import pytest

from src.agents import DQNAgent, EpsilonSchedule, RandomAgent, TabularQAgent
from src.envs import EnvConfig, GridWorld, RewardConfig
from src.training import (CHECKPOINT_FILE, EPISODES_FILE, LOG_FIELDS, MISTAKES_FILE,
                          TrainConfig, Trainer, build_agent, build_env, compare_phases,
                          format_phase_table, load_agent, load_history, load_mistakes,
                          mistake_hotspots, reward_breakdown, run_episode)

TABULAR = dict(agent="tabular_q", agent_kwargs={"epsilon": {"decay_episodes": 20}},
               episodes=30, seed=3, checkpoint_every=5)
DQN = dict(agent="dqn", episodes=16, seed=1, checkpoint_every=4,
           env_overrides={"max_steps": 40},
           agent_kwargs={"hidden": [32], "learning_starts": 50, "buffer_size": 2000,
                         "batch_size": 16, "epsilon": {"decay_episodes": 10}})


def read(path):
    return path.read_text()


# -- configuration ------------------------------------------------------------------

def test_train_config_json_round_trip():
    cfg = TrainConfig(**DQN)
    assert TrainConfig.from_json(cfg.to_json()) == cfg


def test_build_env_and_agent_convert_json_values():
    cfg = TrainConfig(env_preset="maze",
                      env_overrides={"rewards": {"goal": 2.0}, "layout_pool": [1, 2]},
                      agent="dqn", agent_kwargs={"hidden": [16], "epsilon": {"end": 0.1}})
    env = build_env(cfg)
    assert env.config.rewards == RewardConfig(goal=2.0)
    assert env.config.layout_pool == (1, 2)
    agent = build_agent(cfg, env)
    assert isinstance(agent, DQNAgent)
    assert agent.config.hidden == (16,) and agent.config.epsilon.end == 0.1
    with pytest.raises(ValueError):
        build_env(TrainConfig(env_preset="nope"))


# -- single-episode statistics ---------------------------------------------------------

def test_reward_breakdown_sums_to_total():
    trainer = Trainer(TrainConfig(env_preset="maze_hazards", **{**TABULAR, "episodes": 15}))
    rows = trainer.train()
    for row in rows:
        parts = sum(row[k] for k in LOG_FIELDS if k.startswith("r_"))
        assert parts == pytest.approx(row["total_reward"], abs=1e-6)


def test_reward_breakdown_formula():
    summary = {"success": True, "steps": 10, "collisions": 2, "hazards": 1,
               "revisits": 3, "timeout": 0}
    rb = reward_breakdown(summary, RewardConfig())
    assert rb == pytest.approx({"r_goal": 1.0, "r_step": -0.1, "r_collision": -0.1,
                                "r_hazard": -0.5, "r_revisit": -0.06, "r_timeout": 0.0})


def test_run_episode_update_stats_and_efficiency():
    env = GridWorld.from_ascii("#####\n#S.G#\n#####", EnvConfig(max_steps=20))
    random_summary = run_episode(env, RandomAgent(4, seed=0), train=True)
    assert math.isnan(random_summary["td_error"]) and math.isnan(random_summary["loss"])

    agent = TabularQAgent(4, epsilon=EpsilonSchedule(decay_episodes=10), seed=0)
    s = run_episode(env, agent, train=True)
    assert s["td_error"] >= 0 and math.isnan(s["loss"])   # tabular has no loss
    for _ in range(30):
        run_episode(env, agent, train=True)
    greedy = run_episode(env, agent, train=False, greedy=True)
    assert greedy["path_efficiency"] == pytest.approx(1.0)


# -- logs on disk -------------------------------------------------------------------------

def test_trainer_writes_logs_and_checkpoint(tmp_path):
    run = tmp_path / "run"
    trainer = Trainer(TrainConfig(**TABULAR), run)
    rows = trainer.train()
    assert len(rows) == 30 and trainer.episodes_done == 30
    assert (run / CHECKPOINT_FILE).exists()
    assert json.loads((run / "config.json").read_text())["agent"] == "tabular_q"

    history = load_history(run)
    assert [r["episode"] for r in history] == list(range(30))
    def same(a, b):  # NaN (no loss for tabular agents) never equals itself
        return a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))

    for disk, memory in zip(history, trainer.history):     # disk == memory, types included
        assert disk.keys() == memory.keys()
        assert all(same(disk[k], memory[k]) and type(disk[k]) is type(memory[k]) for k in disk)
    assert isinstance(history[0]["success"], bool) and isinstance(history[0]["steps"], int)

    mistakes = load_mistakes(run)
    assert len(mistakes) == sum(r["mistakes"] for r in history)
    assert set(mistakes[0]) == {"episode", "layout_seed", "step", "event", "pos", "action"}


def test_run_dir_refuses_a_different_config(tmp_path):
    Trainer(TrainConfig(**TABULAR), tmp_path)
    with pytest.raises(FileExistsError):
        Trainer(TrainConfig(**{**TABULAR, "seed": 99}), tmp_path)


def test_in_memory_trainer_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = Trainer(TrainConfig(**{**TABULAR, "episodes": 5}))
    trainer.train()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        trainer.save_checkpoint()


# -- checkpoints and exact resumption -------------------------------------------------------

@pytest.mark.parametrize("cfg", [TABULAR, DQN], ids=["tabular", "dqn"])
def test_resumed_run_matches_uninterrupted_run(tmp_path, cfg):
    full = tmp_path / "full"
    Trainer(TrainConfig(**cfg), full).train()

    split = tmp_path / "split"
    first = Trainer(TrainConfig(**cfg), split)
    first.train(episodes=cfg["episodes"] // 2 + 1)  # stop at an odd point
    del first
    resumed = Trainer.resume(split)
    assert resumed.episodes_done == cfg["episodes"] // 2 + 1
    resumed.train()

    assert read(split / EPISODES_FILE) == read(full / EPISODES_FILE)
    assert read(split / MISTAKES_FILE) == read(full / MISTAKES_FILE)


def test_resume_discards_logs_written_after_last_checkpoint(tmp_path):
    full = tmp_path / "full"
    Trainer(TrainConfig(**TABULAR), full).train()

    run = tmp_path / "crash"
    trainer = Trainer(TrainConfig(**TABULAR), run)
    trainer.train(episodes=10)
    shutil.copy(run / CHECKPOINT_FILE, tmp_path / "ckpt_at_10.pt")
    trainer.train(episodes=3)                      # logs now reach episode 12
    # Simulate a crash after episode 12 whose last checkpoint was episode 10.
    shutil.copy(tmp_path / "ckpt_at_10.pt", run / CHECKPOINT_FILE)

    resumed = Trainer.resume(run)
    assert resumed.episodes_done == 10
    assert len(load_history(run)) == 10
    assert all(m["episode"] < 10 for m in load_mistakes(run))
    resumed.train()
    assert read(run / EPISODES_FILE) == read(full / EPISODES_FILE)


def test_resume_without_checkpoint_restarts(tmp_path):
    trainer = Trainer(TrainConfig(**TABULAR), tmp_path)
    trainer.train(episodes=3)
    (tmp_path / CHECKPOINT_FILE).unlink()
    resumed = Trainer.resume(tmp_path)
    assert resumed.episodes_done == 0 and load_history(tmp_path) == []


def test_callback_can_stop_training_and_checkpoint_is_saved(tmp_path):
    seen = []

    def stop_after_four(row):
        seen.append(row)
        return len(seen) == 4

    trainer = Trainer(TrainConfig(**{**TABULAR, "checkpoint_every": 100}), tmp_path)
    rows = trainer.train(callback=stop_after_four)
    assert len(rows) == 4 and trainer.episodes_done == 4
    assert "mistake_log" in seen[0]
    assert Trainer.resume(tmp_path).episodes_done == 4


def test_checkpoint_buffer_is_optional(tmp_path):
    trainer = Trainer(TrainConfig(**DQN))
    trainer.train(episodes=3)
    assert "buffer" in trainer.checkpoint_state()["agent"]
    assert "buffer" not in trainer.checkpoint_state(include_buffer=False)["agent"]
    small = trainer.save_checkpoint(tmp_path / "model.pt", include_buffer=False)
    big = trainer.save_checkpoint(tmp_path / "full.pt")
    assert small.stat().st_size < big.stat().st_size
    assert not list(tmp_path.glob("*.tmp"))       # atomic save cleans up


@pytest.mark.parametrize("cfg", [TABULAR, DQN], ids=["tabular", "dqn"])
def test_load_agent_restores_policy(tmp_path, cfg):
    trainer = Trainer(TrainConfig(**cfg), tmp_path)
    trainer.train(episodes=6)
    agent, env, config = load_agent(tmp_path / CHECKPOINT_FILE)
    assert config == trainer.config and agent.episodes == 6
    obs, _ = env.reset()
    np.testing.assert_allclose(agent.q_values(obs), trainer.agent.q_values(obs))


def test_load_checkpoint_rejects_other_agent(tmp_path):
    Trainer(TrainConfig(**TABULAR), tmp_path / "a").train(episodes=2)
    other = Trainer(TrainConfig(**{**TABULAR, "agent": "random", "agent_kwargs": {}}))
    with pytest.raises(ValueError):
        other.load_checkpoint(tmp_path / "a" / CHECKPOINT_FILE)


# -- analysis helpers ----------------------------------------------------------------------

def fake_row(i, success, mistakes):
    return {"episode": i, "success": success, "steps": 10, "total_reward": 0.5,
            "path_efficiency": 1.0 if success else 0.0, "mistakes": mistakes,
            "collisions": mistakes, "hazards": 0, "revisits": 0, "timeout": 0}


def test_compare_phases():
    history = [fake_row(i, False, 10) for i in range(5)] + [fake_row(i, True, 2) for i in range(5, 10)]
    cmp = compare_phases(history, window=5)
    assert cmp["early"]["success"] == 0.0 and cmp["late"]["success"] == 1.0
    assert cmp["change"]["mistakes"] == pytest.approx(-8.0)
    assert compare_phases(history[:3], window=50)["window"] == 3
    table = format_phase_table(cmp)
    assert "mistakes" in table and "-8.000" in table
    with pytest.raises(ValueError):
        compare_phases([])


def test_mistake_hotspots():
    mistakes = ([{"episode": 0, "event": "collision", "pos": [1, 1]}] * 3
                + [{"episode": 0, "event": "revisit", "pos": [1, 2]}] * 2
                + [{"episode": 5, "event": "hazard", "pos": [3, 3]}])
    hot = mistake_hotspots(mistakes, top=2)
    assert hot == [{"event": "collision", "pos": (1, 1), "count": 3},
                   {"event": "revisit", "pos": (1, 2), "count": 2}]
    late = mistake_hotspots(mistakes, episodes=range(5, 10))
    assert late == [{"event": "hazard", "pos": (3, 3), "count": 1}]


def test_resume_can_extend_a_finished_run(tmp_path):
    full = tmp_path / "full"
    Trainer(TrainConfig(**TABULAR), full).train()

    run = tmp_path / "short"
    Trainer(TrainConfig(**{**TABULAR, "episodes": 12}), run).train()
    extended = Trainer.resume(run, episodes=30)
    assert extended.config.episodes == 30
    extended.train()
    assert read(run / EPISODES_FILE) == read(full / EPISODES_FILE)
    assert TrainConfig.from_json((run / "config.json").read_text()).episodes == 30


def test_cli_trains_and_resumes(tmp_path, capsys):
    from src.training import main
    run = str(tmp_path / "cli")
    main(["--agent", "tabular_q", "--episodes", "60", "--run-dir", run, "--decay-episodes", "40"])
    out = capsys.readouterr().out
    assert "episode    50" in out and "first 50" in out
    main(["--resume", run, "--episodes", "100"])
    out = capsys.readouterr().out
    assert "Resuming" in out and "at episode 60 of 100" in out
    assert len(load_history(run)) == 100


def test_path_efficiency_never_rewards_hazard_shortcuts():
    layout = """
    #######
    #S.H.G#
    #.###.#
    #.....#
    #######
    """
    env = GridWorld.from_ascii(layout, EnvConfig(hazard_terminal=False, max_steps=30))

    class Scripted:                      # walks straight through the hazard
        rng = np.random.default_rng(0)
        epsilon = 0.0
        def act(self, obs, greedy=False):
            return 1                     # right

    shortcut = run_episode(env, Scripted(), train=False)
    assert shortcut["success"] and shortcut["hazards"] == 1
    assert shortcut["steps"] == 4 < shortcut["optimal_length"] == 8
    assert shortcut["path_efficiency"] == 0.0


def test_checkpoint_loading_refuses_arbitrary_objects(tmp_path):
    import pickle

    class Evil:
        def __reduce__(self):
            return (print, ("this must never run",))

    path = tmp_path / "evil.pt"
    import torch
    torch.save({"agent": Evil()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_agent(path)
