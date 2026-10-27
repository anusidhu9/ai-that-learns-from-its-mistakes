import json
import math

import numpy as np
import pytest

from src.agents import EpsilonSchedule, RandomAgent, TabularQAgent
from src.envs import make_env
from src.evaluation import (CURVE_FILE, FINAL_FILE, METRICS, Experiment, compare_agents,
                            curve_means,
                            evaluate_agent, format_summary, load_experiment, mean_ci,
                            permutation_test, run_experiment, summarize)
from src.training import train_agent

TINY = dict(name="tiny", env_preset="maze", agents=("random", "tabular_q"), seeds=(0, 1),
            episodes=20, decay_episodes=15, eval_every=10, eval_episodes=3)
TINY_GEN = dict(name="tiny_gen", env_preset="maze",
                env_overrides={"layout_pool": [0, 1, 2], "obs_mode": "local"},
                agents=("tabular_q",), seeds=(0,), episodes=20, decay_episodes=15,
                eval_every=10, unseen_layouts=(100, 101))


def quiet(_msg):
    pass


# -- statistics ------------------------------------------------------------------------

def test_mean_ci_matches_t_interval():
    mean, ci = mean_ci([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert ci == pytest.approx(4.303 * 1.0 / math.sqrt(3), rel=1e-4)
    m1, c1 = mean_ci([5.0])
    assert m1 == 5.0 and math.isnan(c1)
    assert mean_ci([2.0] * 4)[1] == 0.0
    assert all(math.isnan(v) for v in mean_ci([]))


def test_permutation_test():
    assert permutation_test([1.0] * 5, [0.0] * 5) == pytest.approx(2 / 252)
    assert permutation_test([0.3, 0.5], [0.3, 0.5]) == 1.0
    a, b = [0.1, 0.9, 0.5, 0.2], [0.4, 0.6, 0.3, 0.8]
    assert permutation_test(a, b) == pytest.approx(permutation_test(b, a))
    assert permutation_test(a, b) > 0.5   # overlapping samples: no evidence
    big = permutation_test(np.ones(15), np.zeros(15), n_resamples=2000)   # Monte Carlo path
    assert big < 0.01


# -- greedy evaluation ---------------------------------------------------------------------

def test_evaluate_agent_reports_means_and_episodes():
    env = make_env("maze")
    result = evaluate_agent(RandomAgent(4, seed=0), env, episodes=4)
    assert result["n_episodes"] == 4 and len(result["episodes"]) == 4
    assert set(METRICS) <= set(result)
    assert result["success"] == np.mean([e["success"] for e in result["episodes"]])


def test_evaluate_agent_on_listed_layouts():
    env = make_env("maze", layout_pool=(0, 1))
    result = evaluate_agent(RandomAgent(4, seed=0), env, layout_seeds=[7, 8, 9])
    assert [e["layout_seed"] for e in result["episodes"]] == [7, 8, 9]


def test_evaluation_is_deterministic():
    env = make_env("maze")
    agent = RandomAgent(4, seed=0)
    first = evaluate_agent(agent, env, episodes=3, seed=5)
    second = evaluate_agent(agent, env, episodes=3, seed=5)
    assert [e["steps"] for e in first["episodes"]] == [e["steps"] for e in second["episodes"]]


def test_evaluating_mid_training_does_not_change_training():
    def train(with_eval):
        env = make_env("maze")
        agent = TabularQAgent(4, epsilon=EpsilonSchedule(decay_episodes=20), seed=0)
        eval_env = make_env("maze")
        train_agent(env, agent, 10, seed=0)
        if with_eval:
            evaluate_agent(agent, eval_env, episodes=5)
        rows = train_agent(env, agent, 10)
        return agent, [r["steps"] for r in rows]

    plain, steps_plain = train(False)
    evaluated, steps_eval = train(True)
    assert steps_plain == steps_eval
    assert plain.q.keys() == evaluated.q.keys()
    assert all(np.array_equal(plain.q[k], evaluated.q[k]) for k in plain.q)
    assert evaluated.updates == plain.updates


# -- experiments ----------------------------------------------------------------------------

def test_run_experiment_writes_results(tmp_path):
    out = run_experiment(Experiment(**TINY), tmp_path, log=quiet)
    exp_dir = tmp_path / "tiny"
    for name in ("experiment.json", "results.csv", "curves.csv", "summary.csv"):
        assert (exp_dir / name).exists()
    assert len(out["results"]) == 2 * 2                      # agents x seeds, one split
    assert {r["split"] for r in out["results"]} == {"train"}
    episodes = sorted({r["episode"] for r in out["curves"]})
    assert episodes == [0, 10, 20]                           # untrained, mid, final
    assert {r["n_seeds"] for r in out["summary"]} == {2}
    assert "tabular_q" in format_summary(out["summary"])


def test_generalization_experiment_has_unseen_split(tmp_path):
    out = run_experiment(Experiment(**TINY_GEN), tmp_path, log=quiet)
    assert {r["split"] for r in out["results"]} == {"train", "unseen"}


def test_rerun_skips_finished_runs_and_matches(tmp_path):
    first = run_experiment(Experiment(**TINY), tmp_path, log=quiet)
    calls = []
    second = run_experiment(Experiment(**TINY), tmp_path, log=calls.append)
    assert len(calls) == 1                                   # only the header line: no training
    assert second["results"] == first["results"]


def test_interrupted_experiment_resumes_to_same_results(tmp_path):
    full = run_experiment(Experiment(**TINY), tmp_path / "full", log=quiet)

    part = tmp_path / "part"
    run_experiment(Experiment(**{**TINY, "episodes": 10}), part, log=quiet)
    resumed = run_experiment(Experiment(**TINY), part, log=quiet)   # extend 10 -> 20
    assert resumed["results"] == full["results"]
    assert resumed["curves"] == full["curves"]
    final = json.loads((part / "tiny" / "tabular_q_s0" / FINAL_FILE).read_text())
    assert final["episodes_trained"] == 20


def test_changed_settings_are_rejected(tmp_path):
    run_experiment(Experiment(**TINY), tmp_path, log=quiet)
    with pytest.raises(ValueError):
        run_experiment(Experiment(**{**TINY, "decay_episodes": 5}), tmp_path, log=quiet)


def test_load_experiment_and_compare(tmp_path):
    out = run_experiment(Experiment(**TINY), tmp_path, log=quiet)
    loaded = load_experiment(tmp_path / "tiny")
    assert loaded["results"] == out["results"]
    assert loaded["experiment"].name == "tiny"
    c = compare_agents(out["results"], "tabular_q", "random")
    assert c["difference"] == pytest.approx(c["mean_a"] - c["mean_b"])
    assert 0.0 <= c["p_value"] <= 1.0
    with pytest.raises(ValueError):
        compare_agents(out["results"], "dqn", "random")


def test_steps_to_goal_counts_only_successful_episodes():
    env = make_env("maze")
    agent = TabularQAgent(env.n_actions, seed=0, epsilon=EpsilonSchedule(decay_episodes=150))
    train_agent(env, agent, 200, seed=0)
    res = evaluate_agent(agent, env, episodes=3, seed=1)
    wins = [e["steps"] for e in res["episodes"] if e["success"]]
    assert wins and res["steps_to_goal"] == pytest.approx(np.mean(wins))
    lost = evaluate_agent(RandomAgent(env.n_actions, seed=0), env, episodes=2, seed=0)
    if not any(e["success"] for e in lost["episodes"]):
        assert lost["steps_to_goal"] is None  # undefined, not 100 and not 0


def test_summarize_skips_undefined_and_missing_steps_to_goal():
    base = {"agent": "a", "split": "train", **{m: 1.0 for m in METRICS}}
    results = [{**base, "steps_to_goal": 10.0}, {**base, "steps_to_goal": 20.0},
               {**base, "steps_to_goal": None}]
    old = {k: v for k, v in base.items() if k != "steps_to_goal"}  # saved before the metric
    row = summarize(results + [old])[0]
    assert row["n_seeds"] == 4 and row["steps_to_goal_n"] == 2
    assert row["steps_to_goal_mean"] == pytest.approx(15.0)
    c = compare_agents(results + [{**base, "agent": "b", "steps_to_goal": 30.0},
                                  {**base, "agent": "b", "steps_to_goal": 40.0}],
                       "b", "a", "steps_to_goal")
    assert c["difference"] == pytest.approx(20.0)


def test_undefined_metric_survives_csv_round_trip(tmp_path):
    from src.evaluation import _read_csv, _write_csv
    rows = [{"agent": "a", "seed": 0, "steps_to_goal": None, "success": 0.0},
            {"agent": "a", "seed": 1, "steps_to_goal": 12.0, "success": 1.0}]
    _write_csv(tmp_path / "r.csv", rows, ["agent", "seed", "steps_to_goal", "success"])
    assert _read_csv(tmp_path / "r.csv") == rows


def test_mistake_experiment_matches_hazard_maze_settings():
    from src.evaluation import EXPERIMENTS
    base, new = EXPERIMENTS["hazard_maze"], EXPERIMENTS["mistakes_hazard_maze"]
    for field in ("env_preset", "env_overrides", "seeds", "episodes", "decay_episodes",
                  "eval_every", "eval_episodes", "agent_kwargs"):
        assert getattr(new, field) == getattr(base, field), field
    assert new.agents == ("dqn", "dqn_per", "dqn_mistake", "dqn_per_mistake")


def test_mistake_experiment_smoke(tmp_path):
    exp = Experiment(name="tiny_mistakes", env_preset="maze_hazards",
                     agents=("dqn", "dqn_mistake"), seeds=(0, 1), episodes=6,
                     decay_episodes=4, eval_every=3, eval_episodes=2,
                     agent_kwargs={a: {"hidden": [16], "learning_starts": 50, "batch_size": 16}
                                   for a in ("dqn", "dqn_mistake")})
    out = run_experiment(exp, tmp_path, log=quiet)
    assert {r["agent"] for r in out["results"]} == {"dqn", "dqn_mistake"}
    assert all("steps_to_goal" in r for r in out["results"])
    loaded = load_experiment(tmp_path / "tiny_mistakes")
    assert loaded["results"] == out["results"]


def test_summarize_groups_by_agent_and_split():
    results = [{"agent": "a", "split": "train", **{m: 1.0 for m in METRICS}},
               {"agent": "a", "split": "train", **{m: 3.0 for m in METRICS}},
               {"agent": "a", "split": "unseen", **{m: 0.0 for m in METRICS}}]
    rows = {(r["agent"], r["split"]): r for r in summarize(results)}
    assert rows[("a", "train")]["success_mean"] == 2.0 and rows[("a", "train")]["n_seeds"] == 2
    assert math.isnan(rows[("a", "unseen")]["success_ci"])


def test_each_seed_gets_its_own_evaluation_randomness(tmp_path):
    exp = Experiment(**{**TINY, "agents": ("random",), "seeds": (0, 1, 2)})
    out = run_experiment(exp, tmp_path, log=quiet)
    steps = [r["steps"] for r in out["results"]]
    assert len(set(steps)) > 1, "random baseline must not replay identical moves per seed"


def test_cli_seeds_option_can_extend_beyond_defined_seeds(tmp_path, capsys):
    from src.evaluation import main
    main(["--experiment", "fixed_maze", "--seeds", "1", "--agents", "random",
          "--root", str(tmp_path)])
    assert (tmp_path / "fixed_maze" / "random_s0").exists()
    exp = json.loads((tmp_path / "fixed_maze" / "experiment.json").read_text())
    assert exp["seeds"] == [0]


def test_format_p_never_shows_zero():
    from src.evaluation import format_p
    assert format_p(0.00015) == "p < 0.001"
    assert format_p(0.008) == "p = 0.008"
    assert format_p(1.0) == "p = 1.000"


# -- long-memory study: protocol, leakage and runner -----------------------------------

def test_long_memory_protocol_settings():
    from src.evaluation import EXPERIMENTS
    exp = EXPERIMENTS["long_memory_mistakes"]
    assert exp.agents == ("dqn_5k", "dqn_mistake_5k", "dqn_10k")
    assert exp.seeds == tuple(range(100, 110))
    assert (exp.episodes, exp.decay_episodes, exp.eval_every) == (3000, 2000, 500)
    configs = {a: exp.train_config(a, 100) for a in exp.agents}
    assert {a: c.agent for a, c in configs.items()} == {
        "dqn_5k": "dqn", "dqn_mistake_5k": "dqn_mistake", "dqn_10k": "dqn"}
    # identical except for buffer size and the mistake reserve
    def rest(c):
        kw = {k: v for k, v in c.agent_kwargs.items()
              if k not in ("buffer_size", "mistake_capacity", "mistake_fraction")}
        return (c.env_preset, c.env_overrides, kw, c.episodes, c.seed)
    assert len({json.dumps(rest(c), sort_keys=True) for c in configs.values()}) == 1
    assert configs["dqn_5k"].agent_kwargs["buffer_size"] == 5000
    assert configs["dqn_mistake_5k"].agent_kwargs == {
        "epsilon": {"decay_episodes": 2000}, "buffer_size": 5000,
        "mistake_capacity": 5000, "mistake_fraction": 0.25}
    assert configs["dqn_10k"].agent_kwargs["buffer_size"] == 10_000


def _layout_keys(exp, layout_seeds):
    from src.training import build_env
    env = build_env(exp.train_config(exp.agents[0], exp.seeds[0]))
    keys = set()
    for s in layout_seeds:
        env.reset(layout_seed=s)
        keys.add((env.grid.tobytes(), env.start, env.goal))
    return keys


def test_held_out_layouts_are_new_mazes_and_seeds_are_fresh():
    from src.evaluation import EXPERIMENTS, POOL, UNSEEN
    exp, pilot = EXPERIMENTS["long_memory_mistakes"], EXPERIMENTS["long_memory_pilot"]
    pool = tuple(exp.env_overrides["layout_pool"])
    train, held = _layout_keys(exp, pool), _layout_keys(exp, exp.unseen_layouts)
    assert len(train) == 50 and len(held) == 50       # no repeated maze within a split
    assert not train & held                           # no held-out maze seen in training
    used_layouts = {0} | set(POOL) | set(UNSEEN)
    for layouts in (pool, exp.unseen_layouts, pilot.env_overrides["layout_pool"],
                    pilot.unseen_layouts):
        assert not set(layouts) & used_layouts
    pilot_layouts = set(pilot.env_overrides["layout_pool"]) | set(pilot.unseen_layouts)
    assert not set(pool) & pilot_layouts and not set(exp.unseen_layouts) & pilot_layouts
    earlier_seeds = set().union(*(e.seeds for n, e in EXPERIMENTS.items()
                                  if not n.startswith("long_memory")))
    assert not set(exp.seeds) & earlier_seeds and not set(pilot.seeds) & set(exp.seeds)


TINY_ARMS = dict(name="tiny_arms", env_preset="maze_hazards",
                 env_overrides={"layout_pool": [3000, 3001, 3002], "obs_mode": "local"},
                 agents=("small", "small_mistake", "big"), seeds=(0, 1), episodes=6,
                 decay_episodes=4, eval_every=3, unseen_layouts=(4000, 4001),
                 agent_types={"small": "dqn", "small_mistake": "dqn_mistake", "big": "dqn"},
                 agent_kwargs={"small": {"buffer_size": 150, "hidden": [16], "learning_starts": 50},
                               "small_mistake": {"buffer_size": 150, "mistake_capacity": 150,
                                                 "hidden": [16], "learning_starts": 50},
                               "big": {"buffer_size": 300, "hidden": [16], "learning_starts": 50}})


def test_arms_and_buffer_diagnostics(tmp_path):
    out = run_experiment(Experiment(**TINY_ARMS), tmp_path, log=quiet)
    assert {r["agent"] for r in out["results"]} == {"small", "small_mistake", "big"}
    cfg = json.loads((tmp_path / "tiny_arms" / "big_s0" / "config.json").read_text())
    assert cfg["agent"] == "dqn" and cfg["agent_kwargs"]["buffer_size"] == 300
    rows = {(r["agent"], r["seed"]): r for r in out["buffers"]}
    assert rows[("small", 0)]["main_capacity"] == 150 and rows[("big", 0)]["main_capacity"] == 300
    assert rows[("small", 0)]["main_overwritten"] > 0
    assert rows[("small_mistake", 0)]["reserve_size"] != "" and rows[("small", 0)]["reserve_size"] == ""
    assert (tmp_path / "tiny_arms" / "buffer_diagnostics.csv").exists()
    loaded = load_experiment(tmp_path / "tiny_arms")
    assert loaded["experiment"].agent_types == TINY_ARMS["agent_types"]


def test_training_only_ever_resets_to_pool_layouts(tmp_path):
    from src.training import load_history
    run_experiment(Experiment(**TINY_ARMS), tmp_path, log=quiet)
    for run in (tmp_path / "tiny_arms").glob("*_s*"):
        seen = {r["layout_seed"] for r in load_history(run)}
        assert seen and seen <= set(TINY_ARMS["env_overrides"]["layout_pool"]), run.name


def test_held_out_layouts_are_only_evaluated_once_at_the_end(tmp_path, monkeypatch):
    import src.evaluation as ev
    calls = []
    real = ev.evaluate_agent

    def spy(agent, env, layout_seeds=None, **kw):
        calls.append(tuple(layout_seeds) if layout_seeds is not None else None)
        return real(agent, env, layout_seeds=layout_seeds, **kw)

    monkeypatch.setattr(ev, "evaluate_agent", spy)
    exp = Experiment(**{**TINY_ARMS, "agents": ("small_mistake",)})
    run_experiment(exp, tmp_path, log=quiet)
    held = tuple(TINY_ARMS["unseen_layouts"])
    assert calls.count(held) == len(exp.seeds)          # once per run, at the end
    for i, c in enumerate(calls):
        if c == held:
            assert calls[i - 1] == tuple(TINY_ARMS["env_overrides"]["layout_pool"])


def test_evaluation_never_changes_the_agent_or_its_buffer():
    import pickle
    import torch
    from src.agents import make_agent
    env = make_env("maze_hazards", obs_mode="local", layout_pool=(3000, 3001))
    agent = make_agent("dqn_mistake", env.obs_size, env.n_actions, seed=0, hidden=(16,),
                       buffer_size=200, mistake_capacity=100, learning_starts=50)
    train_agent(env, agent, 4, seed=0)
    before = pickle.dumps(agent.state_dict(include_buffer=True))
    counters = (agent.env_steps, agent.updates, agent.episodes, agent.buffer.added)
    evaluate_agent(agent, env, layout_seeds=(4000, 4001, 4002), seed=1)
    after = agent.state_dict(include_buffer=True)
    assert (agent.env_steps, agent.updates, agent.episodes, agent.buffer.added) == counters
    old = pickle.loads(before)
    for name in ("online", "target"):
        for k in old[name]:
            assert torch.equal(old[name][k], after[name][k])
    np.testing.assert_array_equal(old["buffer"]["main"]["obs"], after["buffer"]["main"]["obs"])
    np.testing.assert_array_equal(old["buffer"]["reserve"]["obs"], after["buffer"]["reserve"]["obs"])


def test_hazard_rate_is_share_of_episodes_touching_a_hazard():
    env = make_env("maze_hazards")
    res = evaluate_agent(RandomAgent(env.n_actions, seed=0), env, episodes=6, seed=0)
    expected = np.mean([e["hazards"] > 0 for e in res["episodes"]])
    assert res["hazard_rate"] == pytest.approx(expected)


def test_parallel_workers_give_the_same_results(tmp_path):
    exp = Experiment(**{**TINY_ARMS, "agents": ("small", "small_mistake")})
    serial = run_experiment(exp, tmp_path / "serial", log=quiet)
    parallel = run_experiment(exp, tmp_path / "parallel", log=quiet, workers=2)
    assert parallel["results"] == serial["results"]
    assert parallel["curves"] == serial["curves"]
    assert parallel["buffers"] == serial["buffers"]


def test_arm_env_overrides_and_curve_means():
    exp = Experiment("x", "maze", env_overrides={"layout_seed": 3},
                     agents=("dqn", "dqn_shaped"), agent_types={"dqn_shaped": "dqn"},
                     arm_env_overrides={"dqn_shaped": {"rewards": {"shaping": 1.0}}})
    plain, shaped = exp.train_config("dqn", 0), exp.train_config("dqn_shaped", 0)
    assert plain.env_overrides == {"layout_seed": 3}
    assert shaped.env_overrides == {"layout_seed": 3, "rewards": {"shaping": 1.0}}
    assert shaped.agent == "dqn"
    curves = [{"agent": "a", "seed": 0, "episode": e, "success": s}
              for e, s in [(0, 0.0), (25, 0.5), (50, 1.0)]]
    assert curve_means(curves) == [{"agent": "a", "seed": 0, "split": "curve", "success": 0.75}]
