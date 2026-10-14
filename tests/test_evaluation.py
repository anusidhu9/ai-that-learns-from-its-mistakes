import json
import math

import numpy as np
import pytest

from src.agents import EpsilonSchedule, RandomAgent, TabularQAgent
from src.envs import make_env
from src.evaluation import (CURVE_FILE, FINAL_FILE, METRICS, Experiment, compare_agents,
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
