"""Stage K runner: re-evaluates saved networks in several modes without changing them."""
import json

import pytest
import torch

from src import stochastic_study as ks
from src.training import CHECKPOINT_FILE, Trainer, TrainConfig, export_weights, load_agent

SMALL = dict(episodes=4, seed=3, checkpoint_every=2,
             env_overrides={"obs_mode": "local", "layout_pool": [0, 1], "max_steps": 30})


def saved_model(tmp_path, agent, **kwargs):
    trainer = Trainer(TrainConfig(agent=agent, **SMALL, agent_kwargs=kwargs), tmp_path / agent)
    trainer.train()
    return export_weights(tmp_path / agent / CHECKPOINT_FILE, tmp_path / f"{agent}.pt")


def test_the_study_covers_every_saved_network_once():
    nets = ks.networks()
    assert len(nets) == 80 and len(set((s, a, seed) for s, a, seed, _ in nets)) == 80
    assert ks.modes_for("ppo")[-1] == "sample" and "sample" not in ks.modes_for("dqn")


def test_evaluate_network_runs_every_mode_and_leaves_the_network_alone(tmp_path):
    model = saved_model(tmp_path, "ppo", hidden=[8], n_steps=32, minibatch_size=8)
    before = {k: v.clone() for k, v in load_agent(model)[0].actor.state_dict().items()}
    out = ks.evaluate_network("ppo_generalization", "ppo", 700, model,
                              results_root=None, layouts=[5, 6])
    assert list(out["modes"]) == ["greedy", "eps0.05", "eps0.1", "eps0.2", "sample"]
    assert len(out["modes"]["greedy"]["episodes"]) == 2
    eps = out["modes"]["eps0.05"]["episodes"]
    assert len(eps) == 2 * ks.REPEATS
    assert [e["repeat"] for e in eps] == [r for r in range(ks.REPEATS) for _ in range(2)]
    assert [e["layout_seed"] for e in eps] == [5, 6] * ks.REPEATS
    again = ks.evaluate_network("ppo_generalization", "ppo", 700, model,
                                results_root=None, layouts=[5, 6])
    assert json.dumps(out) == json.dumps(again)   # reproducible
    after = load_agent(model)[0].actor.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_greedy_must_reproduce_saved_results(tmp_path, monkeypatch):
    model = saved_model(tmp_path, "dqn", hidden=[8], learning_starts=10)
    monkeypatch.setattr(ks, "saved_heldout", lambda *a, **k: {m: "-1" for m in ks.CHECKED})
    with pytest.raises(RuntimeError, match="does not reproduce"):
        ks.evaluate_network("cnn_generalization", "dqn", 500, model,
                            results_root=tmp_path, layouts=[5])
