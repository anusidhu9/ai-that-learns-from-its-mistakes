import numpy as np
import pytest

from src.replay import PrioritizedReplayBuffer, ReplayBuffer, SumTree


def fill(buffer, n):
    for i in range(n):
        buffer.add([float(i)], i % 4, float(i), [float(i + 1)], i % 2 == 0)


def test_buffer_is_circular():
    b = ReplayBuffer(3, 1, seed=0)
    fill(b, 5)
    assert len(b) == 3
    assert sorted(b.obs[:, 0].tolist()) == [2.0, 3.0, 4.0]  # oldest two overwritten


def test_uniform_sample_shapes_and_weights():
    b = ReplayBuffer(10, 1, seed=0)
    fill(b, 6)
    batch = b.sample(8)
    assert batch["obs"].shape == (8, 1) and batch["next_obs"].shape == (8, 1)
    assert batch["actions"].dtype == np.int64 and batch["obs"].dtype == np.float32
    assert (batch["weights"] == 1).all()
    assert (batch["indices"] < 6).all()
    # stored transitions stay consistent: reward == obs, next_obs == obs + 1
    np.testing.assert_array_equal(batch["rewards"], batch["obs"][:, 0])
    np.testing.assert_array_equal(batch["next_obs"][:, 0], batch["obs"][:, 0] + 1)


def test_empty_buffer_cannot_sample():
    with pytest.raises(ValueError):
        ReplayBuffer(4, 1).sample(2)
    with pytest.raises(ValueError):
        PrioritizedReplayBuffer(4, 1).sample(2)


def test_sum_tree_sums_and_finds():
    t = SumTree(5)
    t.set(np.arange(5), np.array([1.0, 2.0, 3.0, 4.0, 0.0]))
    assert t.total == pytest.approx(10.0)
    assert t.find(np.array([0.5, 1.5, 3.5, 9.9])).tolist() == [0, 1, 2, 3]
    t.set(np.array([1]), np.array([0.0]))
    assert t.total == pytest.approx(8.0)


def test_prioritized_sampling_follows_priorities():
    b = PrioritizedReplayBuffer(5, 1, alpha=1.0, eps=0.0, seed=0)
    fill(b, 5)
    b.update_priorities(np.arange(5), np.array([1.0, 2.0, 3.0, 4.0, 0.0]))
    counts = np.zeros(5)
    for _ in range(500):
        counts += np.bincount(b.sample(10)["indices"], minlength=5)
    np.testing.assert_allclose(counts / counts.sum(), [0.1, 0.2, 0.3, 0.4, 0.0], atol=0.02)


def test_alpha_zero_is_uniform():
    b = PrioritizedReplayBuffer(4, 1, alpha=0.0, seed=0)
    fill(b, 4)
    b.update_priorities(np.arange(4), np.array([100.0, 1.0, 1.0, 1.0]))
    assert (b.sample(16, beta=1.0)["weights"] == pytest.approx(1.0))


def test_importance_weights_match_formula():
    b = PrioritizedReplayBuffer(2, 1, alpha=1.0, eps=0.0, seed=0)
    fill(b, 2)
    b.update_priorities(np.arange(2), np.array([1.0, 3.0]))
    batch = b.sample(64, beta=1.0)
    # P = [0.25, 0.75]; w = (N*P)^-1 = [2, 2/3]; normalised by max -> [1, 1/3]
    expected = np.where(batch["indices"] == 0, 1.0, 1.0 / 3.0)
    np.testing.assert_allclose(batch["weights"], expected, rtol=1e-6)


def test_new_transitions_get_max_priority():
    b = PrioritizedReplayBuffer(4, 1, alpha=1.0, eps=0.0, seed=0)
    fill(b, 2)
    b.update_priorities(np.arange(2), np.array([5.0, 0.5]))
    b.add([9.0], 0, 0.0, [9.0], False)
    assert b.tree.get(np.array([2]))[0] == pytest.approx(5.0)


@pytest.mark.parametrize("cls", [ReplayBuffer, PrioritizedReplayBuffer])
def test_state_dict_round_trip(cls):
    b = cls(6, 1, seed=0)
    fill(b, 4)
    b.update_priorities(np.arange(4), np.array([0.1, 0.2, 0.3, 0.4]))
    c = cls(6, 1, seed=0)
    c.load_state_dict(b.state_dict())
    assert len(c) == 4 and c.pos == b.pos
    np.testing.assert_array_equal(c.obs[:4], b.obs[:4])
    if cls is PrioritizedReplayBuffer:
        np.testing.assert_allclose(c.tree.get(np.arange(4)), b.tree.get(np.arange(4)))
        assert c.tree.total == pytest.approx(b.tree.total)
