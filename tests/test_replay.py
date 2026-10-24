import numpy as np
import pytest

from src.replay import MistakeReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer, SumTree


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


# ---------------------------------------------------------------------------
# Mistake-aware replay
# ---------------------------------------------------------------------------

def fill_mixed(buffer, n, mistake_every=3):
    """obs = i; every `mistake_every`-th transition is a mistake."""
    for i in range(n):
        buffer.add([float(i)], i % 4, -float(i), [float(i + 1)], False,
                   mistake=i % mistake_every == 0)


def test_mistakes_are_copied_and_ordinary_transitions_kept():
    b = MistakeReplayBuffer(ReplayBuffer(20, 1, seed=0), capacity=10, seed=0)
    fill_mixed(b, 9)
    assert len(b) == 9 and len(b.main) == 9          # main buffer stores everything
    assert sorted(b.reserve.obs[:len(b.reserve), 0].tolist()) == [0.0, 3.0, 6.0]
    assert b.mistakes_seen == 3


def test_reserve_is_circular_and_outlives_the_main_buffer():
    b = MistakeReplayBuffer(ReplayBuffer(4, 1, seed=0), capacity=2, seed=0)
    fill_mixed(b, 10)                                # mistakes at 0, 3, 6, 9
    assert len(b.reserve) == 2
    assert sorted(b.reserve.obs[:2, 0].tolist()) == [6.0, 9.0]
    assert sorted(b.main.obs[:, 0].tolist()) == [6.0, 7.0, 8.0, 9.0]
    for i in range(10, 14):                          # four ordinary transitions
        b.add([float(i)], 0, 0.0, [float(i + 1)], False)
    assert 6.0 not in b.main.obs[:, 0] and 6.0 in b.reserve.obs[:2, 0]


@pytest.mark.parametrize("fraction,batch,reserve,expected", [
    (0.25, 8, 100, 2),     # round(0.25 * 8)
    (0.25, 64, 100, 16),
    (0.25, 64, 5, 5),      # never more than the reserve holds
    (0.25, 64, 0, 0),      # empty reserve: whole batch from the main buffer
    (0.9, 4, 100, 3),      # at least one main-buffer transition per batch
    (0.0, 64, 100, 0),
])
def test_batch_proportions(fraction, batch, reserve, expected):
    b = MistakeReplayBuffer(ReplayBuffer(200, 1, seed=0), capacity=200, fraction=fraction, seed=0)
    for i in range(reserve):
        b.add([1.0], 0, -1.0, [1.0], False, mistake=True)
    for i in range(20):
        b.add([0.0], 0, 0.0, [0.0], False)
    out = b.sample(batch)
    assert out["obs"].shape == (batch, 1)
    assert out["from_reserve"].sum() == expected
    # reserve rows really are mistakes (obs 1), and come last in the batch
    assert (out["obs"][out["from_reserve"], 0] == 1.0).all()
    assert (out["indices"][out["from_reserve"]] >= b.main.capacity).all()
    assert (out["indices"][~out["from_reserve"]] < b.main.capacity).all()


def test_fraction_must_leave_room_for_ordinary_experience():
    with pytest.raises(ValueError):
        MistakeReplayBuffer(ReplayBuffer(4, 1), capacity=4, fraction=1.0)


def test_with_no_mistakes_sampling_matches_the_plain_buffer():
    plain = ReplayBuffer(10, 1, seed=3)
    wrapped = MistakeReplayBuffer(ReplayBuffer(10, 1, seed=3), capacity=10, seed=3)
    fill(plain, 7)
    fill(wrapped, 7)
    a, b = plain.sample(16), wrapped.sample(16)
    for key in ("obs", "actions", "rewards", "next_obs", "terminated", "weights", "indices"):
        np.testing.assert_array_equal(a[key], b[key])


def test_prioritized_main_buffer_and_reserve_weights():
    main = PrioritizedReplayBuffer(8, 1, alpha=1.0, eps=0.0, seed=0)
    b = MistakeReplayBuffer(main, capacity=8, fraction=0.5, seed=0)
    fill_mixed(b, 8, mistake_every=2)
    b.update_priorities(np.arange(8), np.arange(1.0, 9.0))
    out = b.sample(8, beta=1.0)
    assert out["from_reserve"].sum() == 4
    assert (out["weights"][out["from_reserve"]] == 1.0).all()
    assert (out["weights"] <= 1.0).all() and out["weights"].dtype == np.float32


def test_reserve_samples_never_touch_main_priorities():
    main = PrioritizedReplayBuffer(8, 1, alpha=1.0, eps=0.0, seed=0)
    b = MistakeReplayBuffer(main, capacity=8, fraction=0.5, seed=0)
    fill_mixed(b, 8, mistake_every=2)
    b.update_priorities(np.arange(8), np.ones(8))
    before = main.tree.get(np.arange(8)).copy()
    out = b.sample(8, beta=1.0)
    td = np.where(out["from_reserve"], 100.0, 1.0)   # huge errors on reserve rows only
    b.update_priorities(out["indices"], td)
    np.testing.assert_allclose(main.tree.get(np.arange(8)), before)
    assert main.max_priority == pytest.approx(1.0)
    only_reserve = out["indices"][out["from_reserve"]]
    b.update_priorities(only_reserve, np.full(len(only_reserve), 5.0))  # no error, no change
    np.testing.assert_allclose(main.tree.get(np.arange(8)), before)


@pytest.mark.parametrize("cls", [ReplayBuffer, PrioritizedReplayBuffer])
def test_mistake_buffer_state_dict_round_trip(cls):
    b = MistakeReplayBuffer(cls(10, 1, seed=0), capacity=4, fraction=0.5, seed=0)
    fill_mixed(b, 9)
    b.update_priorities(np.arange(9), np.linspace(0.1, 0.9, 9))
    c = MistakeReplayBuffer(cls(10, 1, seed=0), capacity=4, fraction=0.5, seed=99)
    c.load_state_dict(b.state_dict())
    assert len(c) == 9 and len(c.reserve) == 3 and c.mistakes_seen == 3
    c.main.rng.bit_generator.state = b.main.rng.bit_generator.state
    x, y = b.sample(6, beta=1.0), c.sample(6, beta=1.0)  # same generators -> same batch
    for key in x:
        np.testing.assert_array_equal(x[key], y[key])


# -- long memory: insertion stamps and diagnostics ---------------------------------------

def test_diagnostics_show_mistakes_kept_after_main_buffer_forgot_them():
    b = MistakeReplayBuffer(ReplayBuffer(4, 1, seed=0), capacity=3, seed=0)
    fill_mixed(b, 10)                     # transitions 0..9, mistakes at 0, 3, 6, 9
    d = b.diagnostics()
    assert d["transitions_added"] == 10 and d["main_size"] == 4
    assert d["main_overwritten"] == 6     # main holds 6, 7, 8, 9
    assert d["reserve_size"] == 3 and d["mistakes_seen"] == 4
    assert d["reserve_overwritten"] == 1  # mistake 0 fell out of the full reserve
    assert sorted(b.reserve_added_at[:3].tolist()) == [3, 6, 9]
    assert d["reserve_only"] == 1         # mistake 3: gone from main, kept in reserve
    assert d["reserve_span"] == 7         # reserve reaches back to transition 3


def test_reserve_never_exceeds_capacity_and_keeps_the_newest_mistakes():
    b = MistakeReplayBuffer(ReplayBuffer(8, 1, seed=0), capacity=5, seed=0)
    for i in range(200):
        b.add([float(i)], 0, 0.0, [0.0], False, mistake=i % 2 == 1)
        assert len(b.reserve) <= 5 and len(b.main) <= 8
    kept = sorted(b.reserve.obs[:5, 0].tolist())
    assert kept == [191.0, 193.0, 195.0, 197.0, 199.0]
    np.testing.assert_array_equal(np.sort(b.reserve_added_at[:5]), [191, 193, 195, 197, 199])


def test_each_transition_is_stored_once():
    rng = np.random.default_rng(0)
    flags = rng.random(300) < 0.3
    b = MistakeReplayBuffer(ReplayBuffer(1000, 1, seed=0), capacity=1000, seed=0)
    for i, flag in enumerate(flags):
        b.add([float(i)], 0, 0.0, [0.0], False, mistake=bool(flag))
    main = b.main.obs[:len(b.main), 0]
    reserve = b.reserve.obs[:len(b.reserve), 0]
    assert len(main) == len(set(main.tolist())) == 300
    assert sorted(reserve.tolist()) == [float(i) for i in np.flatnonzero(flags)]
    assert len(set(reserve.tolist())) == len(reserve) == b.mistakes_seen


def test_stamps_survive_round_trip_and_old_states_load():
    b = MistakeReplayBuffer(ReplayBuffer(4, 1, seed=0), capacity=3, seed=0)
    fill_mixed(b, 10)
    c = MistakeReplayBuffer(ReplayBuffer(4, 1, seed=0), capacity=3, seed=0)
    c.load_state_dict(b.state_dict())
    assert c.diagnostics() == b.diagnostics()
    old = b.state_dict()
    del old["added"], old["reserve_added_at"]          # saved before stamps existed
    d = MistakeReplayBuffer(ReplayBuffer(4, 1, seed=0), capacity=3, seed=0)
    d.load_state_dict(old)
    assert d.added == 4 and len(d.reserve) == 3
