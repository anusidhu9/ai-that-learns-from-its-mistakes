import numpy as np
import pytest

from src.envs import (EMPTY, GOAL, HAZARD, PRESETS, WALL, EnvConfig, GridWorld,
                      RewardConfig, check_layout, generate_layout, layout_to_ascii, make_env,
                      parse_layout, shortest_path_length)

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
RW = RewardConfig()

SMALL = """
#####
#S..#
#.#H#
#..G#
#####
"""


def small_env(**overrides):
    return GridWorld.from_ascii(SMALL, EnvConfig(**overrides))


# -- configuration ------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(layout="cave"),
    dict(obs_mode="pixels"),
    dict(layout="maze", height=8),
    dict(height=2, width=5, layout="open"),
    dict(layout_pool=()),
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        make_env("nope")


# -- layout generation ----------------------------------------------------------

def test_layout_is_deterministic_per_seed():
    cfg = EnvConfig(**PRESETS["maze_hazards"])
    g1, s1, t1 = generate_layout(cfg, 7)
    g2, s2, t2 = generate_layout(cfg, 7)
    g3, _, _ = generate_layout(cfg, 8)
    assert np.array_equal(g1, g2) and s1 == s2 and t1 == t2
    assert not np.array_equal(g1, g3)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_layout_is_valid_and_solvable(preset):
    cfg = EnvConfig(**PRESETS[preset])
    for seed in range(50):
        grid, start, goal = generate_layout(cfg, seed)
        assert grid.shape == (cfg.height, cfg.width)
        # border is solid wall
        assert (grid[0] == WALL).all() and (grid[-1] == WALL).all()
        assert (grid[:, 0] == WALL).all() and (grid[:, -1] == WALL).all()
        assert start != goal
        assert grid[start] == EMPTY and grid[goal] == GOAL
        assert np.count_nonzero(grid == GOAL) == 1
        assert np.count_nonzero(grid == HAZARD) <= cfg.n_hazards
        # a hazard-free route to the goal always exists
        assert shortest_path_length(grid, start, goal, avoid_hazards=True) is not None


def test_hazard_presets_actually_contain_hazards():
    cfg = EnvConfig(**PRESETS["maze_hazards"])
    counts = [np.count_nonzero(generate_layout(cfg, s)[0] == HAZARD) for s in range(20)]
    assert sum(counts) > 0


# -- observations ---------------------------------------------------------------

@pytest.mark.parametrize("obs_mode", ["grid", "local"])
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_observation_shape_and_dtype(preset, obs_mode):
    env = make_env(preset, obs_mode=obs_mode)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_size,)
    assert obs.dtype == np.float32
    obs, *_ = env.step(RIGHT)
    assert obs.shape == (env.obs_size,)


def test_grid_observation_channels():
    env = small_env(obs_mode="grid")
    obs, _ = env.reset()
    planes = obs.reshape(4, 5, 5)
    assert planes[0].sum() == np.count_nonzero(env.grid == WALL)
    assert planes[1][2, 3] == 1 and planes[1].sum() == 1   # hazard
    assert planes[2][3, 3] == 1 and planes[2].sum() == 1   # goal
    assert planes[3][1, 1] == 1 and planes[3].sum() == 1   # agent


def test_local_observation_is_egocentric():
    env = small_env(obs_mode="local", view_radius=1)
    obs, _ = env.reset()
    window = obs[:27].reshape(3, 3, 3)
    walls = window[0]
    # agent at (1,1): row above and column to the left are wall
    assert walls[0].all() and walls[:, 0].all()
    assert walls[1, 1] == 0
    np.testing.assert_allclose(obs[27:], [2 / 4, 2 / 4])  # goal direction


# -- dynamics, rewards and mistakes -----------------------------------------------

def test_collision_keeps_agent_in_place():
    env = small_env()
    env.reset()
    _, reward, term, trunc, info = env.step(UP)
    assert env.pos == (1, 1)
    assert info["events"] == ["collision"]
    assert reward == pytest.approx(RW.step + RW.collision)
    assert not term and not trunc


def test_revisit_is_penalised():
    env = small_env()
    env.reset()
    env.step(RIGHT)
    _, reward, _, _, info = env.step(LEFT)
    assert info["events"] == ["revisit"]
    assert reward == pytest.approx(RW.step + RW.revisit)


def test_hazard_terminates_by_default():
    env = small_env()
    env.reset()
    env.step(RIGHT)
    env.step(RIGHT)
    _, reward, term, trunc, info = env.step(DOWN)
    assert info["events"] == ["hazard"]
    assert term and not trunc
    assert reward == pytest.approx(RW.step + RW.hazard)
    assert info["episode"]["hazards"] == 1 and not info["episode"]["success"]


def test_non_terminal_hazard_continues():
    env = small_env(hazard_terminal=False)
    env.reset()
    for a in (RIGHT, RIGHT, DOWN):
        _, _, term, _, _ = env.step(a)
    assert not term
    _, reward, term, _, info = env.step(DOWN)
    assert info["events"] == ["goal"] and term


def test_reaching_goal_succeeds():
    env = small_env()
    env.reset()
    total = 0.0
    for a in (DOWN, DOWN, RIGHT, RIGHT):
        _, r, term, trunc, info = env.step(a)
        total += r
    assert term and not trunc
    assert info["events"] == ["goal"]
    ep = info["episode"]
    assert ep["success"] and ep["steps"] == 4 and ep["mistakes"] == 0
    assert ep["optimal_length"] == 4
    assert total == pytest.approx(4 * RW.step + RW.goal)


def test_timeout_truncates_episode():
    env = small_env(max_steps=3)
    env.reset()
    for _ in range(2):
        _, _, term, trunc, _ = env.step(UP)
        assert not term and not trunc
    _, reward, term, trunc, info = env.step(UP)
    assert trunc and not term
    assert info["events"] == ["collision", "timeout"]
    assert reward == pytest.approx(RW.step + RW.collision + RW.timeout)
    assert info["episode"]["timeout"] == 1 and info["episode"]["collisions"] == 3


def test_mistake_log_records_each_mistake():
    env = small_env()
    env.reset()
    env.step(UP)       # collision
    env.step(RIGHT)
    env.step(LEFT)     # revisit
    events = [m["event"] for m in env.mistake_log]
    assert events == ["collision", "revisit"]
    assert env.mistake_log[0] == {"step": 1, "event": "collision", "pos": (1, 1), "action": "up"}


def test_step_after_done_and_invalid_action_raise():
    env = small_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)
    for a in (DOWN, DOWN, RIGHT, RIGHT):
        env.step(a)
    with pytest.raises(RuntimeError):
        env.step(UP)


def test_reset_clears_episode_state():
    env = small_env()
    env.reset()
    env.step(UP)
    env.step(RIGHT)
    env.reset()
    assert env.pos == env.start and env.steps == 0
    assert env.mistake_log == [] and env.visited == {env.start}


# -- layout selection (training pools and unseen layouts) -----------------------

def test_fixed_layout_by_default():
    env = make_env("maze")
    seeds = {env.reset(seed=s)[1]["layout_seed"] for s in range(5)}
    assert seeds == {0}


def test_layout_pool_only_uses_pool_seeds():
    pool = (3, 5, 11)
    env = make_env("maze", layout_pool=pool, seed=0)
    seen = {env.reset()[1]["layout_seed"] for _ in range(60)}
    assert seen == set(pool)


def test_layout_seed_override_for_unseen_layouts():
    env = make_env("maze", layout_pool=(0, 1, 2))
    _, info = env.reset(layout_seed=1000)
    assert info["layout_seed"] == 1000
    expected, _, _ = generate_layout(env.config, 1000)
    assert np.array_equal(env.grid, expected)


def test_randomized_layouts_are_reproducible_with_seed():
    a = make_env("open", randomize_layout=True)
    b = make_env("open", randomize_layout=True)
    seq_a = [a.reset(seed=42)[1]["layout_seed"]] + [a.reset()[1]["layout_seed"] for _ in range(4)]
    seq_b = [b.reset(seed=42)[1]["layout_seed"]] + [b.reset()[1]["layout_seed"] for _ in range(4)]
    assert seq_a == seq_b
    assert len(set(seq_a)) == 5


# -- random-policy smoke test ----------------------------------------------------

@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_random_rollouts_respect_limits(preset):
    env = make_env(preset, randomize_layout=True, max_steps=50)
    rng = np.random.default_rng(0)
    for ep in range(20):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(int(rng.integers(env.n_actions)))
            done = term or trunc
        summary = info["episode"]
        assert summary["steps"] <= 50
        assert summary["success"] or summary["timeout"] or summary["hazards"] > 0


# -- hand-drawn layouts ------------------------------------------------------------

@pytest.mark.parametrize("obs_mode", ["grid", "local"])
def test_observation_at_matches_walking_there(obs_mode):
    walker, probe = small_env(obs_mode=obs_mode), small_env(obs_mode=obs_mode)
    walker.reset(seed=0)
    probe.reset(seed=0)
    obs, *_ = walker.step(RIGHT)
    assert np.array_equal(probe.observation_at((1, 2)), obs)
    assert probe.pos == (1, 1) and probe.steps == 0 and probe.visited == {(1, 1)}


def test_ascii_round_trip():
    env = make_env("maze_hazards")
    env.reset(seed=0)
    copy = GridWorld.from_ascii(env.to_ascii())
    copy.reset(seed=0)
    assert np.array_equal(copy.grid, env.grid)
    assert (copy.start, copy.goal) == (env.start, env.goal)
    assert layout_to_ascii(*parse_layout(SMALL)) == SMALL.strip()


@pytest.mark.parametrize("text", ["#S.#\n#..G#", "#S.S#\n#..G#", "#S..#\n#...#",
                                  "#S.X#\n#..G#", ""])
def test_parse_layout_rejects_bad_text(text):
    with pytest.raises(ValueError):
        parse_layout(text)


def test_check_layout():
    assert check_layout(*parse_layout(SMALL)) is None
    walled = parse_layout("#####\n#S#G#\n#####")
    only_via_hazard = parse_layout("#####\n#SHG#\n#####")
    assert "can't be reached" in check_layout(*walled)
    assert "can't be reached" in check_layout(*only_via_hazard)
    grid, start, _ = parse_layout(SMALL)
    assert "different" in check_layout(grid, start, start)


# -- reward shaping ---------------------------------------------------------------------

def shaped_pair(text=SMALL, **overrides):
    """The same layout without and with shaping (strength 1)."""
    plain = GridWorld.from_ascii(text, EnvConfig(**overrides))
    shaped = GridWorld.from_ascii(text, EnvConfig(rewards=RewardConfig(shaping=1.0), **overrides))
    return plain, shaped


def test_potential_is_scaled_maze_distance():
    _, env = shaped_pair()
    env.reset(seed=0)
    assert env.potential(env.goal) == 0.0
    # SMALL: start (1,1) is 4 steps from the goal (3,3); the farthest cell is 4 away.
    assert env.potential(env.start) == pytest.approx(-1.0)
    assert env.potential((2, 1)) == pytest.approx(-0.75)


def test_shaping_signs_and_task_reward_unchanged():
    plain, shaped = shaped_pair()
    plain.reset(seed=0)
    shaped.reset(seed=0)
    for action in (DOWN, UP, UP):   # toward the goal, away from it, into a wall
        _, r_plain, *_ = plain.step(action)
        _, r_shaped, *_ = shaped.step(action)
        bonus = r_shaped - r_plain
        if action == DOWN:
            assert bonus > 0
        elif shaped.steps == 2:
            assert bonus < 0
        else:  # collision: stays put, so the bonus is (gamma - 1) * phi >= 0 and tiny
            assert 0 <= bonus < 0.02
    assert shaped.stats["total_reward"] == pytest.approx(plain.stats["total_reward"])


@pytest.mark.parametrize("seed", range(5))
def test_discounted_shaping_telescopes(seed):
    """sum_t gamma^t F_t = gamma^T phi(s_T) - phi(s_0), with phi(s_T) = 0 when the
    episode terminates: going round in circles earns nothing."""
    plain, shaped = shaped_pair(hazard_terminal=False, max_steps=30)
    rng = np.random.default_rng(seed)
    plain.reset(seed=0)
    shaped.reset(seed=0)
    gamma = shaped.config.rewards.shaping_gamma
    phi0 = shaped.potential(shaped.start)
    total, t, done = 0.0, 0, False
    while not done:
        action = int(rng.integers(4))
        _, r_plain, *_ = plain.step(action)
        _, r_shaped, terminated, truncated, info = shaped.step(action)
        total += gamma ** t * (r_shaped - r_plain)
        t += 1
        done = terminated or truncated
    end = 0.0 if terminated else shaped.potential(shaped.pos)
    assert total == pytest.approx(gamma ** t * end - phi0)
    assert info["episode"]["total_reward"] == plain.episode_summary()["total_reward"]
    assert info["episode"]["shaping"] != 0.0 and plain.episode_summary()["shaping"] == 0.0
