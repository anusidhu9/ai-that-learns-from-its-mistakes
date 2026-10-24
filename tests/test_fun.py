import pytest

from src import fun
from src.agents import make_agent
from src.training import TrainConfig, Trainer


def row(i=0, success=False, steps=100, mistakes=50, collisions=0, hazards=0, revisits=0,
        timeout=1, optimal_length=12):
    return {"episode": i, "success": success, "steps": steps, "mistakes": mistakes,
            "collisions": collisions, "hazards": hazards, "revisits": revisits,
            "timeout": timeout, "optimal_length": optimal_length}


def test_every_agent_has_a_nickname_and_tagline():
    for name in ("random", "tabular_q", "dqn", "dqn_per", "dqn_mistake", "dqn_per_mistake"):
        agent = make_agent(name, 8, 4, hidden=(8,)) if "dqn" in name else make_agent(name, 8, 4)
        assert agent.nickname == fun.NICKNAMES[name]
        assert fun.TAGLINES[name]


@pytest.mark.parametrize("event", ["collision", "hazard", "revisit", "timeout", "goal"])
def test_event_lines_are_deterministic_ascii(event):
    a = fun.event_line(event, (1, 2), key=7)
    assert a == fun.event_line(event, (1, 2), key=7)
    assert a and a.isascii()
    if event in ("collision", "hazard", "revisit"):
        assert "(1, 2)" in a
    assert fun.event_line("unknown", (0, 0)) == ""


def test_episode_line_matches_what_happened():
    perfect = fun.episode_line(row(success=True, steps=12, mistakes=0, timeout=0))
    assert any(w in perfect for w in ("Flawless", "Speedrun", "Perfect"))
    lava = fun.episode_line(row(hazards=1, timeout=0, steps=6, mistakes=1))
    assert "lava" in lava.lower() or "hazard" in lava.lower()
    assert fun.mood(row(success=True, mistakes=0)) == "smug"
    assert fun.mood(row(hazards=2, timeout=0)) == "singed"
    assert fun.mood(row()) == "lost"


def test_achievements_are_earned_from_the_log():
    history = ([row(i, mistakes=95, collisions=3) for i in range(3)]
               + [row(3, success=True, steps=12, mistakes=0, timeout=0)]
               + [row(i, success=True, steps=12, mistakes=0, timeout=0) for i in range(4, 30)])
    earned = {a["key"]: a["episode"] for a in fun.achievements(history)}
    assert earned["first_goal"] == 3
    assert earned["clean_run"] == 3 and earned["speedrunner"] == 3
    assert earned["comeback_kid"] == 3            # success right after a 95-mistake episode
    assert earned["wall_whisperer"] == 27         # 25th collision-free episode in a row
    assert "lava_enthusiast" not in earned
    assert fun.achievements([]) == []


def test_hall_of_shame_uses_real_counts():
    trainer = Trainer(TrainConfig(episodes=10, agent_kwargs={"epsilon": {"decay_episodes": 10}}))
    trainer.train()
    shame = fun.hall_of_shame(trainer.mistakes, top=3)
    assert [s["rank"] for s in shame] == [1, 2, 3]
    counts = [s["count"] for s in shame]
    assert counts == sorted(counts, reverse=True)
    top = shame[0]
    real = sum(1 for m in trainer.mistakes
               if m["event"] == top["event"] and tuple(m["pos"]) == top["pos"])
    assert top["count"] == real
    assert "HALL OF SHAME" in fun.format_hall_of_shame(shame)
    assert "empty" in fun.format_hall_of_shame([]).lower()


def test_verdicts_follow_the_statistics():
    win = fun.verdict(0.008, 0.9, "tabular_q", "random")
    assert "The Spreadsheet beats Chaos Monkey" in win
    loss = fun.verdict(0.048, -0.8, "dqn", "tabular_q")
    assert "The Spreadsheet beats Neural Ned" in loss
    unclear = fun.verdict(0.21, 0.6, "dqn", "tabular_q")
    assert "jury is still out" in unclear and "beats" not in unclear
    assert "tied" in fun.verdict(1.0, 0.0, "dqn", "tabular_q")
    # lower is better for mistakes: more mistakes must not "win"
    more_mistakes = fun.verdict(0.01, 5.0, "random", "tabular_q", metric="mistakes")
    assert "The Spreadsheet beats Chaos Monkey" in more_mistakes


def test_fun_never_changes_training():
    def run(with_fun):
        t = Trainer(TrainConfig(episodes=15, seed=2))
        callback = (lambda r: fun.episode_line(r) and None) if with_fun else None
        t.train(callback=callback)
        if with_fun:
            fun.achievements(t.history)
            fun.hall_of_shame(t.mistakes)
        return [(r["steps"], r["mistakes"]) for r in t.history]

    assert run(True) == run(False)
