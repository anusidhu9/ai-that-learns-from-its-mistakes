"""The lighter side of MIND: commentary, nicknames, achievements, Hall of Shame.

Everything here only *reads* episode summaries, mistake logs and test results
and turns them into text. Nothing changes the environment, the agents or any
metric, so the jokes can never affect an experiment.

Choices are deterministic (seeded by the inputs), so the same episode always
gets the same line: reproducible research, reproducible puns. All text is
plain ASCII so it prints on any console, including Windows.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

from src.agents import DQN_NICKNAMES, RandomAgent, TabularQAgent
from src.evaluation import format_p
from src.training import mistake_hotspots

# Nicknames live on the agent classes (Stages 2-3); collected here for display.
NICKNAMES: Dict[str, str] = {
    RandomAgent.name: RandomAgent.nickname,
    TabularQAgent.name: TabularQAgent.nickname,
    **DQN_NICKNAMES,
}

TAGLINES: Dict[str, str] = {
    "random": "Never learns, never regrets.",
    "tabular_q": "Remembers every square it has ever been embarrassed on.",
    "dqn": "A neural network that learned walls are load-bearing.",
    "dqn_per": "Replays its most painful memories first. Relatable.",
    "dqn_mistake": "Writes down every wall it walked into, then rereads the list.",
    "dqn_per_mistake": "Keeps a diary of its mistakes and rereads the worst pages first.",
    "dqn_dueling": "Asks \"is this a good place to be?\" before \"which way now?\"",
    "dqn_nstep": "Counts the next three rewards before trusting its own guess.",
    "dqn_dueling_nstep": "Two heads, three steps of patience.",
}

_EVENT_LINES: Dict[str, Sequence[str]] = {
    "collision": (
        "Bonk! The wall at {pos} wins again.",
        "Walked into the wall at {pos}. The wall did not move.",
        "Wall at {pos}: 1, Agent: 0.",
        "Tried to phase through {pos}. Physics said no.",
    ),
    "hazard": (
        "Stepped in the lava at {pos}. That's -0.5 and some dignity.",
        "Hazard at {pos} located. Using its feet.",
        "The hazard at {pos} was clearly marked. Clearly.",
    ),
    "revisit": (
        "Back at {pos} again. Did we leave the stove on?",
        "{pos} again. Sightseeing is not the objective.",
        "Revisited {pos}. Nostalgia costs -0.02.",
    ),
    "timeout": (
        "Out of time. The maze remains undefeated.",
        "Timeout. Somewhere, the goal waits patiently.",
        "Ran out of steps. Tomorrow is another episode.",
    ),
    "goal": (
        "GOAL! Somebody frame this episode.",
        "Reached the goal. Act natural, like that was the plan.",
        "Goal reached. The maze has been informed.",
    ),
}


def _pick(options: Sequence[str], *key) -> str:
    digest = hashlib.sha256(repr(key).encode()).digest()
    return options[digest[0] % len(options)]


def _fmt_pos(pos) -> str:
    return f"({pos[0]}, {pos[1]})"


def event_line(event: str, pos=None, key=0) -> str:
    """One-liner for a single step event (collision, hazard, revisit, timeout, goal)."""
    options = _EVENT_LINES.get(event)
    if options is None:
        return ""
    return _pick(options, event, tuple(pos) if pos is not None else None, key).format(
        pos=_fmt_pos(pos) if pos is not None else "?")


def episode_line(summary: dict) -> str:
    """One-liner describing a whole episode from its summary."""
    key = (summary.get("episode"), summary.get("steps"), summary.get("mistakes"))
    steps, mistakes = summary["steps"], summary["mistakes"]
    opt = summary.get("optimal_length")
    if summary["success"] and mistakes == 0 and opt and steps == opt:
        return _pick(("Flawless: shortest path, zero mistakes. Is this even the same agent?",
                      f"Speedrun! {steps} steps, the theoretical minimum.",
                      "Perfect run. The maze would like a rematch."), *key)
    if summary["success"]:
        return _pick((f"Made it in {steps} steps with {mistakes} mistakes along the way.",
                      f"Success, eventually. {mistakes} mistakes, but who's counting? (We are.)",
                      f"Reached the goal after {steps} steps of character building."), *key)
    if summary.get("hazards", 0) and not summary.get("timeout"):
        return _pick(("Chose the lava over the long walk. Bold strategy.",
                      "Ended it all on a hazard. The maze did not even have to try."), *key)
    return _pick((f"{mistakes} mistakes in {steps} steps. A learning experience.",
                  "Wandered until the clock ran out. Exploration, technically.",
                  f"Timeout after {mistakes} mistakes. Epsilon made me do it."), *key)


def mood(summary: dict) -> str:
    """A short emoji-free mood label for an episode."""
    if summary["success"] and summary["mistakes"] == 0:
        return "smug"
    if summary["success"]:
        return "relieved"
    if summary.get("hazards", 0):
        return "singed"
    return "lost"


# ---------------------------------------------------------------------------
# Achievements
# ---------------------------------------------------------------------------

ACHIEVEMENTS: Dict[str, str] = {
    "first_goal": "First Contact: reached the goal for the first time",
    "clean_run": "Spotless: an episode with zero mistakes",
    "speedrunner": "Speedrunner: matched the shortest possible path",
    "wall_whisperer": "Wall Whisperer: 25 episodes in a row without a collision",
    "lava_enthusiast": "Lava Enthusiast: 5+ hazards in a single episode",
    "groundhog_day": "Groundhog Day: 50+ revisits in a single episode",
    "comeback_kid": "Comeback Kid: succeeded right after a 90+ mistake episode",
}


def achievements(history: List[dict]) -> List[dict]:
    """Achievements unlocked over a training history, in the order earned.

    Returns [{"key", "title", "episode"}]. Purely descriptive; computed from
    the logged rows only.
    """
    earned: Dict[str, int] = {}
    streak = 0
    for i, row in enumerate(history):
        ep = row.get("episode", i)
        opt = row.get("optimal_length")
        checks = {
            "first_goal": row["success"],
            "clean_run": row["mistakes"] == 0,
            "speedrunner": bool(row["success"] and opt and row["steps"] == opt),
            "lava_enthusiast": row.get("hazards", 0) >= 5,
            "groundhog_day": row.get("revisits", 0) >= 50,
            "comeback_kid": bool(row["success"] and i > 0 and history[i - 1]["mistakes"] >= 90),
        }
        streak = streak + 1 if row.get("collisions", 0) == 0 else 0
        checks["wall_whisperer"] = streak >= 25
        for key, ok in checks.items():
            if ok and key not in earned:
                earned[key] = ep
    return [{"key": k, "title": ACHIEVEMENTS[k], "episode": e}
            for k, e in sorted(earned.items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------
# Hall of Shame
# ---------------------------------------------------------------------------

_SHAME_TITLES = {
    "collision": ("Most Loyal Wall", "Favourite Wall to Hug", "Wall of Fame"),
    "revisit": ("Favourite Holiday Spot", "Most Visited Landmark", "Comfort Square"),
    "hazard": ("Preferred Lava Pool", "Hot Tub of Regret", "Signature Hazard"),
    "timeout": ("Where Time Stood Still", "Final Resting Square", "The Waiting Room"),
}


def hall_of_shame(mistakes: List[dict], top: int = 5,
                  episodes: Optional[range] = None) -> List[dict]:
    """The agent's most repeated mistakes, each with a ceremonial title.

    Built on training.mistake_hotspots, so the counts are the real ones.
    """
    rows = []
    for rank, spot in enumerate(mistake_hotspots(mistakes, top=top, episodes=episodes), 1):
        titles = _SHAME_TITLES.get(spot["event"], ("Mystery Mistake",))
        rows.append({"rank": rank, "award": _pick(titles, spot["event"], spot["pos"]),
                     "event": spot["event"], "pos": spot["pos"], "count": spot["count"]})
    return rows


def format_hall_of_shame(rows: List[dict]) -> str:
    if not rows:
        return "Hall of Shame: empty. Suspiciously well behaved."
    lines = ["HALL OF SHAME"]
    for r in rows:
        lines.append(f"  #{r['rank']} {r['award']:<24} {r['event']:<9} at "
                     f"{_fmt_pos(r['pos'])}  x{r['count']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training progress and experiment verdicts
# ---------------------------------------------------------------------------

def progress_quip(success_rate: float, mistakes: float, episode: int) -> str:
    """Comment on a block of training episodes (e.g. the last 50)."""
    key = (episode, round(success_rate, 2))
    if success_rate >= 0.95 and mistakes < 2:
        return _pick(("Basically a professional now.", "The maze has become a formality.",
                      "Showing off at this point."), *key)
    if success_rate >= 0.5:
        return _pick(("Getting the hang of it.", "Mistakes falling, confidence rising.",
                      "Starting to look like it knows what a goal is."), *key)
    if success_rate > 0:
        return _pick(("Occasionally lucky, mostly lost.", "There are signs of hope.",
                      "Found the goal once or twice. Probably by accident."), *key)
    return _pick(("Still introducing itself to every wall.", "Learning, allegedly.",
                  "Pure vibes, no plan."), *key)


LOWER_IS_BETTER = {"mistakes", "hazards", "steps", "collisions", "revisits", "timeout"}


def verdict(p_value: float, difference: float, a: str, b: str, metric: str = "success") -> str:
    """Translate a permutation test into an honest one-liner.

    The joke always matches the statistics: no celebrating a p-value that
    does not support a difference.
    """
    na, nb = NICKNAMES.get(a, a), NICKNAMES.get(b, b)
    if p_value < 0.05 and difference != 0:
        a_higher = difference > 0
        a_better = a_higher != (metric in LOWER_IS_BETTER)
        winner, loser = (na, nb) if a_better else (nb, na)
        return f"{winner} beats {loser} on {metric} ({format_p(p_value)}). Not a fluke."
    if difference == 0:
        return f"{na} and {nb} tied on {metric}. Shake hands."
    return (f"{na} vs {nb} on {metric}: difference {difference:+.2f}, {format_p(p_value)}. "
            f"The jury is still out; more seeds needed before anyone gloats.")
