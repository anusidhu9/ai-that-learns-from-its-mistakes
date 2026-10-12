"""Training loops for MIND.

Stage 2 provides the core episode loop. Stage 4 adds persistent logs,
checkpoints, and resumption on top of it.
"""
from __future__ import annotations

from typing import List, Optional

from src.envs import GridWorld


def run_episode(env: GridWorld, agent, train: bool = True, greedy: bool = False,
                seed: Optional[int] = None, layout_seed: Optional[int] = None) -> dict:
    """Play one episode and return the environment's episode summary.

    train=True calls agent.update after every step and agent.end_episode at the
    end. greedy=True disables exploration (use with train=False for evaluation).
    The summary also includes the agent's epsilon during the episode.
    """
    obs, _ = env.reset(seed=seed, layout_seed=layout_seed)
    epsilon = 0.0 if greedy else agent.epsilon
    done = False
    while not done:
        action = agent.act(obs, greedy=greedy)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if train:
            agent.update(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
        done = terminated or truncated
    if train:
        agent.end_episode()
    summary = dict(info["episode"])
    summary["epsilon"] = epsilon
    summary["mistake_log"] = list(env.mistake_log)
    return summary


def train_agent(env: GridWorld, agent, episodes: int,
                seed: Optional[int] = None) -> List[dict]:
    """Train for a number of episodes; returns one summary per episode.

    `seed` seeds the environment's RNG once at the start, so a run is
    reproducible given the same agent seed.
    """
    history = []
    for i in range(episodes):
        summary = run_episode(env, agent, train=True, seed=seed if i == 0 else None)
        summary["episode"] = agent.episodes - 1
        history.append(summary)
    return history
