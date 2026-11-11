"""Check before the ppo_replication run (from the project root, PYTHONPATH=.): 300 episodes
of ppo_generalization DQN seed 700 and PPO seed 700 reproduce the first 300 rows of their
saved episodes.csv exactly (compared as text). Prints no success values of the new study."""
import tempfile
import time
from pathlib import Path

from src.evaluation import EXPERIMENTS
from src.training import EPISODES_FILE, Trainer

for agent in ("dqn", "ppo"):
    cfg = EXPERIMENTS["ppo_generalization"].train_config(agent, 700)
    cfg.episodes = 300
    with tempfile.TemporaryDirectory() as d:
        start = time.time()
        Trainer(cfg, d).train()
        new = (Path(d) / EPISODES_FILE).read_text().splitlines()
    saved = Path(f"runs/experiments_ppo/ppo_generalization/{agent}_s700/episodes.csv") \
        .read_text().splitlines()[:301]
    print(f"{agent} reproduces ppo_generalization {agent} s700 (300 episodes): {new == saved} "
          f"({len(new)} lines, {time.time() - start:.0f}s)", flush=True)
