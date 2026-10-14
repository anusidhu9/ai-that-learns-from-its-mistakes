---
title: MIND - An AI That Learns From Its Mistakes
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: RL agents that learn from their own mistakes
---

# MIND: an AI that learns from its own mistakes

Reinforcement-learning agents explore grid worlds, get rewarded for reaching the goal and penalised for mistakes (walls, hazards, revisits, timeouts), and are then tested to see whether they actually improved.

**Try it:** open **🏋️ Train**, pick *Tabular Q* on *maze* and press **Train**. Then see where it went wrong in **🔥 Mistake history** and watch it play in **🎬 Watch it play**. The **📊 Results** tab has the multi-seed experiments with confidence intervals and significance tests.

This Space runs on a free CPU: tabular Q-learning trains in seconds, DQN in about a minute per 300 episodes. If several people train at once, requests wait in a queue.

Source code, full results and the final report: https://github.com/anusidhu9/ai-that-learns-from-its-mistakes
