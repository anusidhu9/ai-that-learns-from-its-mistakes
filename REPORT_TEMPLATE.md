# Report template

A fill-in template for writing up a new MIND experiment, for example a rerun with more seeds, new hyperparameters or a new environment. [REPORT.md](REPORT.md) is the completed report for the experiments in `results/`; use it as a worked example.

Each section says where its numbers come from. **Fill in only numbers you have produced**, and keep every claim tied to a file in `results/`.

---

# <Title>

## 1. Summary

<3–5 sentences: the question, what was compared, and the main findings, each with its effect size and p-value.>

## 2. Question

<What are you testing, and why? For example: "Does prioritized replay reduce greedy mistakes on the hazard maze with 10 seeds?">

## 3. Setup

| Setting | Value | Source |
|---|---|---|
| Environment preset and overrides | <e.g. `maze_hazards`, `hazard_terminal: false`> | `results/<name>/experiment.json` → `env_preset`, `env_overrides` |
| Agents | <e.g. random, tabular_q, dqn, dqn_per> | `experiment.json` → `agents` |
| Seeds | <e.g. 0–9> | `experiment.json` → `seeds` |
| Training episodes per run | <e.g. 400> | `experiment.json` → `episodes` |
| Exploration decay (episodes) | <e.g. 300> | `experiment.json` → `decay_episodes` |
| Agent overrides | <e.g. `{"dqn": {"lr": 5e-4}}`, or none> | `experiment.json` → `agent_kwargs` |
| Unseen layouts | <e.g. 1000–1049, or none> | `experiment.json` → `unseen_layouts` |

Reward table and algorithm details: unchanged from [REPORT.md](REPORT.md), Sections 3–4, unless you changed them. If you did, say exactly what changed.

**How to run it:** add an `Experiment(...)` entry to `EXPERIMENTS` in `src/evaluation.py`, then:

```bash
python -m src.evaluation --experiment <name>
```

## 4. Evaluation protocol

Unchanged from [REPORT.md](REPORT.md), Section 5 (greedy evaluation, mean ± 95% t-interval across seeds, exact permutation tests), unless stated here.

<Note anything different, e.g. a different number of evaluation episodes.>

## 5. Results

### Final greedy performance

Copy from `results/<name>/summary.csv` (columns `<metric>_mean`, `<metric>_ci`), or from the printout at the end of the experiment command.

| Agent | Split | Success | Path efficiency | Steps | Mistakes | Hazards |
|---|---|---|---|---|---|---|
| <agent> | <train / unseen> | <mean ± ci> | <mean ± ci> | <mean ± ci> | <mean ± ci> | <mean ± ci> |

### Per-seed values

From `results/<name>/results.csv` (one row per agent, seed and split). List them for any comparison you discuss, because means hide outliers.

<e.g. "DQN success per seed: 1.0, 1.0, 0.0, 1.0, 1.0">

### Comparisons

Printed by the experiment command under "Success-rate differences", or computed with:

```python
from src.evaluation import load_experiment, compare_agents
r = load_experiment("results/<name>")["results"]
compare_agents(r, "<agent a>", "<agent b>", metric="success", split="train")
```

| Comparison | Metric | Difference | p-value | Established? |
|---|---|---|---|---|
| <a − b> | <success> | <+0.00> | <0.000> | <yes if p < 0.05, otherwise no> |

Remember the smallest attainable p-value: 0.008 with 5 seeds per agent, 0.10 with 3. A difference that cannot reach significance with your seed count is "not established", however large it looks.

### Figures

Generate with `src/visualization.py`, for example:

```python
from src.evaluation import load_experiment
from src.visualization import experiment_curves_figure, experiment_bars_figure
exp = load_experiment("results/<name>")
experiment_curves_figure(exp["curves"], "success").savefig("docs/figures/<name>_success.png", dpi=110)
experiment_bars_figure(exp["summary"], "success").savefig("docs/figures/<name>_bars.png", dpi=110)
```

![<description>](docs/figures/<name>_success.png)

## 6. Discussion

<What the results mean. Check, as REPORT.md did:>
- <Is a lower mistake count matched by a higher success rate, or did the agent find a shortcut, such as ending episodes early?>
- <Was learning stable across seeds, or did some runs fail? Look at the per-seed values and `curves.csv`.>
- <Does training-time behaviour (`episodes.csv` in each run directory) agree with greedy evaluation?>

## 7. Limitations

<Seed count, task size, tuning, training budget, and anything specific to this experiment.>

## 8. Reproducibility

```bash
python -m src.evaluation --experiment <name>
```

Software versions: <`python --version`, and `pip show torch numpy gradio` for the others>. Hardware: <CPU or Colab>.

Saved outputs: `results/<name>/` (`summary.csv`, `results.csv`, `curves.csv`, `experiment.json`, `run.log`).
