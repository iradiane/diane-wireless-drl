# diane-wireless-drl

[![arXiv](https://img.shields.io/badge/arXiv-2601.04842-b31b1b.svg)](https://arxiv.org/abs/2601.04842)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

**Companion code for the paper** *"Intelligent resource allocation in wireless
networks via deep reinforcement learning"* by Marie Diane Iradukunda (AIMS
Rwanda & AIRINA Labs, Bénin), Chabi F. Élégbédé (UNSTIM, Bénin), and Yaé Ulrich
Gaba (Sefako Makgatho Health Sciences University, South Africa & AIRINA Labs,
Bénin), July 2026.

**Preprint:** [arXiv:2601.04842](https://arxiv.org/abs/2601.04842)

This repository is **code only** — it exists to reproduce the numbers, tables,
and figures reported in the paper. It contains the environment, all classical
and learned agents, the experiment drivers, and the figure-generation scripts.
The LaTeX source of the paper is not included; only the outputs the code
produces (figures + JSON summaries) are.

## Cite as

```bibtex
@article{iradukunda2026wirelessdrl,
  author        = {Iradukunda, Marie Diane and \'El\'egb\'ed\'e, Chabi F. and Gaba, Ya\'e Ulrich},
  title         = {Intelligent resource allocation in wireless networks via deep reinforcement learning},
  year          = {2026},
  eprint        = {2601.04842},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NI},
  url           = {https://arxiv.org/abs/2601.04842},
  note          = {Code: \url{https://github.com/iradiane/diane-wireless-drl}}
}
```

Related work: the source master's thesis
[Reinforcement Learning in Communication Networks: Optimization of Wireless Resource Allocation](https://github.com/iradiane/DQN-Implementation)
(Iradukunda, AIMS Rwanda, 2025).

## Layout

```
diane-wireless-drl/
├── README.md
├── LICENSE                             ← MIT
├── .gitignore
│
└── reproduction/                       ← Python 3.13 stack
    ├── dqn_wireless.py                 ← env + all agents (see below)
    ├── requirements.txt                ← pinned deps
    │
    ├── make_figures.py                 ← generates paper Figs 2–5
    ├── make_analysis.py                ← generates Figs 6–8 + Tables 6, 7, 8
    ├── rliable_analysis.py             ← rliable-style aggregate statistics
    ├── update_paper_numbers.py         ← writes numeric values into paper macros
    │
    ├── run_final_parallel.py           ← main comparison, N=3 & N=5, 10 seeds
    ├── run_pathB.py                    ← Rainbow-lite + Neural Bandit
    ├── run_followup.py                 ← WF-Discrete + ε-decay ablation
    ├── run_budget_extension.py         ← N=5 with 4× training budget
    ├── run_rayleigh.py                 ← Rayleigh + interference, 10 seeds + WMMSE
    ├── run_iql_regnn.py                ← MARL (IQL) + REGNN-lite scaling
    ├── run_multicell.py                ← K=7 multi-cell environment
    ├── run_n10_topup.py                ← N=10 top-up runner
    ├── run_topup_combined.py           ← combined N=10 + multi-cell top-up
    ├── salvage_iql.py                  ← rescue partial results from a killed log
    │
    └── results/                        ← cached JSONs of every experiment
```

## Reproduce the figures

Requirements: Python 3.13, `pip install -r reproduction/requirements.txt`.
CPU-only is fine; no GPU needed.

### Fast path (~30 seconds, uses cached JSONs)

Reproduces every figure in the paper from the cached experiment output:

```bash
cd reproduction/
python make_figures.py            # paper Figs 2–5
python make_analysis.py           # paper Figs 6–8 (extended analysis) + Tables 6, 7, 8
python rliable_analysis.py        # rliable-style aggregate figures
```

Output PDFs are written next to the scripts (paths configurable via
environment variables — see the top of each script).

### Full re-run from scratch (~7 h wall time, 6 CPU cores)

Regenerates every JSON in `results/` and then every figure:

```bash
cd reproduction/
python run_final_parallel.py       # main comparison (Table 4)         ~90 min
python run_pathB.py                # Rainbow-lite + Neural Bandit      ~95 min
python run_followup.py             # WF-Discrete + ε-decay (Table 5)   ~30 min
python run_budget_extension.py     # N=5 4× budget                     ~55 min
python run_rayleigh.py             # Rayleigh + interference 10 seeds  ~60 min
python run_iql_regnn.py            # MARL + GNN scaling                ~2 h
python run_multicell.py            # multi-cell K=7 environment        ~30 min

python make_figures.py
python make_analysis.py
python rliable_analysis.py
```

Smaller runs when wall time is tight:

```bash
python run_topup_combined.py    # 3-seed learned x 150 eps at N=10 + K=7 (~40 min)
python run_n10_topup.py         # 5-seed N=10 top-up (~1 h)
python salvage_iql.py           # rescue results from a killed run's log
```

All scripts overwrite outputs idempotently and use fixed random seeds; numbers
reproduce exactly on re-run.

## What `dqn_wireless.py` contains

Environment, agents, and utilities in a single ~1300-line file:

| Symbol | Role |
|---|---|
| `WirelessEnv` | Single-cell downlink environment. Supports `channel_model ∈ {"uniform", "rayleigh"}` and `interference ∈ {False, True}`. |
| `MultiCellEnv` | $K$-cell environment: path loss, Rayleigh fading, inter-cell interference. |
| `EnvConfig`, `MultiCellConfig`, `DQNConfig`, `RunResult` | Config dataclasses. |
| `DQNAgent` | Vanilla DQN. `use_double`, `use_dueling` turn it into Rainbow-lite. |
| `DuelingQNet` | Dueling architecture (Wang et al., 2016). |
| `NeuralBanditAgent` | Neural regressor over actions; no bootstrapping, no target net (Riquelme et al., 2018). |
| `TabularQAgent` | Q-learning on a discretised channel state. |
| `IndependentQLAgent` | Per-user DQN, own-channel observation, shared reward (Tan, 1993; Nasir & Guo, 2019). |
| `REGNNAgent`, `GraphQNet` | Message-passing GNN policy (Eisen & Ribeiro, 2020; Shen et al., 2020, discrete-action distillation). |
| `act_random`, `act_fixed`, `act_waterfilling`, `act_wmmse` | Non-learning baselines. |
| `wmmse_powers`, `wmmse_multicell` | WMMSE fixed-point iteration (Shi et al., 2011), scalar and multi-cell variants. |
| `train_dqn`, `train_tabular_q`, `train_neural_bandit`, `train_iql`, `train_regnn` (+ multi-cell variants) | Training loops. |
| `evaluate_policy`, `evaluate_multicell_policy` | Unified evaluation (20 episodes × 100 steps). |
| `jain`, `summarize`, `dump_json` | Utilities. |

## Corresponding author

`mariediane.iradukunda@aims.ac.rw`
