"""Generate publication-quality figures from the reproduction results.

Reads JSON output from run_all.py, writes clean PDF figures directly into
source/ so main.tex can \\includegraphics them.

Figures produced:
  - training_N3.pdf, training_N5.pdf   (DQN cumulative-reward curves w/ CI)
  - perf_N3.pdf, perf_N5.pdf           (method × metric bar chart w/ std)
  - per_user_N3.pdf                    (per-user throughput + latency)
  - eps_ablation_N3.pdf, eps_ablation_N5.pdf  (ε-decay reward curves)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

RESULTS = Path(__file__).parent / "results"
OUT     = Path(__file__).parent.parent / "source"


# ---- palette (matches typical IEEE-paper conventions) --------------------
COLOR = {
    "DQN":            "#1f77b4",  # blue
    "Fixed":          "#2ca02c",  # green
    "Random":         "#d62728",  # red
    "Water-Filling":  "#7f7f7f",  # grey
}


def load(name: str):
    # Prefer final_* over main_* when they exist
    if name.startswith("main_"):
        final_alt = RESULTS / name.replace("main_", "final_")
        if final_alt.exists():
            with open(final_alt) as f:
                return json.load(f)
    with open(RESULTS / name) as f:
        return json.load(f)


# ---- Training curves -----------------------------------------------------

def plot_training(N: int) -> None:
    raw = load(f"main_N{N}_raw.json")
    curves = [r["train_curve"] for r in raw if r["method"] == "DQN"]
    curves = np.array(curves)  # (n_seeds, n_episodes)
    ep = np.arange(1, curves.shape[1] + 1)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)

    # smooth with a small moving average for readability
    win = 10
    kern = np.ones(win) / win
    mean_s = np.convolve(mean, kern, mode="valid")
    std_s  = np.convolve(std,  kern, mode="valid")
    ep_s   = ep[win - 1:]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ep_s, mean_s, color=COLOR["DQN"], lw=1.4, label=f"DQN (mean over {curves.shape[0]} seeds)")
    ax.fill_between(ep_s, mean_s - std_s, mean_s + std_s,
                    color=COLOR["DQN"], alpha=0.2, lw=0, label="±1 s.d.")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative episode reward")
    ax.set_title(f"DQN training curve (N={N})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = OUT / f"training_N{N}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ---- Method comparison bar chart -----------------------------------------

def plot_perf(N: int) -> None:
    summary = load(f"main_N{N}_summary.json")
    # Include all methods available in the summary; tabular Q may be N=3 only
    all_methods = ["DQN", "RainbowLite", "NeuralBandit", "TabularQ", "Fixed", "Random",
                   "Water-Filling-Discrete", "Water-Filling"]
    methods = [m for m in all_methods if m in summary]
    display_names = {
        "TabularQ": "Tabular Q",
        "RainbowLite": "DQN+D+D",
        "NeuralBandit": "N. bandit",
        "Water-Filling-Discrete": "WF (disc)",
        "Water-Filling": "WF (cont)",
    }
    labels = [display_names.get(m, m) for m in methods]
    metrics = [
        ("throughput_mbps",   "Throughput (bits/use)"),
        ("jain",              "Jain's fairness"),
        ("energy_efficiency", "Energy efficiency (bits/J)"),
    ]
    xs = np.arange(len(methods))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, (key, label) in enumerate(metrics):
        vals = [summary[m][key]["mean"] for m in methods]
        errs = [summary[m][key]["std"]  for m in methods]
        ax.bar(xs + (i - 1) * width, vals, width,
               yerr=errs, capsize=3,
               label=label, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("Metric value")
    ax.set_title(f"Method comparison (N={N})")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    fig.tight_layout()
    path = OUT / f"perf_N{N}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ---- Per-user chart ------------------------------------------------------

def plot_per_user(N: int = 3) -> None:
    raw = load(f"main_N{N}_raw.json")
    dqn_runs = [r for r in raw if r["method"] == "DQN"]
    thr = np.array([r["per_user_throughput"] for r in dqn_runs])  # (n_seeds, N)
    lat = np.array([r["per_user_latency"]    for r in dqn_runs])
    users = np.arange(1, N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    axes[0].bar(users, lat.mean(axis=0), yerr=lat.std(axis=0, ddof=1),
                capsize=4, color=COLOR["DQN"], edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(users)
    axes[0].set_xlabel("User index")
    axes[0].set_ylabel("Mean queue length (packets)")
    axes[0].set_title(f"Per-user latency proxy (N={N})")

    axes[1].bar(users, thr.mean(axis=0), yerr=thr.std(axis=0, ddof=1),
                capsize=4, color=COLOR["DQN"], edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(users)
    axes[1].set_xlabel("User index")
    axes[1].set_ylabel("Mean per-user rate (bits/use)")
    axes[1].set_title(f"Per-user throughput (N={N})")

    fig.tight_layout()
    path = OUT / f"per_user_N{N}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ---- Epsilon-decay ablation ----------------------------------------------

def plot_eps(N: int) -> None:
    data = load(f"eps_ablation_N{N}.json")
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    win = 20
    kern = np.ones(win) / win
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
    for color, d in zip(colors, [0.99, 0.98, 0.95, 0.90]):
        m = np.array(data[str(d)]["mean_per_episode"])
        s = np.array(data[str(d)]["std_per_episode"])
        m_s = np.convolve(m, kern, mode="valid")
        s_s = np.convolve(s, kern, mode="valid")
        ep = np.arange(1, len(m) + 1)[win - 1:]
        ax.plot(ep, m_s, label=f"decay={d}", color=color, lw=1.4)
        ax.fill_between(ep, m_s - s_s, m_s + s_s, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative episode reward (smoothed)")
    ax.set_title(f"Effect of $\\epsilon$-decay schedule (N={N})")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    path = OUT / f"eps_ablation_N{N}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    for N in [3, 5]:
        plot_training(N)
        plot_perf(N)
        plot_eps(N)
    plot_per_user(3)
    plot_per_user(5)


if __name__ == "__main__":
    main()
