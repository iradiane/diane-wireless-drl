"""Analysis figures and tables for the paper.

Produces:
  - training_all_N3.pdf   : learning curves for all learned methods overlaid at N=3
  - training_all_N5.pdf   : same at N=5
  - seed_cdf_N3.pdf       : empirical CDF of per-seed throughput at N=3
  - seed_cdf_N5.pdf       : same at N=5
  - pct_of_wf_N3.pdf      : throughput as % of continuous WF upper bound, per method
  - pct_of_wf_N5.pdf      : same at N=5
  - stats_table.tex       : pairwise Wilcoxon p-values across methods (LaTeX table)
  - effect_sizes.tex      : Cohen's d and Cliff's delta between key pairs
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    "legend.fontsize": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent.parent / "source"

METHOD_COLOR = {
    "DQN":              "#d62728",  # red — losing baseline
    "RainbowLite":      "#ff7f0e",  # orange — middle
    "NeuralBandit":     "#1f77b4",  # blue — winner
    "TabularQ":         "#2ca02c",  # green — stable classical
    "Fixed":            "#7f7f7f",  # grey — heuristic
    "Random":           "#c5b0d5",  # light purple — floor
    "Water-Filling":    "#9467bd",  # purple — ceiling
    "Water-Filling-Discrete": "#8c564b",  # brown
    "WMMSE":            "#17becf",  # cyan — classical iterative
    "IQL":              "#e377c2",  # pink — MARL
    "REGNN":            "#bcbd22",  # olive — GNN
    "WF-multi":         "#9467bd",
    "WMMSE-multi":      "#17becf",
}
DISPLAY_NAME = {
    "DQN": "Vanilla DQN",
    "RainbowLite": "DQN + D + D",
    "NeuralBandit": "Neural bandit",
    "TabularQ": "Tabular Q",
    "Fixed": "Fixed (2 W)",
    "Random": "Random",
    "Water-Filling": "WF (cont.)",
    "Water-Filling-Discrete": "WF (disc.)",
    "WMMSE": "WMMSE",
    "IQL": "IQL (MARL)",
    "REGNN": "REGNN-lite",
    "WF-multi": "WF-multi",
    "WMMSE-multi": "WMMSE-multi",
}


def load_raw(N: int) -> list[dict]:
    return json.loads((RESULTS / f"final_N{N}_raw.json").read_text())


def load_summary(N: int) -> dict:
    return json.loads((RESULTS / f"final_N{N}_summary.json").read_text())


def _smooth(x, win=15):
    kern = np.ones(win) / win
    return np.convolve(x, kern, mode="valid")


def plot_training_overlay(N: int) -> None:
    """Overlay learning curves for all methods that have train_curve."""
    raw = load_raw(N)
    # Group train_curves by method
    by_method: dict[str, list[list[float]]] = {}
    for r in raw:
        if r.get("train_curve"):
            by_method.setdefault(r["method"], []).append(r["train_curve"])
    if not by_method:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for method, curves in by_method.items():
        # Curves may have different lengths (e.g. Tabular Q has n_episodes rewards);
        # trim to the minimum common length
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)
        win = 15
        if len(mean) > win:
            m_s = _smooth(mean, win)
            s_s = _smooth(std, win)
            ep = np.arange(1, len(mean) + 1)[win - 1:]
        else:
            m_s, s_s, ep = mean, std, np.arange(1, len(mean) + 1)
        color = METHOD_COLOR.get(method, "black")
        ax.plot(ep, m_s, color=color, lw=1.5, label=DISPLAY_NAME.get(method, method))
        ax.fill_between(ep, m_s - s_s, m_s + s_s, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative episode reward (smoothed)")
    ax.set_title(f"Learning curves, all methods ($N={N}$)")
    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    p = OUT / f"training_all_N{N}.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p}")


def plot_seed_cdf(N: int) -> None:
    """Empirical CDF of per-seed throughput for each method."""
    raw = load_raw(N)
    by_method: dict[str, list[float]] = {}
    for r in raw:
        by_method.setdefault(r["method"], []).append(r["throughput_mbps"])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    order = ["DQN", "RainbowLite", "NeuralBandit", "TabularQ", "Fixed", "Random",
             "Water-Filling-Discrete", "Water-Filling"]
    for method in order:
        if method not in by_method:
            continue
        vals = np.sort(by_method[method])
        ecdf_y = np.arange(1, len(vals) + 1) / len(vals)
        color = METHOD_COLOR.get(method, "black")
        ax.step(vals, ecdf_y, where="post", color=color, lw=1.5,
                label=DISPLAY_NAME.get(method, method))
    ax.set_xlabel("Throughput (bits/use)")
    ax.set_ylabel("Empirical CDF over seeds")
    ax.set_title(f"Per-seed throughput distribution ($N={N}$)")
    ax.legend(loc="lower right", ncol=1, fontsize=7)
    fig.tight_layout()
    p = OUT / f"seed_cdf_N{N}.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p}")


def plot_pct_of_wf(N: int) -> None:
    """Throughput as % of continuous WF upper bound, ranked."""
    summary = load_summary(N)
    if "Water-Filling" not in summary:
        return
    wf = summary["Water-Filling"]["throughput_mbps"]["mean"]
    order = ["Random", "DQN", "TabularQ", "Fixed", "RainbowLite",
             "Water-Filling-Discrete", "NeuralBandit", "Water-Filling"]
    methods = [m for m in order if m in summary]
    pcts = [100 * summary[m]["throughput_mbps"]["mean"] / wf for m in methods]
    stds_pct = [100 * summary[m]["throughput_mbps"]["std"] / wf for m in methods]
    labels = [DISPLAY_NAME.get(m, m) for m in methods]
    colors = [METHOD_COLOR.get(m, "gray") for m in methods]
    ys = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(ys, pcts, xerr=stds_pct, color=colors, edgecolor="black", linewidth=0.5,
            capsize=3)
    ax.axvline(100, color="black", ls=":", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Throughput / continuous WF (%)")
    ax.set_title(f"Distance from theoretical optimum ($N={N}$)")
    ax.set_xlim(left=40, right=110)
    for y, p, s in zip(ys, pcts, stds_pct):
        ax.text(p + s + 1, y, f"{p:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    p = OUT / f"pct_of_wf_N{N}.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p}")


def stats_table() -> None:
    """Pairwise Wilcoxon signed-rank at N=3 across all methods with matched seeds."""
    raw = load_raw(3)
    by = {}
    for r in raw:
        by.setdefault(r["method"], {})[r["seed"]] = r["throughput_mbps"]
    methods = ["DQN", "RainbowLite", "NeuralBandit", "TabularQ", "Fixed", "Random",
               "Water-Filling-Discrete", "Water-Filling"]
    methods = [m for m in methods if m in by]
    # Only compare methods that share all 10 seeds
    common = set.intersection(*(set(by[m].keys()) for m in methods))
    common = sorted(common)

    header = [""] + [DISPLAY_NAME.get(m, m).replace("&","\\&") for m in methods]
    lines = ["\\begin{tabular}{l" + "c" * len(methods) + "}", "\\hline"]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")
    for m1 in methods:
        row = [DISPLAY_NAME.get(m1, m1).replace("&","\\&")]
        for m2 in methods:
            if m1 == m2:
                row.append("---")
            else:
                x = np.array([by[m1][s] for s in common])
                y = np.array([by[m2][s] for s in common])
                try:
                    _, p = stats.wilcoxon(x - y)
                    if p < 0.001:
                        row.append("$<10^{-3}$")
                    elif p < 0.01:
                        row.append(f"{p:.3f}")
                    else:
                        row.append(f"{p:.2f}")
                except Exception:
                    row.append("N/A")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    tex = "\n".join(lines)
    p = OUT / "stats_table.tex"
    p.write_text(tex)
    print(f"wrote {p}")


def effect_sizes_table() -> None:
    """Cohen's d and Cliff's delta between the key head-to-head pairs at N=3."""
    raw = load_raw(3)
    by = {}
    for r in raw:
        by.setdefault(r["method"], []).append(r["throughput_mbps"])

    def cohens_d(a, b):
        na, nb = len(a), len(b)
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        sp = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
        return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0

    def cliffs_delta(a, b):
        a, b = np.asarray(a), np.asarray(b)
        diff_sign = np.sign(a[:, None] - b[None, :])
        return float(diff_sign.mean())

    pairs = [
        ("NeuralBandit", "DQN"),
        ("NeuralBandit", "RainbowLite"),
        ("NeuralBandit", "Fixed"),
        ("NeuralBandit", "Water-Filling-Discrete"),
        ("RainbowLite", "DQN"),
        ("TabularQ", "DQN"),
        ("Fixed", "DQN"),
    ]
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Pair (A vs B)} & \\textbf{Cohen's $d$} & \\textbf{Cliff's $\\delta$} & \\textbf{Interpretation} \\\\",
        "\\hline",
    ]

    def interp(d):
        ad = abs(d)
        if ad < 0.147: return "negligible"
        if ad < 0.33: return "small"
        if ad < 0.474: return "medium"
        return "large"

    for a, b in pairs:
        if a not in by or b not in by:
            continue
        d = cohens_d(by[a], by[b])
        cd = cliffs_delta(by[a], by[b])
        pair_label = f"{DISPLAY_NAME.get(a,a)} vs.\\ {DISPLAY_NAME.get(b,b)}"
        lines.append(
            f"{pair_label} & ${d:+.2f}$ & ${cd:+.2f}$ & {interp(cd)} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    p = OUT / "effect_sizes.tex"
    p.write_text("\n".join(lines))
    print(f"wrote {p}")


def rayleigh_summary_table() -> None:
    """LaTeX table for Rayleigh + interference results, if data available.
    Now includes WMMSE row when present."""
    p = RESULTS / "rayleigh_interference_summary.json"
    if not p.exists():
        return
    summary = json.loads(p.read_text())
    order = ["Random", "DQN", "TabularQ", "Fixed", "RainbowLite",
             "WMMSE", "NeuralBandit", "Water-Filling"]
    lines = [
        "\\begin{tabular}{|l|c|c|c|}",
        "\\hline",
        "\\textbf{Method} & \\textbf{Throughput} & \\textbf{Fairness (Jain)} & \\textbf{Energy efficiency} \\\\",
        "\\hline",
    ]
    for m in order:
        if m not in summary:
            continue
        s = summary[m]
        lines.append(
            f"{DISPLAY_NAME.get(m, m)} & "
            f"${s['throughput_mbps']['mean']:.3f} \\pm {s['throughput_mbps']['std']:.3f}$ & "
            f"${s['jain']['mean']:.3f} \\pm {s['jain']['std']:.3f}$ & "
            f"${s['energy_efficiency']['mean']:.3f} \\pm {s['energy_efficiency']['std']:.3f}$ \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    outp = OUT / "rayleigh_table.tex"
    outp.write_text("\n".join(lines))
    print(f"wrote {outp}")


def iql_regnn_scaling_table() -> None:
    """LaTeX table for IQL/REGNN scaling across N in {3, 5, 10}."""
    p = RESULTS / "iql_regnn_summary.json"
    if not p.exists():
        return
    summary = json.loads(p.read_text())
    order = ["Random", "Fixed", "Water-Filling", "WMMSE",
             "DQN", "RainbowLite", "NeuralBandit", "TabularQ", "IQL", "REGNN"]

    all_Ns = sorted(int(k) for k in summary.keys())
    lines = ["\\begin{tabular}{|l|" + "c|" * len(all_Ns) + "}", "\\hline"]
    header = "\\textbf{Method} & " + " & ".join(
        [f"$N={N}$" for N in all_Ns]
    ) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for m in order:
        row_bits = [DISPLAY_NAME.get(m, m)]
        any_present = False
        for N in all_Ns:
            key = str(N)
            if m in summary[key]:
                any_present = True
                s = summary[key][m]["throughput_mbps"]
                row_bits.append(f"${s['mean']:.3f} \\pm {s['std']:.3f}$")
            else:
                row_bits.append("---")
        if any_present:
            lines.append(" & ".join(row_bits) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    outp = OUT / "iql_regnn_scaling_table.tex"
    outp.write_text("\n".join(lines))
    print(f"wrote {outp}")


def multicell_summary_table() -> None:
    """LaTeX table for K=7 multi-cell results."""
    p = RESULTS / "multicell_summary.json"
    if not p.exists():
        return
    summary = json.loads(p.read_text())
    order = ["Random", "Fixed", "WF-multi", "WMMSE-multi",
             "DQN", "NeuralBandit", "IQL", "REGNN"]
    lines = [
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Method} & \\textbf{Throughput} & \\textbf{Fairness (Jain)} "
        "& \\textbf{Energy eff.} & \\textbf{Avg. latency} \\\\",
        "\\hline",
    ]
    for m in order:
        if m not in summary:
            continue
        s = summary[m]
        lines.append(
            f"{DISPLAY_NAME.get(m, m)} & "
            f"${s['throughput_mbps']['mean']:.3f} \\pm {s['throughput_mbps']['std']:.3f}$ & "
            f"${s['jain']['mean']:.3f} \\pm {s['jain']['std']:.3f}$ & "
            f"${s['energy_efficiency']['mean']:.3f} \\pm {s['energy_efficiency']['std']:.3f}$ & "
            f"${s['avg_latency']['mean']:.3f} \\pm {s['avg_latency']['std']:.3f}$ \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    outp = OUT / "multicell_table.tex"
    outp.write_text("\n".join(lines))
    print(f"wrote {outp}")


def main() -> None:
    for N in [3, 5]:
        plot_training_overlay(N)
        plot_seed_cdf(N)
        plot_pct_of_wf(N)
    stats_table()
    effect_sizes_table()
    rayleigh_summary_table()
    iql_regnn_scaling_table()
    multicell_summary_table()


if __name__ == "__main__":
    main()
