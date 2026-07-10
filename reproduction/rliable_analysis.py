"""rliable-style statistical analysis (Agarwal et al., NeurIPS 2021).

Computes:
- Interquartile Mean (IQM) with stratified bootstrap 95% CIs
- Median and mean with CIs
- Optimality gap w.r.t. a reference (e.g., Water-Filling)
- Probability of improvement (Cliff's delta variant)
- Performance profiles: P[X_method >= tau] over a threshold sweep

Renders three figures:
- Aggregate metrics (IQM/mean/median with CI bars per method)
- Performance profiles
- Probability-of-improvement matrix (heatmap)

Consumes the multi-experiment result JSONs and writes to source/ so LaTeX
can \\input them.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPRO = Path(__file__).parent
RESULTS = REPRO / "results"
SOURCE = REPRO.parent / "source"


def interquartile_mean(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    lo = int(np.ceil(0.25 * n))
    hi = int(np.floor(0.75 * n))
    if hi <= lo:
        return float(np.mean(x))
    return float(np.mean(x[lo:hi]))


def stratified_bootstrap_ci(
    x: np.ndarray, statistic, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for a bootstrap CI on `statistic(x)`."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(x[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1.0 - alpha / 2))
    return float(statistic(x)), lo, hi


def probability_of_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """P[X > Y] estimated from all pairs (x_i, y_j).
    Ties count as 0.5 following the Cliff's delta convention.
    Returns a value in [0, 1]. 0.5 means indifferent; 1 means X always beats Y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    wins = (x[:, None] > y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    total = len(x) * len(y)
    return float((wins + 0.5 * ties) / total)


def performance_profile(x: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """P[X >= tau] at each threshold tau."""
    x = np.asarray(x, dtype=np.float64)
    return np.array([(x >= t).mean() for t in taus])


def _extract_per_seed_throughput(
    result_file: Path, N: int | None = None
) -> dict[str, np.ndarray]:
    """Extract per-seed throughput arrays keyed by method."""
    with open(result_file) as f:
        data = json.load(f)
    by_method: dict[str, list[float]] = {}
    for r in data:
        if N is not None and r.get("N", None) != N:
            continue
        by_method.setdefault(r["method"], []).append(r["throughput_mbps"])
    return {m: np.asarray(v, dtype=np.float64) for m, v in by_method.items()}


def render_aggregate_metrics(
    per_method: dict[str, np.ndarray],
    method_order: list[str],
    outfile: Path,
    title: str,
) -> None:
    """IQM and mean with bootstrap 95% CIs per method."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (fn, label) in zip(
        axes,
        [(interquartile_mean, "IQM"),
         (lambda x: float(np.mean(x)), "Mean")],
    ):
        ys, y_lo, y_hi = [], [], []
        for m in method_order:
            if m not in per_method:
                ys.append(np.nan); y_lo.append(np.nan); y_hi.append(np.nan)
                continue
            p, lo, hi = stratified_bootstrap_ci(per_method[m], fn, seed=42)
            ys.append(p); y_lo.append(lo); y_hi.append(hi)
        xs = np.arange(len(method_order))
        yerr = np.array([[y - lo for y, lo in zip(ys, y_lo)],
                         [hi - y for y, hi in zip(ys, y_hi)]])
        ax.bar(xs, ys, yerr=yerr, capsize=4, color="tab:blue", alpha=0.7,
               edgecolor="black")
        ax.set_xticks(xs)
        ax.set_xticklabels(method_order, rotation=30, ha="right")
        ax.set_ylabel(f"Throughput (Mbps) — {label}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{label} with 95% bootstrap CI")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def render_performance_profile(
    per_method: dict[str, np.ndarray],
    method_order: list[str],
    outfile: Path,
    title: str,
) -> None:
    """P[throughput >= tau] as a function of tau, per method."""
    all_vals = np.concatenate([per_method[m] for m in method_order if m in per_method])
    taus = np.linspace(all_vals.min(), all_vals.max(), 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in method_order:
        if m not in per_method: continue
        prof = performance_profile(per_method[m], taus)
        ax.plot(taus, prof, label=m, linewidth=1.6)
    ax.set_xlabel("Throughput threshold τ (Mbps)")
    ax.set_ylabel("P[throughput ≥ τ]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def render_prob_improvement_matrix(
    per_method: dict[str, np.ndarray],
    method_order: list[str],
    outfile: Path,
    title: str,
) -> None:
    """P[X beats Y] as a heatmap."""
    n = len(method_order)
    P = np.full((n, n), np.nan)
    for i, mi in enumerate(method_order):
        for j, mj in enumerate(method_order):
            if mi not in per_method or mj not in per_method: continue
            if i == j:
                P[i, j] = 0.5
            else:
                P[i, j] = probability_of_improvement(per_method[mi], per_method[mj])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(P, cmap="RdBu_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(method_order, rotation=45, ha="right")
    ax.set_yticklabels(method_order)
    for i in range(n):
        for j in range(n):
            if np.isnan(P[i, j]): continue
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                     color=("white" if abs(P[i, j] - 0.5) > 0.35 else "black"),
                     fontsize=8)
    ax.set_title(title + "\nP[row method > col method]")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def render_rliable_table(
    per_method: dict[str, np.ndarray],
    method_order: list[str],
    outfile: Path,
    caption_key: str,
) -> None:
    """Render a LaTeX table with IQM (95% CI), Mean (95% CI), Median, N seeds."""
    rows = []
    rows.append(r"\begin{tabular}{|l|c|c|c|c|}")
    rows.append(r"\hline")
    rows.append(r"\textbf{Method} & \textbf{IQM [95\% CI]} & \textbf{Mean [95\% CI]} "
                r"& \textbf{Median} & \textbf{n seeds} \\")
    rows.append(r"\hline")
    for m in method_order:
        if m not in per_method:
            rows.append(f"{m} & --- & --- & --- & 0 \\\\"); continue
        vals = per_method[m]
        iqm, iqm_lo, iqm_hi = stratified_bootstrap_ci(vals, interquartile_mean, seed=42)
        mean, mean_lo, mean_hi = stratified_bootstrap_ci(vals, lambda x: float(np.mean(x)), seed=42)
        med = float(np.median(vals))
        rows.append(
            f"{m} & ${iqm:.3f}$ $[{iqm_lo:.3f}, {iqm_hi:.3f}]$ & "
            f"${mean:.3f}$ $[{mean_lo:.3f}, {mean_hi:.3f}]$ & "
            f"${med:.3f}$ & {len(vals)} \\\\"
        )
    rows.append(r"\hline")
    rows.append(r"\end{tabular}")
    outfile.write_text("\n".join(rows))


def render_all_for_regime(
    per_method: dict[str, np.ndarray],
    method_order: list[str],
    prefix: str,
    title_prefix: str,
) -> None:
    render_aggregate_metrics(per_method, method_order,
                              SOURCE / f"{prefix}_aggregate.pdf",
                              title_prefix + " — Aggregate metrics")
    render_performance_profile(per_method, method_order,
                                SOURCE / f"{prefix}_perfprofile.pdf",
                                title_prefix + " — Performance profile")
    render_prob_improvement_matrix(per_method, method_order,
                                    SOURCE / f"{prefix}_probimprovement.pdf",
                                    title_prefix + " — Probability of improvement")
    render_rliable_table(per_method, method_order,
                         SOURCE / f"{prefix}_rliable_table.tex",
                         title_prefix)


def main() -> None:
    method_order_main = ["Random", "Fixed", "TabularQ", "DQN", "RainbowLite",
                          "Neural bandit", "NeuralBandit", "Water-Filling", "WF (cont.)",
                          "WMMSE", "IQL", "REGNN"]

    # 1. Main N=3 comparison (from final_N3_raw.json)
    for N, tag in [(3, "N3"), (5, "N5")]:
        f_final = RESULTS / f"final_{tag}_raw.json"
        if not f_final.exists():
            continue
        per_arr = _extract_per_seed_throughput(f_final)
        per: dict[str, list[float]] = {k: list(v) for k, v in per_arr.items()}
        # Also pull Rainbow-lite + Neural Bandit from pathB
        f_pathB = RESULTS / "pathB_raw.json"
        if f_pathB.exists():
            with open(f_pathB) as f:
                pathB = json.load(f)
            for r in pathB:
                if r["N"] != N: continue
                m_name = r["method"]
                per.setdefault(m_name, []).append(r["throughput_mbps"])
        per_np = {k: np.asarray(v, dtype=np.float64) for k, v in per.items()}
        order = [m for m in method_order_main if m in per_np]
        render_all_for_regime(per_np, order, f"rliable_{tag}",
                               f"Main regime (uniform + orthogonal, N={N})")

    # 2. Rayleigh + interference (10 seeds)
    f_ray = RESULTS / "rayleigh_interference_raw.json"
    if f_ray.exists():
        per_ray = _extract_per_seed_throughput(f_ray)
        # Normalise Neural Bandit name if needed
        order = [m for m in method_order_main if m in per_ray]
        render_all_for_regime(per_ray, order, "rliable_rayleigh",
                               "Rayleigh + interference (N=3)")

    # 3. IQL / REGNN sweep (N=3, 5, 10)
    f_iql = RESULTS / "iql_regnn_raw.json"
    if f_iql.exists():
        for N in (3, 5, 10):
            per = _extract_per_seed_throughput(f_iql, N=N)
            order = [m for m in method_order_main if m in per]
            if not order: continue
            render_all_for_regime(per, order, f"rliable_iqlregnn_N{N}",
                                   f"MARL comparison, Rayleigh+int (N={N})")

    # 4. Multi-cell (K=7)
    f_mc = RESULTS / "multicell_raw.json"
    if f_mc.exists():
        with open(f_mc) as f:
            data = json.load(f)
        per = {}
        for r in data:
            per.setdefault(r["method"], []).append(r["throughput_mbps"])
        per = {k: np.asarray(v, dtype=np.float64) for k, v in per.items()}
        # Multi-cell has its own method names
        mc_order = ["Random", "Fixed", "WF-multi", "WMMSE-multi",
                    "IQL", "REGNN", "DQN", "NeuralBandit"]
        order = [m for m in mc_order if m in per]
        render_all_for_regime(per, order, "rliable_multicell_K7",
                               "Multi-cell K=7 (path loss + Rayleigh + interference)")

    print("rliable analysis rendered.")


if __name__ == "__main__":
    main()
