"""Parse the partial iql_regnn.log and construct a summary JSON from completed
tasks. Also extract per-run data for rliable analysis.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).parent / "results"

# Each line looks like:
#   107/170  IQL            N= 5 seed=3  thr=1.430  (wall 973s, elapsed 2230s)
LINE = re.compile(
    r"\s*\d+/\d+\s+(?P<method>\S+)\s+N=\s*(?P<N>\d+)\s+seed=(?P<seed>\d+)\s+"
    r"thr=(?P<thr>[\d.]+)\s+\(wall\s+(?P<wall>\d+)s"
)


def main() -> None:
    log = RESULTS / "iql_regnn.log"
    if not log.exists():
        print("no log to salvage")
        return

    raw = []
    for line in log.read_text().splitlines():
        m = LINE.search(line)
        if not m:
            continue
        raw.append({
            "method": m.group("method"),
            "N": int(m.group("N")),
            "seed": int(m.group("seed")),
            "regime": "rayleigh_interference",
            "throughput_mbps": float(m.group("thr")),
            "throughput_bits_per_use": float(m.group("thr")),
            "jain_index": 0.5,        # unknown; will drop from tables that need it
            "energy_efficiency": 0.0, # unknown
            "avg_latency": 0.0,       # unknown
            "per_user_throughput": [],
            "per_user_latency": [],
            "wall_time_s": int(m.group("wall")),
        })
    print(f"salvaged {len(raw)} runs from log")

    # Dedupe: same (method, N, seed) — keep first (they should be unique anyway).
    seen = set()
    dedup = []
    for r in raw:
        key = (r["method"], r["N"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    print(f"deduplicated to {len(dedup)} unique runs")

    # Save raw
    (RESULTS / "iql_regnn_raw.json").write_text(json.dumps(dedup, indent=2))

    # Summary
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    Ns = sorted(set(r["N"] for r in dedup))
    methods = sorted(set(r["method"] for r in dedup))
    for N in Ns:
        summary[str(N)] = {}
        for method in methods:
            rs = [r for r in dedup if r["method"] == method and r["N"] == N]
            if not rs:
                continue
            thr = np.array([r["throughput_mbps"] for r in rs])
            summary[str(N)][method] = {
                "throughput_mbps":   {"mean": float(thr.mean()), "std": float(thr.std(ddof=1) if len(thr)>1 else 0.0)},
                "jain":              {"mean": 0.0, "std": 0.0},
                "energy_efficiency": {"mean": 0.0, "std": 0.0},
                "avg_latency":       {"mean": 0.0, "std": 0.0},
                "n_seeds":           len(rs),
            }
    (RESULTS / "iql_regnn_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSalvaged summary (throughput bits/use, n seeds):")
    for N in Ns:
        print(f"  N={N}:")
        for method in methods:
            if method in summary[str(N)]:
                s = summary[str(N)][method]
                thr = s["throughput_mbps"]
                print(f"    {method:15s} {thr['mean']:6.3f} +/- {thr['std']:5.3f} (n={s['n_seeds']})")


if __name__ == "__main__":
    main()
