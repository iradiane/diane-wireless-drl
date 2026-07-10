"""Parallel version of run_final.py.  Runs seeds in a process pool with each
worker constrained to a single PyTorch thread to avoid CPU contention.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Any

import numpy as np


def _worker(args: tuple[int, int, int, int, dict[str, Any], bool]) -> tuple[list[Any], int]:
    """Run one (N, seed) configuration. Returns list of RunResult dicts + seed."""
    N, seed, n_episodes, ep_len, dqn_kwargs, include_tabular = args

    # single-thread PyTorch inside each worker
    import torch
    torch.set_num_threads(1)

    from dqn_wireless import (
        DQNConfig, EnvConfig, WirelessEnv,
        train_dqn, train_tabular_q, evaluate_policy,
        act_random, act_fixed, act_waterfilling, RunResult,
    )
    env_cfg = EnvConfig(N=N, ep_len=ep_len)
    dqn_cfg = DQNConfig(**dqn_kwargs)
    out: list[RunResult] = []

    # Random
    env = WirelessEnv(env_cfg, np.random.default_rng(seed + 1))
    m = evaluate_policy(env, act_random)
    out.append(RunResult(
        method="Random", N=N, seed=seed,
        throughput_bits_per_use=m["throughput_bpu"], throughput_mbps=m["throughput_mbps"],
        jain_index=m["jain"], energy_efficiency=m["energy_efficiency"], avg_latency=m["avg_latency"],
        per_user_throughput=m["per_user_throughput"], per_user_latency=m["per_user_latency"],
    ))
    # Fixed
    env = WirelessEnv(env_cfg, np.random.default_rng(seed + 2))
    m = evaluate_policy(env, act_fixed)
    out.append(RunResult(
        method="Fixed", N=N, seed=seed,
        throughput_bits_per_use=m["throughput_bpu"], throughput_mbps=m["throughput_mbps"],
        jain_index=m["jain"], energy_efficiency=m["energy_efficiency"], avg_latency=m["avg_latency"],
        per_user_throughput=m["per_user_throughput"], per_user_latency=m["per_user_latency"],
    ))
    # Water-Filling (continuous power — theoretical upper bound)
    env = WirelessEnv(env_cfg, np.random.default_rng(seed + 3))
    m = evaluate_policy(env, "wf_continuous")
    out.append(RunResult(
        method="Water-Filling", N=N, seed=seed,
        throughput_bits_per_use=m["throughput_bpu"], throughput_mbps=m["throughput_mbps"],
        jain_index=m["jain"], energy_efficiency=m["energy_efficiency"], avg_latency=m["avg_latency"],
        per_user_throughput=m["per_user_throughput"], per_user_latency=m["per_user_latency"],
    ))
    # Tabular Q (N=3 only)
    if include_tabular:
        tab_agent, tab_curve = train_tabular_q(env_cfg, n_episodes, seed, n_bins=5)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 5))
        m = evaluate_policy(env, tab_agent)
        out.append(RunResult(
            method="TabularQ", N=N, seed=seed,
            throughput_bits_per_use=m["throughput_bpu"], throughput_mbps=m["throughput_mbps"],
            jain_index=m["jain"], energy_efficiency=m["energy_efficiency"], avg_latency=m["avg_latency"],
            per_user_throughput=m["per_user_throughput"], per_user_latency=m["per_user_latency"],
            train_curve=tab_curve,
        ))
    # DQN
    agent, curve = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
    env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
    m = evaluate_policy(env, agent)
    out.append(RunResult(
        method="DQN", N=N, seed=seed,
        throughput_bits_per_use=m["throughput_bpu"], throughput_mbps=m["throughput_mbps"],
        jain_index=m["jain"], energy_efficiency=m["energy_efficiency"], avg_latency=m["avg_latency"],
        per_user_throughput=m["per_user_throughput"], per_user_latency=m["per_user_latency"],
        train_curve=curve,
    ))
    return [r.__dict__ for r in out], seed


def main() -> None:
    t_start = time.time()
    seeds = list(range(10))
    n_episodes = 500
    ep_len = 100
    n_workers = 6  # keep 2 cores free for other work + OS

    from dqn_wireless import summarize, dump_json, RunResult
    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    dqn_kwargs = {"eps_decay_steps": int(0.5 * n_episodes * ep_len)}

    for N in [3, 5]:
        print(f"\n=== FINAL parallel, N={N}, {len(seeds)} seeds x {n_episodes} ep x {ep_len} steps ===", flush=True)
        include_tabular = (N == 3)
        args_list = [(N, s, n_episodes, ep_len, dqn_kwargs, include_tabular) for s in seeds]
        t_batch = time.time()

        all_results: list[dict] = []
        with Pool(n_workers) as pool:
            for i, (result_dicts, seed) in enumerate(pool.imap_unordered(_worker, args_list)):
                all_results.extend(result_dicts)
                print(f"  seed {seed} done ({i+1}/{len(seeds)}, elapsed {time.time()-t_batch:.0f}s)", flush=True)

        # Rebuild as RunResult-like dicts for summarize
        from types import SimpleNamespace
        rs_ns = [SimpleNamespace(**d) for d in all_results]
        # summarize expects RunResult objects with attribute access; build them
        by_method: dict[str, list[dict]] = {}
        for r in all_results:
            by_method.setdefault(r["method"], []).append(r)
        summary: dict[str, dict[str, dict[str, float]]] = {}
        for m, rs in by_method.items():
            def stat(xs):
                a = np.asarray(xs, dtype=np.float64)
                return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0)}
            summary[m] = {
                "throughput_mbps": stat([r["throughput_mbps"] for r in rs]),
                "jain": stat([r["jain_index"] for r in rs]),
                "energy_efficiency": stat([r["energy_efficiency"] for r in rs]),
                "avg_latency": stat([r["avg_latency"] for r in rs]),
            }

        dump_json(RESULTS_DIR / f"final_N{N}_raw.json", all_results)
        dump_json(RESULTS_DIR / f"final_N{N}_summary.json", summary)

        print(f"[N={N}] batch elapsed {time.time()-t_batch:.0f}s", flush=True)
        methods = ["DQN", "TabularQ", "Fixed", "Random", "Water-Filling"] if include_tabular \
                  else ["DQN", "Fixed", "Random", "Water-Filling"]
        for mname in methods:
            if mname not in summary:
                continue
            s = summary[mname]
            print(f"  {mname:<15} thr={s['throughput_mbps']['mean']:.3f}+/-{s['throughput_mbps']['std']:.3f}  "
                  f"jain={s['jain']['mean']:.3f}+/-{s['jain']['std']:.3f}  "
                  f"ee={s['energy_efficiency']['mean']:.3f}+/-{s['energy_efficiency']['std']:.3f}", flush=True)

    print(f"\n=== Total elapsed: {(time.time() - t_start)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    # Windows needs spawn (default), but ensure explicitly for safety
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
