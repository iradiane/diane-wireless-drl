"""Path B experiments: Rainbow-lite (Double + Dueling DQN) and Neural Bandit
at N=3 and N=5, 10 seeds each, matched budget to the main comparison.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np


def _worker(args):
    import torch
    torch.set_num_threads(1)
    method, N, seed, n_episodes, ep_len = args
    from dqn_wireless import (
        DQNConfig, EnvConfig, WirelessEnv,
        train_dqn, train_neural_bandit, evaluate_policy, RunResult,
    )
    env_cfg = EnvConfig(N=N, ep_len=ep_len)
    if method == "RainbowLite":
        dqn_cfg = DQNConfig(
            eps_decay_steps=int(0.5 * n_episodes * ep_len),
            use_double=True, use_dueling=True,
        )
        agent, curve = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
    elif method == "NeuralBandit":
        dqn_cfg = DQNConfig(eps_decay_steps=int(0.5 * n_episodes * ep_len))
        agent, curve = train_neural_bandit(env_cfg, dqn_cfg, n_episodes, seed)
    else:
        raise ValueError(method)
    env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
    m = evaluate_policy(env, agent)
    return RunResult(
        method=method, N=N, seed=seed,
        throughput_bits_per_use=m["throughput_bpu"], throughput_mbps=m["throughput_mbps"],
        jain_index=m["jain"], energy_efficiency=m["energy_efficiency"],
        avg_latency=m["avg_latency"],
        per_user_throughput=m["per_user_throughput"], per_user_latency=m["per_user_latency"],
        train_curve=curve,
    ).__dict__


def main() -> None:
    from dqn_wireless import dump_json
    RESULTS = Path(__file__).parent / "results"

    seeds = list(range(10))
    n_episodes = 500
    ep_len = 100

    tasks: list[tuple[str, int, int, int, int]] = []
    for method in ["RainbowLite", "NeuralBandit"]:
        for N in [3, 5]:
            for s in seeds:
                tasks.append((method, N, s, n_episodes, ep_len))

    t0 = time.time()
    print(f"=== Path B: 2 methods x 2 N x 10 seeds = {len(tasks)} runs ===", flush=True)

    with Pool(6) as pool:
        results: list[dict] = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            print(f"  done {i+1}/{len(tasks)}: {r['method']} N={r['N']} seed={r['seed']} "
                  f"thr={r['throughput_mbps']:.3f} (elapsed {time.time()-t0:.0f}s)", flush=True)

    dump_json(RESULTS / "pathB_raw.json", results)

    # Aggregate and merge into final_N{N}_summary.json
    import json
    for N in [3, 5]:
        summary_path = RESULTS / f"final_N{N}_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = {}
        for method in ["RainbowLite", "NeuralBandit"]:
            rows = [r for r in results if r["N"] == N and r["method"] == method]
            thrs = np.array([r["throughput_mbps"] for r in rows])
            jns  = np.array([r["jain_index"]      for r in rows])
            ees  = np.array([r["energy_efficiency"] for r in rows])
            lats = np.array([r["avg_latency"]     for r in rows])
            summary[method] = {
                "throughput_mbps":   {"mean": float(thrs.mean()), "std": float(thrs.std(ddof=1))},
                "jain":              {"mean": float(jns.mean()),  "std": float(jns.std(ddof=1))},
                "energy_efficiency": {"mean": float(ees.mean()),  "std": float(ees.std(ddof=1))},
                "avg_latency":       {"mean": float(lats.mean()), "std": float(lats.std(ddof=1))},
            }
            print(f"[N={N} {method}] thr {thrs.mean():.3f}+/-{thrs.std(ddof=1):.3f}  "
                  f"jain {jns.mean():.3f}+/-{jns.std(ddof=1):.3f}  "
                  f"ee {ees.mean():.3f}+/-{ees.std(ddof=1):.3f}", flush=True)
        dump_json(summary_path, summary)
        # Also add to raw
        raw_path = RESULTS / f"final_N{N}_raw.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            raw.extend([r for r in results if r["N"] == N])
            dump_json(raw_path, raw)

    print(f"=== Total: {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
