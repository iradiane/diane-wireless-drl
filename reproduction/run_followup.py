"""Follow-up experiments after the main parallel sim completes:
- WF-Discrete (Water-Filling projected to the discrete power set) — very fast,
  10 seeds x 20 eval episodes each
- eps-decay ablation with 5 seeds (up from 2 in the earlier pass)
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np


def _wf_discrete_worker(args):
    N, seed = args
    from dqn_wireless import EnvConfig, WirelessEnv, evaluate_policy, act_waterfilling, RunResult
    env_cfg = EnvConfig(N=N, ep_len=100)
    env = WirelessEnv(env_cfg, np.random.default_rng(seed + 6))
    m = evaluate_policy(env, act_waterfilling)
    return RunResult(
        method="Water-Filling-Discrete", N=N, seed=seed,
        throughput_bits_per_use=m["throughput_bpu"], throughput_mbps=m["throughput_mbps"],
        jain_index=m["jain"], energy_efficiency=m["energy_efficiency"], avg_latency=m["avg_latency"],
        per_user_throughput=m["per_user_throughput"], per_user_latency=m["per_user_latency"],
    ).__dict__


def _eps_worker(args):
    import torch
    torch.set_num_threads(1)
    N, seed, decay, n_episodes, ep_len = args
    from dqn_wireless import EnvConfig, DQNConfig, train_dqn
    env_cfg = EnvConfig(N=N, ep_len=ep_len)
    decay_steps = int(n_episodes * ep_len * decay)
    dqn_cfg = DQNConfig(eps_decay_steps=decay_steps)
    _, curve = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
    return (N, decay, seed, curve)


def main() -> None:
    RESULTS = Path(__file__).parent / "results"
    from dqn_wireless import dump_json

    # ---------- WF-Discrete for the results table ----------
    print("=== WF-Discrete: 10 seeds x N in {3,5} ===", flush=True)
    t0 = time.time()
    args = [(N, s) for N in [3, 5] for s in range(10)]
    with Pool(6) as pool:
        rows = pool.map(_wf_discrete_worker, args)

    # Bucket by N
    for N in [3, 5]:
        N_rows = [r for r in rows if r["N"] == N]
        thrs = np.array([r["throughput_mbps"] for r in N_rows])
        jns  = np.array([r["jain_index"]      for r in N_rows])
        ees  = np.array([r["energy_efficiency"] for r in N_rows])
        print(f"[N={N}] WF-Discrete: thr {thrs.mean():.3f}+/-{thrs.std(ddof=1):.3f}, "
              f"jain {jns.mean():.3f}+/-{jns.std(ddof=1):.3f}, "
              f"ee {ees.mean():.3f}+/-{ees.std(ddof=1):.3f}", flush=True)
        # merge into existing final_N{N}_summary.json
        import json
        sum_path = RESULTS / f"final_N{N}_summary.json"
        if sum_path.exists():
            summary = json.loads(sum_path.read_text())
        else:
            summary = {}
        summary["Water-Filling-Discrete"] = {
            "throughput_mbps":   {"mean": float(thrs.mean()), "std": float(thrs.std(ddof=1))},
            "jain":              {"mean": float(jns.mean()),  "std": float(jns.std(ddof=1))},
            "energy_efficiency": {"mean": float(ees.mean()),  "std": float(ees.std(ddof=1))},
            "avg_latency":       {"mean": float(np.mean([r["avg_latency"] for r in N_rows])),
                                  "std":  float(np.std([r["avg_latency"] for r in N_rows], ddof=1))},
        }
        dump_json(sum_path, summary)
        # also merge into raw
        raw_path = RESULTS / f"final_N{N}_raw.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            raw.extend(N_rows)
            dump_json(raw_path, raw)
    print(f"WF-Discrete done in {time.time()-t0:.1f}s.\n", flush=True)

    # ---------- eps-decay ablation, 5 seeds ----------
    print("=== eps-decay ablation: 5 seeds x 4 decays x N in {3,5} ===", flush=True)
    t0 = time.time()
    n_episodes = 150
    ep_len = 100
    seeds = list(range(5))
    decays = (0.99, 0.98, 0.95, 0.90)
    tasks = [(N, s, d, n_episodes, ep_len) for N in [3, 5] for d in decays for s in seeds]

    with Pool(6) as pool:
        results = pool.map(_eps_worker, tasks)

    # Aggregate
    for N in [3, 5]:
        agg = {}
        for d in decays:
            curves = [r[3] for r in results if r[0] == N and r[1] == d]
            arr = np.array(curves)  # (n_seeds, n_episodes)
            agg[str(d)] = {
                "mean_per_episode": arr.mean(axis=0).tolist(),
                "std_per_episode":  arr.std(axis=0, ddof=1).tolist(),
                "final_mean": float(arr[:, -30:].mean()),
                "final_std":  float(arr[:, -30:].std(ddof=1)),
            }
        dump_json(RESULTS / f"eps_ablation_N{N}.json", agg)
        print(f"[N={N}]", flush=True)
        for d in decays:
            a = agg[str(d)]
            print(f"  decay={d}: final {a['final_mean']:.2f}+/-{a['final_std']:.2f}", flush=True)
    print(f"eps-decay ablation done in {time.time()-t0:.1f}s.", flush=True)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
