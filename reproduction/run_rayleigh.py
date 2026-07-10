"""Rayleigh + interference regime: 10 seeds x N=3, all methods incl. WMMSE.
Establishes robustness of the Neural Bandit finding under a harder, more
realistic wireless regime, and benchmarks against WMMSE (Shi et al. 2011),
the standard classical iterative baseline in interference-limited systems.
"""
from __future__ import annotations

import time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np


def _worker(args):
    import torch
    torch.set_num_threads(1)
    method, seed, n_episodes, ep_len = args
    from dqn_wireless import (
        DQNConfig, EnvConfig, WirelessEnv,
        train_dqn, train_neural_bandit, train_tabular_q, evaluate_policy,
        act_random, act_fixed, act_wmmse, RunResult,
    )
    env_cfg = EnvConfig(
        N=3, ep_len=ep_len,
        channel_model="rayleigh", interference=True,
    )

    t0 = time.time()
    if method == "Random":
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 1))
        m = evaluate_policy(env, act_random)
    elif method == "Fixed":
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 2))
        m = evaluate_policy(env, act_fixed)
    elif method == "Water-Filling":
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 3))
        m = evaluate_policy(env, "wf_continuous")
    elif method == "WMMSE":
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 6))
        m = evaluate_policy(env, act_wmmse)
    elif method == "DQN":
        dqn_cfg = DQNConfig(eps_decay_steps=int(0.5 * n_episodes * ep_len))
        agent, _ = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "RainbowLite":
        dqn_cfg = DQNConfig(
            eps_decay_steps=int(0.5 * n_episodes * ep_len),
            use_double=True, use_dueling=True,
        )
        agent, _ = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "NeuralBandit":
        dqn_cfg = DQNConfig(eps_decay_steps=int(0.5 * n_episodes * ep_len))
        agent, _ = train_neural_bandit(env_cfg, dqn_cfg, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "TabularQ":
        agent, _ = train_tabular_q(env_cfg, n_episodes, seed, n_bins=5)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 5))
        m = evaluate_policy(env, agent)
    else:
        raise ValueError(method)
    dt = time.time() - t0
    return {
        "method": method, "N": 3, "seed": seed, "regime": "rayleigh_interference",
        "throughput_bits_per_use": m["throughput_bpu"],
        "throughput_mbps": m["throughput_mbps"],
        "jain_index": m["jain"], "energy_efficiency": m["energy_efficiency"],
        "avg_latency": m["avg_latency"],
        "per_user_throughput": m["per_user_throughput"],
        "per_user_latency": m["per_user_latency"],
        "wall_time_s": dt,
    }


def main() -> None:
    from dqn_wireless import dump_json
    RESULTS = Path(__file__).parent / "results"
    seeds = list(range(10))
    n_episodes = 500
    ep_len = 100

    methods = ["Random", "Fixed", "Water-Filling", "WMMSE", "TabularQ",
               "DQN", "RainbowLite", "NeuralBandit"]
    tasks = [(m, s, n_episodes, ep_len) for m in methods for s in seeds]

    t0 = time.time()
    print(f"=== Rayleigh + interference, N=3, {len(methods)} methods x 10 seeds = "
          f"{len(tasks)} runs ===", flush=True)
    with Pool(6) as pool:
        results: list[dict] = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            print(f"  done {i+1}/{len(tasks)}: {r['method']:12s} seed={r['seed']}  "
                  f"thr={r['throughput_mbps']:.3f}  (wall {r['wall_time_s']:.0f}s, "
                  f"elapsed {time.time()-t0:.0f}s)", flush=True)

    dump_json(RESULTS / "rayleigh_interference_raw.json", results)
    print("=== Summary (mean +/- std over 10 seeds) ===", flush=True)
    for method in methods:
        rs = [r for r in results if r["method"] == method]
        thr = np.array([r["throughput_mbps"] for r in rs])
        jn = np.array([r["jain_index"] for r in rs])
        ee = np.array([r["energy_efficiency"] for r in rs])
        print(f"  {method:15s} thr={thr.mean():.3f}+/-{thr.std(ddof=1):.3f}  "
              f"jain={jn.mean():.3f}+/-{jn.std(ddof=1):.3f}  "
              f"ee={ee.mean():.3f}+/-{ee.std(ddof=1):.3f}", flush=True)
    summary = {}
    for method in methods:
        rs = [r for r in results if r["method"] == method]
        thr = np.array([r["throughput_mbps"] for r in rs])
        jn = np.array([r["jain_index"] for r in rs])
        ee = np.array([r["energy_efficiency"] for r in rs])
        lat = np.array([r["avg_latency"] for r in rs])
        summary[method] = {
            "throughput_mbps":   {"mean": float(thr.mean()), "std": float(thr.std(ddof=1))},
            "jain":              {"mean": float(jn.mean()),  "std": float(jn.std(ddof=1))},
            "energy_efficiency": {"mean": float(ee.mean()),  "std": float(ee.std(ddof=1))},
            "avg_latency":       {"mean": float(lat.mean()), "std": float(lat.std(ddof=1))},
        }
    dump_json(RESULTS / "rayleigh_interference_summary_10seed.json", summary)
    dump_json(RESULTS / "rayleigh_interference_summary.json", summary)
    print(f"=== Total: {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
