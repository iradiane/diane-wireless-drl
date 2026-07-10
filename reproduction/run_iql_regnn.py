"""MARL comparison: Independent Q-Learning (IQL) and REGNN (GNN policy), against
centralised DQN, Rainbow-lite, Neural Bandit, WMMSE, Water-Filling.

Runs at N in {3, 5, 10}. Centralised methods are skipped at N=10 (action space
4^10 = 1_048_576 is intractable). MARL/GNN methods scale.

Rayleigh + interference channel model, which is the JSAC/TSP-tier setting
(matches Nasir & Guo 2019 / Shen et al. 2020 / Eisen & Ribeiro 2020).

10 seeds per (method, N) cell.
"""
from __future__ import annotations

import time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np


CENTRALISED_METHODS = {"DQN", "RainbowLite", "NeuralBandit", "TabularQ"}
SCALABLE_METHODS = {"Random", "Fixed", "Water-Filling", "WMMSE", "IQL", "REGNN"}


def _worker(args):
    import torch
    torch.set_num_threads(1)
    method, N, seed, n_episodes, ep_len = args
    from dqn_wireless import (
        DQNConfig, EnvConfig, WirelessEnv,
        train_dqn, train_neural_bandit, train_tabular_q,
        train_iql, train_regnn,
        evaluate_policy,
        act_random, act_fixed, act_wmmse,
    )

    env_cfg = EnvConfig(
        N=N, ep_len=ep_len,
        channel_model="rayleigh", interference=True,
    )
    dqn_cfg_default = DQNConfig(eps_decay_steps=int(0.5 * n_episodes * ep_len))

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
    elif method == "IQL":
        agent, _ = train_iql(env_cfg, dqn_cfg_default, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "REGNN":
        agent, _ = train_regnn(env_cfg, dqn_cfg_default, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "DQN":
        agent, _ = train_dqn(env_cfg, dqn_cfg_default, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "RainbowLite":
        rl_cfg = DQNConfig(
            eps_decay_steps=int(0.5 * n_episodes * ep_len),
            use_double=True, use_dueling=True,
        )
        agent, _ = train_dqn(env_cfg, rl_cfg, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "NeuralBandit":
        agent, _ = train_neural_bandit(env_cfg, dqn_cfg_default, n_episodes, seed)
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
        "method": method, "N": N, "seed": seed,
        "regime": "rayleigh_interference",
        "throughput_bits_per_use": m["throughput_bpu"],
        "throughput_mbps": m["throughput_mbps"],
        "jain_index": m["jain"],
        "energy_efficiency": m["energy_efficiency"],
        "avg_latency": m["avg_latency"],
        "per_user_throughput": m["per_user_throughput"],
        "per_user_latency": m["per_user_latency"],
        "wall_time_s": dt,
    }


def main() -> None:
    from dqn_wireless import dump_json
    RESULTS = Path(__file__).parent / "results"
    # 5 seeds for the scaling table (already 10 seeds for the main results in
    # other tables). Cheaper methods (WMMSE, WF, Random, Fixed) get 10 seeds
    # for free since they don't train.
    seeds_learned = list(range(5))
    seeds_free = list(range(10))

    # 300 episodes is enough for convergence in the MARL regime.
    tasks: list[tuple[str, int, int, int, int]] = []
    for N in (3, 5, 10):
        n_ep = 300
        ep_len = 100
        cheap_methods = ["Random", "Fixed", "Water-Filling", "WMMSE"]
        expensive_methods = ["IQL", "REGNN"]
        if N < 10:
            expensive_methods += ["DQN", "NeuralBandit"]
        for m in cheap_methods:
            for s in seeds_free:
                tasks.append((m, N, s, n_ep, ep_len))
        for m in expensive_methods:
            for s in seeds_learned:
                tasks.append((m, N, s, n_ep, ep_len))

    t0 = time.time()
    print(f"=== IQL/REGNN comparison, N in (3,5,10), Rayleigh+interference, "
          f"{len(tasks)} runs total ===", flush=True)
    with Pool(6) as pool:
        results: list[dict] = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            print(f"  {i+1:3d}/{len(tasks)}  {r['method']:14s} N={r['N']:2d} "
                  f"seed={r['seed']}  thr={r['throughput_mbps']:.3f}  "
                  f"(wall {r['wall_time_s']:.0f}s, elapsed {time.time()-t0:.0f}s)",
                  flush=True)

    dump_json(RESULTS / "iql_regnn_raw.json", results)

    # Summarise
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    all_methods = sorted(set(r["method"] for r in results))
    all_Ns = sorted(set(r["N"] for r in results))
    for N in all_Ns:
        summary[str(N)] = {}
        for method in all_methods:
            rs = [r for r in results if r["method"] == method and r["N"] == N]
            if not rs:
                continue
            thr = np.array([r["throughput_mbps"] for r in rs])
            jn = np.array([r["jain_index"] for r in rs])
            ee = np.array([r["energy_efficiency"] for r in rs])
            lat = np.array([r["avg_latency"] for r in rs])
            summary[str(N)][method] = {
                "throughput_mbps":   {"mean": float(thr.mean()), "std": float(thr.std(ddof=1) if len(thr)>1 else 0.0)},
                "jain":              {"mean": float(jn.mean()),  "std": float(jn.std(ddof=1) if len(jn)>1 else 0.0)},
                "energy_efficiency": {"mean": float(ee.mean()),  "std": float(ee.std(ddof=1) if len(ee)>1 else 0.0)},
                "avg_latency":       {"mean": float(lat.mean()), "std": float(lat.std(ddof=1) if len(lat)>1 else 0.0)},
            }
    dump_json(RESULTS / "iql_regnn_summary.json", summary)

    print("=== Summary ===", flush=True)
    for N in all_Ns:
        print(f"--- N={N} ---")
        for method in all_methods:
            if method not in summary[str(N)]: continue
            s = summary[str(N)][method]
            print(f"  {method:15s} thr={s['throughput_mbps']['mean']:6.3f}+/-{s['throughput_mbps']['std']:5.3f}  "
                  f"jain={s['jain']['mean']:.3f}  ee={s['energy_efficiency']['mean']:.3f}")
    print(f"=== Total: {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
