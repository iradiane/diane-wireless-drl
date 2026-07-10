"""Multi-cell wireless experiment (K=7 hexagonal-cluster analogue): path loss +
Rayleigh + inter-cell interference. Fixed topology per env (distance matrix
sampled once), fast-fading redrawn every step.

At K=7 the centralised joint action space is 4^7 = 16 384 -- large but still
tractable for one seed. IQL and REGNN scale trivially. WMMSE-multi solves the
coupled iteration using the full gain matrix. WF-multi treats interference as
noise (weaker baseline but classical).

10 seeds. Compares all methods at K=7 as the JSAC/TSP-tier setting.
"""
from __future__ import annotations

import time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np


def _worker(args):
    import torch
    torch.set_num_threads(1)
    method, K, seed, n_episodes, ep_len = args
    from dqn_wireless import (
        DQNConfig, MultiCellConfig, MultiCellEnv,
        act_random_multi, act_fixed_multi, act_waterfilling_multi, act_wmmse_multi,
        evaluate_multicell_policy,
        train_iql_multicell, train_regnn_multicell,
        train_dqn_multicell, train_neural_bandit_multicell,
    )
    env_cfg = MultiCellConfig(K=K, ep_len=ep_len)
    dqn_cfg = DQNConfig(eps_decay_steps=int(0.5 * n_episodes * ep_len))

    t0 = time.time()
    if method == "Random":
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 1))
        m = evaluate_multicell_policy(env, act_random_multi)
    elif method == "Fixed":
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 2))
        m = evaluate_multicell_policy(env, act_fixed_multi)
    elif method == "WF-multi":
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 3))
        m = evaluate_multicell_policy(env, act_waterfilling_multi)
    elif method == "WMMSE-multi":
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 3))
        m = evaluate_multicell_policy(env, "wmmse_continuous")
    elif method == "IQL":
        agent, _ = train_iql_multicell(env_cfg, dqn_cfg, n_episodes, seed)
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_multicell_policy(env, agent)
    elif method == "REGNN":
        agent, _ = train_regnn_multicell(env_cfg, dqn_cfg, n_episodes, seed)
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_multicell_policy(env, agent)
    elif method == "DQN":
        agent, _ = train_dqn_multicell(env_cfg, dqn_cfg, n_episodes, seed)
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_multicell_policy(env, agent)
    elif method == "NeuralBandit":
        agent, _ = train_neural_bandit_multicell(env_cfg, dqn_cfg, n_episodes, seed)
        env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_multicell_policy(env, agent)
    else:
        raise ValueError(method)
    dt = time.time() - t0
    return {
        "method": method, "K": K, "seed": seed,
        "regime": "multicell_rayleigh_pathloss",
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
    # 5 seeds for learned methods (matches iql_regnn); 10 seeds free.
    seeds_learned = list(range(5))
    seeds_free = list(range(10))
    K = 7
    n_episodes = 250
    ep_len = 100

    cheap_methods = ["Random", "Fixed", "WF-multi", "WMMSE-multi"]
    learned_methods = ["IQL", "REGNN", "NeuralBandit", "DQN"]

    tasks = []
    for m in cheap_methods:
        for s in seeds_free:
            tasks.append((m, K, s, n_episodes, ep_len))
    for m in learned_methods:
        for s in seeds_learned:
            tasks.append((m, K, s, n_episodes, ep_len))

    methods = cheap_methods + learned_methods
    t0 = time.time()
    print(f"=== Multi-cell K={K}, {len(methods)} methods = "
          f"{len(tasks)} runs ===", flush=True)
    with Pool(6) as pool:
        results: list[dict] = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            print(f"  {i+1:3d}/{len(tasks)}  {r['method']:14s} K={r['K']:2d} "
                  f"seed={r['seed']}  thr={r['throughput_mbps']:.3f}  "
                  f"(wall {r['wall_time_s']:.0f}s, elapsed {time.time()-t0:.0f}s)",
                  flush=True)

    dump_json(RESULTS / "multicell_raw.json", results)

    summary: dict[str, dict[str, dict[str, float]]] = {}
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
    dump_json(RESULTS / "multicell_summary.json", summary)

    print("=== Summary ===", flush=True)
    for method in methods:
        s = summary[method]
        print(f"  {method:15s} thr={s['throughput_mbps']['mean']:6.3f}+/-{s['throughput_mbps']['std']:5.3f}  "
              f"jain={s['jain']['mean']:.3f}  ee={s['energy_efficiency']['mean']:.3f}")
    print(f"=== Total: {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
