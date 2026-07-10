"""N=10 top-up: only the scalable learned methods (IQL, REGNN) that centralised
methods cannot handle. Plus cheap classical baselines for reference. 5 seeds
per learned method, 10 seeds per cheap method. 250 episodes to keep wall time
manageable.
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
        train_iql, train_regnn, evaluate_policy,
        act_random, act_fixed, act_wmmse,
    )
    env_cfg = EnvConfig(
        N=10, ep_len=ep_len,
        channel_model="rayleigh", interference=True,
    )
    dqn_cfg = DQNConfig(eps_decay_steps=int(0.5 * n_episodes * ep_len))

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
        agent, _ = train_iql(env_cfg, dqn_cfg, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    elif method == "REGNN":
        agent, _ = train_regnn(env_cfg, dqn_cfg, n_episodes, seed)
        env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
        m = evaluate_policy(env, agent)
    else:
        raise ValueError(method)
    dt = time.time() - t0
    return {
        "method": method, "N": 10, "seed": seed,
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
    seeds_learned = list(range(5))
    seeds_free = list(range(10))
    n_episodes = 250
    ep_len = 100

    tasks = []
    for m in ("Random", "Fixed", "Water-Filling", "WMMSE"):
        for s in seeds_free:
            tasks.append((m, s, n_episodes, ep_len))
    for m in ("IQL", "REGNN"):
        for s in seeds_learned:
            tasks.append((m, s, n_episodes, ep_len))

    t0 = time.time()
    print(f"=== N=10 top-up, {len(tasks)} runs ===", flush=True)
    with Pool(6) as pool:
        results: list[dict] = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            print(f"  {i+1:3d}/{len(tasks)}  {r['method']:14s} seed={r['seed']:2d}  "
                  f"thr={r['throughput_mbps']:.3f}  (wall {r['wall_time_s']:.0f}s, "
                  f"elapsed {time.time()-t0:.0f}s)", flush=True)

    dump_json(RESULTS / "n10_topup_raw.json", results)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in ("Random", "Fixed", "Water-Filling", "WMMSE", "IQL", "REGNN"):
        rs = [r for r in results if r["method"] == method]
        if not rs: continue
        thr = np.array([r["throughput_mbps"] for r in rs])
        jn = np.array([r["jain_index"] for r in rs])
        ee = np.array([r["energy_efficiency"] for r in rs])
        summary[method] = {
            "throughput_mbps":   {"mean": float(thr.mean()), "std": float(thr.std(ddof=1))},
            "jain":              {"mean": float(jn.mean()),  "std": float(jn.std(ddof=1))},
            "energy_efficiency": {"mean": float(ee.mean()),  "std": float(ee.std(ddof=1))},
        }
    dump_json(RESULTS / "n10_topup_summary.json", summary)

    print("=== Summary ===", flush=True)
    for method in ("Random", "Fixed", "Water-Filling", "WMMSE", "IQL", "REGNN"):
        if method not in summary: continue
        s = summary[method]
        print(f"  {method:15s} thr={s['throughput_mbps']['mean']:.3f}+/-{s['throughput_mbps']['std']:.3f}  "
              f"jain={s['jain']['mean']:.3f}  ee={s['energy_efficiency']['mean']:.3f}")
    print(f"=== Total: {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
