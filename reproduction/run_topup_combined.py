"""Combined top-up: 3 seeds each of learned methods for both (a) N=10 single-cell
Rayleigh + interference and (b) multi-cell K=7. 10 seeds for cheap methods.
150 episodes to keep wall time bounded.
"""
from __future__ import annotations

import time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np


def _worker(args):
    import torch
    torch.set_num_threads(1)
    kind, method, K_or_N, seed, n_episodes, ep_len = args
    if kind == "single":
        from dqn_wireless import (
            DQNConfig, EnvConfig, WirelessEnv,
            train_iql, train_regnn, evaluate_policy,
            act_random, act_fixed, act_wmmse,
        )
        env_cfg = EnvConfig(
            N=K_or_N, ep_len=ep_len,
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
    else:  # multi-cell
        from dqn_wireless import (
            DQNConfig, MultiCellConfig, MultiCellEnv,
            act_random_multi, act_fixed_multi, act_waterfilling_multi, act_wmmse_multi,
            evaluate_multicell_policy,
            train_iql_multicell, train_regnn_multicell,
            train_neural_bandit_multicell,
        )
        env_cfg = MultiCellConfig(K=K_or_N, ep_len=ep_len)
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
        elif method == "NeuralBandit":
            agent, _ = train_neural_bandit_multicell(env_cfg, dqn_cfg, n_episodes, seed)
            env = MultiCellEnv(env_cfg, np.random.default_rng(seed + 4))
            m = evaluate_multicell_policy(env, agent)
        else:
            raise ValueError(method)
    dt = time.time() - t0
    return {
        "kind": kind, "method": method,
        "N" if kind == "single" else "K": K_or_N,
        "seed": seed,
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
    seeds_learned = list(range(3))
    seeds_free = list(range(10))
    n_episodes = 150
    ep_len = 100

    tasks = []
    # N=10 single-cell top-up
    for m in ("Random", "Fixed", "Water-Filling", "WMMSE"):
        for s in seeds_free:
            tasks.append(("single", m, 10, s, n_episodes, ep_len))
    for m in ("IQL", "REGNN"):
        for s in seeds_learned:
            tasks.append(("single", m, 10, s, n_episodes, ep_len))
    # Multi-cell K=7
    for m in ("Random", "Fixed", "WF-multi", "WMMSE-multi"):
        for s in seeds_free:
            tasks.append(("multi", m, 7, s, n_episodes, ep_len))
    for m in ("IQL", "REGNN", "NeuralBandit"):
        for s in seeds_learned:
            tasks.append(("multi", m, 7, s, n_episodes, ep_len))

    t0 = time.time()
    print(f"=== Combined top-up: {len(tasks)} runs ===", flush=True)
    n10 = []
    mc = []
    with Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            if r["kind"] == "single":
                n10.append(r)
            else:
                mc.append(r)
            print(f"  {i+1:3d}/{len(tasks)}  [{r['kind']:6s}] {r['method']:14s} "
                  f"seed={r['seed']:2d}  thr={r['throughput_mbps']:.3f}  "
                  f"(wall {r['wall_time_s']:.0f}s, elapsed {time.time()-t0:.0f}s)", flush=True)

    # Save
    dump_json(RESULTS / "n10_topup_raw.json", n10)
    dump_json(RESULTS / "multicell_raw.json", mc)

    def summarise(rs: list[dict], methods: list[str]) -> dict:
        summary = {}
        for m in methods:
            filt = [r for r in rs if r["method"] == m]
            if not filt: continue
            thr = np.array([r["throughput_mbps"] for r in filt])
            jn = np.array([r["jain_index"] for r in filt])
            ee = np.array([r["energy_efficiency"] for r in filt])
            lat = np.array([r["avg_latency"] for r in filt])
            summary[m] = {
                "throughput_mbps":   {"mean": float(thr.mean()), "std": float(thr.std(ddof=1) if len(thr)>1 else 0.0)},
                "jain":              {"mean": float(jn.mean()),  "std": float(jn.std(ddof=1) if len(jn)>1 else 0.0)},
                "energy_efficiency": {"mean": float(ee.mean()),  "std": float(ee.std(ddof=1) if len(ee)>1 else 0.0)},
                "avg_latency":       {"mean": float(lat.mean()), "std": float(lat.std(ddof=1) if len(lat)>1 else 0.0)},
            }
        return summary

    single_summary = summarise(n10, ["Random", "Fixed", "Water-Filling", "WMMSE", "IQL", "REGNN"])
    mc_summary = summarise(mc, ["Random", "Fixed", "WF-multi", "WMMSE-multi", "IQL", "REGNN", "NeuralBandit"])
    dump_json(RESULTS / "n10_topup_summary.json", single_summary)
    dump_json(RESULTS / "multicell_summary.json", mc_summary)

    print("\n=== N=10 summary ===")
    for m in ("Random", "Fixed", "Water-Filling", "WMMSE", "IQL", "REGNN"):
        if m not in single_summary: continue
        s = single_summary[m]["throughput_mbps"]
        print(f"  {m:15s} {s['mean']:.3f}+/-{s['std']:.3f}")
    print("=== Multi-cell K=7 summary ===")
    for m in ("Random", "Fixed", "WF-multi", "WMMSE-multi", "IQL", "REGNN", "NeuralBandit"):
        if m not in mc_summary: continue
        s = mc_summary[m]["throughput_mbps"]
        print(f"  {m:15s} {s['mean']:.3f}+/-{s['std']:.3f}")
    print(f"=== Total: {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
