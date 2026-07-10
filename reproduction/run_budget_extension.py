"""Budget-extension experiment for N=5: train DQN with 4x the standard budget
(2x10^5 transitions per seed vs 5x10^4). 3 seeds is sufficient to establish
whether the failure at N=5 is architectural or a training-budget artefact.
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
    N, seed, n_episodes, ep_len = args
    from dqn_wireless import (
        DQNConfig, EnvConfig, WirelessEnv, train_dqn, evaluate_policy, RunResult
    )
    env_cfg = EnvConfig(N=N, ep_len=ep_len)
    dqn_cfg = DQNConfig(eps_decay_steps=int(0.5 * n_episodes * ep_len))
    t0 = time.time()
    agent, curve = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
    env = WirelessEnv(env_cfg, np.random.default_rng(seed + 4))
    m = evaluate_policy(env, agent)
    train_time_s = time.time() - t0
    return {
        "method": "DQN-BudgetExt",
        "N": N,
        "seed": seed,
        "n_episodes": n_episodes,
        "throughput_bits_per_use": m["throughput_bpu"],
        "throughput_mbps": m["throughput_mbps"],
        "jain_index": m["jain"],
        "energy_efficiency": m["energy_efficiency"],
        "avg_latency": m["avg_latency"],
        "per_user_throughput": m["per_user_throughput"],
        "per_user_latency": m["per_user_latency"],
        "train_curve": curve,
        "train_time_s": train_time_s,
    }


def main() -> None:
    from dqn_wireless import dump_json
    RESULTS = Path(__file__).parent / "results"
    N = 5
    seeds = [0, 1, 2]
    n_episodes = 2000  # 2x10^5 transitions (4x normal budget of 5x10^4)
    ep_len = 100
    print(f"=== N={N} budget-extension: 3 seeds x {n_episodes} ep x {ep_len} steps = "
          f"{n_episodes*ep_len:,} transitions per seed ===", flush=True)
    t0 = time.time()
    args = [(N, s, n_episodes, ep_len) for s in seeds]
    with Pool(3) as pool:
        results = list(pool.imap_unordered(_worker, args))
    print(f"Wall clock: {(time.time()-t0)/60:.1f} min", flush=True)
    thrs = np.array([r["throughput_mbps"]      for r in results])
    jns  = np.array([r["jain_index"]           for r in results])
    ees  = np.array([r["energy_efficiency"]    for r in results])
    print(f"DQN-BudgetExt (N=5): thr {thrs.mean():.3f}+/-{thrs.std(ddof=1):.3f}  "
          f"jain {jns.mean():.3f}+/-{jns.std(ddof=1):.3f}  "
          f"ee {ees.mean():.3f}+/-{ees.std(ddof=1):.3f}", flush=True)
    dump_json(RESULTS / "budget_extension_N5.json", results)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
