# Treating this as: research prototype for paper reproduction. Full typing, no test suite.
"""
Reproduction of the DQN wireless power-allocation study from the Iradukunda et al.
manuscript. Environment: single-cell downlink, N users, discrete power levels
{0,1,2,3} W, i.i.d. uniform fading channels, orthogonal access, no interference,
perfect CSI. Compares DQN against Random, Fixed (2 W), Water-Filling baselines.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Environment
# =============================================================================

@dataclass
class EnvConfig:
    N: int = 3                          # number of users
    power_levels: tuple[int, ...] = (0, 1, 2, 3)  # Watts
    h_min: float = 0.1
    h_max: float = 1.0
    sigma2: float = 1.0                 # noise power
    lam: float = 0.1                    # power-penalty coefficient in reward
    ep_len: int = 100                   # steps per episode
    arrival_rate: float = 1.0           # mean packets per step per user (for latency proxy)
    # Channel and access model:
    channel_model: str = "uniform"      # "uniform" (paper default) or "rayleigh"
    interference: bool = False          # False = orthogonal access (paper default); True = inter-user interference
    rayleigh_mean_gain: float = 0.5     # E[|h|^2] under Rayleigh (mean power gain)


class WirelessEnv:
    """Memoryless i.i.d. fast-fading downlink with orthogonal access and no
    interference. State = channel gains. Action = joint discrete power vector
    encoded as a single index in [0, M^N).
    """

    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.M = len(cfg.power_levels)
        self.n_actions = self.M ** cfg.N
        self._action_table = np.array(
            list(itertools.product(cfg.power_levels, repeat=cfg.N)),
            dtype=np.float32,
        )  # shape (M^N, N)
        self.t = 0
        self.queues = np.zeros(cfg.N, dtype=np.float64)
        self._draw_channels()

    def _draw_channels(self) -> None:
        if self.cfg.channel_model == "uniform":
            self.h = self.rng.uniform(
                self.cfg.h_min, self.cfg.h_max, size=self.cfg.N
            ).astype(np.float32)
        elif self.cfg.channel_model == "rayleigh":
            # h is the magnitude-squared gain |h|^2 with mean rayleigh_mean_gain.
            # For a Rayleigh envelope with parameter sigma_R, |h|^2 is Exponential(1/(2*sigma_R^2)).
            # We just draw |h|^2 directly from Exponential(mean = rayleigh_mean_gain).
            self.h = self.rng.exponential(
                self.cfg.rayleigh_mean_gain, size=self.cfg.N
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown channel_model {self.cfg.channel_model!r}")

    def reset(self) -> np.ndarray:
        self.t = 0
        self.queues[:] = 0.0
        self._draw_channels()
        return self.h.copy()

    def action_vector(self, a: int) -> np.ndarray:
        return self._action_table[a]

    def step(self, a: int) -> tuple[np.ndarray, float, bool, dict]:
        p = self._action_table[a]  # shape (N,)
        if self.cfg.interference:
            # Interference-limited SINR: user i sees noise + interference from all j != i.
            # SINR_i = p_i h_i / (sigma^2 + sum_{j != i} p_j h_j)
            signal = p * self.h
            total_interf_plus_noise = signal.sum() + self.cfg.sigma2 - signal
            sinr = signal / np.clip(total_interf_plus_noise, 1e-9, None)
            rate = np.log2(1.0 + sinr)
        else:
            snr = p * self.h / self.cfg.sigma2
            rate = np.log2(1.0 + snr)  # bits/s/Hz per user
        reward = float(rate.sum() - self.cfg.lam * p.sum())

        # queue-proxy latency (arrival_rate packets/step, served at `rate`)
        arrivals = self.rng.poisson(self.cfg.arrival_rate, size=self.cfg.N).astype(
            np.float64
        )
        self.queues = np.maximum(self.queues + arrivals - rate.astype(np.float64), 0.0)

        info = {
            "rate": rate.copy(),
            "power": p.copy(),
            "queue": self.queues.copy(),
        }
        self.t += 1
        done = self.t >= self.cfg.ep_len
        self._draw_channels()
        return self.h.copy(), reward, done, info


# =============================================================================
# Baselines
# =============================================================================

def act_random(env: WirelessEnv, h: np.ndarray) -> int:
    return int(env.rng.integers(0, env.n_actions))


def act_fixed(env: WirelessEnv, h: np.ndarray) -> int:
    # Constant p_i = 2 W for all users
    target = np.full(env.cfg.N, 2, dtype=np.int64)
    for idx, vec in enumerate(env._action_table):
        if np.array_equal(vec.astype(np.int64), target):
            return idx
    raise ValueError("2W fixed action not in action table")


def waterfilling_powers(h: np.ndarray, p_max_total: float, sigma2: float) -> np.ndarray:
    """Continuous water-filling under sum-power constraint P_max_total, then
    return per-user real-valued allocations (not necessarily in the discrete set)."""
    n = h.shape[0]
    # thresholds sigma2 / h_i sorted ascending
    inv = sigma2 / np.clip(h, 1e-8, None)
    order = np.argsort(inv)
    inv_sorted = inv[order]
    # find water level mu so that sum((mu - inv_sorted_k)^+) = P_max_total
    p = np.zeros(n)
    for k in range(n, 0, -1):
        # assume top k users are active
        mu = (p_max_total + inv_sorted[:k].sum()) / k
        if mu >= inv_sorted[k - 1]:
            p_sorted = np.maximum(mu - inv_sorted[:k], 0.0)
            p[order[:k]] = p_sorted
            return p
    return p  # all zero (should not happen for reasonable p_max_total)


# =============================================================================
# Tabular Q-learning baseline
# =============================================================================

class TabularQAgent:
    """Tabular Q-learning with discretized channel-state.
    Discretizes each user's channel gain into `n_bins` bins over [h_min, h_max],
    yielding a state space of size n_bins^N.  The joint action space is 4^N
    (same as DQN).  For N=3 with 5 bins: 125 states * 64 actions = 8 000 entries.
    For N=5 with 5 bins: 3 125 * 1024 = 3.2 M entries — intractable in memory
    and in samples for the training budgets considered here.  We therefore
    report tabular Q-learning only for N=3.
    """

    def __init__(
        self,
        env: WirelessEnv,
        rng: np.random.Generator,
        n_bins: int = 5,
        lr: float = 0.1,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 20_000,
    ):
        self.env = env
        self.rng = rng
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.N = env.cfg.N
        self.n_states = n_bins ** self.N
        self.n_actions = env.n_actions
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.step_count = 0
        self.bin_edges = np.linspace(
            env.cfg.h_min, env.cfg.h_max, n_bins + 1
        )
        # Widen last edge to catch h_max exactly
        self.bin_edges[-1] = env.cfg.h_max + 1e-6

    def _state_index(self, h: np.ndarray) -> int:
        # digitize returns 1..n_bins; we want 0..n_bins-1
        idx = np.clip(np.digitize(h, self.bin_edges) - 1, 0, self.n_bins - 1)
        s = 0
        for k in idx:
            s = s * self.n_bins + int(k)
        return s

    def eps(self) -> float:
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, h: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps():
            return int(self.rng.integers(0, self.n_actions))
        s = self._state_index(h)
        return int(np.argmax(self.Q[s]))

    def update(self, h: np.ndarray, a: int, r: float, h2: np.ndarray, done: bool):
        s = self._state_index(h)
        s2 = self._state_index(h2)
        target = r + (0.0 if done else self.gamma * self.Q[s2].max())
        self.Q[s, a] += self.lr * (target - self.Q[s, a])
        self.step_count += 1


def train_tabular_q(
    env_cfg: EnvConfig,
    n_episodes: int,
    seed: int,
    n_bins: int = 5,
) -> tuple[TabularQAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    env = WirelessEnv(env_cfg, rng_env)
    agent = TabularQAgent(env, rng_agent, n_bins=n_bins)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(h)
            h2, r, done, _ = env.step(a)
            agent.update(h, a, r, h2, done)
            h = h2
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


def act_waterfilling(env: WirelessEnv, h: np.ndarray) -> int:
    """Water-filling with continuous power under sum-power budget = max discrete
    sum (= N * max(power_levels) = N * 3 W). Then project to the nearest
    discrete joint action.
    """
    p_max_total = env.cfg.N * max(env.cfg.power_levels)
    p_cont = waterfilling_powers(h, p_max_total, env.cfg.sigma2)
    p_disc = np.clip(np.round(p_cont), 0, max(env.cfg.power_levels)).astype(np.int64)
    # Encode as base-M index
    M = env.M
    idx = 0
    for pi in p_disc:
        idx = idx * M + int(pi)
    return idx


def waterfilling_continuous_powers(env: WirelessEnv, h: np.ndarray) -> np.ndarray:
    """Return the continuous (non-projected) water-filling power vector for the
    'theoretical upper bound' reference in the paper."""
    p_max_total = env.cfg.N * max(env.cfg.power_levels)
    return waterfilling_powers(h, p_max_total, env.cfg.sigma2)


# =============================================================================
# WMMSE (Shi, Razaviyayn, Luo, He 2011; specialised to scalar interference channel).
# Used as a classical iterative baseline in the interference-limited regime.
# In orthogonal / noise-limited regime it collapses to water-filling.
# =============================================================================

def wmmse_powers(
    h: np.ndarray,
    p_max_per_user: float,
    sigma2: float,
    n_iters: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """WMMSE iteration for sum-rate maximisation on a scalar interference channel.
    Model:  SINR_i = p_i h_i / (sigma^2 + sum_{j != i} p_j h_j),
    with the stylised assumption that user j interferes with user i at gain h_j
    (matching the environment). Iterations (Shi et al. 2011, Sec. II):

        v_i^2 = p_i,  g_i = sqrt(h_i)
        u_i   = g_i v_i / (sum_k h_k v_k^2 + sigma^2)
        w_i   = 1 / (1 - u_i g_i v_i)   (== 1 + SINR_i)
        v_i   = (w_i u_i) / (g_i * sum_k w_k u_k^2), then clipped to [0, sqrt(P_max)].

    Returns the fixed-point per-user power vector. Guaranteed to converge (weakly)
    to a KKT point of sum-rate maximisation; not globally optimal, but a standard
    strong classical baseline in the wireless-DRL literature.
    """
    N = h.shape[0]
    g = np.sqrt(np.clip(h, 1e-12, None))
    p = np.full(N, p_max_per_user * 0.5)   # warm-start at half-max
    v = np.sqrt(p)
    for _ in range(n_iters):
        v_prev = v.copy()
        Iy = (h * v ** 2).sum() + sigma2   # total received energy at each user's "own" front-end
        u = g * v / Iy
        w = 1.0 / np.clip(1.0 - u * g * v, 1e-12, None)
        denom = g * (w * u ** 2).sum()
        v_new = w * u / np.clip(denom, 1e-12, None)
        v = np.clip(v_new, 0.0, np.sqrt(p_max_per_user))
        if np.abs(v - v_prev).max() < tol:
            break
    return v ** 2


def act_wmmse(env: WirelessEnv, h: np.ndarray) -> int:
    """WMMSE action: solve the continuous problem, then project to discrete
    action table. In the interference-free regime this collapses to water-filling
    (same optimum) so `act_wmmse` and `act_waterfilling` will agree there.
    """
    p_max_per_user = float(max(env.cfg.power_levels))
    p_cont = wmmse_powers(h, p_max_per_user, env.cfg.sigma2)
    p_disc = np.clip(np.round(p_cont), 0, max(env.cfg.power_levels)).astype(np.int64)
    idx = 0
    M = env.M
    for pi in p_disc:
        idx = idx * M + int(pi)
    return idx


def wmmse_continuous_powers(env: WirelessEnv, h: np.ndarray) -> np.ndarray:
    """Continuous (non-projected) WMMSE per-user powers, matching the API of
    `waterfilling_continuous_powers`."""
    p_max_per_user = float(max(env.cfg.power_levels))
    return wmmse_powers(h, p_max_per_user, env.cfg.sigma2)


# =============================================================================
# Multi-cell environment: K cells with one user per cell, path loss + Rayleigh
# fading + inter-cell interference. Standard model for the JSAC/TSP-tier
# wireless-DRL literature (Nasir & Guo 2019; Shen et al. 2020; Eisen & Ribeiro
# 2020). Compared with the single-cell env, this is the physically realistic
# setting: signals decay with distance, and neighbouring cells cause structural
# interference that is not resolvable by orthogonalisation.
#
# Design:
#   - K cells, one user per cell (BS-k serves user-k).
#   - Distance matrix D: D[k,k']  = distance from BS-k to user-k'.
#       * D[k,k]   = d_intra (own-cell distance, default 1.0)
#       * D[k,k']  ~ Uniform(d_inter_min, d_inter_max)  (k != k')
#     Distances are sampled once at env init and held fixed (fixed topology).
#   - Path loss: PL(d) = d^{-alpha}, alpha = 3.0 (dense urban).
#   - Per-link Rayleigh fading redrawn every step: |h_{k',k}(t)|^2 ~ Exp(mean).
#   - Effective gain: g_{k',k}(t) = PL(D[k',k]) * |h_{k',k}(t)|^2.
#   - Rate at user-k: log2(1 + p_k g_{k,k} / (sigma^2 + sum_{k' != k} p_{k'} g_{k',k})).
#   - Reward: sum-rate - lambda * sum_power.
# Observation exposed to the policy: for each user, its own effective gain
# g_{k,k}(t) (matches the single-cell env's "own channel" observation).
# =============================================================================


@dataclass
class MultiCellConfig:
    K: int = 7                          # number of cells
    power_levels: tuple[int, ...] = (0, 1, 2, 3)
    sigma2: float = 1.0
    lam: float = 0.1
    ep_len: int = 100
    d_intra: float = 1.0
    d_inter_min: float = 2.0
    d_inter_max: float = 4.0
    path_loss_alpha: float = 3.0
    rayleigh_mean: float = 1.0
    arrival_rate: float = 1.0


class MultiCellEnv:
    """K-cell wireless environment with path loss + Rayleigh + inter-cell
    interference. Topology (distance matrix) is fixed per env instance;
    fast-fading is redrawn every step.
    """

    def __init__(self, cfg: MultiCellConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.M = len(cfg.power_levels)
        self.n_actions = self.M ** cfg.K
        self._action_table = np.array(
            list(itertools.product(cfg.power_levels, repeat=cfg.K)),
            dtype=np.float32,
        )
        # Distance matrix D[k',k]: BS-k' -> user-k
        K = cfg.K
        D = self.rng.uniform(cfg.d_inter_min, cfg.d_inter_max, size=(K, K)).astype(np.float32)
        np.fill_diagonal(D, cfg.d_intra)
        self.D = D
        self.pathloss = np.power(D, -cfg.path_loss_alpha)   # (K, K)
        self.queues = np.zeros(K, dtype=np.float64)
        self.t = 0
        self._draw_fading()

    def _draw_fading(self) -> None:
        K = self.cfg.K
        # |h_{k',k}|^2 ~ Exp(rayleigh_mean) for each link
        self.fading = self.rng.exponential(self.cfg.rayleigh_mean, size=(K, K)).astype(np.float32)
        self.gains = self.pathloss * self.fading            # (K, K) effective link gains
        self.own_gains = np.diag(self.gains).copy()          # what the policy sees per user

    def reset(self) -> np.ndarray:
        self.t = 0
        self.queues[:] = 0.0
        self._draw_fading()
        return self.own_gains.copy()

    def action_vector(self, a: int) -> np.ndarray:
        return self._action_table[a]

    def step(self, a: int) -> tuple[np.ndarray, float, bool, dict]:
        p = self._action_table[a]                            # (K,)
        # signal at user-k = p_k * g_{k,k}
        signal = p * self.own_gains
        # total interference at user-k: sum_{k' != k} p_{k'} * g_{k',k}
        # Received power per user (from each BS): p_{k'} * gains[k',k], summed over k'
        total_rx = self.gains.T @ p                          # (K,) — total received by each user
        interference = total_rx - signal                     # remove own-cell part
        interference = np.clip(interference, 0.0, None)
        sinr = signal / np.clip(interference + self.cfg.sigma2, 1e-9, None)
        rate = np.log2(1.0 + sinr)                            # (K,)
        reward = float(rate.sum() - self.cfg.lam * p.sum())

        arrivals = self.rng.poisson(self.cfg.arrival_rate, size=self.cfg.K).astype(np.float64)
        self.queues = np.maximum(self.queues + arrivals - rate.astype(np.float64), 0.0)

        info = {
            "rate": rate.copy(),
            "power": p.copy(),
            "queue": self.queues.copy(),
            "sinr": sinr.copy(),
        }
        self.t += 1
        done = self.t >= self.cfg.ep_len
        self._draw_fading()
        return self.own_gains.copy(), reward, done, info


def act_random_multi(env: MultiCellEnv, h: np.ndarray) -> int:
    return int(env.rng.integers(0, env.n_actions))


def act_fixed_multi(env: MultiCellEnv, h: np.ndarray) -> int:
    target = np.full(env.cfg.K, 2, dtype=np.int64)
    for idx, vec in enumerate(env._action_table):
        if np.array_equal(vec.astype(np.int64), target):
            return idx
    raise ValueError("2W fixed action not in multi-cell action table")


def waterfilling_multicell(h: np.ndarray, p_max_per_cell: float, sigma2: float) -> np.ndarray:
    """Water-filling in multi-cell **treating interference as noise**: a
    per-cell water-filling using only the own-cell gain. This is a stronger
    baseline than uniform allocation but weaker than a proper coupled algorithm
    that iterates on the interference. Included for reference alongside WMMSE.
    """
    K = h.shape[0]
    p_total = K * p_max_per_cell
    return waterfilling_powers(h, p_total, sigma2)


def act_waterfilling_multi(env: MultiCellEnv, h: np.ndarray) -> int:
    p_max_per_cell = float(max(env.cfg.power_levels))
    p_cont = waterfilling_multicell(h, p_max_per_cell, env.cfg.sigma2)
    p_disc = np.clip(np.round(p_cont), 0, max(env.cfg.power_levels)).astype(np.int64)
    idx = 0
    M = env.M
    for pi in p_disc:
        idx = idx * M + int(pi)
    return idx


def wmmse_multicell(gains: np.ndarray, p_max_per_cell: float, sigma2: float,
                    n_iters: int = 50, tol: float = 1e-6) -> np.ndarray:
    """WMMSE for a multi-cell system, using the full gain matrix (K x K).
    gains[k', k] = effective channel gain from BS-k' to user-k.
    Objective: max sum_k log(1 + p_k gains[k,k] / (sigma^2 + sum_{k'!=k} p_{k'} gains[k',k])).
    """
    K = gains.shape[0]
    own = np.clip(np.diag(gains), 1e-12, None)               # (K,)
    g_own_sqrt = np.sqrt(own)
    v = np.sqrt(np.full(K, p_max_per_cell * 0.5))
    for _ in range(n_iters):
        v_prev = v.copy()
        # Total received power at user-k from all BSs
        # I_k = sum_{k'} gains[k', k] v_{k'}^2  + sigma^2
        rx = gains.T @ (v ** 2) + sigma2                      # (K,)
        u = g_own_sqrt * v / rx                                # (K,)
        w = 1.0 / np.clip(1.0 - u * g_own_sqrt * v, 1e-12, None)
        # tx update: v_k = (w_k u_k g_own_sqrt_k) / (sum_j w_j u_j^2 gains[k, j])
        denom = gains @ (w * u ** 2)                           # (K,) — gains[k,j] * (w u^2)_j
        v_new = (w * u * g_own_sqrt) / np.clip(denom, 1e-12, None)
        v = np.clip(v_new, 0.0, np.sqrt(p_max_per_cell))
        if np.abs(v - v_prev).max() < tol:
            break
    return v ** 2


def act_wmmse_multi(env: MultiCellEnv, h: np.ndarray) -> int:
    p_max_per_cell = float(max(env.cfg.power_levels))
    p_cont = wmmse_multicell(env.gains, p_max_per_cell, env.cfg.sigma2)
    p_disc = np.clip(np.round(p_cont), 0, max(env.cfg.power_levels)).astype(np.int64)
    idx = 0
    M = env.M
    for pi in p_disc:
        idx = idx * M + int(pi)
    return idx


def evaluate_multicell_policy(
    env: MultiCellEnv,
    policy,                # callable, agent, or "wmmse_continuous"
    n_eval_episodes: int = 20,
    bandwidth_mhz: float = 1.0,
) -> dict:
    """Evaluate a multi-cell policy. Same interface as `evaluate_policy`."""
    is_agent = isinstance(policy, (DQNAgent, TabularQAgent, NeuralBanditAgent,
                                    IndependentQLAgent, REGNNAgent))
    is_wmmse_cont = isinstance(policy, str) and policy == "wmmse_continuous"
    per_user_rate_sum = np.zeros(env.cfg.K)
    per_user_queue_sum = np.zeros(env.cfg.K)
    per_step_jain: list[float] = []
    total_power = 0.0
    total_steps = 0
    for _ in range(n_eval_episodes):
        h = env.reset()
        done = False
        while not done:
            if is_wmmse_cont:
                p = wmmse_multicell(env.gains, float(max(env.cfg.power_levels)),
                                     env.cfg.sigma2)
                signal = p * env.own_gains
                total_rx = env.gains.T @ p
                interf = total_rx - signal
                interf = np.clip(interf, 0.0, None)
                sinr = signal / np.clip(interf + env.cfg.sigma2, 1e-9, None)
                rate = np.log2(1.0 + sinr).astype(np.float64)
                arrivals = env.rng.poisson(env.cfg.arrival_rate, size=env.cfg.K).astype(np.float64)
                env.queues = np.maximum(env.queues + arrivals - rate, 0.0)
                env.t += 1
                done = env.t >= env.cfg.ep_len
                env._draw_fading()
                h = env.own_gains.copy()
                per_user_rate_sum += rate
                per_user_queue_sum += env.queues
                per_step_jain.append(jain(rate))
                total_power += float(p.sum())
                total_steps += 1
                continue
            if is_agent:
                a = policy.act(h, greedy=True)
            else:
                a = policy(env, h)
            h, r, done, info = env.step(a)
            per_user_rate_sum += info["rate"]
            per_user_queue_sum += info["queue"]
            per_step_jain.append(jain(info["rate"]))
            total_power += float(info["power"].sum())
            total_steps += 1
    per_user_avg_rate = per_user_rate_sum / total_steps
    per_user_avg_queue = per_user_queue_sum / total_steps
    tot_rate = per_user_avg_rate.sum()
    total_bits = per_user_rate_sum.sum()
    ee = float(total_bits / total_power) if total_power > 0 else 0.0
    return {
        "throughput_bpu": float(tot_rate),
        "throughput_mbps": float(tot_rate * bandwidth_mhz),
        "jain": float(np.mean(per_step_jain)),
        "energy_efficiency": ee,
        "avg_latency": float(per_user_avg_queue.mean()),
        "per_user_throughput": per_user_avg_rate.tolist(),
        "per_user_latency": per_user_avg_queue.tolist(),
    }


def train_iql_multicell(
    env_cfg: MultiCellConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[IndependentQLAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = MultiCellEnv(env_cfg, rng_env)
    agent = IndependentQLAgent(env_cfg.K, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            per_user = agent.act_per_user(h)
            joint_a = 0
            for a in per_user:
                joint_a = joint_a * agent._M + int(a)
            h2, r, done, _ = env.step(joint_a)
            agent.push(h, per_user, r, h2, done)
            h = h2
            agent.step_count += 1
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


def train_regnn_multicell(
    env_cfg: MultiCellConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[REGNNAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = MultiCellEnv(env_cfg, rng_env)
    agent = REGNNAgent(env_cfg.K, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            per_user = agent.act_per_user(h)
            joint_a = 0
            for a in per_user:
                joint_a = joint_a * agent._M + int(a)
            h2, r, done, _ = env.step(joint_a)
            agent.push(h, per_user, r, h2, done)
            h = h2
            agent.step_count += 1
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


def train_dqn_multicell(
    env_cfg: MultiCellConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[DQNAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = MultiCellEnv(env_cfg, rng_env)
    agent = DQNAgent(env_cfg.K, env.n_actions, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(h)
            h2, r, done, _ = env.step(a)
            agent.buffer.push(h, a, r, h2, done)
            h = h2
            agent.step_count += 1
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


def train_neural_bandit_multicell(
    env_cfg: MultiCellConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[NeuralBanditAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = MultiCellEnv(env_cfg, rng_env)
    agent = NeuralBanditAgent(env_cfg.K, env.n_actions, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(h)
            h2, r, done, _ = env.step(a)
            agent.buffer.push(h, a, r, h2, done)
            h = h2
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


# =============================================================================
# DQN
# =============================================================================

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, int] = (64, 128)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling DQN architecture (Wang et al., ICML 2016): factors Q(s,a) into
    V(s) + A(s,a) - mean_a A(s,a). Improves value estimation on problems where
    the advantage of each action is small relative to state value."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, int] = (64, 128)):
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(h2, 1)
        self.advantage_head = nn.Linear(h2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        v = self.value_head(z)                     # (B, 1)
        a = self.advantage_head(z)                 # (B, |A|)
        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a) -- keeps V and A identifiable
        return v + a - a.mean(dim=1, keepdim=True)


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 10_000
    batch_size: int = 32
    target_update_steps: int = 100
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    hidden: tuple[int, int] = (64, 128)
    warmup_steps: int = 500
    # Rainbow-lite toggles (default off = vanilla DQN):
    use_double: bool = False       # Double DQN target (van Hasselt 2016)
    use_dueling: bool = False      # Dueling architecture (Wang 2016)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def push(self, s, a, r, s2, d):
        self.obs[self.ptr] = s
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = s2
        self.done[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
            torch.from_numpy(self.rew[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.done[idx]),
        )


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: DQNConfig,
        rng: np.random.Generator,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.n_actions = n_actions
        net_cls = DuelingQNet if cfg.use_dueling else QNet
        self.q = net_cls(obs_dim, n_actions, cfg.hidden).to(device)
        self.q_target = net_cls(obs_dim, n_actions, cfg.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, rng)
        self.step_count = 0

    def eps(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps():
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q = self.q(x)[0]
        return int(torch.argmax(q).item())

    def update(self) -> float | None:
        if self.buffer.size < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        s2 = s2.to(self.device); d = d.to(self.device)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.use_double:
                # Double DQN: select action with online net, evaluate with target net.
                a_next = self.q(s2).argmax(dim=1, keepdim=True)
                q_next = self.q_target(s2).gather(1, a_next).squeeze(1)
            else:
                q_next = self.q_target(s2).max(dim=1).values
            target = r + (1.0 - d) * self.cfg.gamma * q_next
        loss = F.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.step_count % self.cfg.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())


# =============================================================================
# Neural Contextual Bandit (Riquelme et al. 2018-style)
# =============================================================================

class NeuralBanditAgent:
    """Contextual bandit with a neural regressor over |A| actions.
    Predicts E[r | s, a] directly (no bootstrapping, no discount, no target net).
    Uses epsilon-greedy exploration. Since our transition kernel is state-
    independent, this is the theoretically appropriate function class.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: DQNConfig,
        rng: np.random.Generator,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.n_actions = n_actions
        self.q = QNet(obs_dim, n_actions, cfg.hidden).to(device)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        # No target network -- bandit has no bootstrapping.
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, rng)
        self.step_count = 0

    def eps(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps():
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q = self.q(x)[0]
        return int(torch.argmax(q).item())

    def update(self) -> float | None:
        if self.buffer.size < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        s, a, r, _, _ = self.buffer.sample(self.cfg.batch_size)  # ignore s2, d for bandit
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_sa, r)   # direct regression on observed reward
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.step_count += 1
        return float(loss.item())


class IndependentQLAgent:
    """Independent Q-Learning (Tan 1993, canonical MARL baseline; Nasir & Guo 2019
    template for wireless). Per-user DQN with own-channel-only observation and
    joint reward. Trivially scales to any N because the per-user action space is
    fixed at |power_levels|, independent of N.

    Trade-off: non-stationarity — from the perspective of user i, the environment
    changes as the other users' policies evolve. This is a well-known MARL
    pathology; IQL still works well when the cooperative reward is shared and the
    problem admits a broadly-independent per-user structure.

    Design choices (matched to the paper's centralised DQN):
    - Same hidden widths (64, 128) and same Adam LR
    - Shared eps schedule (all users decay together)
    - Optional Double / Dueling per user (Rainbow-lite)
    - Target-net sync every ``target_update_steps``.
    """

    def __init__(
        self,
        N: int,
        cfg: DQNConfig,
        rng: np.random.Generator,
        device: str = "cpu",
    ):
        self.N = N
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.n_actions_per_user = 4  # {0,1,2,3} W
        net_cls = DuelingQNet if cfg.use_dueling else QNet
        self.qs = [net_cls(1, self.n_actions_per_user, cfg.hidden).to(device) for _ in range(N)]
        self.q_targets = [net_cls(1, self.n_actions_per_user, cfg.hidden).to(device) for _ in range(N)]
        for qt, q in zip(self.q_targets, self.qs):
            qt.load_state_dict(q.state_dict())
            for p in qt.parameters():
                p.requires_grad_(False)
        self.opts = [torch.optim.Adam(q.parameters(), lr=cfg.lr) for q in self.qs]
        self.buffers = [ReplayBuffer(cfg.buffer_size, 1, rng) for _ in range(N)]
        self.step_count = 0
        # For encoding joint action: base-M positional index (matches env action table).
        self._M = self.n_actions_per_user

    def eps(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def _per_user_action(self, h: np.ndarray, greedy: bool = False) -> np.ndarray:
        eps = 0.0 if greedy else self.eps()
        out = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N):
            if self.rng.random() < eps:
                out[i] = int(self.rng.integers(0, self.n_actions_per_user))
            else:
                with torch.no_grad():
                    x = torch.tensor([[float(h[i])]], dtype=torch.float32, device=self.device)
                    q = self.qs[i](x)[0]
                out[i] = int(torch.argmax(q).item())
        return out

    def act(self, h: np.ndarray, greedy: bool = False) -> int:
        """Return the joint action index (matches env action-table encoding)."""
        per_user = self._per_user_action(h, greedy=greedy)
        idx = 0
        for a in per_user:
            idx = idx * self._M + int(a)
        return idx

    def act_per_user(self, h: np.ndarray, greedy: bool = False) -> np.ndarray:
        return self._per_user_action(h, greedy=greedy)

    def push(self, h: np.ndarray, actions_per_user: np.ndarray,
             r: float, h2: np.ndarray, done: bool) -> None:
        for i in range(self.N):
            self.buffers[i].push(
                np.array([h[i]], dtype=np.float32),
                int(actions_per_user[i]),
                float(r),
                np.array([h2[i]], dtype=np.float32),
                done,
            )

    def update(self) -> float | None:
        losses: list[float] = []
        for i in range(self.N):
            buf = self.buffers[i]
            if buf.size < max(self.cfg.batch_size, self.cfg.warmup_steps):
                continue
            s, a, r, s2, d = buf.sample(self.cfg.batch_size)
            s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
            s2 = s2.to(self.device); d = d.to(self.device)
            q_sa = self.qs[i](s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if self.cfg.use_double:
                    a_next = self.qs[i](s2).argmax(dim=1, keepdim=True)
                    q_next = self.q_targets[i](s2).gather(1, a_next).squeeze(1)
                else:
                    q_next = self.q_targets[i](s2).max(dim=1).values
                target = r + (1.0 - d) * self.cfg.gamma * q_next
            loss = F.mse_loss(q_sa, target)
            self.opts[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.qs[i].parameters(), 10.0)
            self.opts[i].step()
            if self.step_count % self.cfg.target_update_steps == 0:
                self.q_targets[i].load_state_dict(self.qs[i].state_dict())
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else None


def train_iql(
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[IndependentQLAgent, list[float]]:
    """Train an Independent Q-Learning MARL agent (per-user DQN, shared reward).
    Scales to any N without exploding the per-user action space.
    """
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = WirelessEnv(env_cfg, rng_env)
    agent = IndependentQLAgent(env_cfg.N, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            per_user = agent.act_per_user(h)
            joint_a = 0
            for a in per_user:
                joint_a = joint_a * agent._M + int(a)
            h2, r, done, _ = env.step(joint_a)
            agent.push(h, per_user, r, h2, done)
            h = h2
            agent.step_count += 1
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


# =============================================================================
# REGNN-lite: permutation-equivariant GNN policy over the interference graph.
# Inspired by Eisen & Ribeiro (TSP 2020) and Shen et al. (JSAC 2020); simplified
# to a two-layer message-passing net with edge features = neighbour channel gain.
# Outputs per-user logits over the discrete power set. Trained via IQL-style
# Q-learning (shared params across users, joint reward). One network handles any
# N, since message-passing is permutation-equivariant and shape-agnostic.
# =============================================================================


class GraphQNet(nn.Module):
    """Permutation-equivariant GNN Q-function over N users.
    Node feature: own channel gain h_i.
    Edge feature: h_j (interferer's gain).
    Output: Q-values over per-user action set, shape (B, N, n_actions_per_user).
    """

    def __init__(self, n_actions_per_user: int = 4, hidden: int = 32):
        super().__init__()
        # Message: concat(h_i, edge_ij, h_j) -> hidden
        self.mp1 = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Layer 2: concat(node_prev, edge_ij, neighbor_prev) -> hidden
        self.mp2 = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions_per_user)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N), channel gains
        B, N = h.shape
        h_ = h.unsqueeze(-1)                         # (B, N, 1)
        h_i = h_.unsqueeze(2).expand(B, N, N, 1)      # (B, N, N, 1) each row is user i
        h_j = h_.unsqueeze(1).expand(B, N, N, 1)      # (B, N, N, 1) each column is user j
        edge = h_j                                    # edge_ij = interferer's gain

        eye = torch.eye(N, device=h.device).bool().view(1, N, N, 1).expand(B, N, N, 1)

        msg_in = torch.cat([h_i, edge, h_j], dim=-1)  # (B, N, N, 3)
        msg = self.mp1(msg_in)                        # (B, N, N, hidden)
        msg = msg.masked_fill(eye.expand_as(msg), 0.0)
        # Aggregate: mean over neighbors (excluding self) -> divide by (N-1)
        aggr1 = msg.sum(dim=2) / max(N - 1, 1)         # (B, N, hidden)

        aggr1_i = aggr1.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, hidden)
        aggr1_j = aggr1.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, hidden)
        msg2_in = torch.cat([aggr1_i, edge, aggr1_j], dim=-1)
        msg2 = self.mp2(msg2_in)
        msg2 = msg2.masked_fill(eye.expand_as(msg2), 0.0)
        aggr2 = msg2.sum(dim=2) / max(N - 1, 1)        # (B, N, hidden)

        return self.head(aggr2)                        # (B, N, n_actions_per_user)


class REGNNAgent:
    """GNN policy trained via IQL-style Q-learning with shared parameters across
    all users (permutation-equivariance means one network policies any user).
    Uses one shared replay buffer whose entries are (h_full, i, a_i, r, h2_full, d):
    at update time each user's Q-value is read off the GNN's per-user output.

    Scales in N essentially for free: the GNN parameter count is independent of N,
    and the per-user action space is fixed at |power_levels|.
    """

    def __init__(
        self,
        N: int,
        cfg: DQNConfig,
        rng: np.random.Generator,
        device: str = "cpu",
        hidden_gnn: int = 32,
    ):
        self.N = N
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.n_actions_per_user = 4
        self.q = GraphQNet(self.n_actions_per_user, hidden_gnn).to(device)
        self.q_target = GraphQNet(self.n_actions_per_user, hidden_gnn).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        # One shared buffer: obs = full h vector, action = per-user vector packed as N ints.
        self.obs = np.zeros((cfg.buffer_size, N), dtype=np.float32)
        self.acts = np.zeros((cfg.buffer_size, N), dtype=np.int64)
        self.rew = np.zeros(cfg.buffer_size, dtype=np.float32)
        self.next_obs = np.zeros((cfg.buffer_size, N), dtype=np.float32)
        self.done = np.zeros(cfg.buffer_size, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.step_count = 0
        self._M = self.n_actions_per_user

    def eps(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def _per_user_action(self, h: np.ndarray, greedy: bool = False) -> np.ndarray:
        eps = 0.0 if greedy else self.eps()
        if not greedy and self.rng.random() < eps:
            return self.rng.integers(0, self.n_actions_per_user, size=self.N).astype(np.int64)
        with torch.no_grad():
            x = torch.from_numpy(h).float().unsqueeze(0).to(self.device)   # (1, N)
            q = self.q(x)[0]                                                # (N, n_a)
            return q.argmax(dim=1).cpu().numpy().astype(np.int64)

    def act(self, h: np.ndarray, greedy: bool = False) -> int:
        per_user = self._per_user_action(h, greedy=greedy)
        idx = 0
        for a in per_user:
            idx = idx * self._M + int(a)
        return idx

    def act_per_user(self, h: np.ndarray, greedy: bool = False) -> np.ndarray:
        return self._per_user_action(h, greedy=greedy)

    def push(self, h: np.ndarray, per_user: np.ndarray, r: float,
             h2: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = h
        self.acts[self.ptr] = per_user
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = h2
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.cfg.buffer_size
        self.size = min(self.size + 1, self.cfg.buffer_size)

    def update(self) -> float | None:
        if self.size < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        idx = self.rng.integers(0, self.size, size=self.cfg.batch_size)
        s = torch.from_numpy(self.obs[idx]).to(self.device)       # (B, N)
        a = torch.from_numpy(self.acts[idx]).to(self.device)      # (B, N)
        r = torch.from_numpy(self.rew[idx]).to(self.device)       # (B,)
        s2 = torch.from_numpy(self.next_obs[idx]).to(self.device)
        d = torch.from_numpy(self.done[idx]).to(self.device)

        q = self.q(s)                                              # (B, N, n_a)
        # Per-user Q(s, a_i)
        q_sa = q.gather(2, a.unsqueeze(-1)).squeeze(-1)            # (B, N)

        with torch.no_grad():
            if self.cfg.use_double:
                a_next = self.q(s2).argmax(dim=2, keepdim=True)    # (B, N, 1)
                q_next = self.q_target(s2).gather(2, a_next).squeeze(-1)
            else:
                q_next = self.q_target(s2).max(dim=2).values        # (B, N)
            # Shared reward: broadcast r over the N users; standard IQL trick.
            r_bc = r.unsqueeze(1).expand_as(q_sa)
            d_bc = d.unsqueeze(1).expand_as(q_sa)
            target = r_bc + (1.0 - d_bc) * self.cfg.gamma * q_next

        loss = F.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.step_count % self.cfg.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())


def train_regnn(
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[REGNNAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = WirelessEnv(env_cfg, rng_env)
    agent = REGNNAgent(env_cfg.N, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            per_user = agent.act_per_user(h)
            joint_a = 0
            for a in per_user:
                joint_a = joint_a * agent._M + int(a)
            h2, r, done, _ = env.step(joint_a)
            agent.push(h, per_user, r, h2, done)
            h = h2
            agent.step_count += 1
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


def train_neural_bandit(
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[NeuralBanditAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = WirelessEnv(env_cfg, rng_env)
    agent = NeuralBanditAgent(env_cfg.N, env.n_actions, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for _ in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(h)
            h2, r, done, _ = env.step(a)
            agent.buffer.push(h, a, r, h2, done)
            h = h2
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


# =============================================================================
# Training and evaluation
# =============================================================================

@dataclass
class RunResult:
    method: str
    N: int
    seed: int
    throughput_bits_per_use: float          # avg sum-rate per step (bits/channel-use)
    throughput_mbps: float                  # scaled by bandwidth to Mbps for paper
    jain_index: float
    energy_efficiency: float                # bits per Joule (dimensionless in this sim)
    avg_latency: float                      # avg queue occupancy
    per_user_throughput: list[float] = field(default_factory=list)
    per_user_latency: list[float] = field(default_factory=list)
    train_curve: list[float] = field(default_factory=list)   # per-episode reward (DQN only)


def jain(rates: np.ndarray) -> float:
    s = rates.sum()
    if s <= 0:
        return 0.0
    return float(s ** 2 / (len(rates) * (rates ** 2).sum() + 1e-12))


def evaluate_policy(
    env: WirelessEnv,
    policy,                 # callable(env, h) -> action, or a DQNAgent, or "wf_continuous"
    n_eval_episodes: int = 20,
    bandwidth_mhz: float = 1.0,
) -> dict:
    """Evaluate a policy for `n_eval_episodes` × ep_len steps. Returns:
    - throughput (avg sum-rate per step, bits/channel-use)
    - Jain's fairness index averaged per-step over rates > 0 users (standard
      wireless-literature convention: instantaneous fairness, then time-average)
    - energy efficiency = total bits transmitted / total energy consumed
    - per-user average rates and queue lengths

    The special string `"wf_continuous"` evaluates the theoretical
    continuous-power water-filling upper bound (not projected to discrete set).
    """
    is_agent = isinstance(policy, (DQNAgent, TabularQAgent, NeuralBanditAgent,
                                    IndependentQLAgent, REGNNAgent))
    is_wf_continuous = isinstance(policy, str) and policy == "wf_continuous"
    per_user_rate_sum = np.zeros(env.cfg.N)
    per_user_queue_sum = np.zeros(env.cfg.N)
    per_step_jain: list[float] = []
    total_power = 0.0
    total_steps = 0
    for _ in range(n_eval_episodes):
        h = env.reset()
        done = False
        while not done:
            if is_wf_continuous:
                # continuous water-filling: bypass action table entirely
                p = waterfilling_continuous_powers(env, h)
                if env.cfg.interference:
                    signal = p * h
                    total_ipn = signal.sum() + env.cfg.sigma2 - signal
                    sinr = signal / np.clip(total_ipn, 1e-9, None)
                    rate = np.log2(1.0 + sinr).astype(np.float64)
                else:
                    snr = p * h / env.cfg.sigma2
                    rate = np.log2(1.0 + snr).astype(np.float64)
                # step the queues and channels manually to preserve dynamics
                arrivals = env.rng.poisson(env.cfg.arrival_rate, size=env.cfg.N).astype(np.float64)
                env.queues = np.maximum(env.queues + arrivals - rate, 0.0)
                env.t += 1
                done = env.t >= env.cfg.ep_len
                env._draw_channels()
                h = env.h.copy()
                per_user_rate_sum += rate
                per_user_queue_sum += env.queues
                per_step_jain.append(jain(rate))
                total_power += float(p.sum())
                total_steps += 1
                continue

            if is_agent:
                a = policy.act(h, greedy=True)
            else:
                a = policy(env, h)
            h, r, done, info = env.step(a)
            per_user_rate_sum += info["rate"]
            per_user_queue_sum += info["queue"]
            per_step_jain.append(jain(info["rate"]))
            total_power += float(info["power"].sum())
            total_steps += 1

    per_user_avg_rate = per_user_rate_sum / total_steps
    per_user_avg_queue = per_user_queue_sum / total_steps
    tot_rate = per_user_avg_rate.sum()
    total_bits = per_user_rate_sum.sum()
    ee = float(total_bits / total_power) if total_power > 0 else 0.0
    return {
        "throughput_bpu": float(tot_rate),
        "throughput_mbps": float(tot_rate * bandwidth_mhz),
        "jain": float(np.mean(per_step_jain)),
        "energy_efficiency": ee,
        "avg_latency": float(per_user_avg_queue.mean()),
        "per_user_throughput": per_user_avg_rate.tolist(),
        "per_user_latency": per_user_avg_queue.tolist(),
    }


def train_dqn(
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> tuple[DQNAgent, list[float]]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed)
    env = WirelessEnv(env_cfg, rng_env)
    agent = DQNAgent(env_cfg.N, env.n_actions, dqn_cfg, rng_agent, device=device)
    curve: list[float] = []
    for ep in range(n_episodes):
        h = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(h)
            h2, r, done, _ = env.step(a)
            agent.buffer.push(h, a, r, h2, done)
            h = h2
            agent.step_count += 1
            agent.update()
            ep_ret += r
        curve.append(ep_ret)
    return agent, curve


def run_experiment(
    N: int,
    seeds: Iterable[int],
    n_episodes: int = 300,
    ep_len: int = 100,
    dqn_cfg: DQNConfig | None = None,
    bandwidth_mhz: float = 1.0,
) -> list[RunResult]:
    dqn_cfg = dqn_cfg or DQNConfig()
    env_cfg = EnvConfig(N=N, ep_len=ep_len)
    results: list[RunResult] = []

    for seed in seeds:
        # -- Random --
        rng = np.random.default_rng(seed + 1)
        env_r = WirelessEnv(env_cfg, rng)
        m = evaluate_policy(env_r, act_random, bandwidth_mhz=bandwidth_mhz)
        results.append(RunResult(
            method="Random", N=N, seed=seed,
            throughput_bits_per_use=m["throughput_bpu"],
            throughput_mbps=m["throughput_mbps"],
            jain_index=m["jain"],
            energy_efficiency=m["energy_efficiency"],
            avg_latency=m["avg_latency"],
            per_user_throughput=m["per_user_throughput"],
            per_user_latency=m["per_user_latency"],
        ))

        # -- Fixed 2W --
        rng = np.random.default_rng(seed + 2)
        env_f = WirelessEnv(env_cfg, rng)
        m = evaluate_policy(env_f, act_fixed, bandwidth_mhz=bandwidth_mhz)
        results.append(RunResult(
            method="Fixed", N=N, seed=seed,
            throughput_bits_per_use=m["throughput_bpu"],
            throughput_mbps=m["throughput_mbps"],
            jain_index=m["jain"],
            energy_efficiency=m["energy_efficiency"],
            avg_latency=m["avg_latency"],
            per_user_throughput=m["per_user_throughput"],
            per_user_latency=m["per_user_latency"],
        ))

        # -- Water-Filling (continuous, theoretical upper bound) --
        rng = np.random.default_rng(seed + 3)
        env_w = WirelessEnv(env_cfg, rng)
        m = evaluate_policy(env_w, "wf_continuous", bandwidth_mhz=bandwidth_mhz)
        results.append(RunResult(
            method="Water-Filling", N=N, seed=seed,
            throughput_bits_per_use=m["throughput_bpu"],
            throughput_mbps=m["throughput_mbps"],
            jain_index=m["jain"],
            energy_efficiency=m["energy_efficiency"],
            avg_latency=m["avg_latency"],
            per_user_throughput=m["per_user_throughput"],
            per_user_latency=m["per_user_latency"],
        ))

        # -- DQN --
        agent, curve = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
        rng_eval = np.random.default_rng(seed + 4)
        env_eval = WirelessEnv(env_cfg, rng_eval)
        m = evaluate_policy(env_eval, agent, bandwidth_mhz=bandwidth_mhz)
        results.append(RunResult(
            method="DQN", N=N, seed=seed,
            throughput_bits_per_use=m["throughput_bpu"],
            throughput_mbps=m["throughput_mbps"],
            jain_index=m["jain"],
            energy_efficiency=m["energy_efficiency"],
            avg_latency=m["avg_latency"],
            per_user_throughput=m["per_user_throughput"],
            per_user_latency=m["per_user_latency"],
            train_curve=curve,
        ))

    return results


# =============================================================================
# Epsilon-decay ablation
# =============================================================================

def run_eps_decay_ablation(
    N: int,
    seeds: Iterable[int],
    decay_fractions: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90),
    n_episodes: int = 300,
    ep_len: int = 100,
) -> dict:
    """Interpret decay = 0.99 as 'linearly decay ε over decay_steps such that
    total_steps * (1 - decay) ~= decay_steps'. In practice we sweep the total
    number of decay steps proportional to (1 - decay); slower decay = more
    exploration."""
    curves: dict[float, list[list[float]]] = {d: [] for d in decay_fractions}
    for d in decay_fractions:
        # Map decay to eps_decay_steps: smaller (1-d) ⇒ longer decay
        # Use full training horizon; larger d ⇒ longer decay window
        decay_steps = int(n_episodes * ep_len * d)
        dqn_cfg = DQNConfig(eps_decay_steps=decay_steps)
        env_cfg = EnvConfig(N=N, ep_len=ep_len)
        for seed in seeds:
            _, curve = train_dqn(env_cfg, dqn_cfg, n_episodes, seed)
            curves[d].append(curve)
    return curves


# =============================================================================
# Main entry point (invoked by run_all.py)
# =============================================================================

def summarize(results: list[RunResult]) -> dict:
    by_method: dict[str, list[RunResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for m, rs in by_method.items():
        def stat(xs):
            xs = np.asarray(xs, dtype=np.float64)
            return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1) if len(xs) > 1 else 0.0)}
        summary[m] = {
            "throughput_mbps": stat([r.throughput_mbps for r in rs]),
            "jain": stat([r.jain_index for r in rs]),
            "energy_efficiency": stat([r.energy_efficiency for r in rs]),
            "avg_latency": stat([r.avg_latency for r in rs]),
        }
    return summary


def dump_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        if hasattr(o, "__dict__"):     return o.__dict__
        raise TypeError(f"Not JSON serializable: {type(o)}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


__all__ = [
    "EnvConfig", "WirelessEnv", "DQNConfig", "DQNAgent",
    "act_random", "act_fixed", "act_waterfilling", "act_wmmse",
    "wmmse_powers", "wmmse_continuous_powers",
    "IndependentQLAgent", "train_iql",
    "GraphQNet", "REGNNAgent", "train_regnn",
    "NeuralBanditAgent", "train_neural_bandit",
    "TabularQAgent", "train_tabular_q",
    "train_dqn", "evaluate_policy", "run_experiment", "run_eps_decay_ablation",
    "summarize", "dump_json", "RunResult",
    # Multi-cell
    "MultiCellConfig", "MultiCellEnv",
    "act_random_multi", "act_fixed_multi",
    "act_waterfilling_multi", "act_wmmse_multi",
    "wmmse_multicell", "waterfilling_multicell",
    "evaluate_multicell_policy",
    "train_iql_multicell", "train_regnn_multicell",
    "train_dqn_multicell", "train_neural_bandit_multicell",
]
