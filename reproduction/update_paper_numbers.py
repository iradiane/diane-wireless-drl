"""Read reproduction results and rewrite the LaTeX macros in main.tex."""
from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
SOURCE  = Path(__file__).parent.parent / "source" / "main.tex"


def fmt(mean: float, std: float, dec: int = 3) -> str:
    return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"


def load_summary(N: int) -> dict:
    # Prefer the final (10-seed, long-training) results if they exist.
    final = RESULTS / f"final_N{N}_summary.json"
    if final.exists():
        return json.loads(final.read_text())
    return json.loads((RESULTS / f"main_N{N}_summary.json").read_text())


def build_macros() -> dict[str, str]:
    m3 = load_summary(3)
    m5 = load_summary(5)
    macros: dict[str, str] = {}
    for N, m in [(3, m3), (5, m5)]:
        suffix = "Three" if N == 3 else "Five"
        for method, key in [
            ("DQN", "DQN"), ("Fix", "Fixed"), ("Rnd", "Random"),
            ("WF", "Water-Filling"), ("WFD", "Water-Filling-Discrete"),
            ("Rainbow", "RainbowLite"), ("Bandit", "NeuralBandit"),
            ("IQL", "IQL"), ("REGNN", "REGNN"), ("WMMSE", "WMMSE"),
        ]:
            if key not in m:
                continue
            s = m[key]
            macros[f"{method}Thr{suffix}"]  = fmt(s["throughput_mbps"]["mean"],   s["throughput_mbps"]["std"])
            macros[f"{method}Jain{suffix}"] = fmt(s["jain"]["mean"],              s["jain"]["std"])
            macros[f"{method}EE{suffix}"]   = fmt(s["energy_efficiency"]["mean"], s["energy_efficiency"]["std"])
    # Tabular Q-learning: N=3 only
    if "TabularQ" in m3:
        s = m3["TabularQ"]
        macros["TabThrThree"]  = fmt(s["throughput_mbps"]["mean"],   s["throughput_mbps"]["std"])
        macros["TabJainThree"] = fmt(s["jain"]["mean"],              s["jain"]["std"])
        macros["TabEEThree"]   = fmt(s["energy_efficiency"]["mean"], s["energy_efficiency"]["std"])

    # Scalar convenience macros for abstract prose
    dqn3 = m3["DQN"]
    rnd3 = m3["Random"]
    fix3 = m3["Fixed"]
    wf3  = m3["Water-Filling"]
    macros["DQNThrScalar"]  = f"{dqn3['throughput_mbps']['mean']:.2f}"
    macros["JainScalar"]    = f"{dqn3['jain']['mean']:.2f}"
    if rnd3["throughput_mbps"]["mean"] > 0:
        macros["GainRandomPct"] = f"{100 * (dqn3['throughput_mbps']['mean'] - rnd3['throughput_mbps']['mean']) / rnd3['throughput_mbps']['mean']:.0f}"
    if fix3["throughput_mbps"]["mean"] > 0:
        macros["GainFixedPct"]  = f"{100 * (dqn3['throughput_mbps']['mean'] - fix3['throughput_mbps']['mean']) / fix3['throughput_mbps']['mean']:.0f}"
    if wf3["throughput_mbps"]["mean"] > 0:
        macros["PctOfWFThree"] = f"{100 * dqn3['throughput_mbps']['mean'] / wf3['throughput_mbps']['mean']:.0f}"
    if "DQN" in m5 and "Water-Filling" in m5 and m5["Water-Filling"]["throughput_mbps"]["mean"] > 0:
        macros["PctOfWFFive"] = f"{100 * m5['DQN']['throughput_mbps']['mean'] / m5['Water-Filling']['throughput_mbps']['mean']:.0f}"

    # ---------- IQL / REGNN scaling (Rayleigh + interference, MARL comparison) ----------
    iql_path = RESULTS / "iql_regnn_summary.json"
    if iql_path.exists():
        iql_summary = json.loads(iql_path.read_text())
        for N in (3, 5, 10):
            key = str(N)
            if key not in iql_summary: continue
            suf = {3: "Three", 5: "Five", 10: "Ten"}[N]
            for method, tag in [("IQL", "IQLRay"), ("REGNN", "REGNNRay"),
                                  ("DQN", "DQNRay"), ("NeuralBandit", "BanditRay"),
                                  ("WMMSE", "WMMSERay"), ("Water-Filling", "WFRay")]:
                if method in iql_summary[key]:
                    s = iql_summary[key][method]
                    macros[f"{tag}Thr{suf}"] = fmt(s["throughput_mbps"]["mean"],
                                                     s["throughput_mbps"]["std"])
                    macros[f"{tag}Jain{suf}"] = fmt(s["jain"]["mean"], s["jain"]["std"])

    # ---------- Rayleigh + interference (10 seeds, WMMSE included) ----------
    ray_10 = RESULTS / "rayleigh_interference_summary_10seed.json"
    if ray_10.exists():
        ray = json.loads(ray_10.read_text())
        for method, tag in [("Water-Filling", "WFRayl"), ("WMMSE", "WMMSERayl"),
                             ("NeuralBandit", "BanditRayl"), ("DQN", "DQNRayl"),
                             ("RainbowLite", "RainbowRayl"), ("Fixed", "FixRayl"),
                             ("Random", "RndRayl"), ("TabularQ", "TabRayl")]:
            if method in ray:
                s = ray[method]
                macros[f"{tag}Thr"] = fmt(s["throughput_mbps"]["mean"], s["throughput_mbps"]["std"])
                macros[f"{tag}Jain"] = fmt(s["jain"]["mean"], s["jain"]["std"])

    # ---------- Multi-cell K=7 ----------
    mc_path = RESULTS / "multicell_summary.json"
    if mc_path.exists():
        mc = json.loads(mc_path.read_text())
        for method, tag in [("WF-multi", "WFMulti"), ("WMMSE-multi", "WMMSEMulti"),
                              ("IQL", "IQLMulti"), ("REGNN", "REGNNMulti"),
                              ("DQN", "DQNMulti"), ("NeuralBandit", "BanditMulti"),
                              ("Fixed", "FixMulti"), ("Random", "RndMulti")]:
            if method in mc:
                s = mc[method]
                macros[f"{tag}Thr"] = fmt(s["throughput_mbps"]["mean"], s["throughput_mbps"]["std"])
                macros[f"{tag}Jain"] = fmt(s["jain"]["mean"], s["jain"]["std"])

    # Epsilon-decay ablation
    decay_map = {"0.99": ("EpsDecayNine", "EpsDecayNineFive"),
                 "0.98": ("EpsDecayEightN", "EpsDecayEightNFive"),
                 "0.95": ("EpsDecayFive", "EpsDecayFiveFive"),
                 "0.9":  ("EpsDecayZero", "EpsDecayZeroFive")}
    for N, tag_idx in [(3, 0), (5, 1)]:
        ablation_path = RESULTS / f"eps_ablation_N{N}.json"
        if ablation_path.exists():
            ablation = json.loads(ablation_path.read_text())
            for key, tags in decay_map.items():
                if key in ablation:
                    mean = ablation[key]["final_mean"]
                    std  = ablation[key]["final_std"]
                    macros[tags[tag_idx]] = f"${mean:.1f} \\pm {std:.1f}$"
    return macros


def rewrite_tex(macros: dict[str, str]) -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for name, val in macros.items():
        pattern = re.compile(r"\\newcommand\{\\" + re.escape(name) + r"\}\{[^}]*\}")
        # Use a lambda so backslashes in val are not reinterpreted as regex backrefs
        new_text, n = pattern.subn(lambda _m, v=val, nm=name: f"\\newcommand{{\\{nm}}}{{{v}}}", text)
        if n == 0:
            print(f"WARNING: macro \\{name} not found in main.tex")
        else:
            text = new_text
    SOURCE.write_text(text, encoding="utf-8")
    print(f"Updated {SOURCE} with {len(macros)} macros.")


def main() -> None:
    macros = build_macros()
    print("Populated macros:")
    for k, v in macros.items():
        print(f"  \\{k} = {v}")
    rewrite_tex(macros)


if __name__ == "__main__":
    main()
