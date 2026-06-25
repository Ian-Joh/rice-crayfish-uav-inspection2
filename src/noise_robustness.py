# -*- coding: utf-8 -*-
"""Step 8: robustness experiment under perception noise."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route
from planners import avoid_obstacles, count_obstacle_collisions, proposed_route


def load_layers() -> dict:
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    return {k: z[k].copy() for k in z.files}


def normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def add_noise_layers(base: dict, noise_level: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: v.copy() if hasattr(v, "copy") else v for k, v in base.items()}
    for k in ["C", "V", "T", "S"]:
        out[k] = np.clip(out[k] + rng.normal(0, noise_level, out[k].shape), 0, 1)
    # K is structural annotation, not perception noise.
    R = 0.24 * out["C"] + 0.20 * out["V"] + 0.16 * out["T"] + 0.20 * out["S"] + 0.20 * out["K"]
    out["R"] = normalize(R)
    rc = np.zeros_like(R, dtype=np.int8)
    rc[(out["R"] >= 0.4) & (out["R"] < 0.7)] = 1
    rc[out["R"] >= 0.7] = 2
    out["risk_class"] = rc
    return out


def run_noise_robustness() -> pd.DataFrame:
    base = load_layers()
    eval_layers = base
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    rows = []
    for nl in [0.00, 0.05, 0.10, 0.20, 0.30]:
        metrics_list = []
        for rep in range(5):
            planning_layers = add_noise_layers(base, nl, config.RANDOM_SEED + 100 + rep)
            route = avoid_obstacles(proposed_route(planning_layers, risk_threshold=0.7), farm)
            m = evaluate_route(route, eval_layers)
            m["obstacle_collisions"] = count_obstacle_collisions(route, farm)
            metrics_list.append(m)
        mean = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        std = {k + "_std": float(np.std([m[k] for m in metrics_list])) for k in metrics_list[0]}
        rows.append({"noise_level": nl, **mean, **std})
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "table7_noise_robustness.csv", index=False, encoding="utf-8-sig")
    return df


def plot_noise(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax2 = ax1.twinx()
    x = df["noise_level"] * 100

    e1 = ax1.errorbar(x, df["high_risk_coverage_rate"], yerr=df["high_risk_coverage_rate_std"], marker="o", linewidth=2.0, color="#d32f2f", capsize=3, label="High-risk coverage")
    e2 = ax1.errorbar(x, df["abnormality_detection_rate"], yerr=df["abnormality_detection_rate_std"], marker="s", linewidth=2.0, color="#1976d2", capsize=3, label="Detection rate")
    e3 = ax2.errorbar(x, df["total_distance_m"], yerr=df["total_distance_m_std"], marker="^", linewidth=1.7, linestyle="--", color="#388e3c", capsize=3, label="Distance")

    ax1.set_xlabel("Perception noise level (%)")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Total distance (m)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Noise robustness: rates and route cost")
    handles = [e1, e2, e3]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig10_noise_robustness.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    clean = df[df["noise_level"] == 0.0].iloc[0]
    noisy = df[df["noise_level"] == 0.3].iloc[0]
    text = f"""# Checkpoint 09 - Noise robustness

Status: completed

Noise levels:
- 0%
- 5%
- 10%
- 20%
- 30%

Each noise level repeated 5 times with different random seeds.

Key comparison:
- Clean detection rate: {clean['abnormality_detection_rate']:.4f}
- 30% noise detection rate: {noisy['abnormality_detection_rate']:.4f}
- Clean high-risk coverage: {clean['high_risk_coverage_rate']:.4f}
- 30% noise high-risk coverage: {noisy['high_risk_coverage_rate']:.4f}

Outputs:
- results/tables/table7_noise_robustness.csv
- results/figures/fig10_noise_robustness.png
"""
    (config.CHECKPOINT_DIR / "09_noise_robustness.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step8] Running noise robustness experiment...")
    df = run_noise_robustness()
    plot_noise(df)
    write_checkpoint(df)
    config.log(f"[Step8] Saved noise table: {config.TABLE_DIR / 'table7_noise_robustness.csv'}")
    config.log(f"[Step8] Saved noise figure: {config.FIG_DIR / 'fig10_noise_robustness.png'}")
    config.log(df[["noise_level", "total_distance_m", "high_risk_coverage_rate", "abnormality_detection_rate", "missed_risk_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
