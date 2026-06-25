# -*- coding: utf-8 -*-
"""Supplementary Experiment B: risk-weight sensitivity."""
from __future__ import annotations

import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route
from planners import avoid_obstacles, count_obstacle_collisions, proposed_route

WEIGHT_SCHEMES = {
    "Balanced": {"C": 0.24, "V": 0.20, "T": 0.16, "S": 0.20, "K": 0.20},
    "Equal weights": {"C": 0.20, "V": 0.20, "T": 0.20, "S": 0.20, "K": 0.20},
    "Vision-dominant": {"C": 0.30, "V": 0.25, "T": 0.20, "S": 0.15, "K": 0.10},
    "Sensor-dominant": {"C": 0.17, "V": 0.16, "T": 0.12, "S": 0.35, "K": 0.20},
    "Structure-dominant": {"C": 0.17, "V": 0.16, "T": 0.12, "S": 0.15, "K": 0.40},
}
ORDER = list(WEIGHT_SCHEMES.keys())


def normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def load_layers() -> Dict[str, np.ndarray]:
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    return {k: z[k].copy() for k in z.files}


def rebuild_with_weights(base: Dict[str, np.ndarray], weights: Dict[str, float]) -> Dict[str, np.ndarray]:
    total = sum(weights.values())
    R = sum(weights[k] * base[k] for k in ["C", "V", "T", "S", "K"]) / max(total, 1e-9)
    R = normalize(R)
    out = {k: v.copy() if hasattr(v, "copy") else v for k, v in base.items()}
    out["R"] = R
    rc = np.zeros_like(R, dtype=np.int8)
    rc[(R >= 0.4) & (R < 0.7)] = 1
    rc[R >= 0.7] = 2
    out["risk_class"] = rc
    return out


def run_weight_sensitivity() -> pd.DataFrame:
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    base = load_layers()
    eval_layers = base
    rows = []
    for scheme, weights in WEIGHT_SCHEMES.items():
        planning_layers = rebuild_with_weights(base, weights)
        route = avoid_obstacles(proposed_route(planning_layers, risk_threshold=0.7), farm)
        m = evaluate_route(route, eval_layers)
        rows.append({
            "weight_scheme": scheme,
            "w_C": weights["C"],
            "w_V": weights["V"],
            "w_T": weights["T"],
            "w_S": weights["S"],
            "w_K": weights["K"],
            "planning_high_risk_cells": int((planning_layers["risk_class"] == 2).sum()),
            "obstacle_collisions": count_obstacle_collisions(route, farm),
            **m,
        })
    df = pd.DataFrame(rows)
    df["weight_scheme"] = pd.Categorical(df["weight_scheme"], categories=ORDER, ordered=True)
    df = df.sort_values("weight_scheme").reset_index(drop=True)
    df.to_csv(config.TABLE_DIR / "table11_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    return df


def plot_weight(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ax2 = ax1.twinx()
    x = np.arange(len(df))
    width = 0.34

    b1 = ax1.bar(x - width / 2, df["high_risk_coverage_rate"], width=width, color="#d32f2f", alpha=0.82, label="High-risk coverage")
    b2 = ax1.bar(x + width / 2, df["abnormality_detection_rate"], width=width, color="#1976d2", alpha=0.82, label="Detection rate")
    l1 = ax2.plot(x, df["total_distance_m"], marker="^", color="#388e3c", linewidth=1.8, linestyle="--", label="Distance")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["weight_scheme"].astype(str), rotation=25, ha="right")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Total distance (m)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("Risk-weight sensitivity")
    handles = [b1, b2, l1[0]]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig14_weight_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    best = df.sort_values(["abnormality_detection_rate", "high_risk_coverage_rate"], ascending=False).iloc[0]
    text = f"""# Checkpoint 16 - Risk-weight sensitivity

Status: completed

Weight schemes:
- Balanced
- Equal weights
- Vision-dominant
- Sensor-dominant
- Structure-dominant

Best by detection rate:
- scheme: {best['weight_scheme']}
- detection rate: {best['abnormality_detection_rate']:.4f}
- high-risk coverage: {best['high_risk_coverage_rate']:.4f}
- distance: {best['total_distance_m']:.2f} m

Collision check:
- Max obstacle collisions: {int(df['obstacle_collisions'].max())}

Outputs:
- results/tables/table11_weight_sensitivity.csv
- results/figures/fig14_weight_sensitivity.png
"""
    (config.CHECKPOINT_DIR / "16_weight_sensitivity.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Supplement B] Running risk-weight sensitivity experiment...")
    df = run_weight_sensitivity()
    plot_weight(df)
    write_checkpoint(df)
    config.log(f"[Supplement B] Saved table: {config.TABLE_DIR / 'table11_weight_sensitivity.csv'}")
    config.log(f"[Supplement B] Saved figure: {config.FIG_DIR / 'fig14_weight_sensitivity.png'}")
    config.log(df[["weight_scheme", "total_distance_m", "high_risk_coverage_rate", "abnormality_detection_rate", "missed_risk_rate", "obstacle_collisions"]].to_string(index=False))


if __name__ == "__main__":
    main()
