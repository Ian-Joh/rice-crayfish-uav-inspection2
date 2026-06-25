# -*- coding: utf-8 -*-
"""Step 6: sensitivity analysis for high-risk threshold tau."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route, route_distance
from planners import avoid_obstacles, count_obstacle_collisions, proposed_route


def load_layers() -> dict:
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    return {k: z[k].copy() for k in z.files}


def apply_threshold(layers: dict, tau: float) -> dict:
    out = {k: v.copy() if hasattr(v, "copy") else v for k, v in layers.items()}
    R = out["R"]
    rc = np.zeros_like(R, dtype=np.int8)
    rc[(R >= 0.4) & (R < tau)] = 1
    rc[R >= tau] = 2
    out["risk_class"] = rc
    return out


def run_threshold_sensitivity() -> pd.DataFrame:
    base = load_layers()
    # Use original high-risk/truth as common evaluation target; tau changes route generation.
    eval_layers = base
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    rows = []
    for tau in [0.5, 0.6, 0.7, 0.8]:
        planning_layers = apply_threshold(base, tau)
        route = avoid_obstacles(proposed_route(planning_layers, risk_threshold=tau), farm)
        m = evaluate_route(route, eval_layers)
        rows.append({"tau": tau, "inserted_points": len(route) - len(__import__('planners').standard_cpp_route()), "obstacle_collisions": count_obstacle_collisions(route, farm), **m})
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "table4_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    return df


def plot_threshold(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax2 = ax1.twinx()
    x = df["tau"]

    l1 = ax1.plot(x, df["high_risk_coverage_rate"], marker="o", linewidth=2.0, color="#d32f2f", label="High-risk coverage")
    l2 = ax1.plot(x, df["abnormality_detection_rate"], marker="s", linewidth=2.0, color="#1976d2", label="Detection rate")
    l3 = ax2.plot(x, df["total_distance_m"], marker="^", linewidth=1.8, linestyle="--", color="#388e3c", label="Distance")

    for _, row in df.iterrows():
        ax1.annotate(f"+{int(row['inserted_points'])}", (row["tau"], row["abnormality_detection_rate"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#333333")

    ax1.set_xlabel("Risk threshold τ")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Total distance (m)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Threshold sensitivity: coverage, detection and distance")
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig7_threshold_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    best = df.sort_values(["abnormality_detection_rate", "high_risk_coverage_rate"], ascending=False).iloc[0]
    text = f"""# Checkpoint 06 - Threshold sensitivity

Status: completed

Tested tau values:
- 0.5
- 0.6
- 0.7
- 0.8

Best by detection rate:
- tau: {best['tau']}
- detection rate: {best['abnormality_detection_rate']:.4f}
- high-risk coverage: {best['high_risk_coverage_rate']:.4f}
- total distance: {best['total_distance_m']:.2f} m

Outputs:
- results/tables/table4_threshold_sensitivity.csv
- results/figures/fig7_threshold_sensitivity.png
"""
    (config.CHECKPOINT_DIR / "06_threshold_sensitivity.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step6] Running threshold sensitivity experiment...")
    df = run_threshold_sensitivity()
    plot_threshold(df)
    write_checkpoint(df)
    config.log(f"[Step6] Saved threshold table: {config.TABLE_DIR / 'table4_threshold_sensitivity.csv'}")
    config.log(f"[Step6] Saved threshold figure: {config.FIG_DIR / 'fig7_threshold_sensitivity.png'}")
    config.log(df[["tau", "inserted_points", "total_distance_m", "high_risk_coverage_rate", "abnormality_detection_rate", "missed_risk_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
