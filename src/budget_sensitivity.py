# -*- coding: utf-8 -*-
"""Step 7: sensitivity analysis for maximum additional flight-distance budget."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route
from planners import avoid_obstacles, count_obstacle_collisions, proposed_route, standard_cpp_route


def load_layers() -> dict:
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    return {k: z[k].copy() for k in z.files}


def run_budget_sensitivity() -> pd.DataFrame:
    layers = load_layers()
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    base_len = len(standard_cpp_route())
    rows = []
    # The dense CPP route makes insertion detours cheap; use fine-grained
    # low-budget ratios to reveal the transition before the candidate set saturates.
    for ratio in [0.002, 0.005, 0.010, 0.020, 0.050]:
        route = avoid_obstacles(proposed_route(layers, extra_budget_ratio=ratio, risk_threshold=0.5, min_benefit_ratio=0.0001), farm)
        m = evaluate_route(route, layers)
        rows.append({"extra_budget_ratio": ratio, "inserted_points": len(route) - base_len, "obstacle_collisions": count_obstacle_collisions(route, farm), **m})
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "table6_extra_budget_sensitivity.csv", index=False, encoding="utf-8-sig")
    return df


def plot_budget(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax2 = ax1.twinx()
    x = df["extra_budget_ratio"] * 100

    l1 = ax1.plot(x, df["high_risk_coverage_rate"], marker="o", linewidth=2.0, color="#d32f2f", label="High-risk coverage")
    l2 = ax1.plot(x, df["abnormality_detection_rate"], marker="s", linewidth=2.0, color="#1976d2", label="Detection rate")
    l3 = ax2.plot(x, df["inserted_points"], marker="^", linewidth=1.8, linestyle="--", color="#388e3c", label="Inserted points")

    ax1.set_xlabel("Extra flight-distance budget (%)")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Inserted revisiting points")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Extra-budget sensitivity: benefit saturation under dense CPP")
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig9_extra_budget_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    best = df.sort_values(["abnormality_detection_rate", "high_risk_coverage_rate"], ascending=False).iloc[0]
    text = f"""# Checkpoint 08 - Extra budget sensitivity

Status: completed

Tested extra distance budgets:
- 0.2%
- 0.5%
- 1%
- 2%
- 5%

Best by detection rate:
- extra budget: {best['extra_budget_ratio'] * 100:.0f}%
- inserted points: {int(best['inserted_points'])}
- detection rate: {best['abnormality_detection_rate']:.4f}
- high-risk coverage: {best['high_risk_coverage_rate']:.4f}
- total distance: {best['total_distance_m']:.2f} m

Outputs:
- results/tables/table6_extra_budget_sensitivity.csv
- results/figures/fig9_extra_budget_sensitivity.png
"""
    (config.CHECKPOINT_DIR / "08_extra_budget_sensitivity.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step7] Running extra budget sensitivity experiment...")
    df = run_budget_sensitivity()
    plot_budget(df)
    write_checkpoint(df)
    config.log(f"[Step7] Saved budget table: {config.TABLE_DIR / 'table6_extra_budget_sensitivity.csv'}")
    config.log(f"[Step7] Saved budget figure: {config.FIG_DIR / 'fig9_extra_budget_sensitivity.png'}")
    config.log(df[["extra_budget_ratio", "inserted_points", "total_distance_m", "high_risk_coverage_rate", "abnormality_detection_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
