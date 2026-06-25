# -*- coding: utf-8 -*-
"""Step 10: battery / maximum route-length constraint experiment."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route, route_distance
from planners import avoid_obstacles, count_obstacle_collisions, standard_cpp_route, proposed_route, fixed_route


def load_layers() -> dict:
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    return {k: z[k].copy() for k in z.files}


def run_battery_constraint() -> pd.DataFrame:
    layers = load_layers()
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    base_dist = route_distance(avoid_obstacles(standard_cpp_route(), farm))
    rows = []
    # Budgets are relative to the dense CPP route. Values below 1.0 mean the UAV
    # cannot afford full dense coverage, so we reduce adaptive extra budget to zero.
    for budget_factor in [0.80, 0.90, 1.00, 1.05, 1.10]:
        max_distance = base_dist * budget_factor
        if budget_factor < 1.0:
            # If battery cannot support dense CPP, start from a coarser fixed route
            # and spend the remaining distance on risk-driven revisiting points.
            base = avoid_obstacles(fixed_route(), farm)
            base_dist_short = route_distance(base)
            extra_ratio = max(0.0, (max_distance - base_dist_short) / max(base_dist_short, 1e-9))
            route = avoid_obstacles(proposed_route(layers, extra_budget_ratio=extra_ratio, risk_threshold=0.7, base_route=base), farm)
        else:
            extra_ratio = budget_factor - 1.0
            route = avoid_obstacles(proposed_route(layers, extra_budget_ratio=extra_ratio, risk_threshold=0.7), farm)
        # If numerical/path choices slightly exceed budget, revert to the base route.
        if route_distance(route) > max_distance:
            route = avoid_obstacles(fixed_route(), farm) if budget_factor < 1.0 else avoid_obstacles(standard_cpp_route(), farm)
        m = evaluate_route(route, layers)
        rows.append({"budget_factor": budget_factor, "max_distance_m": max_distance, "within_budget": route_distance(route) <= max_distance + 1e-6, "obstacle_collisions": count_obstacle_collisions(route, farm), **m})
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "table9_battery_constraint.csv", index=False, encoding="utf-8-sig")
    return df


def plot_battery(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax2 = ax1.twinx()
    x = df["budget_factor"] * 100

    l1 = ax1.plot(x, df["high_risk_coverage_rate"], marker="o", linewidth=2.0, color="#d32f2f", label="High-risk coverage")
    l2 = ax1.plot(x, df["abnormality_detection_rate"], marker="s", linewidth=2.0, color="#1976d2", label="Detection rate")
    l3 = ax2.plot(x, df["total_distance_m"], marker="^", linewidth=1.8, linestyle="--", color="#388e3c", label="Actual distance")
    l4 = ax2.plot(x, df["max_distance_m"], marker="x", linewidth=1.8, linestyle=":", color="#f57c00", label="Battery budget")

    for _, row in df.iterrows():
        status = "OK" if bool(row["within_budget"]) else "over"
        ax2.annotate(status, (row["budget_factor"] * 100, row["total_distance_m"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#333333")

    ax1.set_xlabel("Battery budget (% of CPP distance)")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Distance (m)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Battery constraint: inspection performance and route length")
    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig12_battery_constraint.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    strict = df[df["budget_factor"] == 0.8].iloc[0]
    loose = df[df["budget_factor"] == 1.1].iloc[0]
    text = f"""# Checkpoint 11 - Battery constraint

Status: completed

Battery budgets:
- 80% of CPP distance
- 90% of CPP distance
- 100% of CPP distance
- 105% of CPP distance
- 110% of CPP distance

Key comparison:
- 80% budget detection rate: {strict['abnormality_detection_rate']:.4f}
- 110% budget detection rate: {loose['abnormality_detection_rate']:.4f}
- 80% budget high-risk coverage: {strict['high_risk_coverage_rate']:.4f}
- 110% budget high-risk coverage: {loose['high_risk_coverage_rate']:.4f}

Outputs:
- results/tables/table9_battery_constraint.csv
- results/figures/fig12_battery_constraint.png
"""
    (config.CHECKPOINT_DIR / "11_battery_constraint.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step10] Running battery constraint experiment...")
    df = run_battery_constraint()
    plot_battery(df)
    write_checkpoint(df)
    config.log(f"[Step10] Saved battery table: {config.TABLE_DIR / 'table9_battery_constraint.csv'}")
    config.log(f"[Step10] Saved battery figure: {config.FIG_DIR / 'fig12_battery_constraint.png'}")
    config.log(df[["budget_factor", "max_distance_m", "total_distance_m", "within_budget", "high_risk_coverage_rate", "abnormality_detection_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
