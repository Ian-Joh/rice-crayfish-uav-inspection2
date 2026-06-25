# -*- coding: utf-8 -*-
"""Step 4: main comparison experiment for all routes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route

Point = Tuple[float, float]


def load_routes() -> Dict[str, List[Point]]:
    path = config.DATA_DIR / "routes.json"
    if not path.exists():
        raise FileNotFoundError("Missing routes.json. Run planners.py first.")
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: [(float(x), float(y)) for x, y in v] for k, v in d.items()}


def load_layers() -> Dict[str, np.ndarray]:
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    return {k: z[k] for k in z.files}


def write_simulation_settings() -> None:
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    facility_counts = {}
    for f in farm["facilities"]:
        facility_counts[f["kind"]] = facility_counts.get(f["kind"], 0) + 1
    unit_counts = {}
    for p in farm["ponds"]:
        unit_counts[p.get("unit_type", "unknown")] = unit_counts.get(p.get("unit_type", "unknown"), 0) + 1
    rows = [
        ["scenario", farm["metadata"].get("scenario", "unknown")],
        ["map_width_m", farm["metadata"].get("map_width_m", config.MAP_WIDTH_M)],
        ["map_height_m", farm["metadata"].get("map_height_m", config.MAP_HEIGHT_M)],
        ["grid_size_m", farm["metadata"].get("grid_size_m", config.GRID_SIZE_M)],
        ["random_seed", farm["metadata"].get("seed", config.RANDOM_SEED)],
        ["n_production_units", len(farm["ponds"])],
        ["n_rice_crayfish_paddy_units", unit_counts.get("rice_crayfish_paddy", 0)],
        ["n_crayfish_pond_units", unit_counts.get("crayfish_pond", 0)],
        ["n_canals", len(farm.get("canals", []))],
        ["n_inlet_outlet", facility_counts.get("inlet_outlet", 0)],
        ["n_feeding_points", facility_counts.get("feeding_point", 0)],
        ["n_aerators", facility_counts.get("aerator", 0)],
        ["n_water_sensors", facility_counts.get("water_sensor", 0)],
        ["n_obstacles", len(farm["obstacles"])],
        ["uav_speed_mps", 6.0],
        ["camera_footprint_radius_m", 35.0],
    ]
    pd.DataFrame(rows, columns=["parameter", "value"]).to_csv(config.TABLE_DIR / "table1_simulation_settings.csv", index=False, encoding="utf-8-sig")


def run_main_comparison() -> pd.DataFrame:
    layers = load_layers()
    routes = load_routes()
    rows = []
    for method, route in routes.items():
        m = evaluate_route(route, layers)
        rows.append({"method": method, **m})
    df = pd.DataFrame(rows)
    # Friendly ordering and rounded copy for paper table.
    order = ["Fixed route", "Standard CPP", "Key-point TSP", "IPP baseline", "Proposed method"]
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)
    df.to_csv(config.TABLE_DIR / "table2_main_comparison.csv", index=False, encoding="utf-8-sig")
    return df


def plot_main_metrics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    method_labels = df["method"].astype(str).tolist()
    colors = ["#90caf9", "#a5d6a7", "#ffcc80", "#ce93d8", "#ef5350"]

    plots = [
        ("total_distance_m", "Total distance (m)", False),
        ("high_risk_coverage_rate", "High-risk coverage rate", True),
        ("abnormality_detection_rate", "Abnormality detection rate", True),
        ("detections_per_km", "Detected abnormal cells per km", True),
    ]
    for ax, (col, title, ylim01) in zip(axes.flat, plots):
        ax.bar(method_labels, df[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        if ylim01:
            ymax = max(1.0, float(df[col].max()) * 1.15)
            ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig5_main_metrics_bar.png")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    proposed = df[df["method"].astype(str) == "Proposed method"].iloc[0]
    cpp = df[df["method"].astype(str) == "Standard CPP"].iloc[0]
    fixed = df[df["method"].astype(str) == "Fixed route"].iloc[0]
    high_gain_vs_cpp = proposed["high_risk_coverage_rate"] - cpp["high_risk_coverage_rate"]
    det_gain_vs_fixed = proposed["abnormality_detection_rate"] - fixed["abnormality_detection_rate"]
    text = f"""# Checkpoint 04 - Main comparison experiment

Status: completed

Methods compared:
- Fixed route
- Standard CPP
- Key-point TSP
- IPP baseline
- Proposed method

Key observations:
- Proposed high-risk coverage: {proposed['high_risk_coverage_rate']:.4f}
- Standard CPP high-risk coverage: {cpp['high_risk_coverage_rate']:.4f}
- Absolute gain over CPP: {high_gain_vs_cpp:.4f}
- Proposed abnormality detection rate: {proposed['abnormality_detection_rate']:.4f}
- Fixed-route abnormality detection rate: {fixed['abnormality_detection_rate']:.4f}
- Absolute gain over fixed route: {det_gain_vs_fixed:.4f}
- Proposed total distance: {proposed['total_distance_m']:.2f} m

Outputs:
- results/tables/table1_simulation_settings.csv
- results/tables/table2_main_comparison.csv
- results/figures/fig5_main_metrics_bar.png
"""
    (config.CHECKPOINT_DIR / "04_main_experiment.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step4] Running main comparison experiment...")
    write_simulation_settings()
    df = run_main_comparison()
    plot_main_metrics(df)
    write_checkpoint(df)
    config.log(f"[Step4] Saved settings table: {config.TABLE_DIR / 'table1_simulation_settings.csv'}")
    config.log(f"[Step4] Saved main comparison table: {config.TABLE_DIR / 'table2_main_comparison.csv'}")
    config.log(f"[Step4] Saved main metrics figure: {config.FIG_DIR / 'fig5_main_metrics_bar.png'}")
    config.log("[Step4] Main comparison preview:")
    preview_cols = ["method", "total_distance_m", "high_risk_coverage_rate", "abnormality_detection_rate", "missed_risk_rate"]
    config.log(df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
