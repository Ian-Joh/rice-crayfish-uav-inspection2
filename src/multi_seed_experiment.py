# -*- coding: utf-8 -*-
"""Supplementary Experiment A: multi-seed / multi-scenario robustness.

This experiment evaluates whether the proposed method is robust across multiple
randomly generated farm scenarios. It does not overwrite baseline data files.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route
from planners import avoid_obstacles, count_obstacle_collisions, fixed_route, ipp_route, keypoint_tsp_route, proposed_route, standard_cpp_route
import simulation
import risk_model

Point = Tuple[float, float]

METHOD_ORDER = ["Fixed route", "Standard CPP", "Key-point TSP", "IPP baseline", "Proposed method"]
SEEDS = [20260612, 20260613, 20260614, 20260615, 20260616]


def _layers_without_metadata(layers: Dict) -> Dict[str, np.ndarray]:
    return {k: v for k, v in layers.items() if isinstance(v, np.ndarray)}


def generate_scenario(seed: int) -> Tuple[Dict, Dict[str, np.ndarray]]:
    old_seed = config.RANDOM_SEED
    old_data_dir = risk_model.config.DATA_DIR
    try:
        config.RANDOM_SEED = seed
        risk_model.config.RANDOM_SEED = seed
        farm = simulation.generate_farm_map()

        # risk_model.generate_risk_layers() loads farm from disk, so replicate by
        # temporarily writing map would be intrusive. Instead, monkeypatch its
        # private loader for this controlled experiment.
        old_loader = risk_model._load_farm
        risk_model._load_farm = lambda: farm
        layers_full = risk_model.generate_risk_layers()
        risk_model._load_farm = old_loader
        return farm, _layers_without_metadata(layers_full)
    finally:
        config.RANDOM_SEED = old_seed
        risk_model.config.RANDOM_SEED = old_seed


def generate_routes_for_scenario(farm: Dict, layers: Dict[str, np.ndarray]) -> Dict[str, List[Point]]:
    raw = {
        "Fixed route": fixed_route(),
        "Standard CPP": standard_cpp_route(),
        "Key-point TSP": keypoint_tsp_route(farm),
        "IPP baseline": ipp_route(layers),
        "Proposed method": proposed_route(layers),
    }
    return {name: avoid_obstacles(route, farm) for name, route in raw.items()}


def run_multi_seed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        farm, layers = generate_scenario(seed)
        routes = generate_routes_for_scenario(farm, layers)
        scenario_time = time.perf_counter() - t0
        for method in METHOD_ORDER:
            route = routes[method]
            m = evaluate_route(route, layers)
            rows.append({
                "seed": seed,
                "method": method,
                "scenario_generation_time_s": scenario_time,
                "obstacle_collisions": count_obstacle_collisions(route, farm),
                **m,
            })
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["seed", "method"]).reset_index(drop=True)
    df.to_csv(config.TABLE_DIR / "table10_multi_seed_statistics.csv", index=False, encoding="utf-8-sig")

    metric_cols = [
        "total_distance_m",
        "high_risk_coverage_rate",
        "abnormality_detection_rate",
        "missed_risk_rate",
        "detections_per_km",
        "obstacle_collisions",
    ]
    summary_rows = []
    for method, g in df.groupby("method", observed=True):
        row = {"method": method}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)
    summary.to_csv(config.TABLE_DIR / "table10_multi_seed_summary.csv", index=False, encoding="utf-8-sig")
    return df, summary


def plot_multi_seed(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    colors = ["#90caf9", "#a5d6a7", "#ffcc80", "#ce93d8", "#ef5350"]

    data_detection = [df[df["method"] == m]["abnormality_detection_rate"].values for m in METHOD_ORDER]
    data_high = [df[df["method"] == m]["high_risk_coverage_rate"].values for m in METHOD_ORDER]

    bp1 = axes[0].boxplot(data_detection, tick_labels=METHOD_ORDER, patch_artist=True, showmeans=True)
    bp2 = axes[1].boxplot(data_high, tick_labels=METHOD_ORDER, patch_artist=True, showmeans=True)
    for bp in [bp1, bp2]:
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)

    axes[0].set_title("Detection rate across random scenarios")
    axes[1].set_title("High-risk coverage across random scenarios")
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig13_multi_seed_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    proposed = summary[summary["method"] == "Proposed method"].iloc[0]
    cpp = summary[summary["method"] == "Standard CPP"].iloc[0]
    text = f"""# Checkpoint 15 - Multi-seed / multi-scenario robustness

Status: completed

Seeds:
- {', '.join(str(s) for s in SEEDS)}

Methods:
- {', '.join(METHOD_ORDER)}

Key mean results:
- Proposed detection rate: {proposed['abnormality_detection_rate_mean']:.4f} ± {proposed['abnormality_detection_rate_std']:.4f}
- Standard CPP detection rate: {cpp['abnormality_detection_rate_mean']:.4f} ± {cpp['abnormality_detection_rate_std']:.4f}
- Proposed high-risk coverage: {proposed['high_risk_coverage_rate_mean']:.4f} ± {proposed['high_risk_coverage_rate_std']:.4f}
- Standard CPP high-risk coverage: {cpp['high_risk_coverage_rate_mean']:.4f} ± {cpp['high_risk_coverage_rate_std']:.4f}

Collision check:
- Max obstacle collisions: {int(df['obstacle_collisions'].max())}

Outputs:
- results/tables/table10_multi_seed_statistics.csv
- results/tables/table10_multi_seed_summary.csv
- results/figures/fig13_multi_seed_boxplot.png
"""
    (config.CHECKPOINT_DIR / "15_multi_seed_experiment.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Supplement A] Running multi-seed robustness experiment...")
    df, summary = run_multi_seed()
    plot_multi_seed(df)
    write_checkpoint(df, summary)
    config.log(f"[Supplement A] Saved detailed table: {config.TABLE_DIR / 'table10_multi_seed_statistics.csv'}")
    config.log(f"[Supplement A] Saved summary table: {config.TABLE_DIR / 'table10_multi_seed_summary.csv'}")
    config.log(f"[Supplement A] Saved figure: {config.FIG_DIR / 'fig13_multi_seed_boxplot.png'}")
    preview = summary[["method", "high_risk_coverage_rate_mean", "high_risk_coverage_rate_std", "abnormality_detection_rate_mean", "abnormality_detection_rate_std", "obstacle_collisions_mean"]]
    config.log(preview.to_string(index=False))


if __name__ == "__main__":
    main()
