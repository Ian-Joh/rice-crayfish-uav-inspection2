# -*- coding: utf-8 -*-
"""Step 7b: grid-resolution sensitivity experiment.

This script does not overwrite the baseline 20 m risk layers. It rebuilds risk
layers at alternative grid sizes using the same farm map and the same synthetic
risk-generation logic, then evaluates the proposed planner with obstacle
avoidance under a common 20 m baseline evaluation target.
"""
from __future__ import annotations

import json
import random
import time
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route
from planners import avoid_obstacles, count_obstacle_collisions, proposed_route

Point = Tuple[float, float]

WEIGHTS = {"C": 0.24, "V": 0.20, "T": 0.16, "S": 0.20, "K": 0.20}


def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def grid_centers(grid_size: float) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(grid_size / 2, config.MAP_WIDTH_M, grid_size)
    ys = np.arange(grid_size / 2, config.MAP_HEIGHT_M, grid_size)
    return np.meshgrid(xs, ys)


def add_gaussian(layer: np.ndarray, X: np.ndarray, Y: np.ndarray, cx: float, cy: float, amp: float, sigma: float) -> None:
    layer += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def facility_points(farm: Dict, kinds: Iterable[str] | None = None) -> List[Point]:
    kind_set = set(kinds) if kinds is not None else None
    pts = []
    for f in farm["facilities"]:
        if kind_set is None or f["kind"] in kind_set:
            pts.append((float(f["x"]), float(f["y"])))
    return pts


def nearest_distance_layer(X: np.ndarray, Y: np.ndarray, pts: List[Point]) -> np.ndarray:
    if not pts:
        return np.zeros_like(X, dtype=float)
    dist = np.full_like(X, fill_value=np.inf, dtype=float)
    for x, y in pts:
        dist = np.minimum(dist, np.sqrt((X - x) ** 2 + (Y - y) ** 2))
    return dist


def generate_patch_centers(farm: Dict, n: int, prefer_kinds: List[str], jitter: float) -> List[Point]:
    base = facility_points(farm, prefer_kinds)
    centers: List[Point] = []
    for _ in range(n):
        if base and random.random() < 0.72:
            bx, by = random.choice(base)
            x = max(0.0, min(config.MAP_WIDTH_M, bx + random.uniform(-jitter, jitter)))
            y = max(0.0, min(config.MAP_HEIGHT_M, by + random.uniform(-jitter, jitter)))
        else:
            x = random.uniform(80, config.MAP_WIDTH_M - 80)
            y = random.uniform(80, config.MAP_HEIGHT_M - 80)
        centers.append((round(x, 2), round(y, 2)))
    return centers


def build_layers_for_grid(farm: Dict, grid_size: float) -> Dict[str, np.ndarray]:
    # Same seed for each grid size: differences should mainly come from grid resolution.
    random.seed(config.RANDOM_SEED + 1)
    np.random.seed(config.RANDOM_SEED + 1)
    X, Y = grid_centers(grid_size)
    C = np.zeros_like(X, dtype=float)
    V = np.zeros_like(X, dtype=float)
    T = np.zeros_like(X, dtype=float)
    sensor_signal = np.zeros_like(X, dtype=float)
    abnormal_events: List[Dict] = []

    for idx, (cx, cy) in enumerate(generate_patch_centers(farm, 12, ["inlet_outlet"], 80), start=1):
        amp = random.uniform(0.65, 1.0)
        sigma = random.uniform(25, 55)
        add_gaussian(C, X, Y, cx, cy, amp, sigma)
        abnormal_events.append({"kind": "water_colour", "x": cx, "y": cy, "sigma": sigma})

    pond_centers = [tuple(p["center"]) for p in farm["ponds"]]
    for _ in range(10):
        bx, by = random.choice(pond_centers)
        cx = max(0.0, min(config.MAP_WIDTH_M, bx + random.uniform(-65, 65)))
        cy = max(0.0, min(config.MAP_HEIGHT_M, by + random.uniform(-65, 65)))
        sigma = random.uniform(30, 70)
        add_gaussian(V, X, Y, cx, cy, random.uniform(0.55, 0.95), sigma)
        abnormal_events.append({"kind": "vegetation", "x": round(cx, 2), "y": round(cy, 2), "sigma": sigma})

    for cx, cy in generate_patch_centers(farm, 7, ["feeding_point", "aerator"], 110):
        sigma = random.uniform(40, 85)
        add_gaussian(T, X, Y, cx, cy, random.uniform(0.45, 0.9), sigma)
        abnormal_events.append({"kind": "thermal", "x": cx, "y": cy, "sigma": sigma})

    sensor_pts = facility_points(farm, ["water_sensor"])
    for sid, (sx, sy) in enumerate(sensor_pts, start=1):
        local_base = 0.25 + 0.5 * random.random()
        if random.random() < 0.45:
            local_base += random.uniform(0.35, 0.65)
        local_base = min(1.0, local_base)
        add_gaussian(sensor_signal, X, Y, sx, sy, local_base, sigma=95)
        if local_base >= 0.72:
            abnormal_events.append({"kind": "sensor_water_quality", "x": sx, "y": sy, "sigma": 65.0})

    structural_pts = facility_points(farm, ["inlet_outlet", "feeding_point", "aerator"])
    dist_to_struct = nearest_distance_layer(X, Y, structural_pts)
    K = np.exp(-dist_to_struct / 85.0)
    for kx, ky in random.sample(structural_pts, min(12, len(structural_pts))):
        abnormal_events.append({"kind": "structural_facility", "x": kx, "y": ky, "sigma": 45.0})

    C = normalize(C + np.random.normal(0, 0.015, C.shape))
    V = normalize(V + np.random.normal(0, 0.015, V.shape))
    T = normalize(T + np.random.normal(0, 0.012, T.shape))
    S = normalize(sensor_signal + np.random.normal(0, 0.01, sensor_signal.shape))
    K = normalize(K)
    R = normalize(WEIGHTS["C"] * C + WEIGHTS["V"] * V + WEIGHTS["T"] * T + WEIGHTS["S"] * S + WEIGHTS["K"] * K)

    rc = np.zeros_like(R, dtype=np.int8)
    rc[(R >= 0.4) & (R < 0.7)] = 1
    rc[R >= 0.7] = 2

    truth = np.zeros_like(R, dtype=np.int8)
    for e in abnormal_events:
        d = np.sqrt((X - e["x"]) ** 2 + (Y - e["y"]) ** 2)
        truth[d <= max(30.0, float(e["sigma"]) * 0.9)] = 1

    return {"X": X, "Y": Y, "C": C, "V": V, "T": T, "S": S, "K": K, "R": R, "risk_class": rc, "truth": truth}


def load_eval_layers() -> Dict[str, np.ndarray]:
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    return {k: z[k].copy() for k in z.files}


def run_grid_sensitivity() -> pd.DataFrame:
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    eval_layers = load_eval_layers()
    rows = []
    for grid_size in [10, 20, 30, 40]:
        t0 = time.perf_counter()
        planning_layers = build_layers_for_grid(farm, grid_size)
        route = avoid_obstacles(proposed_route(planning_layers, risk_threshold=0.7), farm)
        elapsed = time.perf_counter() - t0
        m = evaluate_route(route, eval_layers)
        rows.append({
            "grid_size_m": grid_size,
            "n_rows": int(planning_layers["R"].shape[0]),
            "n_cols": int(planning_layers["R"].shape[1]),
            "n_cells": int(planning_layers["R"].size),
            "high_risk_cells_planning_grid": int((planning_layers["risk_class"] == 2).sum()),
            "truth_cells_planning_grid": int(planning_layers["truth"].sum()),
            "runtime_s": elapsed,
            "obstacle_collisions": count_obstacle_collisions(route, farm),
            **m,
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "table5_grid_sensitivity.csv", index=False, encoding="utf-8-sig")
    return df


def plot_grid(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax2 = ax1.twinx()
    x = df["grid_size_m"]

    l1 = ax1.plot(x, df["high_risk_coverage_rate"], marker="o", linewidth=2.0, color="#d32f2f", label="High-risk coverage")
    l2 = ax1.plot(x, df["abnormality_detection_rate"], marker="s", linewidth=2.0, color="#1976d2", label="Detection rate")
    l3 = ax2.plot(x, df["runtime_s"], marker="^", linewidth=1.8, linestyle="--", color="#388e3c", label="Runtime")

    for _, row in df.iterrows():
        ax1.annotate(f"{int(row['n_cells'])} cells", (row["grid_size_m"], row["abnormality_detection_rate"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax1.set_xlabel("Grid size (m)")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Runtime (s)")
    ax1.set_ylim(0, 1.05)
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Grid-resolution sensitivity")
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig8_grid_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    best = df.sort_values(["abnormality_detection_rate", "high_risk_coverage_rate"], ascending=False).iloc[0]
    text = f"""# Checkpoint 07 - Grid-resolution sensitivity

Status: completed

Tested grid sizes:
- 10 m
- 20 m
- 30 m
- 40 m

Best by detection rate:
- grid size: {best['grid_size_m']:.0f} m
- detection rate: {best['abnormality_detection_rate']:.4f}
- high-risk coverage: {best['high_risk_coverage_rate']:.4f}
- runtime: {best['runtime_s']:.4f} s

Collision check:
- Max obstacle collisions across grid settings: {int(df['obstacle_collisions'].max())}

Outputs:
- results/tables/table5_grid_sensitivity.csv
- results/figures/fig8_grid_sensitivity.png
"""
    (config.CHECKPOINT_DIR / "07_grid_sensitivity.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step7b] Running grid-resolution sensitivity experiment...")
    df = run_grid_sensitivity()
    plot_grid(df)
    write_checkpoint(df)
    config.log(f"[Step7b] Saved grid table: {config.TABLE_DIR / 'table5_grid_sensitivity.csv'}")
    config.log(f"[Step7b] Saved grid figure: {config.FIG_DIR / 'fig8_grid_sensitivity.png'}")
    config.log(df[["grid_size_m", "n_cells", "runtime_s", "total_distance_m", "high_risk_coverage_rate", "abnormality_detection_rate", "obstacle_collisions"]].to_string(index=False))


if __name__ == "__main__":
    main()
