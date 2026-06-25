# -*- coding: utf-8 -*-
"""Supplementary Experiment C: abnormality-density sensitivity."""
from __future__ import annotations

import json
import random
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import evaluate_route
from planners import avoid_obstacles, count_obstacle_collisions, fixed_route, proposed_route, standard_cpp_route

Point = Tuple[float, float]
WEIGHTS = {"C": 0.24, "V": 0.20, "T": 0.16, "S": 0.20, "K": 0.20}
DENSITY_SETTINGS = {
    "Low": {"C": 5, "V": 4, "T": 3, "K": 5},
    "Medium": {"C": 12, "V": 10, "T": 7, "K": 12},
    "High": {"C": 20, "V": 16, "T": 12, "K": 20},
    "Very high": {"C": 30, "V": 24, "T": 18, "K": 30},
}
ORDER = list(DENSITY_SETTINGS.keys())


def normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def grid_centers() -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(config.GRID_SIZE_M / 2, config.MAP_WIDTH_M, config.GRID_SIZE_M)
    ys = np.arange(config.GRID_SIZE_M / 2, config.MAP_HEIGHT_M, config.GRID_SIZE_M)
    return np.meshgrid(xs, ys)


def add_gaussian(layer: np.ndarray, X: np.ndarray, Y: np.ndarray, cx: float, cy: float, amp: float, sigma: float) -> None:
    layer += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def facility_points(farm: Dict, kinds: Iterable[str] | None = None) -> List[Point]:
    kind_set = set(kinds) if kinds is not None else None
    return [(float(f["x"]), float(f["y"])) for f in farm["facilities"] if kind_set is None or f["kind"] in kind_set]


def nearest_distance_layer(X: np.ndarray, Y: np.ndarray, pts: List[Point]) -> np.ndarray:
    dist = np.full_like(X, fill_value=np.inf, dtype=float)
    for x, y in pts:
        dist = np.minimum(dist, np.sqrt((X - x) ** 2 + (Y - y) ** 2))
    return dist if pts else np.zeros_like(X, dtype=float)


def generate_patch_centers(farm: Dict, n: int, prefer_kinds: List[str], jitter: float) -> List[Point]:
    base = facility_points(farm, prefer_kinds)
    out: List[Point] = []
    for _ in range(n):
        if base and random.random() < 0.72:
            bx, by = random.choice(base)
            x = max(0.0, min(config.MAP_WIDTH_M, bx + random.uniform(-jitter, jitter)))
            y = max(0.0, min(config.MAP_HEIGHT_M, by + random.uniform(-jitter, jitter)))
        else:
            x = random.uniform(80, config.MAP_WIDTH_M - 80)
            y = random.uniform(80, config.MAP_HEIGHT_M - 80)
        out.append((round(x, 2), round(y, 2)))
    return out


def build_density_layers(farm: Dict, setting: Dict[str, int], seed_offset: int = 500) -> Dict[str, np.ndarray]:
    random.seed(config.RANDOM_SEED + seed_offset)
    np.random.seed(config.RANDOM_SEED + seed_offset)
    X, Y = grid_centers()
    C = np.zeros_like(X, dtype=float)
    V = np.zeros_like(X, dtype=float)
    T = np.zeros_like(X, dtype=float)
    sensor_signal = np.zeros_like(X, dtype=float)
    abnormal_events: List[Dict] = []

    for cx, cy in generate_patch_centers(farm, setting["C"], ["inlet_outlet"], 80):
        sigma = random.uniform(25, 55)
        add_gaussian(C, X, Y, cx, cy, random.uniform(0.65, 1.0), sigma)
        abnormal_events.append({"x": cx, "y": cy, "sigma": sigma})

    pond_centers = [tuple(p["center"]) for p in farm["ponds"]]
    for _ in range(setting["V"]):
        bx, by = random.choice(pond_centers)
        cx = max(0.0, min(config.MAP_WIDTH_M, bx + random.uniform(-65, 65)))
        cy = max(0.0, min(config.MAP_HEIGHT_M, by + random.uniform(-65, 65)))
        sigma = random.uniform(30, 70)
        add_gaussian(V, X, Y, cx, cy, random.uniform(0.55, 0.95), sigma)
        abnormal_events.append({"x": round(cx, 2), "y": round(cy, 2), "sigma": sigma})

    for cx, cy in generate_patch_centers(farm, setting["T"], ["feeding_point", "aerator"], 110):
        sigma = random.uniform(40, 85)
        add_gaussian(T, X, Y, cx, cy, random.uniform(0.45, 0.9), sigma)
        abnormal_events.append({"x": cx, "y": cy, "sigma": sigma})

    for sx, sy in facility_points(farm, ["water_sensor"]):
        local_base = 0.25 + 0.5 * random.random()
        if random.random() < 0.45:
            local_base += random.uniform(0.35, 0.65)
        local_base = min(1.0, local_base)
        add_gaussian(sensor_signal, X, Y, sx, sy, local_base, sigma=95)
        if local_base >= 0.72:
            abnormal_events.append({"x": sx, "y": sy, "sigma": 65.0})

    structural_pts = facility_points(farm, ["inlet_outlet", "feeding_point", "aerator"])
    dist_to_struct = nearest_distance_layer(X, Y, structural_pts)
    K = np.exp(-dist_to_struct / 85.0)
    for kx, ky in random.sample(structural_pts, min(setting["K"], len(structural_pts))):
        abnormal_events.append({"x": kx, "y": ky, "sigma": 45.0})

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


def run_density() -> pd.DataFrame:
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    rows = []
    for level, setting in DENSITY_SETTINGS.items():
        layers = build_density_layers(farm, setting)
        routes = {
            "Fixed route": avoid_obstacles(fixed_route(), farm),
            "Standard CPP": avoid_obstacles(standard_cpp_route(), farm),
            "Proposed method": avoid_obstacles(proposed_route(layers, risk_threshold=0.7), farm),
        }
        for method, route in routes.items():
            m = evaluate_route(route, layers)
            rows.append({
                "density_level": level,
                "method": method,
                "n_event_requested": int(sum(setting.values())),
                "truth_cells": int(layers["truth"].sum()),
                "high_risk_cells": int((layers["risk_class"] == 2).sum()),
                "obstacle_collisions": count_obstacle_collisions(route, farm),
                **m,
            })
    df = pd.DataFrame(rows)
    df["density_level"] = pd.Categorical(df["density_level"], categories=ORDER, ordered=True)
    df.to_csv(config.TABLE_DIR / "table12_abnormality_density.csv", index=False, encoding="utf-8-sig")
    return df


def plot_density(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    x = np.arange(len(ORDER))
    markers = {"Fixed route": "o", "Standard CPP": "s", "Proposed method": "^"}
    colors = {"Fixed route": "#90caf9", "Standard CPP": "#66bb6a", "Proposed method": "#ef5350"}
    for method in ["Fixed route", "Standard CPP", "Proposed method"]:
        g = df[df["method"] == method].sort_values("density_level")
        ax.plot(x, g["abnormality_detection_rate"], marker=markers[method], linewidth=2.0, color=colors[method], label=f"{method} detection")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_xlabel("Abnormality density")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.set_title("Abnormality-density sensitivity")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig15_abnormality_density.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    proposed = df[df["method"] == "Proposed method"]
    cpp = df[df["method"] == "Standard CPP"]
    text = f"""# Checkpoint 17 - Abnormality-density sensitivity

Status: completed

Density levels:
- Low
- Medium
- High
- Very high

Compared methods:
- Fixed route
- Standard CPP
- Proposed method

Mean detection rate across density levels:
- Proposed method: {proposed['abnormality_detection_rate'].mean():.4f}
- Standard CPP: {cpp['abnormality_detection_rate'].mean():.4f}

Collision check:
- Max obstacle collisions: {int(df['obstacle_collisions'].max())}

Outputs:
- results/tables/table12_abnormality_density.csv
- results/figures/fig15_abnormality_density.png
"""
    (config.CHECKPOINT_DIR / "17_abnormality_density.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Supplement C] Running abnormality-density sensitivity experiment...")
    df = run_density()
    plot_density(df)
    write_checkpoint(df)
    config.log(f"[Supplement C] Saved table: {config.TABLE_DIR / 'table12_abnormality_density.csv'}")
    config.log(f"[Supplement C] Saved figure: {config.FIG_DIR / 'fig15_abnormality_density.png'}")
    config.log(df[["density_level", "method", "truth_cells", "high_risk_cells", "high_risk_coverage_rate", "abnormality_detection_rate", "obstacle_collisions"]].to_string(index=False))


if __name__ == "__main__":
    main()
