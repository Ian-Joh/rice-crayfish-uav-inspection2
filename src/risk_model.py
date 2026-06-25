# -*- coding: utf-8 -*-
"""Step 2: generate multisource risk layers for the simulated farm map."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config

Point = Tuple[float, float]

WEIGHTS = {
    "C": 0.24,  # water-colour abnormality
    "V": 0.20,  # aquatic vegetation abnormality
    "T": 0.16,  # thermal anomaly
    "S": 0.20,  # water-quality sensor abnormality
    "K": 0.20,  # structural proximity risk
}


def _load_farm() -> Dict:
    path = config.DATA_DIR / "simulated_farm_map.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing farm map: {path}. Run simulation.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_centers() -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(config.GRID_SIZE_M / 2, config.MAP_WIDTH_M, config.GRID_SIZE_M)
    ys = np.arange(config.GRID_SIZE_M / 2, config.MAP_HEIGHT_M, config.GRID_SIZE_M)
    return np.meshgrid(xs, ys)


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _add_gaussian(layer: np.ndarray, X: np.ndarray, Y: np.ndarray, cx: float, cy: float, amp: float, sigma: float) -> None:
    layer += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def _facility_points(farm: Dict, kinds: Iterable[str] | None = None) -> List[Point]:
    pts = []
    kind_set = set(kinds) if kinds is not None else None
    for f in farm["facilities"]:
        if kind_set is None or f["kind"] in kind_set:
            pts.append((float(f["x"]), float(f["y"])))
    return pts


def _nearest_distance_layer(X: np.ndarray, Y: np.ndarray, pts: List[Point]) -> np.ndarray:
    if not pts:
        return np.zeros_like(X, dtype=float)
    dist = np.full_like(X, fill_value=np.inf, dtype=float)
    for x, y in pts:
        d = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        dist = np.minimum(dist, d)
    return dist


def _generate_patch_centers(farm: Dict, n: int, prefer_kinds: List[str], jitter: float) -> List[Point]:
    base = _facility_points(farm, prefer_kinds)
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


def generate_risk_layers() -> Dict[str, np.ndarray | Dict | List[Dict]]:
    random.seed(config.RANDOM_SEED + 1)
    np.random.seed(config.RANDOM_SEED + 1)
    farm = _load_farm()
    X, Y = _grid_centers()

    C = np.zeros_like(X, dtype=float)
    V = np.zeros_like(X, dtype=float)
    T = np.zeros_like(X, dtype=float)
    sensor_signal = np.zeros_like(X, dtype=float)

    abnormal_events: List[Dict] = []

    # Water-colour anomalies: more likely near inlets/outlets.
    for idx, (cx, cy) in enumerate(_generate_patch_centers(farm, 12, ["inlet_outlet"], 80), start=1):
        amp = random.uniform(0.65, 1.0)
        sigma = random.uniform(25, 55)
        _add_gaussian(C, X, Y, cx, cy, amp, sigma)
        abnormal_events.append({"event_id": f"C{idx}", "kind": "water_colour", "x": cx, "y": cy, "sigma": sigma, "severity": amp})

    # Vegetation anomalies: around ponds but less tied to facilities.
    pond_centers = [tuple(p["center"]) for p in farm["ponds"]]
    for idx in range(1, 11):
        bx, by = random.choice(pond_centers)
        cx = max(0.0, min(config.MAP_WIDTH_M, bx + random.uniform(-65, 65)))
        cy = max(0.0, min(config.MAP_HEIGHT_M, by + random.uniform(-65, 65)))
        amp = random.uniform(0.55, 0.95)
        sigma = random.uniform(30, 70)
        _add_gaussian(V, X, Y, cx, cy, amp, sigma)
        abnormal_events.append({"event_id": f"V{idx}", "kind": "vegetation", "x": round(cx, 2), "y": round(cy, 2), "sigma": sigma, "severity": amp})

    # Thermal anomalies: fewer, smoother hot spots.
    for idx, (cx, cy) in enumerate(_generate_patch_centers(farm, 7, ["feeding_point", "aerator"], 110), start=1):
        amp = random.uniform(0.45, 0.9)
        sigma = random.uniform(40, 85)
        _add_gaussian(T, X, Y, cx, cy, amp, sigma)
        abnormal_events.append({"event_id": f"T{idx}", "kind": "thermal", "x": cx, "y": cy, "sigma": sigma, "severity": amp})

    # Sensor abnormality: point readings diffused by inverse-distance-like Gaussian kernels.
    sensor_pts = _facility_points(farm, ["water_sensor"])
    sensor_readings = []
    for sid, (sx, sy) in enumerate(sensor_pts, start=1):
        # Several sensors are abnormal, especially near existing patches.
        local_base = 0.25 + 0.5 * random.random()
        if random.random() < 0.45:
            local_base += random.uniform(0.35, 0.65)
        local_base = min(1.0, local_base)
        sensor_readings.append({"sensor_id": sid, "x": sx, "y": sy, "abnormal_score": round(local_base, 4)})
        _add_gaussian(sensor_signal, X, Y, sx, sy, local_base, sigma=95)
        # Important: sensor-only events must also enter ground truth; otherwise
        # removing S_i cannot be evaluated fairly in ablation.
        if local_base >= 0.72:
            abnormal_events.append({"event_id": f"S{sid}", "kind": "sensor_water_quality", "x": sx, "y": sy, "sigma": 65.0, "severity": local_base})

    # Structural proximity risk.
    structural_pts = _facility_points(farm, ["inlet_outlet", "feeding_point", "aerator"])
    dist_to_struct = _nearest_distance_layer(X, Y, structural_pts)
    K = np.exp(-dist_to_struct / 85.0)
    # Add structure-driven true-risk events around key facilities. These represent
    # facility-related risks such as drainage blockage, feeding residue or aerator
    # malfunction, and make K_i ablation meaningful.
    for idx, (kx, ky) in enumerate(random.sample(structural_pts, min(12, len(structural_pts))), start=1):
        abnormal_events.append({"event_id": f"K{idx}", "kind": "structural_facility", "x": kx, "y": ky, "sigma": 45.0, "severity": random.uniform(0.65, 0.95)})

    # Normalize and add mild observation noise.
    C = _normalize(C + np.random.normal(0, 0.015, C.shape))
    V = _normalize(V + np.random.normal(0, 0.015, V.shape))
    T = _normalize(T + np.random.normal(0, 0.012, T.shape))
    S = _normalize(sensor_signal + np.random.normal(0, 0.01, sensor_signal.shape))
    K = _normalize(K)

    R = WEIGHTS["C"] * C + WEIGHTS["V"] * V + WEIGHTS["T"] * T + WEIGHTS["S"] * S + WEIGHTS["K"] * K
    R = _normalize(R)
    risk_class = np.zeros_like(R, dtype=np.int8)
    risk_class[(R >= 0.4) & (R < 0.7)] = 1
    risk_class[R >= 0.7] = 2

    # Ground-truth abnormal cells: cells close to generated event centers and severe enough.
    truth = np.zeros_like(R, dtype=np.int8)
    for e in abnormal_events:
        d = np.sqrt((X - e["x"]) ** 2 + (Y - e["y"]) ** 2)
        truth[d <= max(30.0, float(e["sigma"]) * 0.9)] = 1

    metadata = {
        "weights": WEIGHTS,
        "threshold_low_medium": 0.4,
        "threshold_medium_high": 0.7,
        "grid_size_m": config.GRID_SIZE_M,
        "n_rows": int(R.shape[0]),
        "n_cols": int(R.shape[1]),
        "n_abnormal_events": len(abnormal_events),
        "n_truth_cells": int(truth.sum()),
        "n_high_risk_cells": int((risk_class == 2).sum()),
    }

    return {
        "X": X,
        "Y": Y,
        "C": C,
        "V": V,
        "T": T,
        "S": S,
        "K": K,
        "R": R,
        "risk_class": risk_class,
        "truth": truth,
        "metadata": metadata,
        "abnormal_events": abnormal_events,
        "sensor_readings": sensor_readings,
    }


def save_layers(layers: Dict) -> None:
    np.savez_compressed(
        config.DATA_DIR / "risk_layers.npz",
        X=layers["X"],
        Y=layers["Y"],
        C=layers["C"],
        V=layers["V"],
        T=layers["T"],
        S=layers["S"],
        K=layers["K"],
        R=layers["R"],
        risk_class=layers["risk_class"],
        truth=layers["truth"],
    )
    meta = {
        "metadata": layers["metadata"],
        "abnormal_events": layers["abnormal_events"],
        "sensor_readings": layers["sensor_readings"],
    }
    (config.DATA_DIR / "risk_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_risk_layers(layers: Dict) -> None:
    names = [
        ("C", "Water-colour abnormality C_i"),
        ("V", "Vegetation abnormality V_i"),
        ("T", "Thermal anomaly T_i"),
        ("S", "Water-quality sensor abnormality S_i"),
        ("K", "Structural proximity risk K_i"),
        ("truth", "Ground-truth abnormal cells"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)
    for ax, (key, title) in zip(axes.flat, names):
        im = ax.imshow(layers[key], origin="lower", extent=[0, config.MAP_WIDTH_M, 0, config.MAP_HEIGHT_M], cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig2_risk_layers.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(layers["R"], origin="lower", extent=[0, config.MAP_WIDTH_M, 0, config.MAP_HEIGHT_M], cmap="magma", vmin=0, vmax=1)
    ax.contour(layers["X"], layers["Y"], layers["risk_class"] == 2, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title("Integrated risk map R_i with high-risk contour")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="Risk score")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig3_integrated_risk_map.png")
    plt.close(fig)


def write_checkpoint(layers: Dict) -> None:
    m = layers["metadata"]
    text = f"""# Checkpoint 02 - Risk map generation

Status: completed

Risk model:
- R_i = {WEIGHTS['C']} C_i + {WEIGHTS['V']} V_i + {WEIGHTS['T']} T_i + {WEIGHTS['S']} S_i + {WEIGHTS['K']} K_i
- Low risk: R_i < 0.4
- Medium risk: 0.4 <= R_i < 0.7
- High risk: R_i >= 0.7

Grid:
- Rows: {m['n_rows']}
- Cols: {m['n_cols']}
- Grid size: {m['grid_size_m']} m

Generated risk evidence:
- Abnormal events: {m['n_abnormal_events']}
- Ground-truth abnormal cells: {m['n_truth_cells']}
- High-risk cells: {m['n_high_risk_cells']}

Outputs:
- data/risk_layers.npz
- data/risk_metadata.json
- results/figures/fig2_risk_layers.png
- results/figures/fig3_integrated_risk_map.png
"""
    (config.CHECKPOINT_DIR / "02_risk_map_generation.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step2] Generating multisource risk layers...")
    layers = generate_risk_layers()
    save_layers(layers)
    plot_risk_layers(layers)
    write_checkpoint(layers)
    config.log(f"[Step2] Saved risk layers: {config.DATA_DIR / 'risk_layers.npz'}")
    config.log(f"[Step2] Saved risk metadata: {config.DATA_DIR / 'risk_metadata.json'}")
    config.log(f"[Step2] High-risk cells: {layers['metadata']['n_high_risk_cells']}, truth cells: {layers['metadata']['n_truth_cells']}")


if __name__ == "__main__":
    main()
