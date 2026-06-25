# -*- coding: utf-8 -*-
"""Step 9: robustness experiment under missing water-quality sensors."""
from __future__ import annotations

import json
import math

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


def add_gaussian(layer: np.ndarray, X: np.ndarray, Y: np.ndarray, cx: float, cy: float, amp: float, sigma: float) -> None:
    layer += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def rebuild_with_sensor_ratio(base: dict, keep_ratio: float, seed: int) -> dict:
    meta = json.loads((config.DATA_DIR / "risk_metadata.json").read_text(encoding="utf-8"))
    sensors = meta["sensor_readings"]
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(len(sensors) * keep_ratio)))
    keep_idx = set(rng.choice(len(sensors), size=n_keep, replace=False).tolist())
    X, Y = base["X"], base["Y"]
    S = np.zeros_like(base["S"], dtype=float)
    for i, s in enumerate(sensors):
        if i in keep_idx:
            add_gaussian(S, X, Y, float(s["x"]), float(s["y"]), float(s["abnormal_score"]), sigma=95)
    S = normalize(S)
    out = {k: v.copy() if hasattr(v, "copy") else v for k, v in base.items()}
    out["S"] = S
    R = 0.24 * out["C"] + 0.20 * out["V"] + 0.16 * out["T"] + 0.20 * out["S"] + 0.20 * out["K"]
    out["R"] = normalize(R)
    rc = np.zeros_like(out["R"], dtype=np.int8)
    rc[(out["R"] >= 0.4) & (out["R"] < 0.7)] = 1
    rc[out["R"] >= 0.7] = 2
    out["risk_class"] = rc
    out["kept_sensor_count"] = n_keep
    return out


def run_sensor_missing() -> pd.DataFrame:
    base = load_layers()
    eval_layers = base
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    rows = []
    for ratio in [1.00, 0.75, 0.50, 0.25]:
        metrics_list = []
        kept_counts = []
        for rep in range(5):
            planning_layers = rebuild_with_sensor_ratio(base, ratio, config.RANDOM_SEED + 300 + rep)
            route = avoid_obstacles(proposed_route(planning_layers, risk_threshold=0.7), farm)
            m = evaluate_route(route, eval_layers)
            m["obstacle_collisions"] = count_obstacle_collisions(route, farm)
            metrics_list.append(m)
            kept_counts.append(planning_layers["kept_sensor_count"])
        mean = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        std = {k + "_std": float(np.std([m[k] for m in metrics_list])) for k in metrics_list[0]}
        rows.append({"sensor_keep_ratio": ratio, "kept_sensor_count": float(np.mean(kept_counts)), **mean, **std})
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "table8_sensor_missing.csv", index=False, encoding="utf-8-sig")
    return df


def plot_sensor_missing(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    x = df["sensor_keep_ratio"] * 100

    ax.errorbar(x, df["high_risk_coverage_rate"], yerr=df["high_risk_coverage_rate_std"], marker="o", linewidth=2.0, color="#d32f2f", capsize=3, label="High-risk coverage")
    ax.errorbar(x, df["abnormality_detection_rate"], yerr=df["abnormality_detection_rate_std"], marker="s", linewidth=2.0, color="#1976d2", capsize=3, label="Detection rate")
    ax.errorbar(x, df["missed_risk_rate"], yerr=df["missed_risk_rate_std"], marker="^", linewidth=1.8, linestyle="--", color="#388e3c", capsize=3, label="Missed-risk rate")

    for _, row in df.iterrows():
        ax.annotate(f"n={int(row['kept_sensor_count'])}", (row["sensor_keep_ratio"] * 100, row["abnormality_detection_rate"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Sensor keep ratio (%)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.set_title("Sensor-missing robustness")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig11_sensor_missing.png", bbox_inches="tight")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    full = df[df["sensor_keep_ratio"] == 1.0].iloc[0]
    low = df[df["sensor_keep_ratio"] == 0.25].iloc[0]
    text = f"""# Checkpoint 10 - Sensor missing robustness

Status: completed

Sensor keep ratios:
- 100%
- 75%
- 50%
- 25%

Each ratio repeated 5 times with different retained sensor subsets.

Key comparison:
- 100% sensors detection rate: {full['abnormality_detection_rate']:.4f}
- 25% sensors detection rate: {low['abnormality_detection_rate']:.4f}
- 100% sensors high-risk coverage: {full['high_risk_coverage_rate']:.4f}
- 25% sensors high-risk coverage: {low['high_risk_coverage_rate']:.4f}

Outputs:
- results/tables/table8_sensor_missing.csv
- results/figures/fig11_sensor_missing.png
"""
    (config.CHECKPOINT_DIR / "10_sensor_missing.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step9] Running sensor missing robustness experiment...")
    df = run_sensor_missing()
    plot_sensor_missing(df)
    write_checkpoint(df)
    config.log(f"[Step9] Saved sensor missing table: {config.TABLE_DIR / 'table8_sensor_missing.csv'}")
    config.log(f"[Step9] Saved sensor missing figure: {config.FIG_DIR / 'fig11_sensor_missing.png'}")
    config.log(df[["sensor_keep_ratio", "kept_sensor_count", "high_risk_coverage_rate", "abnormality_detection_rate", "missed_risk_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
