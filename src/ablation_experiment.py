# -*- coding: utf-8 -*-
"""Step 5: ablation studies for the proposed method."""
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


def normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def rebuild_risk(layers: dict, use_k: bool = True, use_s: bool = True) -> dict:
    C, V, T, S, K = layers["C"], layers["V"], layers["T"], layers["S"], layers["K"]
    weights = {"C": 0.24, "V": 0.20, "T": 0.16, "S": 0.20 if use_s else 0.0, "K": 0.20 if use_k else 0.0}
    total = sum(weights.values())
    R = weights["C"] * C + weights["V"] * V + weights["T"] * T + weights["S"] * S + weights["K"] * K
    R = R / max(total, 1e-9)
    R = normalize(R)
    out = {k: v.copy() if hasattr(v, "copy") else v for k, v in layers.items()}
    out["R"] = R
    rc = np.zeros_like(R, dtype=np.int8)
    rc[(R >= 0.4) & (R < 0.7)] = 1
    rc[R >= 0.7] = 2
    out["risk_class"] = rc
    return out


def run_ablation() -> pd.DataFrame:
    base_layers = load_layers()
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    cases = []

    # Full risk map is the common evaluation target for all ablations.
    # Ablated maps are used only to generate routes; metrics are computed against
    # the same full-model high-risk cells and the same ground-truth abnormal cells.
    layers_full = rebuild_risk(base_layers, use_k=True, use_s=True)
    eval_layers = layers_full

    # Full model.
    cases.append(("Full model", proposed_route(layers_full)))

    # Without structural proximity risk K_i.
    layers_no_k = rebuild_risk(base_layers, use_k=False, use_s=True)
    cases.append(("w/o K_i", proposed_route(layers_no_k)))

    # Without sensor abnormality S_i.
    layers_no_s = rebuild_risk(base_layers, use_k=True, use_s=False)
    cases.append(("w/o S_i", proposed_route(layers_no_s)))

    # Without adaptive insertion: baseline CPP.
    cases.append(("w/o adaptive insertion", standard_cpp_route()))

    rows = []
    for name, route in cases:
        route = avoid_obstacles(route, farm)
        m = evaluate_route(route, eval_layers)
        rows.append({"ablation": name, "obstacle_collisions": count_obstacle_collisions(route, farm), **m})
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLE_DIR / "table3_ablation.csv", index=False, encoding="utf-8-sig")
    return df


def plot_ablation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=160)
    colors = ["#ef5350", "#ffb74d", "#64b5f6", "#a5d6a7"]
    x = df["ablation"].tolist()
    for ax, col, title in [
        (axes[0], "high_risk_coverage_rate", "High-risk coverage"),
        (axes[1], "abnormality_detection_rate", "Detection rate"),
        (axes[2], "missed_risk_rate", "Missed-risk rate"),
    ]:
        ax.bar(x, df[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, float(df[col].max()) * 1.15))
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig6_ablation_results.png")
    plt.close(fig)


def write_checkpoint(df: pd.DataFrame) -> None:
    full = df[df["ablation"] == "Full model"].iloc[0]
    no_k = df[df["ablation"] == "w/o K_i"].iloc[0]
    no_s = df[df["ablation"] == "w/o S_i"].iloc[0]
    no_adapt = df[df["ablation"] == "w/o adaptive insertion"].iloc[0]
    text = f"""# Checkpoint 05 - Ablation experiment

Status: completed

Ablation cases:
- Full model
- w/o K_i: remove structural proximity risk
- w/o S_i: remove water-quality sensor abnormality
- w/o adaptive insertion: use standard CPP route only

Key results:
- Full detection rate: {full['abnormality_detection_rate']:.4f}
- w/o K_i detection rate: {no_k['abnormality_detection_rate']:.4f}
- w/o S_i detection rate: {no_s['abnormality_detection_rate']:.4f}
- w/o adaptive insertion detection rate: {no_adapt['abnormality_detection_rate']:.4f}
- Full high-risk coverage: {full['high_risk_coverage_rate']:.4f}
- w/o adaptive insertion high-risk coverage: {no_adapt['high_risk_coverage_rate']:.4f}

Outputs:
- results/tables/table3_ablation.csv
- results/figures/fig6_ablation_results.png
"""
    (config.CHECKPOINT_DIR / "05_ablation_experiment.md").write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step5] Running ablation experiment...")
    df = run_ablation()
    plot_ablation(df)
    write_checkpoint(df)
    config.log(f"[Step5] Saved ablation table: {config.TABLE_DIR / 'table3_ablation.csv'}")
    config.log(f"[Step5] Saved ablation figure: {config.FIG_DIR / 'fig6_ablation_results.png'}")
    config.log(df[["ablation", "total_distance_m", "high_risk_coverage_rate", "abnormality_detection_rate", "missed_risk_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
