# -*- coding: utf-8 -*-
"""Step 1: generate a realistic rice-crayfish co-culture farm map.

Design rationale:
- Rice-crayfish farms are usually connected paddy/pond blocks rather than isolated random ponds.
- Water-exchange risks concentrate near canals, inlets/outlets, corners and weak-flow zones.
- The map therefore uses a block layout: rectangular/irregular paddy-crayfish units,
  a main canal, branch canals, embankment gaps, functional points and no-fly obstacles.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

import config

Point = Tuple[float, float]


@dataclass
class Pond:
    pond_id: int
    polygon: List[Point]
    center: Point
    unit_type: str


@dataclass
class Facility:
    facility_id: int
    kind: str
    x: float
    y: float
    pond_id: int | None = None


@dataclass
class Canal:
    canal_id: int
    kind: str
    polyline: List[Point]
    width: float


@dataclass
class Obstacle:
    obstacle_id: int
    kind: str
    x: float
    y: float
    width: float
    height: float


def _jitter_rect(x0: float, y0: float, w: float, h: float, jitter: float = 10.0) -> List[Point]:
    pts = [
        (x0 + random.uniform(-jitter, jitter), y0 + random.uniform(-jitter, jitter)),
        (x0 + w + random.uniform(-jitter, jitter), y0 + random.uniform(-jitter, jitter)),
        (x0 + w + random.uniform(-jitter, jitter), y0 + h + random.uniform(-jitter, jitter)),
        (x0 + random.uniform(-jitter, jitter), y0 + h + random.uniform(-jitter, jitter)),
    ]
    return [(round(max(20, min(config.MAP_WIDTH_M - 20, x)), 2), round(max(20, min(config.MAP_HEIGHT_M - 20, y)), 2)) for x, y in pts]


def _center(poly: List[Point]) -> Point:
    return (round(sum(x for x, _ in poly) / len(poly), 2), round(sum(y for _, y in poly) / len(poly), 2))


def _edge_point(poly: List[Point], side: str) -> Point:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    if side == "left":
        return (min(xs), sum(ys) / len(ys))
    if side == "right":
        return (max(xs), sum(ys) / len(ys))
    if side == "bottom":
        return (sum(xs) / len(xs), min(ys))
    return (sum(xs) / len(xs), max(ys))


def generate_farm_map() -> Dict:
    random.seed(config.RANDOM_SEED)

    ponds: List[Pond] = []
    facilities: List[Facility] = []
    canals: List[Canal] = []
    obstacles: List[Obstacle] = []

    # Main water network: one main canal and four branch canals.
    canals.append(Canal(1, "main_canal", [(90, 500), (910, 500)], 28.0))
    for i, x in enumerate([220, 390, 610, 780], start=2):
        canals.append(Canal(i, "branch_canal", [(x, 130), (x + random.uniform(-20, 20), 870)], 16.0))

    # Connected production blocks arranged in 4 rows x 5 columns.
    # Leave corridors for canals and embankments.
    pid = 1
    x_positions = [110, 270, 445, 620, 790]
    y_positions = [110, 300, 545, 735]
    for row, y0 in enumerate(y_positions):
        for col, x0 in enumerate(x_positions):
            if pid > 18:
                break
            w = random.uniform(115, 145)
            h = random.uniform(115, 145)
            poly = _jitter_rect(x0, y0, w, h, jitter=9.0)
            cx, cy = _center(poly)
            unit_type = "rice_crayfish_paddy" if row in [0, 1, 2] else "crayfish_pond"
            ponds.append(Pond(pid, poly, (cx, cy), unit_type))

            # Inlet/outlet is placed on edge facing nearest canal.
            nearest_side = "left" if abs(cx - 220) < abs(cx - 780) else "right"
            if abs(cy - 500) < 80:
                nearest_side = "bottom" if cy > 500 else "top"
            ix, iy = _edge_point(poly, nearest_side)
            ix += random.uniform(-8, 8)
            iy += random.uniform(-8, 8)
            facilities.append(Facility(len(facilities) + 1, "inlet_outlet", round(ix, 2), round(iy, 2), pid))

            # Feeding point in internal shallow-water area.
            facilities.append(Facility(len(facilities) + 1, "feeding_point", round(cx + random.uniform(-28, 28), 2), round(cy + random.uniform(-28, 28), 2), pid))

            # Aerators not in all units; more likely in crayfish pond units and large blocks.
            if random.random() < (0.55 if unit_type == "rice_crayfish_paddy" else 0.85):
                facilities.append(Facility(len(facilities) + 1, "aerator", round(cx + random.uniform(-32, 32), 2), round(cy + random.uniform(-32, 32), 2), pid))

            # Sensors in about half the units, biased near canal-connected blocks.
            if random.random() < 0.58:
                facilities.append(Facility(len(facilities) + 1, "water_sensor", round(cx + random.uniform(-38, 38), 2), round(cy + random.uniform(-38, 38), 2), pid))
            pid += 1

    # Obstacles/no-fly objects: trees, pump house, wires. Avoid too many random central rectangles.
    obstacles.extend([
        Obstacle(1, "pump_house", 55, 455, 55, 80),
        Obstacle(2, "tree_cluster", 925, 430, 45, 95),
        Obstacle(3, "power_pole_zone", 475, 910, 90, 35),
        Obstacle(4, "storage_shed", 35, 820, 75, 55),
        Obstacle(5, "tree_cluster", 890, 120, 70, 50),
    ])

    farm = {
        "metadata": {
            "seed": config.RANDOM_SEED,
            "map_width_m": config.MAP_WIDTH_M,
            "map_height_m": config.MAP_HEIGHT_M,
            "grid_size_m": config.GRID_SIZE_M,
            "uav_start": config.UAV_START,
            "scenario": "connected_block_rice_crayfish_farm_with_canals",
        },
        "ponds": [asdict(p) for p in ponds],
        "canals": [asdict(c) for c in canals],
        "facilities": [asdict(f) for f in facilities],
        "obstacles": [asdict(o) for o in obstacles],
    }
    return farm


def save_farm_map(farm: Dict, path: Path) -> None:
    path.write_text(json.dumps(farm, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_farm_map(farm: Dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
    ax.set_xlim(0, config.MAP_WIDTH_M)
    ax.set_ylim(0, config.MAP_HEIGHT_M)
    ax.set_aspect("equal")
    ax.set_title("Connected rice-crayfish co-culture farm map")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Canals first.
    for canal in farm["canals"]:
        xs = [p[0] for p in canal["polyline"]]
        ys = [p[1] for p in canal["polyline"]]
        color = "#1565c0" if canal["kind"] == "main_canal" else "#42a5f5"
        ax.plot(xs, ys, color=color, linewidth=canal["width"] / 5.0, alpha=0.65, solid_capstyle="round", label=canal["kind"] if canal["canal_id"] <= 2 else None)

    for pond in farm["ponds"]:
        face = "#c8e6c9" if pond.get("unit_type") == "rice_crayfish_paddy" else "#b3e5fc"
        edge = "#2e7d32" if pond.get("unit_type") == "rice_crayfish_paddy" else "#0277bd"
        patch = Polygon(pond["polygon"], closed=True, facecolor=face, edgecolor=edge, linewidth=1.0, alpha=0.78)
        ax.add_patch(patch)
        cx, cy = pond["center"]
        ax.text(cx, cy, str(pond["pond_id"]), fontsize=7, color="#1b5e20", ha="center", va="center")

    colors = {
        "inlet_outlet": "#0d47a1",
        "feeding_point": "#ef6c00",
        "aerator": "#6a1b9a",
        "water_sensor": "#c62828",
    }
    markers = {
        "inlet_outlet": "s",
        "feeding_point": "^",
        "aerator": "*",
        "water_sensor": "o",
    }
    for kind in colors:
        xs = [f["x"] for f in farm["facilities"] if f["kind"] == kind]
        ys = [f["y"] for f in farm["facilities"] if f["kind"] == kind]
        if xs:
            ax.scatter(xs, ys, s=36, c=colors[kind], marker=markers[kind], label=kind, edgecolors="white", linewidths=0.4, zorder=4)

    for obs in farm["obstacles"]:
        ax.add_patch(Rectangle((obs["x"], obs["y"]), obs["width"], obs["height"], facecolor="#424242", alpha=0.55, edgecolor="black"))
        ax.text(obs["x"] + obs["width"] / 2, obs["y"] + obs["height"] / 2, obs["kind"].split("_")[0], fontsize=5, color="white", ha="center", va="center")

    sx, sy = farm["metadata"]["uav_start"]
    ax.scatter([sx], [sy], s=90, c="#000000", marker="X", label="UAV start", zorder=5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_checkpoint(farm: Dict, path: Path) -> None:
    counts = {}
    for f in farm["facilities"]:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    unit_counts = {}
    for p in farm["ponds"]:
        unit_counts[p.get("unit_type", "unknown")] = unit_counts.get(p.get("unit_type", "unknown"), 0) + 1
    text = f"""# Checkpoint 01 - Map generation

Status: completed

Map setting:
- Scenario: connected block rice-crayfish farm with canals
- Size: {config.MAP_WIDTH_M} m x {config.MAP_HEIGHT_M} m
- Grid size: {config.GRID_SIZE_M} m
- Seed: {config.RANDOM_SEED}

Generated objects:
- Production units: {len(farm['ponds'])}
- Rice-crayfish paddy units: {unit_counts.get('rice_crayfish_paddy', 0)}
- Crayfish pond units: {unit_counts.get('crayfish_pond', 0)}
- Canals: {len(farm['canals'])}
- Inlet/outlet points: {counts.get('inlet_outlet', 0)}
- Feeding points: {counts.get('feeding_point', 0)}
- Aerators: {counts.get('aerator', 0)}
- Water-quality sensors: {counts.get('water_sensor', 0)}
- Obstacles/no-fly rectangles: {len(farm['obstacles'])}

Outputs:
- data/simulated_farm_map.json
- results/figures/fig1_simulated_farm_map.png
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step1] Generating redesigned connected farm map...")
    farm = generate_farm_map()
    json_path = config.DATA_DIR / "simulated_farm_map.json"
    fig_path = config.FIG_DIR / "fig1_simulated_farm_map.png"
    ckpt_path = config.CHECKPOINT_DIR / "01_map_generation.md"
    save_farm_map(farm, json_path)
    plot_farm_map(farm, fig_path)
    write_checkpoint(farm, ckpt_path)
    config.log(f"[Step1] Saved map JSON: {json_path}")
    config.log(f"[Step1] Saved map figure: {fig_path}")
    config.log(f"[Step1] Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
