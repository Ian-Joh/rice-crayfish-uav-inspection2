# -*- coding: utf-8 -*-
"""Step 3: implement baseline and proposed UAV inspection planners."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config
from metrics import evaluate_route, route_distance

Point = Tuple[float, float]
SAFETY_MARGIN_M = 18.0


def load_inputs() -> Tuple[Dict, Dict[str, np.ndarray]]:
    farm = json.loads((config.DATA_DIR / "simulated_farm_map.json").read_text(encoding="utf-8"))
    z = np.load(config.DATA_DIR / "risk_layers.npz")
    layers = {k: z[k] for k in z.files}
    return farm, layers


def lawnmower_route(spacing: float = 80.0, margin: float = 50.0) -> List[Point]:
    route: List[Point] = [tuple(config.UAV_START)]
    ys = np.arange(margin, config.MAP_HEIGHT_M - margin + 1e-6, spacing)
    left, right = margin, config.MAP_WIDTH_M - margin
    for idx, y in enumerate(ys):
        if idx % 2 == 0:
            route.extend([(left, float(y)), (right, float(y))])
        else:
            route.extend([(right, float(y)), (left, float(y))])
    route.append(tuple(config.UAV_START))
    return [(round(x, 2), round(y, 2)) for x, y in route]


def fixed_route() -> List[Point]:
    # Coarser manual route, representing common fixed inspection practice.
    return lawnmower_route(spacing=150.0, margin=80.0)


def standard_cpp_route() -> List[Point]:
    return lawnmower_route(spacing=80.0, margin=50.0)


def nearest_neighbor_route(points: List[Point], start: Point) -> List[Point]:
    remaining = points[:]
    route = [start]
    cur = start
    while remaining:
        j = min(range(len(remaining)), key=lambda i: math.dist(cur, remaining[i]))
        cur = remaining.pop(j)
        route.append(cur)
    route.append(start)
    return [(round(x, 2), round(y, 2)) for x, y in route]


def expanded_rect(obs: Dict, margin: float = SAFETY_MARGIN_M) -> Tuple[float, float, float, float]:
    """Return obstacle rectangle expanded by a safety margin."""
    return (
        float(obs["x"]) - margin,
        float(obs["y"]) - margin,
        float(obs["x"]) + float(obs["width"]) + margin,
        float(obs["y"]) + float(obs["height"]) + margin,
    )


def _ccw(a: Point, b: Point, c: Point) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _point_in_rect(p: Point, rect: Tuple[float, float, float, float]) -> bool:
    x, y = p
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def segment_intersects_rect(a: Point, b: Point, obs: Dict, margin: float = SAFETY_MARGIN_M) -> bool:
    """Check whether a flight segment intersects an expanded no-fly rectangle."""
    rect = expanded_rect(obs, margin)
    if _point_in_rect(a, rect) or _point_in_rect(b, rect):
        return True
    x1, y1, x2, y2 = rect
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    return any(_segments_intersect(a, b, c, d) for c, d in edges)


def count_obstacle_collisions(route: List[Point], farm: Dict, margin: float = SAFETY_MARGIN_M) -> int:
    """Count segment-obstacle intersections for validation."""
    hits = 0
    for a, b in zip(route[:-1], route[1:]):
        for obs in farm.get("obstacles", []):
            if segment_intersects_rect(a, b, obs, margin):
                hits += 1
    return hits


def _clamp_point(p: Point) -> Point:
    x, y = p
    return (
        round(max(0.0, min(config.MAP_WIDTH_M, x)), 2),
        round(max(0.0, min(config.MAP_HEIGHT_M, y)), 2),
    )


def _detour_points(a: Point, b: Point, obs: Dict, margin: float = SAFETY_MARGIN_M) -> List[Point]:
    """Generate two detour waypoints around an expanded rectangular obstacle.

    Candidate detours are placed below, above, left and right of the expanded
    rectangle. The shortest candidate is selected.
    """
    x1, y1, x2, y2 = expanded_rect(obs, margin)
    pad = 12.0
    candidates = [
        [(x1 - pad, y1 - pad), (x2 + pad, y1 - pad)],
        [(x1 - pad, y2 + pad), (x2 + pad, y2 + pad)],
        [(x1 - pad, y1 - pad), (x1 - pad, y2 + pad)],
        [(x2 + pad, y1 - pad), (x2 + pad, y2 + pad)],
    ]
    best = min(candidates, key=lambda pts: math.dist(a, pts[0]) + math.dist(pts[0], pts[1]) + math.dist(pts[1], b))
    return [_clamp_point(best[0]), _clamp_point(best[1])]


def _blocked_grid(farm: Dict, grid: float, margin: float) -> np.ndarray:
    nx = int(config.MAP_WIDTH_M / grid) + 1
    ny = int(config.MAP_HEIGHT_M / grid) + 1
    blocked = np.zeros((ny, nx), dtype=bool)
    for obs in farm.get("obstacles", []):
        x1, y1, x2, y2 = expanded_rect(obs, margin)
        c1, c2 = max(0, int(x1 // grid)), min(nx - 1, int(x2 // grid) + 1)
        r1, r2 = max(0, int(y1 // grid)), min(ny - 1, int(y2 // grid) + 1)
        blocked[r1 : r2 + 1, c1 : c2 + 1] = True
    return blocked


def _point_to_cell(p: Point, grid: float, blocked: np.ndarray) -> Tuple[int, int]:
    ny, nx = blocked.shape
    c = max(0, min(nx - 1, int(round(p[0] / grid))))
    r = max(0, min(ny - 1, int(round(p[1] / grid))))
    if not blocked[r, c]:
        return r, c
    # If a target lies inside an expanded obstacle, move to nearest free cell.
    best = (r, c)
    best_d = float("inf")
    for rr in range(max(0, r - 8), min(ny, r + 9)):
        for cc in range(max(0, c - 8), min(nx, c + 9)):
            if not blocked[rr, cc]:
                d = (rr - r) ** 2 + (cc - c) ** 2
                if d < best_d:
                    best_d = d
                    best = (rr, cc)
    return best


def _cell_to_point(cell: Tuple[int, int], grid: float) -> Point:
    r, c = cell
    return _clamp_point((c * grid, r * grid))


def _astar_segment(a: Point, b: Point, farm: Dict, grid: float = 20.0, margin: float = SAFETY_MARGIN_M) -> List[Point]:
    blocked = _blocked_grid(farm, grid, margin)
    start = _point_to_cell(a, grid, blocked)
    goal = _point_to_cell(b, grid, blocked)
    if start == goal:
        return [b]
    import heapq

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_heap = [(0.0, start)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g = {start: 0.0}
    ny, nx = blocked.shape

    def h(cell: Tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    found = False
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            found = True
            break
        for dr, dc in neighbors:
            nr, nc = cur[0] + dr, cur[1] + dc
            if nr < 0 or nr >= ny or nc < 0 or nc >= nx or blocked[nr, nc]:
                continue
            step = math.hypot(dr, dc)
            ng = g[cur] + step
            nxt = (nr, nc)
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), nxt))
    if not found:
        return _detour_points(a, b, farm.get("obstacles", [])[0], margin) + [b]

    cells = [goal]
    while cells[-1] != start:
        cells.append(came[cells[-1]])
    cells.reverse()
    pts = [_cell_to_point(cell, grid) for cell in cells[1:]]
    # Simplify collinear runs.
    simplified: List[Point] = []
    for p in pts:
        if len(simplified) < 2:
            simplified.append(p)
        else:
            a0, a1 = simplified[-2], simplified[-1]
            v1 = (a1[0] - a0[0], a1[1] - a0[1])
            v2 = (p[0] - a1[0], p[1] - a1[1])
            if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-6:
                simplified[-1] = p
            else:
                simplified.append(p)
    safe_goal = _cell_to_point(goal, grid)
    # If the original endpoint is inside an expanded no-fly area, keep the
    # nearest free grid point as endpoint instead of forcing the route back into
    # the forbidden safety zone.
    endpoint_is_safe = not any(segment_intersects_rect(b, b, obs, margin) for obs in farm.get("obstacles", []))
    final_point = b if endpoint_is_safe else safe_goal
    if simplified:
        simplified[-1] = final_point
    return simplified or [final_point]


def avoid_obstacles(route: List[Point], farm: Dict, margin: float = SAFETY_MARGIN_M, max_iter: int = 6) -> List[Point]:
    """Repair a route by replacing colliding segments with A* grid detours."""
    blocked = _blocked_grid(farm, 20.0, margin)

    def safe_point(p: Point) -> Point:
        if any(segment_intersects_rect(p, p, obs, margin) for obs in farm.get("obstacles", [])):
            return _cell_to_point(_point_to_cell(p, 20.0, blocked), 20.0)
        return tuple(p)

    repaired = [safe_point(tuple(p)) for p in route]
    for _ in range(max_iter):
        changed = False
        new_route: List[Point] = [repaired[0]]
        for a, b in zip(repaired[:-1], repaired[1:]):
            if any(segment_intersects_rect(a, b, obs, margin) for obs in farm.get("obstacles", [])):
                detour = _astar_segment(a, b, farm, grid=20.0, margin=margin)
                new_route.extend(detour)
                changed = True
            else:
                new_route.append(b)
        compact: List[Point] = []
        for p in new_route:
            if not compact or math.dist(compact[-1], p) > 1e-6:
                compact.append(p)
        repaired = compact
        if not changed:
            break
    # Final strict pass with a finer grid for any remaining segment collision.
    strict_route: List[Point] = [repaired[0]]
    for a, b in zip(repaired[:-1], repaired[1:]):
        if any(segment_intersects_rect(a, b, obs, margin) for obs in farm.get("obstacles", [])):
            strict_route.extend(_astar_segment(a, b, farm, grid=10.0, margin=margin))
        else:
            strict_route.append(b)
    return [(round(x, 2), round(y, 2)) for x, y in strict_route]


def keypoint_tsp_route(farm: Dict) -> List[Point]:
    pts = []
    for f in farm["facilities"]:
        if f["kind"] in {"inlet_outlet", "feeding_point", "aerator"}:
            pts.append((float(f["x"]), float(f["y"])))
    return nearest_neighbor_route(pts, tuple(config.UAV_START))


def grid_candidates_from_score(layers: Dict[str, np.ndarray], score: np.ndarray, threshold_quantile: float, max_points: int) -> List[Tuple[Point, float]]:
    X, Y = layers["X"], layers["Y"]
    threshold = float(np.quantile(score, threshold_quantile))
    idxs = np.argwhere(score >= threshold)
    vals = [(int(r), int(c), float(score[r, c])) for r, c in idxs]
    vals.sort(key=lambda t: t[2], reverse=True)
    selected: List[Tuple[Point, float]] = []
    min_sep = 55.0
    for r, c, v in vals:
        p = (float(X[r, c]), float(Y[r, c]))
        if all(math.dist(p, q) >= min_sep for q, _ in selected):
            selected.append((p, v))
        if len(selected) >= max_points:
            break
    return selected


def ipp_route(layers: Dict[str, np.ndarray], max_points: int = 34) -> List[Point]:
    # IPP baseline uses uncertainty/information proxy from perception layers, excluding structural K.
    score = 0.30 * layers["C"] + 0.25 * layers["V"] + 0.20 * layers["T"] + 0.25 * layers["S"]
    candidates = [p for p, _ in grid_candidates_from_score(layers, score, 0.90, max_points)]
    return nearest_neighbor_route(candidates, tuple(config.UAV_START))


def insertion_cost(route: List[Point], p: Point) -> Tuple[float, int]:
    best_cost = float("inf")
    best_idx = 1
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        cost = math.dist(a, p) + math.dist(p, b) - math.dist(a, b)
        if cost < best_cost:
            best_cost = cost
            best_idx = i + 1
    return best_cost, best_idx


def proposed_route(layers: Dict[str, np.ndarray], extra_budget_ratio: float = 0.18, risk_threshold: float = 0.7, min_benefit_ratio: float = 0.003, base_route: List[Point] | None = None) -> List[Point]:
    base = base_route[:] if base_route is not None else standard_cpp_route()
    base_dist = route_distance(base)
    extra_budget = base_dist * extra_budget_ratio
    R = layers["R"]
    X, Y = layers["X"], layers["Y"]
    candidate_idxs = np.argwhere(R >= risk_threshold)
    candidates: List[Tuple[Point, float]] = []
    for r, c in candidate_idxs:
        p = (float(X[r, c]), float(Y[r, c]))
        benefit = float(R[r, c])
        if all(math.dist(p, q) >= 45.0 for q, _ in candidates):
            candidates.append((p, benefit))
    candidates.sort(key=lambda item: item[1], reverse=True)

    route = base[:]
    used_extra = 0.0
    selected = 0
    # Greedy benefit-to-cost insertion.
    while candidates:
        scored = []
        for p, benefit in candidates[:120]:
            cost, idx = insertion_cost(route, p)
            ratio = benefit / max(cost, 1.0)
            scored.append((ratio, cost, idx, p, benefit))
        scored.sort(reverse=True, key=lambda t: t[0])
        ratio, cost, idx, p, benefit = scored[0]
        if used_extra + cost > extra_budget or ratio < min_benefit_ratio:
            break
        route.insert(idx, (round(p[0], 2), round(p[1], 2)))
        used_extra += cost
        selected += 1
        candidates = [(q, b) for q, b in candidates if math.dist(q, p) > 45.0]
        if selected >= 45:
            break
    return route


def generate_routes(farm: Dict, layers: Dict[str, np.ndarray]) -> Dict[str, List[Point]]:
    raw_routes = {
        "Fixed route": fixed_route(),
        "Standard CPP": standard_cpp_route(),
        "Key-point TSP": keypoint_tsp_route(farm),
        "IPP baseline": ipp_route(layers),
        "Proposed method": proposed_route(layers),
    }
    return {name: avoid_obstacles(route, farm) for name, route in raw_routes.items()}


def save_routes(routes: Dict[str, List[Point]]) -> None:
    serializable = {k: [[float(x), float(y)] for x, y in v] for k, v in routes.items()}
    (config.DATA_DIR / "routes.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_routes(farm: Dict, layers: Dict[str, np.ndarray], routes: Dict[str, List[Point]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=160)
    axes_flat = axes.flat
    for ax, (name, route) in zip(axes_flat, routes.items()):
        ax.imshow(layers["R"], origin="lower", extent=[0, config.MAP_WIDTH_M, 0, config.MAP_HEIGHT_M], cmap="magma", alpha=0.55, vmin=0, vmax=1)
        for pond in farm["ponds"]:
            poly = np.array(pond["polygon"])
            ax.plot(*np.r_[poly, poly[:1]].T, color="#2e7d32", linewidth=0.5, alpha=0.7)
        for obs in farm.get("obstacles", []):
            x1, y1, x2, y2 = expanded_rect(obs, SAFETY_MARGIN_M)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="#424242", edgecolor="white", alpha=0.35, linewidth=0.5))
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, color="#00e5ff", linewidth=1.0, alpha=0.9)
        ax.scatter(xs[0], ys[0], c="white", marker="X", s=40, edgecolors="black")
        ax.set_title(f"{name}\nD={route_distance(route):.0f} m, N={len(route)}", fontsize=9)
        ax.set_xlim(0, config.MAP_WIDTH_M)
        ax.set_ylim(0, config.MAP_HEIGHT_M)
        ax.set_aspect("equal")
    axes_flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig4_route_comparison.png")
    plt.close(fig)


def write_checkpoint(routes: Dict[str, List[Point]], layers: Dict[str, np.ndarray], farm: Dict) -> None:
    lines = ["# Checkpoint 03 - Planner implementation", "", "Status: completed", "", "Generated routes:"]
    for name, route in routes.items():
        m = evaluate_route(route, layers)
        collisions = count_obstacle_collisions(route, farm)
        lines.append(f"- {name}: waypoints={len(route)}, distance={m['total_distance_m']:.2f} m, high-risk coverage={m['high_risk_coverage_rate']:.4f}, detection={m['abnormality_detection_rate']:.4f}, obstacle collisions={collisions}")
    lines.extend(["", f"Obstacle/no-fly safety margin: {SAFETY_MARGIN_M} m", "", "Outputs:", "- data/routes.json", "- results/figures/fig4_route_comparison.png"])
    (config.CHECKPOINT_DIR / "03_planner_implementation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    config.log("[Step3] Generating routes for all planners...")
    farm, layers = load_inputs()
    routes = generate_routes(farm, layers)
    save_routes(routes)
    plot_routes(farm, layers, routes)
    write_checkpoint(routes, layers, farm)
    for name, route in routes.items():
        config.log(f"[Step3] {name}: waypoints={len(route)}, distance={route_distance(route):.2f} m, collisions={count_obstacle_collisions(route, farm)}")
    config.log(f"[Step3] Saved routes: {config.DATA_DIR / 'routes.json'}")
    config.log(f"[Step3] Saved route figure: {config.FIG_DIR / 'fig4_route_comparison.png'}")


if __name__ == "__main__":
    main()
