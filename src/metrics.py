# -*- coding: utf-8 -*-
"""Shared route metrics for UAV inspection experiments."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

import config

Point = Tuple[float, float]


def route_distance(route: List[Point]) -> float:
    if len(route) < 2:
        return 0.0
    return float(sum(math.dist(route[i], route[i + 1]) for i in range(len(route) - 1)))


def turning_number(route: List[Point], angle_threshold_deg: float = 25.0) -> int:
    if len(route) < 3:
        return 0
    cnt = 0
    for i in range(1, len(route) - 1):
        ax, ay = route[i - 1]
        bx, by = route[i]
        cx, cy = route[i + 1]
        v1 = np.array([bx - ax, by - ay], dtype=float)
        v2 = np.array([cx - bx, cy - by], dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
        angle = math.degrees(math.acos(cosang))
        if angle >= angle_threshold_deg:
            cnt += 1
    return cnt


def route_to_coverage_mask(route: List[Point], X: np.ndarray, Y: np.ndarray, footprint_radius: float = 35.0) -> np.ndarray:
    covered = np.zeros_like(X, dtype=bool)
    if not route:
        return covered
    # Cover cells near every waypoint and sampled segment point.
    samples: List[Point] = []
    for a, b in zip(route[:-1], route[1:]):
        dist = math.dist(a, b)
        n = max(1, int(dist / (footprint_radius * 0.7)))
        for j in range(n + 1):
            t = j / n
            samples.append((a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t))
    if len(route) == 1:
        samples = route[:]
    for sx, sy in samples:
        covered |= ((X - sx) ** 2 + (Y - sy) ** 2) <= footprint_radius ** 2
    return covered


def evaluate_route(route: List[Point], layers: Dict[str, np.ndarray], speed_mps: float = 6.0) -> Dict[str, float]:
    X, Y = layers["X"], layers["Y"]
    truth = layers["truth"].astype(bool)
    high = layers["risk_class"] == 2
    covered = route_to_coverage_mask(route, X, Y)
    distance = route_distance(route)
    turns = turning_number(route)
    total_cells = covered.size
    truth_cells = int(truth.sum())
    high_cells = int(high.sum())
    detected_truth = int((covered & truth).sum())
    high_covered = int((covered & high).sum())
    coverage_rate = float(covered.sum() / total_cells)
    high_risk_coverage = float(high_covered / high_cells) if high_cells else 0.0
    detection_rate = float(detected_truth / truth_cells) if truth_cells else 0.0
    missed_rate = 1.0 - detection_rate
    flight_time = distance / speed_mps if speed_mps > 0 else 0.0
    energy = distance * 1.0 + turns * 4.0
    detections_per_km = detected_truth / (distance / 1000.0) if distance > 0 else 0.0
    return {
        "total_distance_m": distance,
        "flight_time_s": flight_time,
        "turning_number": float(turns),
        "energy_proxy": energy,
        "overall_coverage_rate": coverage_rate,
        "high_risk_coverage_rate": high_risk_coverage,
        "abnormality_detection_rate": detection_rate,
        "missed_risk_rate": missed_rate,
        "detected_truth_cells": float(detected_truth),
        "detections_per_km": detections_per_km,
        "waypoints": float(len(route)),
    }
