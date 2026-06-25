# -*- coding: utf-8 -*-
"""Run the full simulation workflow.

Execute from the repository root:
    python scripts/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

STEPS = [
    "simulation.py",
    "risk_model.py",
    "planners.py",
    "main_experiment.py",
    "ablation_experiment.py",
    "threshold_sensitivity.py",
    "grid_sensitivity.py",
    "budget_sensitivity.py",
    "noise_robustness.py",
    "sensor_missing.py",
    "battery_constraint.py",
    "multi_seed_experiment.py",
    "weight_sensitivity.py",
    "abnormality_density.py",
]


def run_step(script: str) -> None:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / script)], cwd=str(ROOT), check=True)


def main() -> None:
    for script in STEPS:
        run_step(script)
    print("\nAll experiments completed.")
    print(f"Figures: {ROOT / 'results' / 'figures'}")
    print(f"Tables:  {ROOT / 'results' / 'tables'}")


if __name__ == "__main__":
    main()
