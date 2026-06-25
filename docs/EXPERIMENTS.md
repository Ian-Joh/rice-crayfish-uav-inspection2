# Experiment Guide

This document summarizes the experimental modules in this repository.

## 1. Simulation generation

Script: `src/simulation.py`

Outputs:

- `data/simulated_farm_map.json`
- `results/figures/fig1_simulated_farm_map.png`
- `checkpoints/01_map_generation.md`

Purpose: construct a connected rice-crayfish co-culture farm with production units, canals, functional facilities, sensors and restricted regions.

## 2. Risk-map generation

Script: `src/risk_model.py`

Outputs:

- `data/risk_layers.npz`
- `data/risk_metadata.json`
- `results/figures/fig2_risk_layers.png`
- `results/figures/fig3_integrated_risk_map.png`
- `checkpoints/02_risk_map_generation.md`

Purpose: generate water-colour, vegetation, thermal, water-quality sensor and structural risk layers, then fuse them into an integrated risk map.

## 3. Planner implementation

Script: `src/planners.py`

Outputs:

- `data/routes.json`
- `results/figures/fig4_route_comparison.png`
- `checkpoints/03_planner_implementation.md`

Purpose: generate routes for all compared methods and repair route segments that intersect obstacle/no-fly zones.

## 4. Main comparison

Script: `src/main_experiment.py`

Outputs:

- `results/tables/table1_simulation_settings.csv`
- `results/tables/table2_main_comparison.csv`
- `results/figures/fig5_main_metrics_bar.png`

Purpose: compare route efficiency and inspection effectiveness across all methods.

## 5. Ablation study

Script: `src/ablation_experiment.py`

Outputs:

- `results/tables/table3_ablation.csv`
- `results/figures/fig6_ablation_results.png`

Purpose: test the contribution of structural risk, water-quality sensor risk and adaptive insertion.

## 6. Sensitivity and robustness experiments

Scripts:

- `src/threshold_sensitivity.py`
- `src/grid_sensitivity.py`
- `src/budget_sensitivity.py`
- `src/noise_robustness.py`
- `src/sensor_missing.py`
- `src/battery_constraint.py`

Purpose: evaluate the method under different high-risk thresholds, grid resolutions, additional route budgets, perception noise levels, sensor availability and battery constraints.

## 7. Supplementary robustness experiments

Scripts:

- `src/multi_seed_experiment.py`
- `src/weight_sensitivity.py`
- `src/abnormality_density.py`

Purpose: evaluate multi-scenario robustness, risk-fusion weight sensitivity and abnormality-density sensitivity.

## Evaluation metrics

The main metrics include:

- total route distance
- estimated flight time
- turning number
- energy proxy
- overall coverage rate
- high-risk coverage rate
- abnormality detection rate
- missed-risk rate
- detections per kilometre
- obstacle collision count where applicable

## Important interpretation boundary

All evidence is simulation-based. Results support the computational feasibility of the planning framework, but field deployment requires validation using real UAV imagery, real water-quality sensors and expert-labelled abnormality regions.
