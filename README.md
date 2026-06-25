# Multisource UAV Inspection Path Planning for Rice-Crayfish Co-Culture Areas

This repository contains a simulation experiment project for **multisource perception-driven adaptive UAV inspection path planning** in rice-crayfish co-culture areas.

The project builds a connected rice-crayfish farm simulation, generates multisource risk layers, compares several UAV route-planning strategies, and exports reproducible tables and figures for manuscript preparation.

> Scope note: the current evidence is **simulation-based**. The repository does not include real UAV imagery or private field data.

## Highlights

- Connected rice-crayfish farm simulation with production units, canals, facilities, sensors and no-fly zones.
- Multisource risk map integrating:
  - water-colour abnormality (`C_i`)
  - aquatic-vegetation abnormality (`V_i`)
  - thermal anomaly (`T_i`)
  - water-quality sensor abnormality (`S_i`)
  - structural proximity risk (`K_i`)
- Five route strategies:
  - Fixed route
  - Standard coverage path planning (CPP)
  - Key-point TSP
  - IPP baseline
  - Proposed adaptive risk-aware method
- Obstacle/no-fly-zone repair using expanded rectangles and A* detours.
- Main comparison, ablation, parameter sensitivity and robustness experiments.

## Repository structure

```text
.
├── src/                         # Simulation, risk model, planners and experiments
├── scripts/
│   └── run_all.py               # Run the complete experiment workflow
├── data/                        # Generated map, risk layers and route artifacts
├── results/
│   ├── figures/                 # Generated figures used in the manuscript
│   └── tables/                  # Generated CSV result tables
├── reports/                     # Chinese experiment reports
├── checkpoints/                 # Step-by-step experiment checkpoints
├── docs/                        # Extra documentation placeholder
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Required packages are intentionally minimal: `numpy`, `pandas` and `matplotlib`.

## Quick start

Run the complete workflow from the repository root:

```bash
python scripts/run_all.py
```

The workflow executes the following scripts in order:

1. `src/simulation.py` — generate the simulated rice-crayfish farm map.
2. `src/risk_model.py` — generate multisource risk layers.
3. `src/planners.py` — generate and repair routes.
4. `src/main_experiment.py` — main method comparison.
5. `src/ablation_experiment.py` — ablation study.
6. `src/threshold_sensitivity.py` — high-risk threshold sensitivity.
7. `src/grid_sensitivity.py` — grid-resolution sensitivity.
8. `src/budget_sensitivity.py` — additional-distance budget sensitivity.
9. `src/noise_robustness.py` — perception-noise robustness.
10. `src/sensor_missing.py` — water-quality sensor missingness.
11. `src/battery_constraint.py` — battery/route-distance constraints.
12. `src/multi_seed_experiment.py` — multi-scenario robustness.
13. `src/weight_sensitivity.py` — risk-fusion weight sensitivity.
14. `src/abnormality_density.py` — abnormality-density sensitivity.

## Main formulation

The integrated risk score is defined as:

```text
R_i = 0.24 C_i + 0.20 V_i + 0.16 T_i + 0.20 S_i + 0.20 K_i
```

The default high-risk threshold is:

```text
tau = 0.7
```

The default grid size is:

```text
20 m
```

Obstacle and no-fly zones are expanded by a safety margin before route feasibility checking. In the released result tables, all reported routes have zero obstacle collisions.

## Key outputs

Representative result tables:

- `results/tables/table1_simulation_settings.csv`
- `results/tables/table2_main_comparison.csv`
- `results/tables/table3_ablation.csv`
- `results/tables/table4_threshold_sensitivity.csv`
- `results/tables/table5_grid_sensitivity.csv`
- `results/tables/table10_multi_seed_summary.csv`
- `results/tables/table11_weight_sensitivity.csv`
- `results/tables/table12_abnormality_density.csv`

Representative figures:

- `results/figures/fig1_simulated_farm_map.png`
- `results/figures/fig2_risk_layers.png`
- `results/figures/fig3_integrated_risk_map.png`
- `results/figures/fig4_route_comparison.png`
- `results/figures/fig5_main_metrics_bar.png`
- `results/figures/fig13_multi_seed_boxplot.png`
- `results/figures/fig14_weight_sensitivity.png`
- `results/figures/fig15_abnormality_density.png`

## Latest main comparison

| Method | Distance (m) | High-risk coverage | Detection rate | Missed-risk rate |
|---|---:|---:|---:|---:|
| Fixed route | 6686.42 | 0.6726 | 0.5644 | 0.4356 |
| Standard CPP | 12731.26 | 0.8053 | 0.7343 | 0.2657 |
| Key-point TSP | 5542.60 | 0.6195 | 0.4637 | 0.5363 |
| IPP baseline | 5027.34 | 0.9204 | 0.4967 | 0.5033 |
| Proposed method | 12811.70 | 0.9381 | 0.7937 | 0.2063 |

Multi-scenario robustness summary:

- Proposed high-risk coverage: `0.9717 ± 0.0194`
- Proposed detection rate: `0.8235 ± 0.0386`
- Standard CPP high-risk coverage: `0.7922 ± 0.0801`
- Standard CPP detection rate: `0.7554 ± 0.0372`

## Reproducibility notes

- Random seed defaults are defined in `src/config.py`.
- Generated artifacts are written to `data/`, `results/`, `reports/` and `checkpoints/`.
- The included `data/` and `results/` directories contain one completed simulation run for quick inspection.
- Re-running `python scripts/run_all.py` will overwrite generated outputs with regenerated results.

## Citation

If you use this repository in an academic manuscript, please cite the corresponding paper or project once available.

## License

This project is released under the MIT License. See `LICENSE` for details.
