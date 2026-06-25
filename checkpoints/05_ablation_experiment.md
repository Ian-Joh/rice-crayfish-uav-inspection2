# Checkpoint 05 - Ablation experiment

Status: completed

Ablation cases:
- Full model
- w/o K_i: remove structural proximity risk
- w/o S_i: remove water-quality sensor abnormality
- w/o adaptive insertion: use standard CPP route only

Key results:
- Full detection rate: 0.7937
- w/o K_i detection rate: 0.7261
- w/o S_i detection rate: 0.7541
- w/o adaptive insertion detection rate: 0.7343
- Full high-risk coverage: 0.9381
- w/o adaptive insertion high-risk coverage: 0.8053

Outputs:
- results/tables/table3_ablation.csv
- results/figures/fig6_ablation_results.png
