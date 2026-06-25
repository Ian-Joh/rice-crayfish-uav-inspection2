# Checkpoint 02 - Risk map generation

Status: completed

Risk model:
- R_i = 0.24 C_i + 0.2 V_i + 0.16 T_i + 0.2 S_i + 0.2 K_i
- Low risk: R_i < 0.4
- Medium risk: 0.4 <= R_i < 0.7
- High risk: R_i >= 0.7

Grid:
- Rows: 50
- Cols: 50
- Grid size: 20 m

Generated risk evidence:
- Abnormal events: 44
- Ground-truth abnormal cells: 606
- High-risk cells: 113

Outputs:
- data/risk_layers.npz
- data/risk_metadata.json
- results/figures/fig2_risk_layers.png
- results/figures/fig3_integrated_risk_map.png
