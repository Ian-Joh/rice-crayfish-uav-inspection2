# -*- coding: utf-8 -*-
"""Experiment configuration for rice-crayfish UAV inspection simulation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as _plt

# Publication-grade figure defaults for ICACR 2026 submission.
# All figures must be exported at >= 300 dpi; here we enforce 400 dpi globally
# so that every fig.savefig(...) in the project produces high-resolution output.
FIGURE_EXPORT_DPI = 400
_plt.rcParams.update({
    "savefig.dpi": FIGURE_EXPORT_DPI,
    "figure.dpi": FIGURE_EXPORT_DPI,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
CHECKPOINT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports"
RUN_LOG = REPORT_DIR / "run.log"

RANDOM_SEED = 20260612

# Base simulation setting, used in Step 1.
MAP_WIDTH_M = 1000
MAP_HEIGHT_M = 1000
GRID_SIZE_M = 20
N_PONDS = 18
N_INLETS = 18
N_FEEDING_POINTS = 18
N_AERATORS = 12
N_WATER_SENSORS = 10
N_OBSTACLES = 6

UAV_START = (50.0, 50.0)


def ensure_dirs() -> None:
    for d in [DATA_DIR, FIG_DIR, TABLE_DIR, CHECKPOINT_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    ensure_dirs()
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")
    print(message)
