from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/new_data.xlsx")
OUTPUT_PRED_PATH = Path("data/pred_2025_optimized.csv")
FIG_DIR = Path("figs")
MODEL_DIR = Path("models/gbdt")
MODEL_PATH = MODEL_DIR / "best_model.json"
MODEL_PATH_RECURSIVE = MODEL_DIR / "best_model_recursive.json"
MODEL_PATH_TMINUS2 = MODEL_DIR / "best_model_tminus2.json"
STATS_PATH = MODEL_DIR / "group_stats.json"
CALIB_PATH = MODEL_DIR / "calibration.json"
FORECAST_YEAR = 2025

CNY_DATES = {
    2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
    2025: pd.Timestamp("2025-01-29"),
}

DYNAMIC_COLS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "roll_7",
    "roll_14",
    "roll_30",
    "std_7",
    "std_14",
    "trend_7",
    "slope_7",
    "slope_14",
    "accel_7",
    "recent_change_3d",
    "delta_vs_roll7",
    "delta_vs_lag7",
]

RANDOM_SEARCH_ITERS = 8

BEST_PARAMS = {
    "max_depth": 4,
    "n_estimators": 800,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 0.5,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "max_bin": 256,
}

WEIGHT_GRID = [(1.0, 2.0), (2.0, 4.0)]

# XGBoost objectives (experiment-friendly; keep defaults backward-compatible)
OBJECTIVE_RECURSIVE = "reg:absoluteerror"
OBJECTIVE_TMINUS2 = "reg:squarederror"
EVAL_METRIC = "mae"
