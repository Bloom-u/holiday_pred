import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.gbdt.config import (
    CNY_DATES,
    DATA_PATH,
    DYNAMIC_COLS,
    FORECAST_YEAR,
    MODEL_PATH,
    OUTPUT_PRED_PATH,
    STATS_PATH,
)
from src.gbdt.data import load_data
from src.gbdt.features import apply_group_stats, build_base_features
from src.gbdt.metrics import compute_metrics
from src.gbdt.persistence import load_group_stats


def load_model(path):
    model = XGBRegressor()
    model.load_model(path)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="GBDT inference for holiday traffic.")
    parser.add_argument("--input", default=str(DATA_PATH), help="Input Excel path.")
    parser.add_argument("--output", default=str(OUTPUT_PRED_PATH), help="Output CSV path.")
    parser.add_argument("--year", type=int, default=FORECAST_YEAR, help="Forecast year.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Model JSON path.")
    parser.add_argument("--stats", default=str(STATS_PATH), help="Group stats JSON path.")
    parser.add_argument("--delay", type=int, default=2, help="Label availability delay for t-minus-k forecasting.")
    parser.add_argument(
        "--mode",
        choices=["tminus2", "recursive", "both"],
        default="both",
        help="tminus2: uses history up to t-delay; recursive: uses predicted lags; both outputs both.",
    )
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    stats_path = Path(args.stats)
    forecast_year = args.year

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    raw = load_data(input_path)
    base = build_base_features(raw, CNY_DATES)
    stats = load_group_stats(stats_path)
    base = apply_group_stats(base, stats)

    y = raw["y"]
    test_mask = raw["date"].dt.year == forecast_year

    model = load_model(model_path)
    feature_cols = base.columns.tolist() + DYNAMIC_COLS

    out = raw.loc[test_mask, ["date", "y"]].copy()

    if args.mode in ("tminus2", "both"):
        from src.gbdt.forecasting import predict_tminus2_series

        pred_tminus2 = predict_tminus2_series(
            model,
            base,
            y,
            feature_cols,
            DYNAMIC_COLS,
            test_mask,
            delay=args.delay,
        )
        out["pred_tminus2"] = pred_tminus2.loc[test_mask].values

    if args.mode in ("recursive", "both"):
        from src.gbdt.forecasting import predict_recursive_series

        pred_rec = predict_recursive_series(
            model,
            base,
            y,
            feature_cols,
            DYNAMIC_COLS,
            test_mask,
        )
        out["pred_recursive"] = pred_rec.loc[test_mask].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    cny = CNY_DATES[forecast_year]
    window_start = cny - pd.Timedelta(days=25)
    window_end = cny + pd.Timedelta(days=15)
    window = out[(out["date"] >= window_start) & (out["date"] <= window_end)]

    print("Saved:", output_path)
    for col in ["pred_tminus2", "pred_recursive"]:
        if col not in out.columns:
            continue
        overall = compute_metrics(out["y"], out[col])
        spring = compute_metrics(window["y"], window[col])
        print(f"{forecast_year} {col} metrics:")
        print(f"  MAE={overall.mae:.2f} RMSE={overall.rmse:.2f} MAPE={overall.mape:.2f}%")
        print(f"{forecast_year} {col} spring window metrics:")
        print(f"  MAE={spring.mae:.2f} RMSE={spring.rmse:.2f} MAPE={spring.mape:.2f}%")


if __name__ == "__main__":
    main()
