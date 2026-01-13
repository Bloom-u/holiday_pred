import sys

import numpy as np
import pandas as pd

from src.gbdt.config import (
    BEST_PARAMS,
    CNY_DATES,
    DATA_PATH,
    DYNAMIC_COLS,
    MODEL_PATH_TMINUS2_BASELINE,
    MODEL_PATH_TMINUS2_BASELINE_CF,
    MODEL_PATH_TMINUS2_UPLIFT,
    MODEL_PATH_TMINUS2_UPLIFT_CNY,
    MODEL_PATH_TMINUS2_UPLIFT_HOLIDAY,
    OUTPUT_PRED_PATH,
    STATS_PATH,
    WEIGHT_GRID,
)
from src.gbdt.data import load_data
from src.gbdt.features import (
    apply_group_stats,
    build_base_features,
    build_feature_matrix,
    compute_group_stats,
    make_sample_weight,
)
from src.gbdt.forecasting import predict_tminus2_series
from src.gbdt.metrics import compute_metrics
from src.gbdt.model import build_model
from src.gbdt.persistence import save_group_stats


def _uplift_mask(base: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=base.index)
    if "is_holiday" in base.columns:
        mask = mask | (base["is_holiday"] == 1)
    if "cny_window" in base.columns:
        mask = mask | (base["cny_window"] == 1)
    return mask


def _baseline_cols(base: pd.DataFrame) -> list[str]:
    cols = [
        "year_normalized",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "dow_mean",
        "dow_std",
        "month_mean",
        "month_std",
    ]
    return [c for c in cols if c in base.columns]


def _uplift_cols(base: pd.DataFrame) -> list[str]:
    cols = [
        # calendar/time context
        "year_normalized",
        "month",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        # holiday & proximity
        "holiday_type",
        "is_holiday",
        "is_statutory_holiday",
        "is_adjusted_workday",
        "days_to_next_holiday",
        "next_holiday_type",
        "days_from_prev_holiday",
        "prev_holiday_type",
        "days_to_nearest_holiday",
        "holiday_proximity",
        "holiday_phase",
        "holiday_day_num",
        "total_holiday_length",
        "holiday_progress",
        # CNY-specific
        "days_to_cny",
        "cny_window",
        "cny_pre",
        "cny_post",
        "cny_day",
        # group stats
        "holiday_type_mean",
        "holiday_type_std",
        "cny_offset_mean",
    ]
    return [c for c in cols if c in base.columns]


def train_and_predict_decomposed(delay=2):
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    raw = load_data(DATA_PATH)
    base = build_base_features(raw, CNY_DATES)
    y = raw["y"].astype(float)

    train_mask = raw["date"].dt.year.isin([2023, 2024])
    test_mask = raw["date"].dt.year == 2025

    group_stats = compute_group_stats(base, y, train_mask)
    save_group_stats(STATS_PATH, group_stats)
    base = apply_group_stats(base, group_stats)

    X_full = build_feature_matrix(base, y, DYNAMIC_COLS, delay=delay)
    feature_cols_base = _baseline_cols(base) + DYNAMIC_COLS
    feature_cols_uplift = _uplift_cols(base)

    uplift_mask = _uplift_mask(base)
    uplift_train_mask = train_mask & uplift_mask
    cny_mask = base["cny_window"] == 1 if "cny_window" in base.columns else pd.Series(False, index=base.index)
    holiday_mask = (base["is_holiday"] == 1) if "is_holiday" in base.columns else pd.Series(False, index=base.index)
    holiday_non_cny = holiday_mask & ~cny_mask

    base_train = X_full.loc[train_mask, feature_cols_base].dropna()
    uplift_train = X_full.loc[uplift_train_mask, feature_cols_uplift].dropna()

    if base_train.empty:
        raise ValueError("Baseline training set is empty; check uplift mask.")
    if uplift_train.empty:
        raise ValueError("Uplift training set is empty; check uplift mask.")

    # Use the best weight pair discovered in the standard pipeline (keep it simple here).
    # Fallback: (2.0, 4.0) is included in WEIGHT_GRID.
    best_weights = (2.0, 4.0)
    if best_weights not in WEIGHT_GRID:
        best_weights = WEIGHT_GRID[-1]

    w_window, w_core = best_weights
    weights_all = make_sample_weight(base, w_window, w_core)
    weights_all = pd.Series(weights_all, index=base.index)
    weights_base = weights_all.loc[base_train.index]
    weights_uplift = weights_all.loc[uplift_train.index]

    params = dict(BEST_PARAMS)
    params["eval_metric"] = "mae"

    model_base_normal = build_model(params)
    model_base_normal.fit(
        base_train.astype(np.float32),
        y.loc[base_train.index].astype(float),
        sample_weight=weights_base,
        verbose=False,
    )

    base_cf_train = X_full.loc[train_mask & ~uplift_mask, feature_cols_base].dropna()
    if base_cf_train.empty:
        raise ValueError("Counterfactual baseline training set is empty; check uplift mask.")
    model_base_cf = build_model(params)
    model_base_cf.fit(
        base_cf_train.astype(np.float32),
        y.loc[base_cf_train.index].astype(float),
        sample_weight=weights_all.loc[base_cf_train.index],
        verbose=False,
    )

    pred_base_normal_all = predict_tminus2_series(
        model_base_normal,
        base,
        y,
        feature_cols_base,
        DYNAMIC_COLS,
        train_mask | test_mask,
        delay=delay,
    )
    pred_base_cf_all = predict_tminus2_series(
        model_base_cf,
        base,
        y,
        feature_cols_base,
        DYNAMIC_COLS,
        train_mask | test_mask,
        delay=delay,
    )

    base = base.copy()
    base["tminus2_base_pred_normal"] = pred_base_normal_all.astype(float)
    base["tminus2_base_pred_cf"] = pred_base_cf_all.astype(float)
    X_full_with_base = build_feature_matrix(base, y, DYNAMIC_COLS, delay=delay)
    feature_cols_uplift = (
        _uplift_cols(base) + ["tminus2_base_pred_normal", "tminus2_base_pred_cf"] + DYNAMIC_COLS
    )

    residual_all = (y - pred_base_cf_all).astype(float)
    uplift_target = residual_all.loc[uplift_train_mask]
    uplift_target = uplift_target.dropna()
    uplift_train = X_full_with_base.loc[uplift_target.index, feature_cols_uplift].dropna()
    uplift_target = uplift_target.loc[uplift_train.index]
    weights_uplift = weights_all.loc[uplift_train.index]

    cny_train_idx = uplift_train.index.intersection(base.index[cny_mask & train_mask])
    holiday_train_idx = uplift_train.index.intersection(base.index[holiday_non_cny & train_mask])
    if cny_train_idx.empty or holiday_train_idx.empty:
        raise ValueError("Uplift training split empty; check masks for CNY/holiday.")

    uplift_params = dict(params)
    uplift_params["objective"] = "reg:squarederror"
    model_uplift = build_model(uplift_params)
    model_uplift.fit(
        uplift_train.astype(np.float32),
        uplift_target,
        sample_weight=weights_uplift,
        verbose=False,
    )

    model_uplift_cny = build_model(uplift_params)
    model_uplift_cny.fit(
        uplift_train.loc[cny_train_idx].astype(np.float32),
        uplift_target.loc[cny_train_idx],
        sample_weight=weights_uplift.loc[cny_train_idx] * 2.0,
        verbose=False,
    )

    model_uplift_holiday = build_model(uplift_params)
    model_uplift_holiday.fit(
        uplift_train.loc[holiday_train_idx].astype(np.float32),
        uplift_target.loc[holiday_train_idx],
        sample_weight=weights_uplift.loc[holiday_train_idx],
        verbose=False,
    )

    pred_base = pred_base_normal_all.copy()
    pred_base_cf = pred_base_cf_all.copy()
    pred_uplift_cny = predict_tminus2_series(
        model_uplift_cny,
        base,
        y,
        feature_cols_uplift,
        DYNAMIC_COLS,
        test_mask & cny_mask,
        delay=delay,
    )
    pred_uplift_holiday = predict_tminus2_series(
        model_uplift_holiday,
        base,
        y,
        feature_cols_uplift,
        DYNAMIC_COLS,
        test_mask & holiday_non_cny,
        delay=delay,
    )

    pred_total = pred_base.copy()
    pred_total.loc[test_mask] = pred_base.loc[test_mask]
    pred_total.loc[test_mask & uplift_mask] = pred_base_cf.loc[test_mask & uplift_mask]

    # In the CNY window, baseline_cf can be too low (distribution shift). Blend in a fraction of
    # the normal baseline (which is closer to the current level) for days away from CNY day.
    if "days_to_cny" in base.columns:
        diff = (pred_base - pred_base_cf).astype(float)
        pre = test_mask & cny_mask & (base["days_to_cny"] <= -2)
        post = test_mask & cny_mask & (base["days_to_cny"] >= 1)
        pred_total.loc[pre] = pred_total.loc[pre] + 0.50 * diff.loc[pre]
        pred_total.loc[post] = pred_total.loc[post] + 0.35 * diff.loc[post]
    pred_total.loc[test_mask & cny_mask] = pred_total.loc[test_mask & cny_mask] + pred_uplift_cny.loc[
        test_mask & cny_mask
    ].fillna(0.0)
    pred_total.loc[test_mask & holiday_non_cny] = pred_total.loc[test_mask & holiday_non_cny] + pred_uplift_holiday.loc[
        test_mask & holiday_non_cny
    ].fillna(0.0)
    # Core CNY days are extremely sparse in training (often 1 sample per offset),
    # so shrink predictions toward the historical CNY offset mean to reduce overshoot.
    if "days_to_cny" in base.columns and "cny_offset_mean" in base.columns:
        core = test_mask & cny_mask & base["days_to_cny"].between(-1, 0)
        shrink = 0.30  # keep most of the learned signal, but cap extreme deviations near day 0
        pred_total.loc[core] = (
            shrink * pred_total.loc[core].astype(float)
            + (1.0 - shrink) * base.loc[core, "cny_offset_mean"].astype(float)
        )

    MODEL_PATH_TMINUS2_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    model_base_normal.save_model(MODEL_PATH_TMINUS2_BASELINE)
    model_base_cf.save_model(MODEL_PATH_TMINUS2_BASELINE_CF)
    model_uplift.save_model(MODEL_PATH_TMINUS2_UPLIFT)
    model_uplift_cny.save_model(MODEL_PATH_TMINUS2_UPLIFT_CNY)
    model_uplift_holiday.save_model(MODEL_PATH_TMINUS2_UPLIFT_HOLIDAY)

    out = raw.loc[test_mask, ["date", "y"]].copy()
    out["pred_tminus2_baseline"] = pred_base.loc[test_mask].values
    out["pred_tminus2_baseline_cf"] = pred_base_cf.loc[test_mask].values
    out["pred_tminus2_uplift"] = (pred_uplift_cny.fillna(0.0) + pred_uplift_holiday.fillna(0.0)).loc[test_mask].values
    out["pred_tminus2_uplift_cny"] = pred_uplift_cny.loc[test_mask].values
    out["pred_tminus2_uplift_holiday"] = pred_uplift_holiday.loc[test_mask].values
    out["pred_tminus2_decomposed"] = pred_total.loc[test_mask].values
    OUTPUT_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PRED_PATH, index=False)

    cny = CNY_DATES[2025]
    window_start = cny - pd.Timedelta(days=25)
    window_end = cny + pd.Timedelta(days=15)
    window = out[(out["date"] >= window_start) & (out["date"] <= window_end)]

    overall = compute_metrics(out["y"], out["pred_tminus2_decomposed"])
    spring = compute_metrics(window["y"], window["pred_tminus2_decomposed"])
    print("Saved:", OUTPUT_PRED_PATH)
    print("Saved:", MODEL_PATH_TMINUS2_BASELINE)
    print("Saved:", MODEL_PATH_TMINUS2_BASELINE_CF)
    print("Saved:", MODEL_PATH_TMINUS2_UPLIFT)
    print("Saved:", MODEL_PATH_TMINUS2_UPLIFT_CNY)
    print("Saved:", MODEL_PATH_TMINUS2_UPLIFT_HOLIDAY)
    print("Saved:", STATS_PATH)
    print("2025 pred_tminus2_decomposed metrics:")
    print(f"  MAE={overall.mae:.2f} RMSE={overall.rmse:.2f} MAPE={overall.mape:.2f}%")
    print("2025 pred_tminus2_decomposed spring window metrics:")
    print(f"  MAE={spring.mae:.2f} RMSE={spring.rmse:.2f} MAPE={spring.mape:.2f}%")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    train_and_predict_decomposed(delay=2)


if __name__ == "__main__":
    main()
