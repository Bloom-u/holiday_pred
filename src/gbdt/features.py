import contextlib
import io

import numpy as np
import pandas as pd

from src.holiday_feature import HolidayFeatureEngine


def _linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.any(np.isnan(values)):
        return np.nan
    n = int(values.size)
    if n == 1:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y_mean = float(values.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0.0:
        return 0.0
    num = float(((x - x_mean) * (values - y_mean)).sum())
    return num / denom


def add_cny_features(df, cny_dates, date_col="date"):
    dates = pd.to_datetime(df[date_col])
    cny_map = dates.dt.year.map(cny_dates)
    days_to_cny = (dates - cny_map).dt.days
    df = df.copy()
    df["days_to_cny"] = days_to_cny
    df["cny_window"] = ((days_to_cny >= -25) & (days_to_cny <= 15)).astype(int)
    df["cny_pre"] = np.where((days_to_cny < 0) & (days_to_cny >= -25), -days_to_cny, 0)
    df["cny_post"] = np.where((days_to_cny >= 0) & (days_to_cny <= 15), days_to_cny, 0)
    df["cny_day"] = (days_to_cny == 0).astype(int)
    return df


def build_base_features(raw, cny_dates):
    engine = HolidayFeatureEngine()
    with contextlib.redirect_stdout(io.StringIO()):
        base = engine.create_features(raw[["date"]], date_column="date")

    base = add_cny_features(base, cny_dates, date_col="date")
    drop_cols = ["date", "holiday_name", "next_holiday_name", "prev_holiday_name"]
    base = base.drop(columns=drop_cols)
    base["year_normalized"] = (
        (base["year"] - base["year"].min())
        / max(1, (base["year"].max() - base["year"].min()))
    )
    return base


def compute_group_stats(base, y, train_mask):
    train = base.loc[train_mask]
    y_train = y.loc[train_mask]
    overall_mean = float(y_train.mean())
    overall_std = float(y_train.std())

    def group_stat_map(col):
        mean_map = y_train.groupby(train[col]).mean().to_dict()
        std_map = y_train.groupby(train[col]).std().to_dict()
        return mean_map, std_map

    dow_mean, dow_std = group_stat_map("day_of_week")
    month_mean, month_std = group_stat_map("month")
    holiday_mean, holiday_std = group_stat_map("holiday_type")

    cny_train = pd.DataFrame({"days_to_cny": train["days_to_cny"], "y": y_train})
    cny_train = cny_train[cny_train["days_to_cny"].between(-25, 15)]
    cny_offset_mean = cny_train.groupby("days_to_cny")["y"].mean().to_dict()

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "dow_mean": dow_mean,
        "dow_std": dow_std,
        "month_mean": month_mean,
        "month_std": month_std,
        "holiday_type_mean": holiday_mean,
        "holiday_type_std": holiday_std,
        "cny_offset_mean": cny_offset_mean,
    }


def apply_group_stats(base, stats):
    out = base.copy()

    def map_group_stat(col, name):
        out[f"{name}_mean"] = out[col].map(stats[f"{name}_mean"]).fillna(stats["overall_mean"])
        out[f"{name}_std"] = out[col].map(stats[f"{name}_std"]).fillna(stats["overall_std"])

    map_group_stat("day_of_week", "dow")
    map_group_stat("month", "month")
    map_group_stat("holiday_type", "holiday_type")
    out["cny_offset_mean"] = out["days_to_cny"].map(stats["cny_offset_mean"]).fillna(
        stats["overall_mean"]
    )
    return out


def add_group_stats(base, y, train_mask):
    stats = compute_group_stats(base, y, train_mask)
    return apply_group_stats(base, stats)


def build_dynamic_features(series, dynamic_cols):
    df = pd.DataFrame(index=series.index)
    shifted = series.shift(1)
    df["lag_1"] = shifted
    df["lag_2"] = series.shift(2)
    df["lag_3"] = series.shift(3)
    df["lag_7"] = series.shift(7)
    df["lag_14"] = series.shift(14)
    df["lag_21"] = series.shift(21)
    df["lag_28"] = series.shift(28)
    df["roll_7"] = shifted.rolling(7).mean()
    df["roll_14"] = shifted.rolling(14).mean()
    df["roll_30"] = shifted.rolling(30).mean()
    df["std_7"] = shifted.rolling(7).std()
    df["std_14"] = shifted.rolling(14).std()
    df["trend_7"] = df["roll_7"] - df["roll_7"].shift(7)
    df["slope_7"] = shifted.rolling(7).apply(_linear_slope, raw=True)
    df["slope_14"] = shifted.rolling(14).apply(_linear_slope, raw=True)
    df["accel_7"] = df["slope_7"] - df["slope_7"].shift(7)
    df["recent_change_3d"] = (df["lag_1"] - series.shift(4)) / 3.0
    df["delta_vs_roll7"] = df["lag_1"] - df["roll_7"]
    df["delta_vs_lag7"] = df["lag_1"] - df["lag_7"]
    return df[dynamic_cols]


def build_feature_matrix(base, series, dynamic_cols):
    dynamic = build_dynamic_features(series, dynamic_cols)
    return pd.concat([base, dynamic], axis=1)


def make_sample_weight(base, w_window, w_core):
    weights = np.ones(len(base))
    window = base["days_to_cny"].between(-25, 15)
    core = base["days_to_cny"].between(-3, 3)
    weights[window] += w_window
    weights[core] += w_core
    return weights
