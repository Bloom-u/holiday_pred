import numpy as np
import pandas as pd

from src.gbdt.predict import dynamic_row, predict_one


def _predict_one_day(model, base_row, dyn_features, feature_cols):
    row = pd.concat([base_row, pd.Series(dyn_features)])
    row_df = row[feature_cols].to_frame().T.astype(np.float32)
    return float(predict_one(model, row_df))


def predict_recursive_series(model, base, y, feature_cols, dynamic_cols, target_mask):
    pred = y.astype(float).copy()
    base_cols = [c for c in feature_cols if c not in dynamic_cols]
    for idx in np.where(target_mask)[0]:
        dyn = dynamic_row(pred, idx)
        base_row = base.loc[idx, base_cols]
        pred.iloc[idx] = _predict_one_day(model, base_row, dyn, feature_cols)
    return pred


def predict_tminus2_series(model, base, y, feature_cols, dynamic_cols, target_mask, delay=2):
    """
    Rolling one-step forecast for each target day t:
      - dynamic features use real history up to t-delay
      - predict y(t) directly
    This avoids using any information newer than t-delay.
    """
    pred = pd.Series(np.nan, index=y.index, dtype=float)
    y_real = y.astype(float)
    base_cols = [c for c in feature_cols if c not in dynamic_cols]

    target_indices = np.where(target_mask)[0]
    for idx in target_indices:
        hist_end = idx - delay
        if hist_end < 30:
            continue

        def safe_get(offset):
            j = hist_end - offset
            if j < 0:
                return np.nan
            return float(y_real.iloc[j])

        lag_1 = safe_get(0)
        lag_2 = safe_get(1)
        lag_3 = safe_get(2)
        lag_4 = safe_get(3)
        lag_7 = safe_get(6)
        lag_14 = safe_get(13)
        lag_21 = safe_get(20)
        lag_28 = safe_get(27)

        def window(start_offset, length):
            end = hist_end - start_offset
            start = end - (length - 1)
            if start < 0:
                return None
            return y_real.iloc[start : end + 1].astype(float)

        w7 = window(0, 7)
        w14 = window(0, 14)
        w30 = window(0, 30)
        w7_prev = window(7, 7)
        if w7 is None or w14 is None or w30 is None or w7_prev is None:
            continue

        def slope(arr: pd.Series) -> float:
            values = arr.to_numpy(dtype=float)
            if values.size <= 1 or np.any(np.isnan(values)):
                return np.nan
            n = int(values.size)
            x = np.arange(n, dtype=float)
            x_mean = (n - 1) / 2.0
            y_mean = float(values.mean())
            denom = float(((x - x_mean) ** 2).sum())
            if denom == 0.0:
                return 0.0
            num = float(((x - x_mean) * (values - y_mean)).sum())
            return num / denom

        roll_7 = float(w7.mean())
        roll_14 = float(w14.mean())
        roll_30 = float(w30.mean())
        std_7 = float(w7.std())
        std_14 = float(w14.std())
        trend_7 = float(roll_7 - w7_prev.mean())
        slope_7 = float(slope(w7))
        slope_14 = float(slope(w14))
        accel_7 = float(slope_7 - slope(w7_prev))
        recent_change_3d = float((lag_1 - lag_4) / 3.0)
        delta_vs_roll7 = float(lag_1 - roll_7)
        delta_vs_lag7 = float(lag_1 - lag_7)

        dyn = {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_21": lag_21,
            "lag_28": lag_28,
            "roll_7": roll_7,
            "roll_14": roll_14,
            "roll_30": roll_30,
            "std_7": std_7,
            "std_14": std_14,
            "trend_7": trend_7,
            "slope_7": slope_7,
            "slope_14": slope_14,
            "accel_7": accel_7,
            "recent_change_3d": recent_change_3d,
            "delta_vs_roll7": delta_vs_roll7,
            "delta_vs_lag7": delta_vs_lag7,
        }

        base_row = base.loc[idx, base_cols]
        pred.iloc[idx] = _predict_one_day(model, base_row, dyn, feature_cols)

    return pred

