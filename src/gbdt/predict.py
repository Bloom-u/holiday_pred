import numpy as np
import pandas as pd


def dynamic_row(pred_series, idx):
    lag_1 = pred_series.iloc[idx - 1]
    lag_2 = pred_series.iloc[idx - 2]
    lag_3 = pred_series.iloc[idx - 3]
    lag_4 = pred_series.iloc[idx - 4]
    lag_7 = pred_series.iloc[idx - 7]
    lag_14 = pred_series.iloc[idx - 14]
    lag_21 = pred_series.iloc[idx - 21]
    lag_28 = pred_series.iloc[idx - 28]

    roll_7 = pred_series.iloc[idx - 7:idx].mean()
    roll_14 = pred_series.iloc[idx - 14:idx].mean()
    roll_30 = pred_series.iloc[idx - 30:idx].mean()
    std_7 = pred_series.iloc[idx - 7:idx].std()
    std_14 = pred_series.iloc[idx - 14:idx].std()
    trend_7 = roll_7 - pred_series.iloc[idx - 14:idx - 7].mean()
    recent_change_3d = (lag_1 - lag_4) / 3.0
    delta_vs_roll7 = lag_1 - roll_7
    delta_vs_lag7 = lag_1 - lag_7

    return {
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
        "recent_change_3d": recent_change_3d,
        "delta_vs_roll7": delta_vs_roll7,
        "delta_vs_lag7": delta_vs_lag7,
    }


def predict_one(model, row_df):
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return model.predict(row_df, iteration_range=(0, model.best_iteration + 1))[0]
    return model.predict(row_df)[0]


def recursive_predict(model, base, y, test_mask, feature_cols, dynamic_cols):
    pred = y.astype(float).copy()
    base_cols = [c for c in feature_cols if c not in dynamic_cols]

    for idx in np.where(test_mask)[0]:
        dyn = dynamic_row(pred, idx)
        row = pd.concat([base.loc[idx, base_cols], pd.Series(dyn)])
        row_df = row[feature_cols].to_frame().T.astype(np.float32)
        pred.iloc[idx] = predict_one(model, row_df)
    return pred
