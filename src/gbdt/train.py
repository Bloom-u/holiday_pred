import random
import sys
from itertools import product

import numpy as np
import pandas as pd

from src.gbdt.config import (
    BEST_PARAMS,
    CNY_DATES,
    DATA_PATH,
    DYNAMIC_COLS,
    MODEL_PATH,
    OUTPUT_PRED_PATH,
    RANDOM_SEARCH_ITERS,
    STATS_PATH,
    WEIGHT_GRID,
)
from src.gbdt.data import load_data
from src.gbdt.features import (
    add_group_stats,
    apply_group_stats,
    build_base_features,
    build_feature_matrix,
    compute_group_stats,
    make_sample_weight,
)
from src.gbdt.metrics import compute_metrics
from src.gbdt.model import build_model
from src.gbdt.predict import recursive_predict
from src.gbdt.persistence import save_group_stats


def evaluate_model(model, base, y, test_mask):
    feature_cols = base.columns.tolist() + DYNAMIC_COLS
    pred = recursive_predict(model, base, y, test_mask, feature_cols, DYNAMIC_COLS)
    overall = compute_metrics(y.loc[test_mask], pred.loc[test_mask])

    spring_mask = test_mask & base["days_to_cny"].between(-25, 15)
    spring = compute_metrics(y.loc[spring_mask], pred.loc[spring_mask])
    return pred, overall, spring


def build_param_grid():
    return list(
        product(
            [4, 5, 6, 7],
            [400, 800],
            [0.03, 0.05],
            [0.8, 0.9],
            [0.8, 0.9],
            [1, 5, 10],
            [0.5, 1.0, 2.0],
            [0.0, 0.3],
            [0.0, 0.1],
            [256, 512],
        )
    )


def select_best_model(raw, base):
    y = raw["y"]
    train_mask_2023 = raw["date"].dt.year == 2023
    val_mask_2024 = raw["date"].dt.year == 2024

    base_2023 = add_group_stats(base, y, train_mask_2023)
    X_full_2023 = build_feature_matrix(base_2023, y, DYNAMIC_COLS)
    train_2023 = X_full_2023.loc[train_mask_2023].dropna()
    X_train_2023 = train_2023.astype(np.float32)
    y_train_2023 = y.loc[train_2023.index].astype(float)

    param_grid = build_param_grid()
    random.seed(42)
    if len(param_grid) > RANDOM_SEARCH_ITERS:
        param_grid = random.sample(param_grid, RANDOM_SEARCH_ITERS)

    best_score = None
    best_params = None
    best_weights = None

    for (
        max_depth,
        n_estimators,
        learning_rate,
        subsample,
        colsample_bytree,
        min_child_weight,
        reg_lambda,
        gamma,
        reg_alpha,
        max_bin,
    ) in param_grid:
        params = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "max_bin": max_bin,
        }
        for w_window, w_core in WEIGHT_GRID:
            weights = make_sample_weight(base_2023, w_window, w_core)
            weights = pd.Series(weights, index=base_2023.index).loc[train_2023.index]

            model = build_model(params)
            model.fit(
                X_train_2023,
                y_train_2023,
                sample_weight=weights,
                verbose=False,
            )

            _, overall, spring = evaluate_model(model, base_2023, y, val_mask_2024)
            score = 0.7 * spring.mae + 0.3 * overall.mae

            if best_score is None or score < best_score:
                best_score = score
                best_params = params
                best_weights = (w_window, w_core)

    return best_params, best_weights, best_score


def train_and_predict():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    raw = load_data(DATA_PATH)
    base = build_base_features(raw, CNY_DATES)
    y = raw["y"]

    best_params, best_weights, best_score = select_best_model(raw, base)
    print("Best params on 2024 validation:")
    print(f"  params={best_params} weights={best_weights} score={best_score:.2f}")

    train_mask_final = raw["date"].dt.year.isin([2023, 2024])
    test_mask_2025 = raw["date"].dt.year == 2025

    group_stats = compute_group_stats(base, y, train_mask_final)
    save_group_stats(STATS_PATH, group_stats)
    base_final = apply_group_stats(base, group_stats)
    X_full_final = build_feature_matrix(base_final, y, DYNAMIC_COLS)
    train_final = X_full_final.loc[train_mask_final].dropna()
    X_train_final = train_final.astype(np.float32)
    y_train_final = y.loc[train_final.index].astype(float)

    w_window, w_core = best_weights
    weights_final = make_sample_weight(base_final, w_window, w_core)
    weights_final = pd.Series(weights_final, index=base_final.index).loc[train_final.index]

    final_model = build_model(best_params)
    final_model.fit(
        X_train_final,
        y_train_final,
        sample_weight=weights_final,
        verbose=False,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_PATH)

    pred_all, overall_2025, spring_2025 = evaluate_model(
        final_model, base_final, y, test_mask_2025
    )

    out = raw.loc[test_mask_2025, ["date", "y"]].copy()
    out["pred_optimized"] = pred_all.loc[test_mask_2025].values
    OUTPUT_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PRED_PATH, index=False)

    print("Saved:", OUTPUT_PRED_PATH)
    print("Saved:", MODEL_PATH)
    print("Saved:", STATS_PATH)
    print("2025 Overall metrics:")
    print(f"  MAE={overall_2025.mae:.2f} RMSE={overall_2025.rmse:.2f} MAPE={overall_2025.mape:.2f}%")
    print("2025 Spring window metrics:")
    print(f"  MAE={spring_2025.mae:.2f} RMSE={spring_2025.rmse:.2f} MAPE={spring_2025.mape:.2f}%")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    train_and_predict()


if __name__ == "__main__":
    main()
