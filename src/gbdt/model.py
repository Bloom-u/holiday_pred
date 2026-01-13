import os

import xgboost as xgb
from xgboost import XGBRegressor


def _resolve_device() -> str:
    override = os.environ.get("XGB_DEVICE")
    if override:
        return override
    try:
        info = xgb.build_info()
        if bool(info.get("USE_CUDA")):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_model(params):
    device = _resolve_device()
    return XGBRegressor(
        tree_method="hist",
        device=device,
        objective=params.get("objective", "reg:absoluteerror"),
        eval_metric=params.get("eval_metric", "mae"),
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        max_bin=params["max_bin"],
        random_state=42,
    )
