from xgboost import XGBRegressor


def build_model(params):
    return XGBRegressor(
        tree_method="hist",
        device="cuda",
        objective="reg:absoluteerror",
        eval_metric="mae",
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
