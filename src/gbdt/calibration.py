import pandas as pd


def fit_tminus2_offsets(base: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    df = pd.DataFrame({"y_true": y_true.astype(float), "y_pred": y_pred.astype(float)}, index=y_true.index)
    df = df.dropna()
    if df.empty:
        return {"overall": 0.0}

    target = df["y_true"] - df["y_pred"]
    overall = float(target.mean())

    def offset(mask: pd.Series) -> float:
        values = target.loc[mask.reindex(target.index, fill_value=False)]
        if values.empty:
            return overall
        return float(values.mean())

    holiday_type = base.get("holiday_type")
    is_weekend = base.get("is_weekend")
    is_holiday = base.get("is_holiday")
    is_adjusted_workday = base.get("is_adjusted_workday")
    cny_window = base.get("cny_window")

    if holiday_type is None or is_weekend is None or is_holiday is None or is_adjusted_workday is None:
        return {"overall": overall}

    normal_workday = (
        (holiday_type == 0)
        & (is_weekend == 0)
        & (is_adjusted_workday == 0)
        & (is_holiday == 0)
    )
    statutory = is_holiday == 1
    in_cny_window = cny_window == 1 if cny_window is not None else pd.Series(False, index=base.index)

    return {
        "overall": overall,
        "normal_workday_cny_window": offset(normal_workday & in_cny_window),
        "normal_workday_non_cny": offset(normal_workday & ~in_cny_window),
        "statutory_holiday": offset(statutory),
    }


def apply_tminus2_offsets(
    base: pd.DataFrame, y_pred: pd.Series, offsets: dict[str, float]
) -> pd.Series:
    out = y_pred.astype(float).copy()
    overall = float(offsets.get("overall", 0.0))

    holiday_type = base.get("holiday_type")
    is_weekend = base.get("is_weekend")
    is_holiday = base.get("is_holiday")
    is_adjusted_workday = base.get("is_adjusted_workday")
    cny_window = base.get("cny_window")

    if holiday_type is None or is_weekend is None or is_holiday is None or is_adjusted_workday is None:
        return out + overall

    normal_workday = (
        (holiday_type == 0)
        & (is_weekend == 0)
        & (is_adjusted_workday == 0)
        & (is_holiday == 0)
    )
    statutory = is_holiday == 1
    in_cny_window = cny_window == 1 if cny_window is not None else pd.Series(False, index=base.index)

    out += overall
    out.loc[normal_workday & in_cny_window] += float(offsets.get("normal_workday_cny_window", 0.0)) - overall
    out.loc[normal_workday & ~in_cny_window] += float(offsets.get("normal_workday_non_cny", 0.0)) - overall
    out.loc[statutory] += float(offsets.get("statutory_holiday", 0.0)) - overall
    return out
