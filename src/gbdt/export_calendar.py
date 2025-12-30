import contextlib
import io
from pathlib import Path

import pandas as pd

from src.gbdt.config import CNY_DATES, MODEL_DIR
from src.gbdt.features import add_cny_features
from src.holiday_feature import HolidayFeatureEngine


OUTPUT_PATH = MODEL_DIR / "holiday_calendar.csv"


def build_calendar(start="2023-01-01", end="2025-12-31"):
    dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({"date": dates})
    engine = HolidayFeatureEngine()
    with contextlib.redirect_stdout(io.StringIO()):
        base = engine.create_features(df, date_column="date")
    base = add_cny_features(base, CNY_DATES, date_col="date")
    drop_cols = ["holiday_name", "next_holiday_name", "prev_holiday_name"]
    base = base.drop(columns=drop_cols)
    base["year_normalized"] = (
        (base["year"] - base["year"].min())
        / max(1, (base["year"].max() - base["year"].min()))
    )
    return base


def main():
    cal = build_calendar()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
