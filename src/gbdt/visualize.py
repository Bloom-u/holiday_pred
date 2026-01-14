import contextlib
import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.gbdt.config import (
    CNY_DATES,
    FIG_DIR,
    OUTPUT_PRED_PATH,
)
from src.holiday_feature import HolidayFeatureEngine

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_predictions():
    pred = pd.read_csv(OUTPUT_PRED_PATH, parse_dates=["date"])
    pred = pred.sort_values("date").reset_index(drop=True)
    return pred


def build_holiday_mask(dates):
    engine = HolidayFeatureEngine()
    df = pd.DataFrame({"date": dates})
    with contextlib.redirect_stdout(io.StringIO()):
        feat = engine.create_features(df, date_column="date")
    return feat["holiday_type"] > 1


def plot_full_year(pred):
    cny_2025 = CNY_DATES[2025]
    window_start = cny_2025 - pd.Timedelta(days=25)
    window_end = cny_2025 + pd.Timedelta(days=15)

    has_decomposed = "pred_tminus2_decomposed" in pred.columns

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        pred["date"],
        pred["y"],
        "b-",
        label="actual",
        linewidth=2,
        alpha=0.8,
        marker="o",
        markersize=4,
        markevery=5,
    )
    if "pred_tminus2_decomposed" in pred.columns:
        ax.plot(
            pred["date"],
            pred["pred_tminus2_decomposed"],
            color="#2ca02c",
            linestyle="--",
            label="pred_tminus2_decomposed",
            linewidth=2,
            alpha=0.8,
            marker="D",
            markersize=3,
            markevery=5,
        )
    if "pred_tminus2_baseline_cf" in pred.columns:
        ax.plot(
            pred["date"],
            pred["pred_tminus2_baseline_cf"],
            color="#7f7f7f",
            linestyle=":",
            label="pred_tminus2_baseline_cf",
            linewidth=1.8,
            alpha=0.75,
        )

    holiday_mask = build_holiday_mask(pred["date"])
    holiday_dates = pred.loc[holiday_mask, "date"]
    for d in holiday_dates:
        ax.axvline(d, color="orange", alpha=0.25, linewidth=1.5)

    ax.axvspan(window_start, window_end, color="#d62728", alpha=0.08)
    ax.axvline(cny_2025, color="#d62728", linewidth=1.2, alpha=0.7)

    title = "2025 Forecast - GBDT"
    if has_decomposed:
        title += " (Decomposed t-2)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Flow", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_spring_window(pred):
    cny_2025 = CNY_DATES[2025]
    window_start = cny_2025 - pd.Timedelta(days=25)
    window_end = cny_2025 + pd.Timedelta(days=15)
    window = pred[(pred["date"] >= window_start) & (pred["date"] <= window_end)].copy()

    has_decomposed = "pred_tminus2_decomposed" in window.columns

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        window["date"],
        window["y"],
        "b-",
        label="actual",
        linewidth=2,
        alpha=0.8,
        marker="o",
        markersize=4,
    )
    if "pred_tminus2_decomposed" in window.columns:
        ax.plot(
            window["date"],
            window["pred_tminus2_decomposed"],
            color="#2ca02c",
            linestyle="--",
            label="pred_tminus2_decomposed",
            linewidth=2,
            alpha=0.8,
            marker="D",
            markersize=3,
        )
    if "pred_tminus2_baseline_cf" in window.columns:
        ax.plot(
            window["date"],
            window["pred_tminus2_baseline_cf"],
            color="#7f7f7f",
            linestyle=":",
            label="pred_tminus2_baseline_cf",
            linewidth=1.8,
            alpha=0.75,
        )

    holiday_mask = build_holiday_mask(window["date"])
    holiday_dates = window.loc[holiday_mask, "date"]
    for d in holiday_dates:
        ax.axvline(d, color="orange", alpha=0.25, linewidth=1.5)

    ax.axvline(cny_2025, color="#d62728", linewidth=1.2, alpha=0.7)

    title = "Spring Transport Window (CNY-25 to CNY+15)"
    if has_decomposed:
        title += " - Decomposed t-2"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Flow", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def main():
    if not OUTPUT_PRED_PATH.exists():
        raise FileNotFoundError(OUTPUT_PRED_PATH)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = load_predictions()
    has_decomposed = "pred_tminus2_decomposed" in pred.columns

    fig_full = plot_full_year(pred)
    full_path = FIG_DIR / ("forecast_2025_decomposed_full.png" if has_decomposed else "forecast_2025_optimized_full.png")
    fig_full.savefig(full_path, dpi=160, bbox_inches="tight")
    plt.close(fig_full)

    fig_window = plot_spring_window(pred)
    window_path = FIG_DIR / (
        "forecast_2025_decomposed_spring_window.png"
        if has_decomposed
        else "forecast_2025_optimized_spring_window.png"
    )
    fig_window.savefig(window_path, dpi=160, bbox_inches="tight")
    plt.close(fig_window)

    print("Saved:", full_path)
    print("Saved:", window_path)


if __name__ == "__main__":
    main()
