import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.gbdt.config import (
    CNY_DATES,
    DATA_PATH,
    DYNAMIC_COLS,
    MODEL_DIR,
    MODEL_PATH,
    STATS_PATH,
)
from src.gbdt.data import load_data
from src.gbdt.features import apply_group_stats, build_base_features, build_feature_matrix
from src.gbdt.persistence import load_group_stats


MODEL_BAS_PATH = MODEL_DIR / "xgb_model.bas"
STATS_BAS_PATH = MODEL_DIR / "xgb_stats.bas"
FEATURES_CSV_PATH = MODEL_DIR / "feature_columns.csv"


def load_model(path):
    model = XGBRegressor()
    model.load_model(path)
    return model


def extract_base_score(model):
    config = json.loads(model.get_booster().save_config())
    base_raw = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(base_raw, list):
        return float(base_raw[0])
    if isinstance(base_raw, str) and base_raw.startswith("["):
        base_raw = base_raw.strip("[]")
    return float(base_raw)


def build_feature_columns():
    raw = load_data(DATA_PATH)
    base = build_base_features(raw, CNY_DATES)
    stats = load_group_stats(STATS_PATH)
    base = apply_group_stats(base, stats)
    X_full = build_feature_matrix(base, raw["y"], DYNAMIC_COLS)
    return X_full.columns.tolist()


def export_feature_columns(feature_cols):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"index": list(range(len(feature_cols))), "feature": feature_cols})
    df.to_csv(FEATURES_CSV_PATH, index=False)


def export_stats_vba(stats):
    lines = [
        "Option Explicit",
        "",
        f"Public Const OVERALL_MEAN As Double = {float(stats['overall_mean']):.10f}",
        f"Public Const OVERALL_STD As Double = {float(stats['overall_std']):.10f}",
        "",
        "Private gStatsInitialized As Boolean",
        "",
        "Public dowMean(0 To 6) As Double",
        "Public dowStd(0 To 6) As Double",
        "Public monthMean(0 To 11) As Double",
        "Public monthStd(0 To 11) As Double",
        "Public holidayMean(0 To 9) As Double",
        "Public holidayStd(0 To 9) As Double",
        "Public cnyOffsetMean(-25 To 15) As Double",
        "",
    ]

    def emit_init(name, start, end, values):
        lines.append(f"Private Sub Init_{name}()")
        for k in range(start, end + 1):
            val = values.get(k, stats["overall_mean"])
            lines.append(f"    {name}({k}) = {float(val):.10f}")
        lines.append("End Sub")
        lines.append("")

    emit_init("dowMean", 0, 6, stats["dow_mean"])
    emit_init("dowStd", 0, 6, stats["dow_std"])
    emit_init("monthMean", 0, 11, stats["month_mean"])
    emit_init("monthStd", 0, 11, stats["month_std"])
    emit_init("holidayMean", 0, 9, stats["holiday_type_mean"])
    emit_init("holidayStd", 0, 9, stats["holiday_type_std"])

    lines.append("Private Sub Init_cnyOffsetMean()")
    for k in range(-25, 16):
        val = stats["cny_offset_mean"].get(k, stats["overall_mean"])
        lines.append(f"    cnyOffsetMean({k}) = {float(val):.10f}")
    lines.append("End Sub")
    lines.append("")

    lines.append("Public Sub EnsureStatsInitialized()")
    lines.append("    If gStatsInitialized Then Exit Sub")
    lines.append("    Init_dowMean")
    lines.append("    Init_dowStd")
    lines.append("    Init_monthMean")
    lines.append("    Init_monthStd")
    lines.append("    Init_holidayMean")
    lines.append("    Init_holidayStd")
    lines.append("    Init_cnyOffsetMean")
    lines.append("    gStatsInitialized = True")
    lines.append("End Sub")
    lines.append("")

    STATS_BAS_PATH.write_text("\n".join(lines), encoding="utf-8")


def export_model_vba(model, feature_cols):
    booster = model.get_booster()
    trees = booster.trees_to_dataframe()
    base_score = extract_base_score(model)

    feature_index = {name: idx for idx, name in enumerate(feature_cols)}

    num_trees = int(trees["Tree"].max()) + 1
    node_counts = trees.groupby("Tree")["Node"].max().astype(int) + 1
    tree_start = []
    offset = 0
    for t in range(num_trees):
        tree_start.append(offset)
        offset += int(node_counts.loc[t])

    total_nodes = offset
    node_type = np.zeros(total_nodes, dtype=int)
    split_feature = np.zeros(total_nodes, dtype=int)
    threshold = np.zeros(total_nodes, dtype=float)
    left_child = np.zeros(total_nodes, dtype=int)
    right_child = np.zeros(total_nodes, dtype=int)
    missing_child = np.zeros(total_nodes, dtype=int)
    leaf_value = np.zeros(total_nodes, dtype=float)

    def parse_node_id(value):
        if isinstance(value, str) and "-" in value:
            return int(value.split("-")[1])
        return int(value)

    for _, row in trees.iterrows():
        tree_id = int(row["Tree"])
        node_id = int(row["Node"])
        idx = tree_start[tree_id] + node_id

        if row["Feature"] == "Leaf":
            node_type[idx] = 0
            leaf_value[idx] = float(row["Gain"])
            continue

        feat = row["Feature"]
        if feat not in feature_index:
            raise ValueError(f"Missing feature in map: {feat}")
        node_type[idx] = 1
        split_feature[idx] = feature_index[feat]
        threshold[idx] = float(row["Split"])
        left_child[idx] = tree_start[tree_id] + parse_node_id(row["Yes"])
        right_child[idx] = tree_start[tree_id] + parse_node_id(row["No"])
        missing_child[idx] = tree_start[tree_id] + parse_node_id(row["Missing"])

    lines = [
        "Option Explicit",
        "",
        f"Private Const NUM_TREES As Long = {num_trees}",
        f"Private Const NUM_NODES As Long = {total_nodes}",
        f"Private Const BASE_SCORE As Double = {base_score:.10f}",
        "",
        "Private gModelInitialized As Boolean",
        "Private treeStart(0 To NUM_TREES - 1) As Long",
        "Private nodeType(0 To NUM_NODES - 1) As Integer",
        "Private splitFeature(0 To NUM_NODES - 1) As Integer",
        "Private threshold(0 To NUM_NODES - 1) As Double",
        "Private leftChild(0 To NUM_NODES - 1) As Long",
        "Private rightChild(0 To NUM_NODES - 1) As Long",
        "Private missingChild(0 To NUM_NODES - 1) As Long",
        "Private leafValue(0 To NUM_NODES - 1) As Double",
        "",
        "Private Sub InitModel()",
    ]

    for t, start in enumerate(tree_start):
        lines.append(f"    treeStart({t}) = {start}")

    for i in range(total_nodes):
        lines.append(f"    nodeType({i}) = {node_type[i]}")
        if node_type[i] == 1:
            lines.append(f"    splitFeature({i}) = {split_feature[i]}")
            lines.append(f"    threshold({i}) = {threshold[i]:.10f}")
            lines.append(f"    leftChild({i}) = {left_child[i]}")
            lines.append(f"    rightChild({i}) = {right_child[i]}")
            lines.append(f"    missingChild({i}) = {missing_child[i]}")
        else:
            lines.append(f"    leafValue({i}) = {leaf_value[i]:.10f}")

    lines.append("    gModelInitialized = True")
    lines.append("End Sub")
    lines.append("")
    lines.append("Private Sub EnsureModelInitialized()")
    lines.append("    If gModelInitialized Then Exit Sub")
    lines.append("    InitModel")
    lines.append("End Sub")
    lines.append("")
    lines.append("Public Function XGBPredict(features() As Double) As Double")
    lines.append("    Dim t As Long, idx As Long")
    lines.append("    Dim f As Long, x As Double")
    lines.append("    Dim score As Double")
    lines.append("    EnsureModelInitialized")
    lines.append("    score = BASE_SCORE")
    lines.append("    For t = 0 To NUM_TREES - 1")
    lines.append("        idx = treeStart(t)")
    lines.append("        Do")
    lines.append("            If nodeType(idx) = 0 Then")
    lines.append("                score = score + leafValue(idx)")
    lines.append("                Exit Do")
    lines.append("            End If")
    lines.append("            f = splitFeature(idx)")
    lines.append("            x = features(f)")
    lines.append("            If x <> x Then")
    lines.append("                idx = missingChild(idx)")
    lines.append("            ElseIf x < threshold(idx) Then")
    lines.append("                idx = leftChild(idx)")
    lines.append("            Else")
    lines.append("                idx = rightChild(idx)")
    lines.append("            End If")
    lines.append("        Loop")
    lines.append("    Next t")
    lines.append("    XGBPredict = score")
    lines.append("End Function")

    MODEL_BAS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not STATS_PATH.exists():
        raise FileNotFoundError(STATS_PATH)

    model = load_model(MODEL_PATH)
    feature_cols = build_feature_columns()
    export_feature_columns(feature_cols)
    export_model_vba(model, feature_cols)

    stats = load_group_stats(STATS_PATH)
    export_stats_vba(stats)

    print("Saved:", MODEL_BAS_PATH)
    print("Saved:", STATS_BAS_PATH)
    print("Saved:", FEATURES_CSV_PATH)


if __name__ == "__main__":
    main()
