import json


def _serialize_map(map_obj):
    return {str(k): float(v) for k, v in map_obj.items()}


def _deserialize_map(map_obj, cast_key=int):
    return {cast_key(k): float(v) for k, v in map_obj.items()}


def save_group_stats(path, stats):
    data = {
        "overall_mean": float(stats["overall_mean"]),
        "overall_std": float(stats["overall_std"]),
        "dow_mean": _serialize_map(stats["dow_mean"]),
        "dow_std": _serialize_map(stats["dow_std"]),
        "month_mean": _serialize_map(stats["month_mean"]),
        "month_std": _serialize_map(stats["month_std"]),
        "holiday_type_mean": _serialize_map(stats["holiday_type_mean"]),
        "holiday_type_std": _serialize_map(stats["holiday_type_std"]),
        "cny_offset_mean": _serialize_map(stats["cny_offset_mean"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_group_stats(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "overall_mean": float(data["overall_mean"]),
        "overall_std": float(data["overall_std"]),
        "dow_mean": _deserialize_map(data["dow_mean"], int),
        "dow_std": _deserialize_map(data["dow_std"], int),
        "month_mean": _deserialize_map(data["month_mean"], int),
        "month_std": _deserialize_map(data["month_std"], int),
        "holiday_type_mean": _deserialize_map(data["holiday_type_mean"], int),
        "holiday_type_std": _deserialize_map(data["holiday_type_std"], int),
        "cny_offset_mean": _deserialize_map(data["cny_offset_mean"], int),
    }


def save_calibration(path, calibration):
    data = {str(k): float(v) for k, v in calibration.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}
