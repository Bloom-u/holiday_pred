"""
集成预测模块
结合多种预测策略：
1. 标准预测（14天预测7天，非重叠）
2. 滚动预测（小窗口滚动，取平均）
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def sliding_window_predict_ensemble(model, df, scalers, lag_scaler, dataset_config, device,
                                     start_idx=0, num_days=None):
    """
    集成预测：结合标准预测和滚动平均预测

    策略1: 标准预测 (14天预测7天，非重叠)
    策略2: 滚动预测 (14天预测7天，每次滚动1天，对每个目标日取多次预测的平均)

    Args:
        model: 训练好的模型
        df: DataFrame with features
        scalers: dict with target_scaler and cont_scaler
        lag_scaler: scaler for lag features
        dataset_config: 数据集配置
        device: 计算设备
        start_idx: 起始索引
        num_days: 预测天数
    """
    model.eval()

    seq_len = dataset_config['seq_len']
    pred_len = dataset_config['pred_len']
    target_col = dataset_config['target_col']
    lag_cols = dataset_config.get('lag_cols', ['lag_1d', 'lag_7d', 'lag_14d', 'recent_change_3d'])
    continuous_cols = dataset_config['continuous_cols']
    binary_cols = dataset_config['binary_cols']
    holiday_feature_cols = dataset_config.get('holiday_feature_cols',
                                               ['holiday_day_num', 'total_holiday_length',
                                                'holiday_progress', 'holiday_traffic_ratio'])

    target_scaler = scalers['target_scaler']
    cont_scaler = scalers['cont_scaler']

    end_idx = len(df) if num_days is None else min(start_idx + num_days + seq_len, len(df))

    # ===== 策略1: 标准非重叠预测 =====
    print("\n  [策略1] 标准预测（非重叠窗口）...")
    standard_preds = {}

    with torch.no_grad():
        idx = start_idx
        while idx + seq_len + pred_len <= end_idx:
            x_start, x_end = idx, idx + seq_len
            y_start, y_end = x_end, x_end + pred_len

            # 预测
            pred = _predict_one_window(
                model, df, x_start, x_end, y_start, y_end,
                lag_cols, continuous_cols, binary_cols, holiday_feature_cols,
                lag_scaler, cont_scaler, target_scaler, device
            )

            # 存储预测（按日期索引）
            for i, val in enumerate(pred):
                date_idx = y_start + i
                if date_idx not in standard_preds:
                    standard_preds[date_idx] = []
                standard_preds[date_idx].append(val)

            idx += pred_len  # 非重叠

    # ===== 策略2: 滚动预测取平均 =====
    print("  [策略2] 滚动预测（每次滚动1天，取平均）...")
    rolling_preds = {}
    prediction_counts = {}

    with torch.no_grad():
        idx = start_idx
        pbar = tqdm(total=(end_idx - start_idx - seq_len - pred_len + 1),
                   desc="滚动预测", leave=False)

        while idx + seq_len + pred_len <= end_idx:
            x_start, x_end = idx, idx + seq_len
            y_start, y_end = x_end, x_end + pred_len

            # 预测
            pred = _predict_one_window(
                model, df, x_start, x_end, y_start, y_end,
                lag_cols, continuous_cols, binary_cols, holiday_feature_cols,
                lag_scaler, cont_scaler, target_scaler, device
            )

            # 累加预测（按日期索引）
            for i, val in enumerate(pred):
                date_idx = y_start + i
                if date_idx not in rolling_preds:
                    rolling_preds[date_idx] = 0
                    prediction_counts[date_idx] = 0
                rolling_preds[date_idx] += val
                prediction_counts[date_idx] += 1

            idx += 1  # 滚动1天
            pbar.update(1)

        pbar.close()

    # 计算滚动平均
    for date_idx in rolling_preds:
        rolling_preds[date_idx] /= prediction_counts[date_idx]

    print(f"  滚动预测：每个目标日平均被预测 {np.mean(list(prediction_counts.values())):.1f} 次")

    # ===== 策略3: 集成（加权平均）=====
    print("  [策略3] 集成预测（标准预测60% + 滚动预测40%）...")

    # 收集所有日期的预测结果
    all_dates = sorted(set(list(standard_preds.keys()) + list(rolling_preds.keys())))

    results = {
        'dates': [],
        'true_values': [],
        'predictions_standard': [],
        'predictions_rolling': [],
        'predictions_ensemble': [],
        'prediction_counts': [],
        'holiday_types': [],
        'holiday_names': []
    }

    for date_idx in all_dates:
        if date_idx >= len(df):
            continue

        # 标准预测（可能有多个，取平均）
        std_pred = np.mean(standard_preds[date_idx]) if date_idx in standard_preds else None

        # 滚动预测
        roll_pred = rolling_preds[date_idx] if date_idx in rolling_preds else None

        # 集成预测
        if std_pred is not None and roll_pred is not None:
            # 两者都有：加权平均（标准60%，滚动40%）
            ensemble_pred = 0.6 * std_pred + 0.4 * roll_pred
        elif std_pred is not None:
            ensemble_pred = std_pred
        elif roll_pred is not None:
            ensemble_pred = roll_pred
        else:
            continue

        # 收集结果
        results['dates'].append(df['日期'].iloc[date_idx])
        results['true_values'].append(df[target_col].iloc[date_idx])
        results['predictions_standard'].append(std_pred if std_pred is not None else ensemble_pred)
        results['predictions_rolling'].append(roll_pred if roll_pred is not None else ensemble_pred)
        results['predictions_ensemble'].append(ensemble_pred)
        results['prediction_counts'].append(prediction_counts.get(date_idx, 1))
        results['holiday_types'].append(df['holiday_type'].iloc[date_idx])
        results['holiday_names'].append(df['holiday_name'].iloc[date_idx])

    # 转换为numpy数组
    for key in ['dates', 'true_values', 'predictions_standard', 'predictions_rolling',
                'predictions_ensemble', 'prediction_counts', 'holiday_types', 'holiday_names']:
        if key == 'dates':
            results[key] = pd.to_datetime(results[key])
        elif key == 'holiday_names':
            results[key] = np.array(results[key])
        else:
            results[key] = np.array(results[key])

    print(f"  生成了 {len(results['dates'])} 个预测点")

    return results


def _predict_one_window(model, df, x_start, x_end, y_start, y_end,
                        lag_cols, continuous_cols, binary_cols, holiday_feature_cols,
                        lag_scaler, cont_scaler, target_scaler, device):
    """执行一次窗口预测"""
    # Lag特征
    x_lag = torch.FloatTensor(
        lag_scaler.transform(df[lag_cols].values[x_start:x_end])
    ).unsqueeze(0).to(device)

    # 连续特征
    x_cont = torch.FloatTensor(
        cont_scaler.transform(df[continuous_cols].values[x_start:x_end])
    ).unsqueeze(0).to(device)

    # 二值特征
    x_binary = torch.FloatTensor(
        df[binary_cols].values[x_start:x_end]
    ).unsqueeze(0).to(device)

    # 节假日特征
    x_holiday_type = torch.LongTensor(df['holiday_type'].values[x_start:x_end]).unsqueeze(0).to(device)
    x_holiday_features = torch.FloatTensor(df[holiday_feature_cols].values[x_start:x_end]).unsqueeze(0).to(device)

    # 未来特征
    y_cont = torch.FloatTensor(
        cont_scaler.transform(df[continuous_cols].values[y_start:y_end])
    ).unsqueeze(0).to(device)
    y_binary = torch.FloatTensor(df[binary_cols].values[y_start:y_end]).unsqueeze(0).to(device)
    y_holiday_type = torch.LongTensor(df['holiday_type'].values[y_start:y_end]).unsqueeze(0).to(device)
    y_holiday_features = torch.FloatTensor(df[holiday_feature_cols].values[y_start:y_end]).unsqueeze(0).to(device)

    # 判断是否高/低流量节假日
    y_is_high = torch.FloatTensor([df['is_high_traffic_holiday'].values[y_start:y_end].max()]).to(device)
    y_is_low = torch.FloatTensor([df['is_low_traffic_holiday'].values[y_start:y_end].max()]).to(device)

    # 预测
    pred = model(
        x_lag, x_cont, x_binary,
        y_cont, y_binary,
        x_holiday_type, x_holiday_features,
        y_holiday_type, y_holiday_features,
        y_is_high, y_is_low
    )

    pred_np = pred.cpu().numpy().flatten()
    pred_inv = target_scaler.inverse_transform(pred_np.reshape(-1, 1)).flatten()

    return pred_inv


def compare_prediction_strategies(results):
    """对比不同预测策略的效果"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    true_values = results['true_values']
    pred_std = results['predictions_standard']
    pred_roll = results['predictions_rolling']
    pred_ens = results['predictions_ensemble']
    holiday_types = results['holiday_types']

    print("\n" + "=" * 70)
    print("预测策略对比")
    print("=" * 70)

    strategies = [
        ('标准预测（非重叠）', pred_std),
        ('滚动预测（取平均）', pred_roll),
        ('集成预测（60%+40%）', pred_ens)
    ]

    for name, preds in strategies:
        mae = mean_absolute_error(true_values, preds)
        rmse = np.sqrt(mean_squared_error(true_values, preds))

        # 节假日MAE
        holiday_mask = holiday_types > 1
        if holiday_mask.sum() > 0:
            holiday_mae = mean_absolute_error(
                true_values[holiday_mask],
                preds[holiday_mask]
            )
        else:
            holiday_mae = 0

        print(f"\n【{name}】")
        print(f"  整体 MAE:   {mae:.2f}")
        print(f"  整体 RMSE:  {rmse:.2f}")
        print(f"  节假日 MAE: {holiday_mae:.2f}")

    print("=" * 70)

    return results
