"""
V2模型专用可视化模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def sliding_window_predict_v2(model, df, scalers, lag_scaler, dataset_config, device,
                               start_idx=0, num_days=None):
    """
    V2模型的滑动窗口预测

    Args:
        model: ImprovedTrafficModel
        df: 包含高级特征的DataFrame
        scalers: {'target_scaler': ..., 'cont_scaler': ...}
        lag_scaler: lag特征的scaler
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

    dates = []
    true_values = []
    predictions = []
    holiday_types = []
    holiday_names = []

    end_idx = len(df) if num_days is None else min(start_idx + num_days + seq_len, len(df))

    with torch.no_grad():
        idx = start_idx
        while idx + seq_len + pred_len <= end_idx:
            x_start, x_end = idx, idx + seq_len
            y_start, y_end = x_end, x_end + pred_len

            # 准备输入 - Lag特征
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

            # 收集结果
            for i in range(pred_len):
                if y_start + i < len(df):
                    dates.append(df['日期'].iloc[y_start + i])
                    true_values.append(df[target_col].iloc[y_start + i])
                    predictions.append(pred_inv[i])
                    holiday_types.append(df['holiday_type'].iloc[y_start + i])
                    holiday_names.append(df['holiday_name'].iloc[y_start + i])

            idx += pred_len  # 非重叠滑动

    return {
        'dates': pd.to_datetime(dates),
        'true_values': np.array(true_values),
        'predictions': np.array(predictions),
        'holiday_types': np.array(holiday_types),
        'holiday_names': np.array(holiday_names)
    }


def plot_overall_performance(results, output_dir='./figs_v2'):
    """绘制整体性能图"""
    dates = results['dates']
    true_values = results['true_values']
    predictions = results['predictions']
    holiday_types = results['holiday_types']

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 1. 时间序列对比
    ax = axes[0]
    ax.plot(dates, true_values, 'b-', label='真实值', linewidth=2, alpha=0.8, marker='o', markersize=4, markevery=3)
    ax.plot(dates, predictions, 'r--', label='预测值', linewidth=2, alpha=0.8, marker='s', markersize=4, markevery=3)

    # 标记节假日
    is_holiday = holiday_types > 1
    if is_holiday.sum() > 0:
        holiday_dates = dates[is_holiday]
        for d in holiday_dates:
            ax.axvline(d, color='orange', alpha=0.25, linewidth=1.5)

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('交通流量 (机动车当量)', fontsize=11)
    ax.set_title('V2模型预测结果 - 时间序列对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. 误差分布
    ax = axes[1]
    errors = predictions - true_values
    abs_errors = np.abs(errors)

    ax.hist(abs_errors, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2,
               label=f'平均误差: {abs_errors.mean():.1f}')
    ax.axvline(np.median(abs_errors), color='green', linestyle='--', linewidth=2,
               label=f'中位数: {np.median(abs_errors):.1f}')
    ax.set_xlabel('绝对误差', fontsize=11)
    ax.set_ylabel('频次', fontsize=11)
    ax.set_title('预测误差分布', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 真实值 vs 预测值散点图
    ax = axes[2]

    normal_mask = holiday_types <= 1
    holiday_mask = holiday_types > 1

    ax.scatter(true_values[normal_mask], predictions[normal_mask],
               alpha=0.5, s=25, c='blue', label='平日', edgecolors='navy', linewidth=0.3)
    ax.scatter(true_values[holiday_mask], predictions[holiday_mask],
               alpha=0.7, s=45, c='red', marker='^', label='节假日', edgecolors='darkred', linewidth=0.5)

    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='完美预测线', alpha=0.7)

    ax.set_xlabel('真实值', fontsize=11)
    ax.set_ylabel('预测值', fontsize=11)
    ax.set_title('真实值 vs 预测值', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/overall_performance.png', dpi=150, bbox_inches='tight')
    print(f"  保存: {output_dir}/overall_performance.png")
    plt.close()


# 复用原来的plot_holiday_comparison和plot_specific_holiday
from visualization import plot_holiday_comparison, plot_specific_holiday


def generate_all_visualizations_v2(model, df, scalers, lag_scaler, dataset_config,
                                    device, output_dir='./figs_v2', test_ratio=0.15):
    """
    生成所有可视化图表 - V2版本
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    # 确定测试集起始位置
    start_idx = int(len(df) * (1 - test_ratio))

    # 1. 滑动窗口预测
    print("\n[1/5] 滑动窗口预测...")
    results = sliding_window_predict_v2(
        model, df, scalers, lag_scaler, dataset_config, device,
        start_idx=start_idx
    )

    print(f"  生成了 {len(results['dates'])} 个预测点")

    # 2. 整体性能图
    print("\n[2/5] 绘制整体性能图...")
    plot_overall_performance(results, output_dir)

    # 3. 节假日对比图
    print("\n[3/5] 绘制节假日对比图...")
    plot_holiday_comparison(results, output_dir)

    # 4. 特定节假日分析
    print("\n[4/5] 绘制春节详细分析...")
    plot_specific_holiday(results, '春节', output_dir)

    print("\n[5/5] 绘制国庆节详细分析...")
    plot_specific_holiday(results, '国庆节', output_dir)

    # 打印统计信息
    true_values = results['true_values']
    predictions = results['predictions']
    holiday_types = results['holiday_types']

    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))

    normal_mask = holiday_types <= 1
    holiday_mask = holiday_types > 1

    normal_mae = mean_absolute_error(true_values[normal_mask], predictions[normal_mask]) if normal_mask.sum() > 0 else 0
    holiday_mae = mean_absolute_error(true_values[holiday_mask], predictions[holiday_mask]) if holiday_mask.sum() > 0 else 0

    print(f"\n{'='*60}")
    print("预测性能汇总")
    print(f"{'='*60}")
    print(f"  整体 MAE:   {mae:.2f}")
    print(f"  整体 RMSE:  {rmse:.2f}")
    print(f"  平日 MAE:   {normal_mae:.2f}")
    print(f"  节假日 MAE: {holiday_mae:.2f}")
    print(f"{'='*60}")

    return results
