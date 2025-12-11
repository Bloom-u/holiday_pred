"""
综合评估脚本 - 全方位可视化评估已训练的模型
"""

import sys
sys.path.insert(0, './src')

import pandas as pd
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from model import ImprovedTrafficModel
from feature_engineering import AdvancedFeatureEngine

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_model_and_data():
    """加载训练好的模型和数据"""
    print("[加载模型和数据]")

    # 加载数据
    df = pd.read_pickle('./data/df_with_advanced_features.pkl')

    # 加载模型
    model_path = './models/model_v2_output/best_model.pth'
    with open('./models/model_v2_output/scalers.pkl', 'rb') as f:
        scalers_dict = pickle.load(f)

    scalers = {
        'target_scaler': scalers_dict['target_scaler'],
        'cont_scaler': scalers_dict['cont_scaler']
    }
    lag_scaler = scalers_dict['lag_scaler']

    # 数据集配置
    dataset_config = {
        'seq_len': 14,
        'pred_len': 7,
        'target_col': '机动车当量',
        'lag_cols': ['lag_1d', 'lag_7d', 'lag_14d', 'recent_change_3d'],
        'continuous_cols': [
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'days_to_next_holiday', 'days_from_prev_holiday',
            'holiday_proximity', 'ma_7d', 'ma_14d', 'ma_30d',
            'std_7d', 'trend_7d', 'is_before_holiday_1d',
            'is_before_holiday_2d', 'is_after_holiday_1d',
            'is_after_holiday_2d', 'year_normalized'
        ],
        'binary_cols': [
            'is_weekend', 'is_holiday', 'is_adjusted_workday',
            'is_high_traffic_holiday', 'is_low_traffic_holiday',
            'is_holiday_start', 'is_holiday_end', 'is_2022'
        ],
        'holiday_feature_cols': [
            'holiday_day_num', 'total_holiday_length',
            'holiday_progress', 'holiday_traffic_ratio'
        ]
    }

    # 创建模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = ImprovedTrafficModel(
        num_continuous=len(dataset_config['continuous_cols']),
        num_binary=len(dataset_config['binary_cols']),
        num_lag=len(dataset_config['lag_cols']),
        hidden_dim=64,
        pred_len=7,
        dropout=0.15
    ).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print(f"✓ 模型已加载: {model_path}")
    print(f"✓ 设备: {device}")
    print(f"✓ 数据形状: {df.shape}")

    return model, df, device, scalers, lag_scaler, dataset_config




def evaluate_model(model, df, dataset_config, scalers, lag_scaler, device):
    """评估模型性能"""
    print("\n[评估模型性能]")

    predictions = []
    actuals = []
    dates_list = []

    seq_len = dataset_config['seq_len']
    pred_len = dataset_config['pred_len']
    target_col = dataset_config['target_col']
    lag_cols = dataset_config['lag_cols']
    continuous_cols = dataset_config['continuous_cols']
    binary_cols = dataset_config['binary_cols']
    holiday_feature_cols = dataset_config['holiday_feature_cols']

    target_scaler = scalers['target_scaler']
    cont_scaler = scalers['cont_scaler']

    test_start_idx = int(len(df) * 0.85)

    model.eval()
    with torch.no_grad():
        idx = test_start_idx
        while idx + seq_len + pred_len <= len(df):
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

            pred_np = pred.cpu().numpy()[0]

            # 反规范化
            pred_denorm = target_scaler.inverse_transform(
                pred_np.reshape(-1, 1)
            ).flatten()

            actual = df[target_col].values[y_start:y_end]
            dates = df.index[y_start:y_end]

            predictions.append(pred_denorm)
            actuals.append(actual)
            dates_list.extend(dates)

            idx += pred_len

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 展平数据用于指标计算
    predictions_flat = predictions.flatten()
    actuals_flat = actuals.flatten()

    # 计算指标
    mae = mean_absolute_error(actuals_flat, predictions_flat)
    rmse = np.sqrt(mean_squared_error(actuals_flat, predictions_flat))
    r2 = r2_score(actuals_flat, predictions_flat)
    mape = np.mean(np.abs((actuals_flat - predictions_flat) / actuals_flat)) * 100

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'predictions': predictions,
        'actuals': actuals,
        'dates': dates_list
    }

    print(f"✓ MAE:  {mae:.2f}")
    print(f"✓ RMSE: {rmse:.2f}")
    print(f"✓ R²:   {r2:.4f}")
    print(f"✓ MAPE: {mape:.2f}%")

    return metrics


def plot_time_series(metrics, output_dir='./figs'):
    """绘制时间序列对比"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 6))

    # 展平数据
    actuals_flat = metrics['actuals'].flatten()
    predictions_flat = metrics['predictions'].flatten()
    dates_flat = metrics['dates']

    ax.plot(dates_flat, actuals_flat, 'k-', label='实际值', linewidth=2.5, alpha=0.8, zorder=5)
    ax.plot(dates_flat, predictions_flat, 'r--', label='预测值', linewidth=2, alpha=0.7, marker='o', markersize=3, markevery=7)

    ax.fill_between(dates_flat, actuals_flat, predictions_flat, alpha=0.1, color='red')

    ax.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax.set_ylabel('交通流量 (机动车当量)', fontsize=12, fontweight='bold')
    ax.set_title('测试集时间序列对比 - 实际值 vs 预测值', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_time_series.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存: {output_dir}/01_time_series.png")
    plt.close()


def plot_scatter(metrics, output_dir='./figs'):
    """绘制散点图 - 预测值 vs 实际值"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    actuals_flat = metrics['actuals'].flatten()
    predictions_flat = metrics['predictions'].flatten()

    ax.scatter(actuals_flat, predictions_flat, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

    # 绘制完美预测线
    min_val = min(actuals_flat.min(), predictions_flat.min())
    max_val = max(actuals_flat.max(), predictions_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测', alpha=0.7)

    ax.set_xlabel('实际值', fontsize=12, fontweight='bold')
    ax.set_ylabel('预测值', fontsize=12, fontweight='bold')
    ax.set_title('预测精度散点图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加R²信息
    ax.text(0.05, 0.95, f'R² = {metrics["R2"]:.4f}\nMAE = {metrics["MAE"]:.2f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_scatter.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存: {output_dir}/02_scatter.png")
    plt.close()


def plot_residuals(metrics, output_dir='./figs'):
    """绘制残差分析"""
    os.makedirs(output_dir, exist_ok=True)

    residuals = metrics['actuals'].flatten() - metrics['predictions'].flatten()
    dates_flat = metrics['dates']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 残差时间序列
    ax = axes[0, 0]
    ax.plot(dates_flat, residuals, 'b-', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_between(dates_flat, residuals, 0, alpha=0.2, color='blue')
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('残差', fontsize=11)
    ax.set_title('残差时间序列', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 子图2: 残差直方图
    ax = axes[0, 1]
    ax.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('残差值', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title('残差分布', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 子图3: Q-Q图
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q图', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 子图4: 残差vs预测值
    ax = axes[1, 1]
    predictions_flat = metrics['predictions'].flatten()
    ax.scatter(predictions_flat, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('预测值', fontsize=11)
    ax.set_ylabel('残差', fontsize=11)
    ax.set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_residuals.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存: {output_dir}/03_residuals.png")
    plt.close()


def plot_error_analysis(metrics, output_dir='./figs'):
    """绘制误差分析"""
    os.makedirs(output_dir, exist_ok=True)

    actuals_flat = metrics['actuals'].flatten()
    predictions_flat = metrics['predictions'].flatten()
    errors = np.abs(actuals_flat - predictions_flat)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 误差时间序列
    ax = axes[0, 0]
    dates_flat = metrics['dates']
    ax.plot(dates_flat, errors, 'orange', linewidth=1.5, alpha=0.7)
    ax.axhline(metrics['MAE'], color='r', linestyle='--', linewidth=2, label=f"MAE={metrics['MAE']:.2f}")
    ax.fill_between(dates_flat, errors, alpha=0.2, color='orange')
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('绝对误差', fontsize=11)
    ax.set_title('绝对误差时间序列', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 子图2: 误差箱线图
    ax = axes[0, 1]
    mape = (errors / actuals_flat) * 100
    bp = ax.boxplot([errors, mape], labels=['绝对误差', 'MAPE (%)'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('误差分布对比', fontsize=12, fontweight='bold')

    # 子图3: 百分位数分析
    ax = axes[1, 0]
    percentiles = np.percentile(errors, [10, 25, 50, 75, 90])
    ax.bar(range(len(percentiles)), percentiles, color=['lightgreen', 'lightblue', 'gold', 'orange', 'lightcoral'],
           edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(percentiles)))
    ax.set_xticklabels(['10%', '25%', '50%', '75%', '90%'])
    ax.set_ylabel('绝对误差', fontsize=11)
    ax.set_title('误差百分位数分布', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(percentiles):
        ax.text(i, v + 20, f'{v:.0f}', ha='center', fontweight='bold')

    # 子图4: 误差统计表
    ax = axes[1, 1]
    ax.axis('off')
    stats_data = [
        ['指标', '值'],
        ['平均误差 (MAE)', f'{metrics["MAE"]:.2f}'],
        ['均方根误差 (RMSE)', f'{metrics["RMSE"]:.2f}'],
        ['平均百分比误差 (MAPE)', f'{metrics["MAPE"]:.2f}%'],
        ['最小误差', f'{errors.min():.2f}'],
        ['最大误差', f'{errors.max():.2f}'],
        ['中位误差', f'{np.median(errors):.2f}'],
        ['标准差', f'{errors.std():.2f}'],
    ]
    table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 表头样式
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存: {output_dir}/04_error_analysis.png")
    plt.close()


def plot_performance_summary(metrics, output_dir='./figs'):
    """绘制性能总结"""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 主标题
    fig.suptitle('模型综合性能评估', fontsize=16, fontweight='bold', y=0.98)

    # 1. 关键指标卡片
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')

    metrics_text = f"""
    MAE: {metrics['MAE']:.2f}  |  RMSE: {metrics['RMSE']:.2f}  |  R²: {metrics['R2']:.4f}  |  MAPE: {metrics['MAPE']:.2f}%
    """
    ax.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))

    # 2. 预测vs实际对比
    ax = fig.add_subplot(gs[1, 0])
    actuals_flat = metrics['actuals'].flatten()
    predictions_flat = metrics['predictions'].flatten()
    ax.scatter(actuals_flat, predictions_flat, alpha=0.5, s=20)
    min_val = min(actuals_flat.min(), predictions_flat.min())
    max_val = max(actuals_flat.max(), predictions_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    ax.set_xlabel('实际值', fontsize=10)
    ax.set_ylabel('预测值', fontsize=10)
    ax.set_title('预测精度', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. 误差分布
    ax = fig.add_subplot(gs[1, 1])
    errors = np.abs(actuals_flat - predictions_flat)
    ax.hist(errors, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(metrics['MAE'], color='r', linestyle='--', linewidth=2, label=f"MAE={metrics['MAE']:.0f}")
    ax.set_xlabel('绝对误差', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.set_title('误差分布', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 相对误差分析
    ax = fig.add_subplot(gs[1, 2])
    mape_values = (errors / actuals_flat) * 100
    ax.hist(mape_values, bins=25, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.axvline(metrics['MAPE'], color='darkred', linestyle='--', linewidth=2, label=f"MAPE={metrics['MAPE']:.1f}%")
    ax.set_xlabel('MAPE (%)', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.set_title('相对误差分布', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 5. 残差分析
    ax = fig.add_subplot(gs[2, 0])
    residuals = actuals_flat - predictions_flat
    ax.scatter(predictions_flat, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('预测值', fontsize=10)
    ax.set_ylabel('残差', fontsize=10)
    ax.set_title('残差分析', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6. 性能评级
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')

    # 根据指标评级
    r2_score = metrics['R2']
    mape = metrics['MAPE']

    if r2_score > 0.9 and mape < 5:
        rating = '⭐⭐⭐⭐⭐ 优秀'
        color = 'lightgreen'
    elif r2_score > 0.85 and mape < 8:
        rating = '⭐⭐⭐⭐ 很好'
        color = 'lightblue'
    elif r2_score > 0.80 and mape < 10:
        rating = '⭐⭐⭐ 良好'
        color = 'lightyellow'
    else:
        rating = '⭐⭐ 一般'
        color = 'lightcoral'

    ax.text(0.5, 0.5, rating, ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8, pad=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 7. 性能指标详情
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    detail_text = f"""R²: {r2_score:.4f}
MAPE: {mape:.2f}%
样本数: {len(actuals_flat)}
最大误差: {errors.max():.0f}
最小误差: {errors.min():.0f}"""

    ax.text(0.1, 0.5, detail_text, ha='left', va='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.8))

    plt.savefig(f'{output_dir}/05_performance_summary.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存: {output_dir}/05_performance_summary.png")
    plt.close()


def main():
    print("=" * 70)
    print("综合评估脚本 - 全方位可视化评估已训练的模型")
    print("=" * 70)

    # 1. 加载模型和数据
    model, df, device, scalers, lag_scaler, dataset_config = load_model_and_data()

    # 3. 评估模型
    metrics = evaluate_model(model, df, dataset_config, scalers, lag_scaler, device)

    # 4. 生成可视化
    print("\n[生成可视化评估图表]")
    plot_time_series(metrics)
    plot_scatter(metrics)
    plot_residuals(metrics)
    plot_error_analysis(metrics)
    plot_performance_summary(metrics)

    # 5. 输出总结
    print("\n" + "=" * 70)
    print("✅ 综合评估完成！")
    print("=" * 70)
    print(f"\n【性能指标总结】")
    print(f"  MAE (平均绝对误差):    {metrics['MAE']:.2f}")
    print(f"  RMSE (均方根误差):     {metrics['RMSE']:.2f}")
    print(f"  R² (决定系数):         {metrics['R2']:.4f}")
    print(f"  MAPE (平均百分比误差): {metrics['MAPE']:.2f}%")
    print(f"\n【可视化输出位置】")
    print(f"  ./figs/01_time_series.png      - 时间序列对比")
    print(f"  ./figs/02_scatter.png          - 预测精度散点图")
    print(f"  ./figs/03_residuals.png        - 残差分析")
    print(f"  ./figs/04_error_analysis.png   - 误差分析")
    print(f"  ./figs/05_performance_summary.png - 性能总结")
    print("=" * 70)


if __name__ == '__main__':
    main()
