"""
集成预测可视化模块
对比标准预测、滚动预测和集成预测的效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_ensemble_comparison(results, output_dir='./figs_ensemble'):
    """对比三种预测策略"""
    os.makedirs(output_dir, exist_ok=True)

    dates = results['dates']
    true_values = results['true_values']
    pred_std = results['predictions_standard']
    pred_roll = results['predictions_rolling']
    pred_ens = results['predictions_ensemble']
    holiday_types = results['holiday_types']

    # 创建大图
    fig = plt.subplots(figsize=(18, 14))
    fig, axes = plt.subplots(4, 1, figsize=(18, 16))

    # ===== 子图1: 三种策略的时间序列对比 =====
    ax = axes[0]
    ax.plot(dates, true_values, 'k-', label='真实值', linewidth=2.5, alpha=0.9, zorder=5)
    ax.plot(dates, pred_std, 'b--', label='标准预测（非重叠）', linewidth=1.5, alpha=0.7, marker='o', markersize=3, markevery=5)
    ax.plot(dates, pred_roll, 'g--', label='滚动预测（平均）', linewidth=1.5, alpha=0.7, marker='s', markersize=3, markevery=5)
    ax.plot(dates, pred_ens, 'r-', label='集成预测（60%+40%）', linewidth=2, alpha=0.8, marker='^', markersize=4, markevery=5)

    # 标记节假日
    is_holiday = holiday_types > 1
    if is_holiday.sum() > 0:
        holiday_dates = dates[is_holiday]
        for d in holiday_dates:
            ax.axvline(d, color='orange', alpha=0.2, linewidth=1.5, zorder=1)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('交通流量 (机动车当量)', fontsize=12)
    ax.set_title('三种预测策略对比 - 时间序列', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ===== 子图2: 三种策略的误差对比（箱线图）=====
    ax = axes[1]

    errors_std = pred_std - true_values
    errors_roll = pred_roll - true_values
    errors_ens = pred_ens - true_values

    bp = ax.boxplot([errors_std, errors_roll, errors_ens],
                     labels=['标准预测', '滚动预测', '集成预测'],
                     patch_artist=True,
                     widths=0.6)

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='零误差线')
    ax.set_ylabel('预测误差', fontsize=12)
    ax.set_title('三种预测策略的误差分布对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加MAE标注
    for i, (name, errors) in enumerate([('标准', errors_std), ('滚动', errors_roll), ('集成', errors_ens)]):
        mae = np.abs(errors).mean()
        ax.text(i+1, ax.get_ylim()[1]*0.9, f'MAE={mae:.1f}',
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ===== 子图3: 节假日vs平日的预测对比 =====
    ax = axes[2]

    normal_mask = holiday_types <= 1
    holiday_mask = holiday_types > 1

    # 计算不同策略在不同日期类型下的MAE
    metrics = []
    for name, preds in [('标准', pred_std), ('滚动', pred_roll), ('集成', pred_ens)]:
        normal_mae = mean_absolute_error(true_values[normal_mask], preds[normal_mask]) if normal_mask.sum() > 0 else 0
        holiday_mae = mean_absolute_error(true_values[holiday_mask], preds[holiday_mask]) if holiday_mask.sum() > 0 else 0
        metrics.append((name, normal_mae, holiday_mae))

    x = np.arange(len(metrics))
    width = 0.35

    normal_maes = [m[1] for m in metrics]
    holiday_maes = [m[2] for m in metrics]

    bars1 = ax.bar(x - width/2, normal_maes, width, label='平日MAE', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, holiday_maes, width, label='节假日MAE', color='coral', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('平日 vs 节假日预测效果对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # ===== 子图4: 预测覆盖度分析 =====
    ax = axes[3]

    pred_counts = results['prediction_counts']

    # 绘制每个点被预测的次数
    scatter = ax.scatter(range(len(pred_counts)), pred_counts,
                        c=pred_counts, cmap='viridis',
                        s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax.axhline(pred_counts.mean(), color='red', linestyle='--', linewidth=2,
              label=f'平均覆盖次数: {pred_counts.mean():.1f}')

    ax.set_xlabel('样本索引', fontsize=12)
    ax.set_ylabel('被预测次数', fontsize=12)
    ax.set_title('滚动预测覆盖度分析（每个目标日被预测的次数）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('预测次数', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  保存: {output_dir}/ensemble_comparison.png")
    plt.close()


def plot_holiday_specific_ensemble(results, holiday_name, output_dir='./figs_ensemble'):
    """针对特定节假日的详细对比"""
    os.makedirs(output_dir, exist_ok=True)

    dates = results['dates']
    true_values = results['true_values']
    pred_std = results['predictions_standard']
    pred_roll = results['predictions_rolling']
    pred_ens = results['predictions_ensemble']
    holiday_names = results['holiday_names']

    # 筛选该节假日
    mask = np.array([name == holiday_name for name in holiday_names])

    if mask.sum() == 0:
        print(f"  未找到 {holiday_name} 的数据")
        return

    h_dates = dates[mask]
    h_true = true_values[mask]
    h_std = pred_std[mask]
    h_roll = pred_roll[mask]
    h_ens = pred_ens[mask]

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ===== 子图1: 时间序列对比 =====
    ax = axes[0]
    ax.plot(h_dates, h_true, 'k-o', label='真实值', linewidth=2.5, markersize=8, alpha=0.9)
    ax.plot(h_dates, h_std, 'b--s', label='标准预测', linewidth=2, markersize=6, alpha=0.7)
    ax.plot(h_dates, h_roll, 'g--^', label='滚动预测', linewidth=2, markersize=6, alpha=0.7)
    ax.plot(h_dates, h_ens, 'r-d', label='集成预测', linewidth=2.5, markersize=7, alpha=0.8)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('交通流量', fontsize=12)
    ax.set_title(f'{holiday_name} - 三种预测策略对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ===== 子图2: 误差分析 =====
    ax = axes[1]

    x = np.arange(len(h_dates))
    width = 0.25

    errors_std = h_std - h_true
    errors_roll = h_roll - h_true
    errors_ens = h_ens - h_true

    ax.bar(x - width, errors_std, width, label='标准预测误差', color='blue', alpha=0.7)
    ax.bar(x, errors_roll, width, label='滚动预测误差', color='green', alpha=0.7)
    ax.bar(x + width, errors_ens, width, label='集成预测误差', color='red', alpha=0.7)

    ax.axhline(0, color='black', linewidth=1.5)
    ax.axhline(errors_std.mean(), color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
              label=f'标准MAE: {np.abs(errors_std).mean():.1f}')
    ax.axhline(errors_roll.mean(), color='green', linestyle='--', linewidth=1.5, alpha=0.7,
              label=f'滚动MAE: {np.abs(errors_roll).mean():.1f}')
    ax.axhline(errors_ens.mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.7,
              label=f'集成MAE: {np.abs(errors_ens).mean():.1f}')

    ax.set_xlabel('样本序号', fontsize=12)
    ax.set_ylabel('预测误差 (预测 - 真实)', fontsize=12)
    ax.set_title(f'{holiday_name} - 预测误差对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)

    plt.tight_layout()

    # 文件名处理
    safe_name = holiday_name.replace('/', '_').replace(' ', '_')
    plt.savefig(f'{output_dir}/{safe_name}_ensemble.png', dpi=150, bbox_inches='tight')
    print(f"  保存: {output_dir}/{safe_name}_ensemble.png")
    plt.close()
