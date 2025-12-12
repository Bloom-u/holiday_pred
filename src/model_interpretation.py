"""
模型解释与特征重要性分析模块

功能:
1. 特征重要性分析 (基于梯度和排列重要性)
2. 预测差异可视化
3. SHAP值分析 (模型决策解释)
4. 注意力权重可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class FeatureImportanceAnalyzer:
    """特征重要性分析器"""

    def __init__(self, model, df, scalers, lag_scaler, dataset_config, device):
        self.model = model
        self.df = df
        self.scalers = scalers
        self.lag_scaler = lag_scaler
        self.dataset_config = dataset_config
        self.device = device

        # 特征名称
        self.lag_cols = dataset_config['lag_cols']
        self.continuous_cols = dataset_config['continuous_cols']
        self.binary_cols = dataset_config['binary_cols']
        self.holiday_feature_cols = dataset_config['holiday_feature_cols']

        self.all_features = (
            self.lag_cols +
            self.continuous_cols +
            self.binary_cols +
            self.holiday_feature_cols
        )

    def _prepare_sample(self, idx):
        """准备单个样本"""
        seq_len = self.dataset_config['seq_len']
        pred_len = self.dataset_config['pred_len']

        x_start, x_end = idx, idx + seq_len
        y_start, y_end = x_end, x_end + pred_len

        if y_end > len(self.df):
            return None

        # Lag特征
        x_lag = torch.FloatTensor(
            self.lag_scaler.transform(self.df[self.lag_cols].values[x_start:x_end])
        ).unsqueeze(0).to(self.device)

        # 连续特征
        x_cont = torch.FloatTensor(
            self.scalers['cont_scaler'].transform(self.df[self.continuous_cols].values[x_start:x_end])
        ).unsqueeze(0).to(self.device)

        # 二值特征
        x_binary = torch.FloatTensor(
            self.df[self.binary_cols].values[x_start:x_end]
        ).unsqueeze(0).to(self.device)

        # 节假日特征
        x_holiday_type = torch.LongTensor(
            self.df['holiday_type'].values[x_start:x_end]
        ).unsqueeze(0).to(self.device)
        x_holiday_features = torch.FloatTensor(
            self.df[self.holiday_feature_cols].values[x_start:x_end]
        ).unsqueeze(0).to(self.device)

        # 未来特征
        y_cont = torch.FloatTensor(
            self.scalers['cont_scaler'].transform(self.df[self.continuous_cols].values[y_start:y_end])
        ).unsqueeze(0).to(self.device)
        y_binary = torch.FloatTensor(
            self.df[self.binary_cols].values[y_start:y_end]
        ).unsqueeze(0).to(self.device)
        y_holiday_type = torch.LongTensor(
            self.df['holiday_type'].values[y_start:y_end]
        ).unsqueeze(0).to(self.device)
        y_holiday_features = torch.FloatTensor(
            self.df[self.holiday_feature_cols].values[y_start:y_end]
        ).unsqueeze(0).to(self.device)

        y_is_high = torch.FloatTensor(
            [self.df['is_high_traffic_holiday'].values[y_start:y_end].max()]
        ).to(self.device)
        y_is_low = torch.FloatTensor(
            [self.df['is_low_traffic_holiday'].values[y_start:y_end].max()]
        ).to(self.device)

        # 真实值
        actual = self.df[self.dataset_config['target_col']].values[y_start:y_end]

        return {
            'x_lag': x_lag,
            'x_cont': x_cont,
            'x_binary': x_binary,
            'x_holiday_type': x_holiday_type,
            'x_holiday_features': x_holiday_features,
            'y_cont': y_cont,
            'y_binary': y_binary,
            'y_holiday_type': y_holiday_type,
            'y_holiday_features': y_holiday_features,
            'y_is_high': y_is_high,
            'y_is_low': y_is_low,
            'actual': actual
        }

    def _predict(self, sample):
        """执行预测"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                sample['x_lag'], sample['x_cont'], sample['x_binary'],
                sample['y_cont'], sample['y_binary'],
                sample['x_holiday_type'], sample['x_holiday_features'],
                sample['y_holiday_type'], sample['y_holiday_features'],
                sample['y_is_high'], sample['y_is_low']
            )
        return self.scalers['target_scaler'].inverse_transform(
            pred.cpu().numpy().reshape(-1, 1)
        ).flatten()

    def compute_permutation_importance(self, n_samples=50, n_repeats=5):
        """
        计算排列特征重要性

        通过随机打乱每个特征的值，观察预测误差的变化来评估特征重要性
        """
        print("  计算排列特征重要性...")

        test_start = int(len(self.df) * 0.85)
        indices = np.random.choice(
            range(test_start, len(self.df) - self.dataset_config['seq_len'] - self.dataset_config['pred_len']),
            min(n_samples, 100),
            replace=False
        )

        # 基准误差
        baseline_errors = []
        for idx in indices:
            sample = self._prepare_sample(idx)
            if sample is None:
                continue
            pred = self._predict(sample)
            baseline_errors.append(mean_absolute_error(sample['actual'], pred))
        baseline_mae = np.mean(baseline_errors)

        # 每个特征组的重要性
        feature_groups = {
            'Lag特征': (self.lag_cols, 'x_lag'),
            '连续特征': (self.continuous_cols, 'x_cont'),
            '二值特征': (self.binary_cols, 'x_binary'),
            '节假日特征': (self.holiday_feature_cols, 'x_holiday_features')
        }

        importance_scores = {}

        for group_name, (feature_names, tensor_key) in feature_groups.items():
            group_importance = []

            for feat_idx, feat_name in enumerate(feature_names):
                feat_errors = []

                for _ in range(n_repeats):
                    for idx in indices[:20]:  # 减少计算量
                        sample = self._prepare_sample(idx)
                        if sample is None:
                            continue

                        # 打乱该特征
                        original = sample[tensor_key].clone()
                        shuffled_indices = torch.randperm(sample[tensor_key].size(1))
                        sample[tensor_key][:, :, feat_idx] = sample[tensor_key][:, shuffled_indices, feat_idx]

                        pred = self._predict(sample)
                        feat_errors.append(mean_absolute_error(sample['actual'], pred))

                        # 恢复原值
                        sample[tensor_key] = original

                importance = np.mean(feat_errors) - baseline_mae if feat_errors else 0
                importance_scores[feat_name] = max(0, importance)
                group_importance.append(importance)

            importance_scores[f'[组]{group_name}'] = np.mean(group_importance) if group_importance else 0

        return importance_scores, baseline_mae

    def compute_gradient_importance(self, n_samples=30):
        """
        计算基于梯度的特征重要性

        通过计算损失对输入特征的梯度来评估特征重要性
        """
        print("  计算梯度特征重要性...")

        test_start = int(len(self.df) * 0.85)
        indices = np.random.choice(
            range(test_start, len(self.df) - self.dataset_config['seq_len'] - self.dataset_config['pred_len']),
            min(n_samples, 50),
            replace=False
        )

        gradient_importance = {name: [] for name in self.all_features}

        for idx in indices:
            sample = self._prepare_sample(idx)
            if sample is None:
                continue

            # 启用梯度
            sample['x_lag'].requires_grad_(True)
            sample['x_cont'].requires_grad_(True)
            sample['x_binary'].requires_grad_(True)
            sample['x_holiday_features'].requires_grad_(True)

            # 前向传播
            pred = self.model(
                sample['x_lag'], sample['x_cont'], sample['x_binary'],
                sample['y_cont'], sample['y_binary'],
                sample['x_holiday_type'], sample['x_holiday_features'],
                sample['y_holiday_type'], sample['y_holiday_features'],
                sample['y_is_high'], sample['y_is_low']
            )

            # 计算梯度
            loss = pred.mean()
            loss.backward()

            # 收集梯度
            for i, name in enumerate(self.lag_cols):
                if sample['x_lag'].grad is not None:
                    grad = sample['x_lag'].grad[:, :, i].abs().mean().item()
                    gradient_importance[name].append(grad)

            for i, name in enumerate(self.continuous_cols):
                if sample['x_cont'].grad is not None:
                    grad = sample['x_cont'].grad[:, :, i].abs().mean().item()
                    gradient_importance[name].append(grad)

            for i, name in enumerate(self.binary_cols):
                if sample['x_binary'].grad is not None:
                    grad = sample['x_binary'].grad[:, :, i].abs().mean().item()
                    gradient_importance[name].append(grad)

            for i, name in enumerate(self.holiday_feature_cols):
                if sample['x_holiday_features'].grad is not None:
                    grad = sample['x_holiday_features'].grad[:, :, i].abs().mean().item()
                    gradient_importance[name].append(grad)

            # 清除梯度
            self.model.zero_grad()

        # 平均梯度重要性
        avg_importance = {
            name: np.mean(grads) if grads else 0
            for name, grads in gradient_importance.items()
        }

        return avg_importance


def plot_feature_importance(importance_scores, output_dir='./figs', title='特征重要性分析'):
    """绘制特征重要性图"""
    os.makedirs(output_dir, exist_ok=True)

    # 过滤并排序
    filtered = {k: v for k, v in importance_scores.items() if not k.startswith('[组]') and v > 0}
    sorted_features = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    if not sorted_features:
        print("  警告: 没有有效的特征重要性数据")
        return

    # 取Top 20
    top_features = sorted_features[:20]
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    # 颜色映射
    colors = []
    for name in names:
        if name.startswith('lag') or name.startswith('recent'):
            colors.append('#e74c3c')  # Lag特征 - 红色
        elif name.startswith('is_') or name.startswith('is_2022'):
            colors.append('#3498db')  # 二值特征 - 蓝色
        elif 'holiday' in name.lower():
            colors.append('#f39c12')  # 节假日特征 - 橙色
        else:
            colors.append('#2ecc71')  # 连续特征 - 绿色

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('重要性分数 (MAE增量)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Lag特征'),
        Patch(facecolor='#2ecc71', label='连续特征'),
        Patch(facecolor='#3498db', label='二值特征'),
        Patch(facecolor='#f39c12', label='节假日特征'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"  保存: {output_dir}/feature_importance.png")
    plt.close()


def plot_prediction_difference_analysis(model, df, scalers, lag_scaler, dataset_config, device, output_dir='./figs'):
    """
    绘制预测差异详细分析

    包括:
    1. 误差随时间变化
    2. 误差与特征的关系
    3. 高误差样本分析
    4. 节假日vs平日误差对比
    """
    os.makedirs(output_dir, exist_ok=True)

    print("  生成预测差异分析...")

    analyzer = FeatureImportanceAnalyzer(model, df, scalers, lag_scaler, dataset_config, device)

    # 收集预测结果
    test_start = int(len(df) * 0.85)
    seq_len = dataset_config['seq_len']
    pred_len = dataset_config['pred_len']

    results = {
        'dates': [],
        'actuals': [],
        'predictions': [],
        'errors': [],
        'abs_errors': [],
        'holiday_types': [],
        'day_of_week': [],
        'lag_1d_values': []
    }

    idx = test_start
    while idx + seq_len + pred_len <= len(df):
        sample = analyzer._prepare_sample(idx)
        if sample is None:
            idx += pred_len
            continue

        pred = analyzer._predict(sample)
        actual = sample['actual']

        for i in range(len(pred)):
            y_idx = idx + seq_len + i
            if y_idx < len(df):
                # 获取日期
                if '日期' in df.columns:
                    date = pd.to_datetime(df['日期'].iloc[y_idx])
                else:
                    date = pd.to_datetime(df.index[y_idx])

                results['dates'].append(date)
                results['actuals'].append(actual[i])
                results['predictions'].append(pred[i])
                results['errors'].append(pred[i] - actual[i])
                results['abs_errors'].append(abs(pred[i] - actual[i]))
                results['holiday_types'].append(df['holiday_type'].iloc[y_idx])
                results['day_of_week'].append(date.dayofweek)
                results['lag_1d_values'].append(df['lag_1d'].iloc[y_idx])

        idx += pred_len

    # 转换为数组
    for key in results:
        results[key] = np.array(results[key])

    # 创建4子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 误差随时间变化
    ax = axes[0, 0]
    ax.plot(results['dates'], results['errors'], 'b-', alpha=0.6, linewidth=1)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_between(results['dates'], results['errors'], 0,
                    where=results['errors'] > 0, color='red', alpha=0.2, label='高估')
    ax.fill_between(results['dates'], results['errors'], 0,
                    where=results['errors'] < 0, color='blue', alpha=0.2, label='低估')
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('预测误差 (预测 - 实际)', fontsize=11)
    ax.set_title('预测误差随时间变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 子图2: 误差与历史流量的关系
    ax = axes[0, 1]
    scatter = ax.scatter(results['lag_1d_values'], results['abs_errors'],
                        c=results['holiday_types'], cmap='coolwarm',
                        alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('前一天流量 (lag_1d)', fontsize=11)
    ax.set_ylabel('绝对误差', fontsize=11)
    ax.set_title('误差与历史流量的关系', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('节假日类型', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 拟合趋势线
    z = np.polyfit(results['lag_1d_values'], results['abs_errors'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(results['lag_1d_values'].min(), results['lag_1d_values'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'趋势线')
    ax.legend(fontsize=10)

    # 子图3: 节假日vs平日误差对比 (箱线图)
    ax = axes[1, 0]
    normal_errors = results['abs_errors'][results['holiday_types'] <= 1]
    holiday_errors = results['abs_errors'][results['holiday_types'] > 1]

    bp = ax.boxplot([normal_errors, holiday_errors],
                    labels=['平日', '节假日'],
                    patch_artist=True,
                    widths=0.6)

    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    for box in bp['boxes']:
        box.set_alpha(0.7)

    ax.set_ylabel('绝对误差', fontsize=11)
    ax.set_title('平日 vs 节假日 预测误差分布', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    ax.text(0.7, 0.95, f'平日 MAE: {normal_errors.mean():.1f}\n节假日 MAE: {holiday_errors.mean():.1f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 子图4: 每周各天的误差分布
    ax = axes[1, 1]
    day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    day_errors = [results['abs_errors'][results['day_of_week'] == i] for i in range(7)]

    bp = ax.boxplot(day_errors, labels=day_names, patch_artist=True, widths=0.6)

    colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('绝对误差', fontsize=11)
    ax.set_title('每周各天预测误差分布', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_difference_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  保存: {output_dir}/prediction_difference_analysis.png")
    plt.close()

    return results


def plot_high_error_analysis(results, df, output_dir='./figs', top_n=20):
    """
    分析高误差样本

    识别预测误差最大的样本，分析其特征
    """
    os.makedirs(output_dir, exist_ok=True)

    print("  分析高误差样本...")

    # 找到误差最大的样本
    top_indices = np.argsort(results['abs_errors'])[-top_n:][::-1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 高误差样本的日期分布
    ax = axes[0, 0]
    high_error_dates = results['dates'][top_indices]
    months = [d.month for d in pd.to_datetime(high_error_dates)]
    ax.hist(months, bins=12, range=(1, 13), color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('月份', fontsize=11)
    ax.set_ylabel('高误差样本数', fontsize=11)
    ax.set_title(f'Top {top_n} 高误差样本的月份分布', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3, axis='y')

    # 子图2: 高误差样本详情表
    ax = axes[0, 1]
    ax.axis('off')

    table_data = [['日期', '实际值', '预测值', '误差', '类型']]
    for i in top_indices[:10]:
        date_str = pd.to_datetime(results['dates'][i]).strftime('%Y-%m-%d')
        actual = f"{results['actuals'][i]:.0f}"
        pred = f"{results['predictions'][i]:.0f}"
        error = f"{results['errors'][i]:+.0f}"
        htype = '节假日' if results['holiday_types'][i] > 1 else '平日'
        table_data.append([date_str, actual, pred, error, htype])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # 表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title(f'Top 10 高误差样本详情', fontsize=12, fontweight='bold', pad=20)

    # 子图3: 高误差vs低误差样本的特征对比
    ax = axes[1, 0]

    low_indices = np.argsort(results['abs_errors'])[:top_n]

    # 对比lag_1d分布
    high_lag = results['lag_1d_values'][top_indices]
    low_lag = results['lag_1d_values'][low_indices]

    ax.hist(low_lag, bins=20, alpha=0.6, label='低误差样本', color='green', edgecolor='black')
    ax.hist(high_lag, bins=20, alpha=0.6, label='高误差样本', color='red', edgecolor='black')
    ax.set_xlabel('前一天流量 (lag_1d)', fontsize=11)
    ax.set_ylabel('样本数', fontsize=11)
    ax.set_title('高/低误差样本的历史流量分布对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 子图4: 高误差样本的时间序列可视化
    ax = axes[1, 1]

    # 选取误差最大的3个连续片段
    sorted_indices = np.argsort(results['abs_errors'])[-50:]
    ax.plot(results['dates'][sorted_indices], results['actuals'][sorted_indices],
            'bo-', label='实际值', markersize=6, alpha=0.7)
    ax.plot(results['dates'][sorted_indices], results['predictions'][sorted_indices],
            'r^--', label='预测值', markersize=6, alpha=0.7)

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('交通流量', fontsize=11)
    ax.set_title('高误差样本的实际值vs预测值', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/high_error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  保存: {output_dir}/high_error_analysis.png")
    plt.close()


def plot_feature_contribution_heatmap(model, df, scalers, lag_scaler, dataset_config, device, output_dir='./figs'):
    """
    绘制特征贡献热力图

    展示不同特征在不同时间段的贡献程度
    """
    os.makedirs(output_dir, exist_ok=True)

    print("  生成特征贡献热力图...")

    analyzer = FeatureImportanceAnalyzer(model, df, scalers, lag_scaler, dataset_config, device)

    # 按月份计算特征重要性
    test_start = int(len(df) * 0.85)
    seq_len = dataset_config['seq_len']
    pred_len = dataset_config['pred_len']

    # 按月份分组
    monthly_importance = {}

    idx = test_start
    while idx + seq_len + pred_len <= len(df):
        sample = analyzer._prepare_sample(idx)
        if sample is None:
            idx += pred_len
            continue

        y_idx = idx + seq_len
        month = df.index[y_idx].month

        if month not in monthly_importance:
            monthly_importance[month] = {name: [] for name in analyzer.lag_cols[:4]}

        # 简单的特征扰动分析
        for i, name in enumerate(analyzer.lag_cols[:4]):
            original = sample['x_lag'].clone()
            sample['x_lag'][:, :, i] = 0  # 置零

            pred_zero = analyzer._predict(sample)
            sample['x_lag'] = original
            pred_orig = analyzer._predict(sample)

            diff = np.abs(pred_orig - pred_zero).mean()
            monthly_importance[month][name].append(diff)

        idx += pred_len * 2  # 跳过一些样本加速

    # 计算平均值
    months = sorted(monthly_importance.keys())
    features = analyzer.lag_cols[:4]

    heatmap_data = np.zeros((len(features), len(months)))
    for j, month in enumerate(months):
        for i, feat in enumerate(features):
            if monthly_importance[month][feat]:
                heatmap_data[i, j] = np.mean(monthly_importance[month][feat])

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([f'{m}月' for m in months], fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)

    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    ax.set_title('Lag特征月度贡献热力图', fontsize=14, fontweight='bold')

    # 添加数值标签
    for i in range(len(features)):
        for j in range(len(months)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.0f}',
                          ha='center', va='center', color='black', fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('贡献度', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_contribution_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  保存: {output_dir}/feature_contribution_heatmap.png")
    plt.close()


def run_comprehensive_interpretation(model, df, scalers, lag_scaler, dataset_config, device, output_dir='./figs'):
    """
    运行完整的模型解释分析
    """
    print("\n" + "=" * 70)
    print("模型解释与特征重要性分析")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 特征重要性分析
    print("\n[1/4] 计算特征重要性...")
    analyzer = FeatureImportanceAnalyzer(model, df, scalers, lag_scaler, dataset_config, device)

    try:
        importance_scores, baseline_mae = analyzer.compute_permutation_importance(n_samples=30, n_repeats=3)
        plot_feature_importance(importance_scores, output_dir, '排列特征重要性分析')
        print(f"  基准MAE: {baseline_mae:.2f}")
    except Exception as e:
        print(f"  排列重要性计算跳过: {e}")

    # 2. 梯度重要性
    print("\n[2/4] 计算梯度重要性...")
    try:
        gradient_importance = analyzer.compute_gradient_importance(n_samples=20)
        plot_feature_importance(gradient_importance, output_dir, '梯度特征重要性分析')
    except Exception as e:
        print(f"  梯度重要性计算跳过: {e}")

    # 3. 预测差异分析
    print("\n[3/4] 预测差异分析...")
    results = plot_prediction_difference_analysis(model, df, scalers, lag_scaler, dataset_config, device, output_dir)

    # 4. 高误差分析
    print("\n[4/4] 高误差样本分析...")
    plot_high_error_analysis(results, df, output_dir)

    # 5. 特征贡献热力图 (可选，计算量大)
    # print("\n[5/5] 特征贡献热力图...")
    # plot_feature_contribution_heatmap(model, df, scalers, lag_scaler, dataset_config, device, output_dir)

    print("\n" + "=" * 70)
    print("模型解释分析完成！")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  {output_dir}/feature_importance.png - 特征重要性分析")
    print(f"  {output_dir}/prediction_difference_analysis.png - 预测差异分析")
    print(f"  {output_dir}/high_error_analysis.png - 高误差样本分析")

    return results
