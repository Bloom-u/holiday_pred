"""
增强评估脚本 - 包含模型解释和特征重要性分析

功能:
1. 标准性能评估
2. 特征重要性分析 (排列重要性 + 梯度重要性)
3. 预测差异可视化
4. 高误差样本分析
5. 模型决策解释
"""

import sys
sys.path.insert(0, './src')

import pandas as pd
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from model import ImprovedTrafficModel
from model_interpretation import (
    FeatureImportanceAnalyzer,
    plot_feature_importance,
    plot_prediction_difference_analysis,
    plot_high_error_analysis,
    run_comprehensive_interpretation
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_model_and_data():
    """加载训练好的模型和数据"""
    print("[加载模型和数据]")

    model_dir = './models/model_v2_output'
    model_path = f'{model_dir}/best_model.pth'
    scalers_path = f'{model_dir}/scalers.pkl'

    # 检查模型文件是否存在
    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        print("  模型文件不存在，先运行训练...")
        print("  请运行: python main.py")
        print("  或使用现有模型进行快速训练...")

        # 快速训练
        from feature_engineering import AdvancedFeatureEngine
        from train import train_model_v2

        # 加载或生成特征
        if os.path.exists('./data/df_with_advanced_features.pkl'):
            df = pd.read_pickle('./data/df_with_advanced_features.pkl')
        else:
            df_base = pd.read_pickle('./data/df_with_features.pkl')
            engine = AdvancedFeatureEngine()
            df = engine.create_advanced_features(df_base)
            df.to_pickle('./data/df_with_advanced_features.pkl')

        # 快速训练 (减少epoch)
        result = train_model_v2(
            df,
            save_dir=model_dir,
            num_epochs=50,  # 快速训练
            patience=15
        )

        print("  快速训练完成!")

    # 加载数据
    df = pd.read_pickle('./data/df_with_advanced_features.pkl')

    # 加载模型
    with open(scalers_path, 'rb') as f:
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

    print(f"  模型已加载: {model_path}")
    print(f"  设备: {device}")
    print(f"  数据形状: {df.shape}")

    return model, df, device, scalers, lag_scaler, dataset_config


def generate_interpretation_summary(results, output_dir='./figs'):
    """生成解释摘要报告"""

    os.makedirs(output_dir, exist_ok=True)

    # 计算统计信息
    abs_errors = results['abs_errors']
    holiday_mask = results['holiday_types'] > 1

    summary = {
        '总样本数': len(abs_errors),
        '平均误差 (MAE)': abs_errors.mean(),
        '误差标准差': abs_errors.std(),
        '误差中位数': np.median(abs_errors),
        '误差90%分位': np.percentile(abs_errors, 90),
        '最大误差': abs_errors.max(),
        '最小误差': abs_errors.min(),
        '平日MAE': abs_errors[~holiday_mask].mean() if (~holiday_mask).sum() > 0 else 0,
        '节假日MAE': abs_errors[holiday_mask].mean() if holiday_mask.sum() > 0 else 0,
        '节假日样本占比': holiday_mask.sum() / len(abs_errors) * 100
    }

    # 写入报告
    report_path = f'{output_dir}/interpretation_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型解释与分析摘要报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("【基础统计】\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.2f}\n")
            else:
                f.write(f"  {key}: {value}\n")

        f.write("\n【高误差样本分析】\n")
        top_indices = np.argsort(abs_errors)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            date_str = pd.to_datetime(results['dates'][idx]).strftime('%Y-%m-%d')
            f.write(f"  {i+1}. {date_str}: 误差={results['errors'][idx]:.0f}, ")
            f.write(f"实际={results['actuals'][idx]:.0f}, 预测={results['predictions'][idx]:.0f}\n")

        f.write("\n【关键发现】\n")
        if summary['节假日MAE'] > summary['平日MAE'] * 1.5:
            f.write("  - 节假日预测误差显著高于平日，建议加强节假日特征\n")
        if summary['误差90%分位'] > summary['平均误差 (MAE)'] * 2:
            f.write("  - 存在显著离群值，建议检查极端情况的处理\n")
        if summary['误差标准差'] > summary['平均误差 (MAE)']:
            f.write("  - 误差波动较大，模型稳定性需要改进\n")

        f.write("\n【改进建议】\n")
        f.write("  1. 针对高误差日期进行特殊处理\n")
        f.write("  2. 增强节假日期间的特征表达\n")
        f.write("  3. 考虑加入外部因素（天气、事件等）\n")
        f.write("  4. 对极端值进行异常检测和处理\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"  保存: {report_path}")

    return summary


def main():
    print("=" * 70)
    print("增强评估 - 模型解释与特征重要性分析")
    print("=" * 70)

    output_dir = './figs'

    # 1. 加载模型和数据
    model, df, device, scalers, lag_scaler, dataset_config = load_model_and_data()

    # 2. 运行完整的模型解释分析
    results = run_comprehensive_interpretation(
        model, df, scalers, lag_scaler, dataset_config, device, output_dir
    )

    # 3. 生成摘要报告
    print("\n[生成摘要报告]")
    summary = generate_interpretation_summary(results, output_dir)

    # 4. 输出最终总结
    print("\n" + "=" * 70)
    print("增强评估完成！")
    print("=" * 70)
    print(f"\n【关键指标】")
    print(f"  平均误差 (MAE): {summary['平均误差 (MAE)']:.2f}")
    print(f"  平日 MAE: {summary['平日MAE']:.2f}")
    print(f"  节假日 MAE: {summary['节假日MAE']:.2f}")
    print(f"  误差90%分位: {summary['误差90%分位']:.2f}")

    print(f"\n【生成的文件】")
    print(f"  {output_dir}/feature_importance.png - 特征重要性分析")
    print(f"  {output_dir}/prediction_difference_analysis.png - 预测差异分析")
    print(f"  {output_dir}/high_error_analysis.png - 高误差样本分析")
    print(f"  {output_dir}/interpretation_summary.txt - 分析摘要报告")
    print("=" * 70)


if __name__ == '__main__':
    main()
