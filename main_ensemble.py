"""
集成预测主脚本
对比三种预测策略的效果
"""

import sys
sys.path.insert(0, './src')

import pandas as pd
import pickle
import torch
from model import ImprovedTrafficModel
from ensemble_prediction import sliding_window_predict_ensemble, compare_prediction_strategies
from visualization_ensemble import plot_ensemble_comparison, plot_holiday_specific_ensemble


def main():
    print("=" * 70)
    print("集成预测评估 - 对比三种预测策略")
    print("=" * 70)

    # 1. 加载数据和模型
    print("\n[1/4] 加载数据和模型...")
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
    print(f"  模型已加载: {model_path}")
    print(f"  设备: {device}")

    # 2. 集成预测
    print("\n[2/4] 执行集成预测...")
    print("  注意：滚动预测会比较慢，因为要对每个目标日进行多次预测")

    start_idx = int(len(df) * 0.85)  # 测试集起始位置
    results = sliding_window_predict_ensemble(
        model, df, scalers, lag_scaler, dataset_config, device,
        start_idx=start_idx
    )

    # 3. 对比分析
    print("\n[3/4] 对比分析...")
    results = compare_prediction_strategies(results)

    # 4. 可视化
    print("\n[4/4] 生成可视化...")

    # 整体对比
    print("  [4.1] 绘制整体对比图...")
    plot_ensemble_comparison(results, output_dir='./figs_ensemble')

    # 特定节假日对比
    print("  [4.2] 绘制劳动节对比...")
    plot_holiday_specific_ensemble(results, '劳动节', output_dir='./figs_ensemble')

    print("  [4.3] 绘制国庆节对比...")
    plot_holiday_specific_ensemble(results, '国庆节', output_dir='./figs_ensemble')

    print("\n" + "=" * 70)
    print("✅ 集成预测评估完成！")
    print("=" * 70)
    print(f"\n【文件位置】")
    print(f"  可视化结果: ./figs_ensemble/")
    print(f"    - ensemble_comparison.png  (三种策略整体对比)")
    print(f"    - 劳动节_ensemble.png      (劳动节详细对比)")
    print(f"    - 国庆节_ensemble.png      (国庆节详细对比)")
    print("=" * 70)


if __name__ == '__main__':
    main()
