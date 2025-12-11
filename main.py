"""
优化版主训练脚本 V2
"""

import sys
sys.path.insert(0, './src')

import pandas as pd
import pickle
from feature_engineering import AdvancedFeatureEngine
from train import train_model_v2
from visualization import generate_all_visualizations_v2


def main():
    print("=" * 70)
    print("交通流量预测 V2 - 优化版训练与评估")
    print("=" * 70)

    # 1. 生成高级特征
    print("\n[1/4] 生成高级特征...")
    df_base = pd.read_pickle('./data/df_with_features.pkl')

    engine = AdvancedFeatureEngine()
    df = engine.create_advanced_features(df_base)

    print(f"  原始特征: {len(df_base.columns)}")
    print(f"  增强特征: {len(df.columns)} (+{len(df.columns) - len(df_base.columns)})")

    # 保存增强特征数据
    df.to_pickle('./data/df_with_advanced_features.pkl')

    # 2. 训练优化模型
    print("\n[2/4] 训练优化模型...")
    config = {
        'save_dir': './models/model_v2_output',
        'seq_len': 14,
        'pred_len': 7,
        'num_epochs': 200,
        'hidden_dim': 64,      # 增加模型容量
        'batch_size': 32,
        'patience': 40         # 增加耐心值
    }

    result = train_model_v2(df, **config)

    # 3. 可视化评估
    print("\n[3/4] 生成可视化...")

    # 加载缩放器
    with open(f"{config['save_dir']}/scalers.pkl", 'rb') as f:
        scalers_dict = pickle.load(f)

    # 为可视化准备scalers
    scalers = {
        'target_scaler': scalers_dict['target_scaler'],
        'cont_scaler': scalers_dict['cont_scaler']
    }
    lag_scaler = scalers_dict['lag_scaler']

    # 数据集配置（V2版本）
    dataset_config = {
        'seq_len': config['seq_len'],
        'pred_len': config['pred_len'],
        'target_col': '机动车当量',
        'lag_cols': result['dataset_config']['lag_cols'],
        'continuous_cols': result['dataset_config']['continuous_cols'],
        'binary_cols': result['dataset_config']['binary_cols'],
        'holiday_feature_cols': result['dataset_config']['holiday_feature_cols']
    }

    # 生成可视化（V2版本）
    viz_results = generate_all_visualizations_v2(
        result['model'],
        df,
        scalers,
        lag_scaler,
        dataset_config,
        result['device'],
        output_dir='./figs',
        test_ratio=0.15
    )

    # 4. 性能总结
    print("\n[4/4] 性能对比...")
    print("\n" + "=" * 70)
    print("✅ V2优化版训练完成！")
    print("=" * 70)
    print(f"\n【模型性能】")
    print(f"  Baseline MAE:  {result['baseline_mae']:.2f}")
    print(f"  模型 MAE:      {result['test_metrics']['overall_mae']:.2f}")
    improvement = (1 - result['test_metrics']['overall_mae'] / result['baseline_mae']) * 100
    print(f"  改进幅度:      {improvement:.1f}%")

    if 'holiday_mae' in result['test_metrics']:
        print(f"\n【节假日性能】")
        print(f"  节假日 MAE:    {result['test_metrics']['holiday_mae']:.2f}")
    if 'spring_mae' in result['test_metrics']:
        print(f"  春节 MAE:      {result['test_metrics']['spring_mae']:.2f}")
    if 'national_mae' in result['test_metrics']:
        print(f"  国庆 MAE:      {result['test_metrics']['national_mae']:.2f}")

    print(f"\n【文件位置】")
    print(f"  模型:          {config['save_dir']}/best_model.pth")
    print(f"  可视化:        ./figs/")
    print(f"  增强特征数据:   ./data/df_with_advanced_features.pkl")
    print("=" * 70)


if __name__ == '__main__':
    main()
