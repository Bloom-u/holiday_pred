"""
增强特征工程模块 V2
专注于提升节假日预测性能
"""

import pandas as pd
import numpy as np
from typing import Dict


class AdvancedFeatureEngine:
    """高级特征工程引擎"""

    def __init__(self):
        # 节假日历史流量统计（基于2020-2024数据的先验知识）
        self.holiday_stats = {
            'Spring Festival': {'mean_ratio': 0.71, 'is_high_traffic': False},  # 春节大幅下降
            'National Day': {'mean_ratio': 1.17, 'is_high_traffic': True},      # 国庆大幅上升
            'Labour Day': {'mean_ratio': 1.15, 'is_high_traffic': True},        # 劳动节上升
            'Tomb-sweeping Day': {'mean_ratio': 1.03, 'is_high_traffic': False}, # 清明略升
            'Dragon Boat Festival': {'mean_ratio': 1.07, 'is_high_traffic': False}, # 端午略升
            'Mid-autumn Festival': {'mean_ratio': 1.03, 'is_high_traffic': False}, # 中秋略升
            "New Year's Day": {'mean_ratio': 0.95, 'is_high_traffic': False},   # 元旦略降
        }

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建高级特征

        Args:
            df: 必须包含基础特征的DataFrame（来自holiday_feature.py）

        Returns:
            添加了高级特征的DataFrame
        """
        result = df.copy()

        print("\n正在生成高级特征...")

        # ========== 1. 节假日类型细分 ==========
        print("  [1/7] 节假日类型细分...")
        result['is_high_traffic_holiday'] = result['holiday_name'].apply(
            lambda x: 1 if x in ['国庆节', '劳动节'] else 0
        )
        result['is_low_traffic_holiday'] = result['holiday_name'].apply(
            lambda x: 1 if x == '春节' else 0
        )

        # 节假日历史流量比例特征
        result['holiday_traffic_ratio'] = result['holiday_name'].apply(
            lambda x: self._get_holiday_ratio(x)
        )

        # ========== 2. 时序滚动特征 ==========
        print("  [2/7] 时序滚动特征...")
        target_col = '机动车当量'

        # 短期趋势（7天）
        result['ma_7d'] = result[target_col].rolling(window=7, min_periods=1).mean()
        result['std_7d'] = result[target_col].rolling(window=7, min_periods=1).std().fillna(0)

        # 中期趋势（14天）
        result['ma_14d'] = result[target_col].rolling(window=14, min_periods=1).mean()

        # 长期趋势（30天）
        result['ma_30d'] = result[target_col].rolling(window=30, min_periods=1).mean()

        # 趋势特征
        result['trend_7d'] = result['ma_7d'] - result['ma_7d'].shift(7)
        result['trend_7d'] = result['trend_7d'].fillna(0)

        # ========== 3. 滞后特征（Lag Features）==========
        print("  [3/7] 滞后特征...")

        # 昨天的流量
        result['lag_1d'] = result[target_col].shift(1).fillna(method='bfill')

        # 上周同一天的流量（周期性）
        result['lag_7d'] = result[target_col].shift(7).fillna(method='bfill')

        # 两周前同一天
        result['lag_14d'] = result[target_col].shift(14).fillna(method='bfill')

        # 最近3天的平均变化
        result['recent_change_3d'] = (
            (result[target_col] - result[target_col].shift(1)) +
            (result[target_col].shift(1) - result[target_col].shift(2)) +
            (result[target_col].shift(2) - result[target_col].shift(3))
        ) / 3
        result['recent_change_3d'] = result['recent_change_3d'].fillna(0)

        # ========== 4. 节假日前后效应 ==========
        print("  [4/7] 节假日前后效应...")

        # 节假日前的"预热"效应（提前1-3天）
        result['is_before_holiday_1d'] = (result['days_to_next_holiday'] == 1).astype(int)
        result['is_before_holiday_2d'] = (result['days_to_next_holiday'] == 2).astype(int)
        result['is_before_holiday_3d'] = (result['days_to_next_holiday'] == 3).astype(int)

        # 节假日后的"余热"效应（之后1-2天）
        result['is_after_holiday_1d'] = (result['days_from_prev_holiday'] == 1).astype(int)
        result['is_after_holiday_2d'] = (result['days_from_prev_holiday'] == 2).astype(int)

        # ========== 5. 节假日强度特征 ==========
        print("  [5/7] 节假日强度特征...")

        # 连续假期强度（假期越长，影响越大）
        result['holiday_intensity'] = result['total_holiday_length'] * result['is_holiday']

        # 节假日位置特征（开始/中间/结束）
        result['is_holiday_start'] = ((result['holiday_day_num'] == 1) & (result['is_holiday'] == 1)).astype(int)
        result['is_holiday_end'] = ((result['holiday_progress'] == 1.0) & (result['is_holiday'] == 1)).astype(int)
        result['is_holiday_middle'] = ((result['holiday_progress'] > 0.3) &
                                       (result['holiday_progress'] < 0.7) &
                                       (result['is_holiday'] == 1)).astype(int)

        # ========== 6. 年度趋势特征 ==========
        print("  [6/7] 年度趋势特征...")

        # 年份归一化（考虑年度变化）
        result['year_normalized'] = (result['year'] - result['year'].min()) / max(1, (result['year'].max() - result['year'].min()))

        # 是否2022年（COVID异常年）
        result['is_2022'] = (result['year'] == 2022).astype(int)

        # ========== 7. 组合特征 ==========
        print("  [7/7] 组合特征...")

        # 节假日 × 周期性
        result['holiday_dow_sin'] = result['is_holiday'] * result['dow_sin']
        result['holiday_dow_cos'] = result['is_holiday'] * result['dow_cos']

        # 高流量节假日 × 假期长度
        result['high_traffic_intensity'] = result['is_high_traffic_holiday'] * result['total_holiday_length']

        # 低流量节假日 × 假期进度
        result['low_traffic_progress'] = result['is_low_traffic_holiday'] * result['holiday_progress']

        print(f"✅ 高级特征生成完成！新增 {len(result.columns) - len(df.columns)} 个特征")

        return result

    def _get_holiday_ratio(self, holiday_name: str) -> float:
        """获取节假日历史流量比例"""
        if holiday_name == 'none' or pd.isna(holiday_name):
            return 1.0

        # 中文到英文映射
        name_map = {
            '春节': 'Spring Festival',
            '国庆节': 'National Day',
            '劳动节': 'Labour Day',
            '清明节': 'Tomb-sweeping Day',
            '端午节': 'Dragon Boat Festival',
            '中秋节': 'Mid-autumn Festival',
            '元旦': "New Year's Day"
        }

        eng_name = name_map.get(holiday_name, holiday_name)
        return self.holiday_stats.get(eng_name, {}).get('mean_ratio', 1.0)

    def get_feature_groups(self) -> Dict[str, list]:
        """返回特征分组（用于模型配置）"""
        return {
            'lag_features': [
                'lag_1d', 'lag_7d', 'lag_14d', 'recent_change_3d'
            ],
            'trend_features': [
                'ma_7d', 'ma_14d', 'ma_30d', 'std_7d', 'trend_7d'
            ],
            'holiday_type_features': [
                'is_high_traffic_holiday', 'is_low_traffic_holiday',
                'holiday_traffic_ratio', 'holiday_intensity'
            ],
            'holiday_context_features': [
                'is_before_holiday_1d', 'is_before_holiday_2d', 'is_before_holiday_3d',
                'is_after_holiday_1d', 'is_after_holiday_2d',
                'is_holiday_start', 'is_holiday_end', 'is_holiday_middle'
            ],
            'temporal_features': [
                'year_normalized', 'is_2022'
            ],
            'interaction_features': [
                'holiday_dow_sin', 'holiday_dow_cos',
                'high_traffic_intensity', 'low_traffic_progress'
            ]
        }


if __name__ == '__main__':
    # 测试
    import os
    data_file = 'data/df_with_features.pkl'
    if not os.path.exists(data_file):
        data_file = '../' + data_file
    df = pd.read_pickle(data_file)

    engine = AdvancedFeatureEngine()
    df_enhanced = engine.create_advanced_features(df)

    print("\n" + "="*60)
    print("特征统计：")
    print("="*60)
    print(f"原始特征数: {len(df.columns)}")
    print(f"增强后特征数: {len(df_enhanced.columns)}")
    print(f"新增特征数: {len(df_enhanced.columns) - len(df.columns)}")

    # 显示部分数据
    print("\n新增特征示例（国庆期间）：")
    national_day = df_enhanced[
        (df_enhanced['日期'] >= '2023-09-28') &
        (df_enhanced['日期'] <= '2023-10-08')
    ]

    display_cols = ['日期', 'is_high_traffic_holiday', 'holiday_traffic_ratio',
                   'is_before_holiday_1d', 'is_holiday_start', 'holiday_intensity']
    print(national_day[display_cols].to_string(index=False))

    # 保存
    save_file = 'data/df_with_advanced_features.pkl'
    if not os.path.exists('data'):
        save_file = '../' + save_file
    df_enhanced.to_pickle(save_file)
    print(f"\n✅ 已保存到: {save_file}")
