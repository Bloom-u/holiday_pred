"""
节假日特征工程 - 优化版

核心改进：
1. 区分"休息日"和"法定节假日"
2. 调休工作日不再被视为节假日
3. 更准确的节假日距离和阶段特征
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from chinese_calendar import is_holiday, is_workday, get_holiday_detail
    HAS_CHINESE_CALENDAR = True
except ImportError:
    print("请安装 chinese_calendar: pip install chinese_calendar")
    HAS_CHINESE_CALENDAR = False


# ==================== 节假日类型定义 ====================
HOLIDAY_TYPES = {
    'workday': 0,              # 普通工作日
    'weekend': 1,              # 普通周末
    "New Year's Day": 2,       # 元旦
    "Spring Festival": 3,      # 春节
    "Tomb-sweeping Day": 4,    # 清明
    "Labour Day": 5,           # 五一
    "Dragon Boat Festival": 6, # 端午
    "Mid-autumn Festival": 7,  # 中秋
    "National Day": 8,         # 国庆
    "adjusted_workday": 9,     # 调休工作日（不是节假日！）
}

# 中文节假日名称映射
HOLIDAY_CN_NAMES = {
    "New Year's Day": "元旦",
    "Spring Festival": "春节",
    "Tomb-sweeping Day": "清明节",
    "Labour Day": "劳动节",
    "Dragon Boat Festival": "端午节",
    "Mid-autumn Festival": "中秋节",
    "National Day": "国庆节",
}


class HolidayFeatureEngine:
    """节假日特征工程类"""

    def __init__(self):
        self.holiday_type_map = HOLIDAY_TYPES

    def _to_date(self, date):
        """统一转换为 date 对象"""
        if isinstance(date, str):
            return pd.to_datetime(date).date()
        elif isinstance(date, (datetime, pd.Timestamp)):
            return date.date()
        return date

    def get_holiday_info(self, date):
        """
        获取日期的节假日信息

        返回: (is_rest_day, is_statutory_holiday, holiday_name, holiday_type_code)
        - is_rest_day: 是否休息（包括周末、节假日，但不包括调休工作日）
        - is_statutory_holiday: 是否法定节假日（不包括普通周末）
        - holiday_name: 节假日名称
        - holiday_type_code: 类型编码
        """
        date = self._to_date(date)

        try:
            is_off, holiday_name = get_holiday_detail(date)
        except:
            is_weekend = date.weekday() >= 5
            return is_weekend, False, None, 1 if is_weekend else 0

        # 核心逻辑：区分不同情况
        if is_off:  # 休息日
            if holiday_name:  # 法定节假日
                type_code = self.holiday_type_map.get(holiday_name, 10)
                cn_name = HOLIDAY_CN_NAMES.get(holiday_name, holiday_name)
                return True, True, cn_name, type_code
            else:  # 普通周末
                return True, False, None, self.holiday_type_map['weekend']
        else:  # 工作日
            if holiday_name:  # 调休工作日
                # 关键：调休工作日不是"节假日"，也不是"休息日"
                return False, False, None, self.holiday_type_map['adjusted_workday']
            else:  # 普通工作日
                return False, False, None, self.holiday_type_map['workday']

    def _find_holiday_boundary(self, date, holiday_name, direction='backward'):
        """
        查找某个节假日的边界日期
        direction: 'backward' 找起始日, 'forward' 找结束日
        """
        date = self._to_date(date)
        delta = -1 if direction == 'backward' else 1
        boundary = date

        check = date + timedelta(days=delta)
        for _ in range(30):  # 最多查30天
            try:
                is_off, name = get_holiday_detail(check)
                # 只找连续的同名节假日
                if is_off and name == holiday_name:
                    boundary = check
                    check = check + timedelta(days=delta)
                else:
                    break
            except:
                break

        return boundary

    def _find_nearest_statutory_holiday(self, date, direction='forward', max_days=60):
        """
        查找最近的法定节假日（排除普通周末和调休工作日）
        返回: (天数, 节假日类型编码, 节假日名称)
        """
        date = self._to_date(date)
        delta = 1 if direction == 'forward' else -1

        for i in range(0, max_days + 1):
            check_date = date + timedelta(days=i * delta)
            try:
                is_off, holiday_name = get_holiday_detail(check_date)
                # 只找法定节假日，不包括普通周末
                if is_off and holiday_name:
                    type_code = self.holiday_type_map.get(holiday_name, 10)
                    cn_name = HOLIDAY_CN_NAMES.get(holiday_name, holiday_name)
                    return i, type_code, cn_name
            except:
                continue

        return max_days, 0, None

    def create_features(self, df, date_column='日期'):
        """生成所有节假日特征"""
        result = df.copy()
        result[date_column] = pd.to_datetime(result[date_column])

        print("正在生成节假日特征...")

        # ========== 1. 基础时间特征 ==========
        print("  [1/5] 基础时间特征...")
        result['year'] = result[date_column].dt.year
        result['month'] = result[date_column].dt.month - 1  # 0-11
        result['day'] = result[date_column].dt.day
        result['day_of_week'] = result[date_column].dt.dayofweek  # 0-6
        result['day_of_year'] = result[date_column].dt.dayofyear
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)

        # ========== 2. 周期性编码 ==========
        print("  [2/5] 周期性编码...")
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        result['doy_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
        result['doy_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)

        # ========== 3. 节假日核心特征（重点改进）==========
        print("  [3/5] 节假日核心特征...")
        holiday_info = result[date_column].apply(self.get_holiday_info)

        # 是否法定节假日（不包括普通周末，不包括调休工作日）
        result['is_holiday'] = holiday_info.apply(lambda x: int(x[1]))

        # 是否法定节假日（与is_holiday相同，保持向后兼容）
        result['is_statutory_holiday'] = holiday_info.apply(lambda x: int(x[1]))

        # 节假日名称
        result['holiday_name'] = holiday_info.apply(lambda x: x[2] if x[2] else 'none')

        # 节假日类型编码
        result['holiday_type'] = holiday_info.apply(lambda x: x[3])

        # 是否调休工作日
        result['is_adjusted_workday'] = (result['holiday_type'] == 9).astype(int)

        # ========== 4. 节假日距离特征 ==========
        print("  [4/5] 节假日距离特征...")
        distance_features = result[date_column].apply(self._compute_distance_features)
        result['days_to_next_holiday'] = distance_features.apply(lambda x: x[0])
        result['next_holiday_type'] = distance_features.apply(lambda x: x[1])
        result['next_holiday_name'] = distance_features.apply(lambda x: x[2])

        result['days_from_prev_holiday'] = distance_features.apply(lambda x: x[3])
        result['prev_holiday_type'] = distance_features.apply(lambda x: x[4])
        result['prev_holiday_name'] = distance_features.apply(lambda x: x[5])

        result['days_to_nearest_holiday'] = result[['days_to_next_holiday', 'days_from_prev_holiday']].min(axis=1)
        result['holiday_proximity'] = np.exp(-result['days_to_nearest_holiday'] / 7)

        # ========== 5. 节假日阶段特征 + 连续假期特征 ==========
        print("  [5/5] 节假日阶段与连续假期特征...")
        phase_features = result.apply(
            lambda row: self._compute_phase_features(
                row[date_column],
                row['holiday_name'],
                row['is_statutory_holiday']
            ),
            axis=1
        )
        result['holiday_phase'] = phase_features.apply(lambda x: x[0])
        result['holiday_day_num'] = phase_features.apply(lambda x: x[1])
        result['total_holiday_length'] = phase_features.apply(lambda x: x[2])
        result['holiday_progress'] = phase_features.apply(lambda x: x[3])

        # ========== 数据验证 ==========
        print("\n✅ 特征生成完成！正在验证数据...")
        self._validate_features(result)

        return result

    def _validate_features(self, df):
        """验证生成的特征是否符合预期范围"""
        print("\n数据验证：")
        print(f"  总样本数: {len(df)}")
        print(f"  休息日数量: {df['is_holiday'].sum()} ({df['is_holiday'].mean()*100:.1f}%)")
        print(f"  法定节假日数量: {df['is_statutory_holiday'].sum()} ({df['is_statutory_holiday'].mean()*100:.1f}%)")
        print(f"  调休工作日数量: {df['is_adjusted_workday'].sum()} ({df['is_adjusted_workday'].mean()*100:.1f}%)")
        print(f"  普通周末数量: {((df['is_weekend'] == 1) & (df['is_statutory_holiday'] == 0)).sum()}")
        print(f"  普通工作日数量: {((df['is_holiday'] == 0) & (df['is_adjusted_workday'] == 0)).sum()}")

        # 节假日类型分布
        print("\n节假日类型分布:")
        for htype, count in df['holiday_type'].value_counts().sort_index().items():
            type_name = [k for k, v in HOLIDAY_TYPES.items() if v == htype]
            type_name = type_name[0] if type_name else f"未知({htype})"
            print(f"  {htype:2d} - {type_name:20s}: {count:4d} 天")

        # 范围检查
        checks = {
            'month': (0, 11, "月份"),
            'day_of_week': (0, 6, "星期"),
            'holiday_type': (0, 10, "节假日类型"),
        }

        print("\n范围验证:")
        all_valid = True
        for col, (min_val, max_val, desc) in checks.items():
            if col in df.columns:
                actual_min = df[col].min()
                actual_max = df[col].max()

                status = "✅" if (actual_min >= min_val and actual_max <= max_val) else "⚠️"
                print(f"  {status} {desc:15s}: [{actual_min:2d}, {actual_max:2d}] (期望: [{min_val:2d}, {max_val:2d}])")

                if df[col].isna().any():
                    print(f"  ❌ {desc} 包含 {df[col].isna().sum()} 个 NaN 值")
                    all_valid = False

        if all_valid:
            print("\n  ✅ 所有特征验证通过！")

    def _compute_distance_features(self, date):
        """计算距离节假日的天数和类型"""
        days_to_next, next_type, next_name = self._find_nearest_statutory_holiday(date, 'forward')
        days_from_prev, prev_type, prev_name = self._find_nearest_statutory_holiday(date, 'backward')

        return (
            days_to_next, next_type, next_name if next_name else 'none',
            days_from_prev, prev_type, prev_name if prev_name else 'none'
        )

    def _compute_phase_features(self, date, holiday_name, is_statutory_holiday):
        """
        计算节假日阶段和连续假期特征

        返回: (phase, day_num, total_length, progress)
        phase:
          0  - 法定节假日期间
          -1 - 节前3天
          -2 - 节前一周
          1  - 节后3天
          2  - 节后一周
          99 - 普通日
        """
        date = self._to_date(date)

        # 如果是法定节假日（不包括普通周末和调休工作日）
        if is_statutory_holiday and holiday_name and holiday_name != 'none':
            # 找到这个节假日的起始和结束日期
            start_date = self._find_holiday_boundary(date, holiday_name, 'backward')
            end_date = self._find_holiday_boundary(date, holiday_name, 'forward')

            total_length = (end_date - start_date).days + 1
            day_num = (date - start_date).days + 1
            progress = day_num / total_length if total_length > 0 else 0

            return (0, day_num, total_length, progress)

        # 非法定节假日，计算阶段
        days_to_next, _, _ = self._find_nearest_statutory_holiday(date, 'forward')
        days_from_prev, _, _ = self._find_nearest_statutory_holiday(date, 'backward')

        if days_to_next <= 3:
            phase = -1  # 节前3天
        elif days_to_next <= 7:
            phase = -2  # 节前一周
        elif days_from_prev <= 3:
            phase = 1   # 节后3天
        elif days_from_prev <= 7:
            phase = 2   # 节后一周
        else:
            phase = 99  # 普通日

        return (phase, 0, 0, 0)

    def get_feature_summary(self):
        """返回特征说明"""
        return {
            '基础时间特征': [
                'year',
                'month (0-11)',
                'day',
                'day_of_week (0-6, 0=周一)',
                'day_of_year',
                'is_weekend'
            ],
            '周期性编码': [
                'dow_sin', 'dow_cos',
                'month_sin', 'month_cos',
                'doy_sin', 'doy_cos'
            ],
            '节假日核心特征 (优化)': [
                'holiday_type (0-10)',
                'is_holiday (休息日，包括周末和法定节假日)',
                'is_statutory_holiday (仅法定节假日)',
                'is_adjusted_workday (调休工作日，不是节假日！)',
                'holiday_name'
            ],
            '节假日距离特征': [
                'days_to_next_holiday',
                'next_holiday_type',
                'next_holiday_name',
                'days_from_prev_holiday',
                'prev_holiday_type',
                'prev_holiday_name',
                'days_to_nearest_holiday',
                'holiday_proximity'
            ],
            '节假日阶段特征': [
                'holiday_phase (-2,-1,0,1,2,99)',
            ],
            '连续假期特征': [
                'holiday_day_num (节假日第几天)',
                'total_holiday_length (整个假期总长度)',
                'holiday_progress (假期进度 0-1)'
            ]
        }


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 读取真实数据（参照 main.ipynb）
    import os
    # 确保路径正确（支持从 src/ 或项目根目录运行）
    data_file = 'data/1.5 2020-2024日变化特征-分行政等级客货汽加权当量分日-干线公路.xlsx'
    if not os.path.exists(data_file):
        data_file = '../' + data_file

    raw_data = pd.read_excel(data_file)
    raw_data = raw_data.rename(columns={
        'GCRQ': '日期',
        'KC_DL': '客车当量',
        'HC_DL': '货车当量',
        'KC_CS': '客车车速',
        'HC_CS': '货车车速',
        'JDC_DL': '机动车当量',
        'JDC_CS': '机动车车速',
        'VC': 'VC'
    })
    raw_data['日期'] = pd.to_datetime(raw_data['日期'], format='%Y-%m-%d')
    df = raw_data[['日期', '客车当量', '货车当量', '客车车速', '货车车速', '机动车当量', '机动车车速', 'VC']]

    # 生成特征
    engine = HolidayFeatureEngine()
    df_result = engine.create_features(df, date_column='日期')

    # 打印特征说明
    print("\n" + "="*60)
    print("特征一览：")
    print("="*60)
    for group, features in engine.get_feature_summary().items():
        print(f"\n【{group}】")
        for f in features:
            print(f"  - {f}")

    # 查看国庆期间数据（包含调休）
    print("\n" + "="*60)
    print("2023国庆期间示例（含调休）：")
    print("="*60)
    national_day = df_result[
        (df_result['日期'] >= '2023-09-28') &
        (df_result['日期'] <= '2023-10-08')
    ]

    cols = ['日期', 'day_of_week', 'is_weekend', 'is_holiday', 'is_statutory_holiday',
            'is_adjusted_workday', 'holiday_name', 'holiday_type',
            'holiday_day_num', 'total_holiday_length']

    print(national_day[cols].to_string(index=False))

    print("\n说明：")
    print("  - is_holiday=1: 法定节假日（不包括普通周末）")
    print("  - is_statutory_holiday=1: 法定节假日（与is_holiday相同）")
    print("  - is_adjusted_workday=1: 调休工作日（虽然可能是周末，但要上班）")

    # 保存处理后的数据
    import pickle
    save_file = 'data/df_with_features.pkl'
    if not os.path.exists('data'):
        save_file = '../' + save_file

    with open(save_file, 'wb') as f:
        pickle.dump(df_result, f)
    print(f"\n✅ 已保存特征数据到: {save_file}")
