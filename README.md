# 🚗 交通流量预测 - V2优化版完整项目

## 📋 快速开始

### 项目概述
这是一个基于PyTorch的交通流量预测系统，采用深度学习和节假日感知的混合专家架构，能够准确预测道路交通流量，特别是在节假日期间。

**性能指标:**
- **MAE**: 496.43 (平均绝对误差)
- **MAPE**: 3.49% (平均百分比误差 - 优秀)
- **R²**: 0.3046 (决定系数)
- **节假日改进**: +30-35% (相比V1)

### 文件结构
```
project/
├── README.md (本文件)
├── main.py                          # 主训练脚本
├── main_ensemble.py                 # 集成预测脚本
├── comprehensive_evaluation.py       # 综合评估脚本
│
├── src/
│   ├── model.py                    # V2优化模型 (148K参数)
│   ├── train.py                    # V2训练模块
│   ├── feature_engineering.py      # 高级特征工程 (27个特征)
│   ├── visualization.py            # V2可视化
│   ├── ensemble_prediction.py      # 集成预测策略
│   ├── visualization_ensemble.py   # 集成对比可视化
│   ├── dataset.py                  # 数据集处理
│   └── holiday_feature.py          # 节假日特征
│
├── data/
│   ├── df_with_features.pkl        # 原始特征数据
│   └── df_with_advanced_features.pkl # 高级特征数据 (850K)
│
├── models/model_v2_output/
│   ├── best_model.pth              # 训练好的模型权重 (604K)
│   └── scalers.pkl                 # 数据归一化器
│
├── figs/                           # 输出可视化
│   ├── 01_time_series.png          # 时间序列对比
│   ├── 02_scatter.png              # 散点图
│   ├── 03_residuals.png            # 残差分析
│   ├── 04_error_analysis.png       # 误差分析
│   └── 05_performance_summary.png  # 性能总结
│
├── figs_ensemble/                  # 集成预测输出
│   ├── ensemble_comparison.png
│   ├── 劳动节_ensemble.png
│   └── 国庆节_ensemble.png
│
└── 📄 文档
    ├── EVALUATION_SUMMARY.txt           # 评估总结
    ├── COMPREHENSIVE_EVALUATION_REPORT.md # 详细报告
    ├── OPTIMIZATION_REPORT.md            # 优化历程
    └── model_evaluation_dashboard.html  # 可视化仪表板
```

---

## 🚀 使用指南

### 1. 训练模型

```bash
python main.py
```

**输出:**
- 训练好的模型: `./models/model_v2_output/best_model.pth`
- 高级特征数据: `./data/df_with_advanced_features.pkl`
- 可视化结果: `./figs/overall_performance.png` 等

**耗时:** ~5-10分钟 (M1/M2 MacBook)

### 2. 综合评估

```bash
python comprehensive_evaluation.py
```

**生成文件:**
- `./figs/01_time_series.png` - 时间序列对比
- `./figs/02_scatter.png` - 预测精度散点图
- `./figs/03_residuals.png` - 残差分析 (4子图)
- `./figs/04_error_analysis.png` - 误差分析 (4子图)
- `./figs/05_performance_summary.png` - 性能总结

### 3. 集成预测评估

```bash
python main_ensemble.py
```

**策略对比:**
- **标准预测** (stride=7): 快速但可能遗漏边界信息
- **滚动预测** (stride=1): 每个目标日预测6.8次，准确但慢
- **集成预测** (60%+40%): 平衡速度和精度

---

## 🎯 核心模型特点

### 1️⃣ 特征工程创新 (27个高级特征)

#### Lag特征
- `lag_1d`: 1日历史交通量
- `lag_7d`: 7日历史(周周期)
- `lag_14d`: 14日历史(两周周期)
- `recent_change_3d`: 最近3日变化

#### 滚动统计
- `ma_7d`, `ma_14d`, `ma_30d`: 移动平均
- `std_7d`: 波动率
- `trend_7d`: 本地趋势

#### 节假日特征
- `is_high_traffic_holiday`: 国庆/劳动节 (流量+15-17%)
- `is_low_traffic_holiday`: 春节 (流量-29%)
- `holiday_traffic_ratio`: 历史流量比率
- `holiday_day_num`: 节假日第N天
- `holiday_progress`: 节假日进度

#### 上下文特征
- `is_before_holiday_1d/2d/3d`: 节假日前3天
- `is_after_holiday_1d/2d`: 节假日后2天
- 周期特征: `dow_sin`, `dow_cos`, `month_sin`, `month_cos`

### 2️⃣ 模型架构 (混合专家 + 注意力)

```
输入: [14天历史数据] → 预测 [7天未来数据]

┌─────────────────────────────────────────┐
│         改进的编码器 (Improved Encoder)   │
├─────────────────────────────────────────┤
│  Lag特征 [50%容量] ────→ ┐              │
│  连续特征 [25%容量] ────→ ├→ [融合]     │
│  二值特征 [25%容量] ────→ ┘              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  节假日感知注意力 × 2层                  │
│  (HolidayAwareAttention)                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      自适应预测器 (Adaptive Predictor)    │
├─────────────────────────────────────────┤
│  ├─ 基础预测器 (常规日期)                │
│  ├─ 高流量专家 (国庆/劳动节)            │
│  └─ 低流量专家 (春节)                   │
│           ↓ (Gate机制 - Softmax)       │
│      最终预测 (加权平均)                 │
└─────────────────────────────────────────┘
```

**参数量:** 148K (轻量级，推理快)

### 3️⃣ 训练策略 (极端过采样 + 自适应损失)

#### 样本权重
```
春节 (25个样本) → 30.0x → 750个样本/epoch
国庆 (36个样本) → 25.0x → 900个样本/epoch
劳动节 (25个样本) → 25.0x → 625个样本/epoch
其他节假日 → 10.0x
普通日 (1700个样本) → 1.0x → 1700个样本/epoch
```

#### 损失函数权重
```python
if is_spring_festival:
    loss_weight = 20.0  # 额外20倍
elif is_high_traffic_holiday:
    loss_weight = 15.0  # 额外15倍
else:
    loss_weight = 1.0
```

#### Huber Loss
处理离群值，平衡L1和L2损失

---

## 📊 性能对比

### 版本演进
| 模型 | MAE | MAPE | 节假日MAE | 优点 |
|------|-----|------|---------|------|
| 基线 | 571.39 | ~5.0% | 1428 | 简单快速 |
| V1 | 580.15 | ~4.8% | 1316 | 初步优化 |
| **V2 (当前)** | **496.43** | **3.49%** | **903** | 混合专家+精细特征 |

### 节假日具体改进
```
        基线 → V1 → V2(当前) → 改进%
国庆节  1807 → 1800 → 1172    (-35%)
劳动节  1769 → 1620 → 1242    (-30%)
春节    ~1400 → ~1350 → 1290  (-8%)
```

---

## 🔍 评估结果详解

### ✅ 优势
1. **MAPE优秀** (3.49%)
   - 相对误差极小
   - 适合实际应用

2. **节假日改进显著** (+30-35%)
   - 混合专家有效
   - 特征工程充分

3. **时间序列跟踪准确**
   - 捕捉趋势好
   - 无严重滞后

4. **统计假设符合**
   - 残差近似正态
   - 无系统偏差

### ⚠️ 改进空间
1. **R²相对较低** (0.3046)
   - 需要外部特征
   - 可考虑更复杂架构

2. **离群值处理**
   - 最大误差可达几千
   - 需要异常检测

3. **高值预测偏差**
   - 峰值可能被低估
   - 需要特殊处理

---

## 📈 可视化解读

### 01_time_series.png
```
X轴: 时间  Y轴: 交通流量
─ 黑线: 实际值
─ 红线: 预测值
━ 橙色竖线: 节假日期间
```
**看点:** 预测线是否贴近实际值，是否有延迟

### 02_scatter.png
```
X: 实际值  Y: 预测值
点分布: 越接近45°直线越好
R²和MAE显示在图中
```
**看点:** 点的密集度，离群点位置

### 03_residuals.png (4子图)
1. **残差时间序列**: 检查系统偏差
2. **分布直方图**: 检查正态性
3. **Q-Q图**: 尾部重性检查
4. **残差vs预测**: 检查异方差性

### 04_error_analysis.png (4子图)
1. **误差时间序列**: 趋势和波动
2. **误差分布**: MAE和MAPE的分布
3. **百分位分析**: 10/25/50/75/90分位
4. **统计表**: 最小/最大/中位误差

### 05_performance_summary.png
**综合仪表板**包含:
- 关键指标卡片
- 预测精度散点
- 误差分布直方图
- 相对误差分布
- 残差分析
- 性能评级

---

## 🛠️ 进阶使用

### 修改模型参数

**编辑 `main.py`:**
```python
config = {
    'seq_len': 14,           # 输入序列长度
    'pred_len': 7,           # 预测长度
    'hidden_dim': 64,        # 隐藏维度
    'num_epochs': 200,       # 训练轮数
    'batch_size': 32,        # 批大小
    'patience': 40,          # 早停轮数
}
```

### 自定义特征

**编辑 `src/feature_engineering.py`:**
```python
def create_advanced_features(self, df):
    # 添加自定义特征
    df['your_feature'] = ...
    return df
```

### 外部特征集成

**在 `src/dataset.py` 中:**
```python
# 加载天气、事件等外部数据
external_data = pd.read_csv('weather.csv')
df = df.merge(external_data, on='date')
```

---

## 🚨 常见问题

### Q: 为什么R²这么低?
**A:** 时间序列预测本身就有较高的噪声。3.49% MAPE说明相对误差很小，更重要。

### Q: 如何改进节假日预测?
**A:** 
1. 增加节假日样本权重 (当前: 25-30x)
2. 收集更多历年节假日数据
3. 加入天气、活动等外部特征

### Q: 能用于生产环境吗?
**A:** 可以。建议:
1. A/B测试 (30天)
2. 实时监控性能指标
3. 每月更新一次模型

### Q: 如何处理预测中的离群值?
**A:** 在 `comprehensive_evaluation.py` 中:
```python
# 检测离群值
Q1 = errors.quantile(0.25)
Q3 = errors.quantile(0.75)
IQR = Q3 - Q1
outliers = (errors > Q3 + 1.5*IQR)
```

---

## 📚 文档导航

| 文档 | 内容 | 用途 |
|-----|------|------|
| `EVALUATION_SUMMARY.txt` | 评估总结 | 快速了解性能 |
| `COMPREHENSIVE_EVALUATION_REPORT.md` | 详细报告 | 深入分析评估 |
| `OPTIMIZATION_REPORT.md` | 优化历程 | 了解优化过程 |
| `model_evaluation_dashboard.html` | 交互仪表板 | 可视化展示 |

### 快速查看
```bash
# 文本总结 (1分钟)
cat EVALUATION_SUMMARY.txt

# 详细报告 (10分钟)  
cat COMPREHENSIVE_EVALUATION_REPORT.md

# HTML仪表板 (打开浏览器)
open model_evaluation_dashboard.html
```

---

## 🔧 技术栈

- **框架**: PyTorch 2.0+
- **GPU**: MPS (Apple Silicon) / CUDA
- **特征工程**: Pandas, Scikit-learn
- **可视化**: Matplotlib, Seaborn
- **数据处理**: Numpy, Scikit-learn scalers

### 环境要求
```bash
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

---

## 📞 支持与反馈

### 遇到问题?
1. 检查数据格式是否正确
2. 验证Python版本 >= 3.8
3. 确保PyTorch正确安装
4. 查看错误日志中的具体信息

### 性能下降?
1. 检查输入数据质量
2. 验证模型权重是否被修改
3. 确认是否有异常值污染
4. 重新训练以更新模型

---

## 📝 更新日志

### V2 (2025-12-11) ✨
- ✅ 混合专家架构
- ✅ 27个高级特征
- ✅ 节假日感知注意力
- ✅ 极端过采样
- ✅ 自适应损失函数
- ✅ 综合评估工具

### V1 (2025-12-10)
- 基础模型
- 初步特征工程

---

## 📄 许可证

MIT License - 自由使用和修改

---

**最后更新**: 2025-12-11 14:42 UTC  
**维护者**: Your Team  
**版本**: V2.0
