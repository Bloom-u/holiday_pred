# 当前 GBDT（XGBoost）交通流量预测模型：详尽介绍文档

本文档面向“阅读代码/复现实验”的使用场景，描述当前分支的端到端实现：**数据读取 → 特征工程（节假日 + 春节窗口 + 动态时序）→ t-2 分解建模（baseline + uplift）→ 评估与可视化 → 产物与接口**。

理论基础与建模动机见：`docs/当前GBDT交通流量预测模型理论基础.md`。

---

## 1. 适用范围与设计目标

适用范围：
- 输入为单序列日频数据（全国每日一个值），字段为 `date`（日期）与 `y`（流量）。
- 默认数据文件：`data/new_data.xlsx`（见 `src/gbdt/config.py`）。
- 模型重点：春节窗口（CNY-25 ~ CNY+15）的形态与误差控制，同时兼顾全年指标。

设计目标：
- **无信息泄露**：训练/测试严格按年份拆分；动态特征用 `shift(delay)` 避免同日泄露；t-2 预测严格只用到 t-2 及之前真实值。
- **聚焦分解建模**：输出 baseline（常态水平）与 uplift（节假日/春运增量），并组合为最终预测。

非目标（当前版本不提供）：
- 多维多路段（多序列）统一建模
- 概率预测与置信区间
- 在线学习/增量训练

---

## 2. 代码结构与主要入口

核心目录：
- `src/gbdt/`：训练、可视化、特征与工具
- `src/holiday_feature.py`：节假日特征引擎（基于 `chinese_calendar`）

主要入口：
- 训练：`python -m src.gbdt.train_decompose`
- 可视化：`python -m src.gbdt.visualize`

---

## 3. 数据输入输出约定

### 3.1 输入格式

默认输入：`data/new_data.xlsx`。

读取逻辑：`src/gbdt/data.py`
- 第一列 → `date`（可被 `pandas.to_datetime` 解析）
- 第二列 → `y`（连续数值）
- 按日期排序后重置索引（以确保后续 shift/rolling 正确）

建议：
- 至少覆盖 2023/2024/2025 三个年份：2023 训练，2024 验证/校准，2025 测试/展示（默认约定）。
- 允许存在缺失，但缺失会在动态特征构建/滚动窗口时传播为 NaN，从而减少可用训练样本。

### 3.2 训练输出（产物）

分解训练（`src/gbdt/train_decompose.py`）会产生：
  - 模型（均为 t-2 / delay=2）：
    - `models/gbdt/best_model_tminus2_baseline.json`
    - `models/gbdt/best_model_tminus2_baseline_cf.json`
    - `models/gbdt/best_model_tminus2_uplift.json`
    - `models/gbdt/best_model_tminus2_uplift_cny.json`
    - `models/gbdt/best_model_tminus2_uplift_holiday.json`
  - 统计量：`models/gbdt/group_stats.json`
- 预测结果：
  - `data/pred_2025_optimized.csv`

`data/pred_2025_optimized.csv` 字段（分解版本）：
- `date`：日期
- `y`：真实值
- `pred_tminus2_baseline`：常态基线（baseline_normal）
- `pred_tminus2_baseline_cf`：反事实基线（baseline_cf）
- `pred_tminus2_uplift` / `pred_tminus2_uplift_cny` / `pred_tminus2_uplift_holiday`：增量项
- `pred_tminus2_decomposed`：最终组合预测

---

## 4. 特征工程（Feature Engineering）

特征由三部分构成：
1) 节假日/日历基础特征（日期 → 特征）
2) 春节窗口特征（日期 → days_to_cny 等）
3) 动态时序特征（历史 y → lag/rolling/trend）
4) 训练集统计聚合（group stats）

最终特征矩阵构建入口：`src/gbdt/features.py:build_feature_matrix(...)`。

### 4.1 HolidayFeatureEngine（日历与节假日特征）

实现：`src/holiday_feature.py`，依赖 `chinese_calendar`。

核心思想：
- 明确区分：工作日/周末/法定节假日/调休工作日。
- 输出大量“距离/阶段/周期”特征，帮助模型学习非平稳结构。

输出特征（按类别概括）：
- 基础时间：`year/month/day/day_of_week/day_of_year/is_weekend`
- 周期编码：`dow_sin/dow_cos/month_sin/month_cos/doy_sin/doy_cos`
- 节假日核心：`is_holiday/is_statutory_holiday/holiday_type/is_adjusted_workday`
- 距离类：`days_to_next_holiday/next_holiday_type/days_from_prev_holiday/.../holiday_proximity`
- 阶段类：`holiday_phase/holiday_day_num/total_holiday_length/holiday_progress`

### 4.2 春节窗口（CNY）派生特征

春节日期表：`src/gbdt/config.py` 中 `CNY_DATES`。

由 `src/gbdt/features.py:add_cny_features` 生成：
- `days_to_cny`：距离当年春节的天数（春节当天=0；节前为负；节后为正）
- `cny_window`：是否处于 [-25,+15]
- `cny_pre`：节前倒数天数（仅窗口内；节前用正数编码）
- `cny_post`：节后天数（仅窗口内）
- `cny_day`：是否春节当天

这些特征的作用：
- 给模型一个“春运窗口坐标系”，更容易学到峰值时点与非对称形态。

### 4.3 Group Stats（训练集统计先验）

实现：`src/gbdt/features.py:compute_group_stats/apply_group_stats`，持久化见 `src/gbdt/persistence.py`。

统计对象：仅在指定训练掩码（例如 2023，或 2023+2024）上统计，然后映射到全量样本。

统计项：
- `overall_mean/overall_std`
- `dow_mean/dow_std`（按 `day_of_week`）
- `month_mean/month_std`（按 `month`）
- `holiday_type_mean/holiday_type_std`（按 `holiday_type`）
- `cny_offset_mean`（按 `days_to_cny`，仅 [-25,+15] 范围内）

作用：
- 把“类别基线水平/波动”作为弱先验输入模型，让树更稳定、更易泛化。
- `cny_offset_mean` 相当于“历史平均春运形状基线”，能帮助对齐春节窗口的整体轮廓。

泄露风险说明：
- `cny_offset_mean` 是用训练掩码内的 y 统计得到的，只要训练掩码不包含 2025，就不会泄露 2025 信息。

### 4.4 动态时序特征（Dynamic Features）

动态列定义：`src/gbdt/config.py` 的 `DYNAMIC_COLS`（共 19 个）。

构建实现：`src/gbdt/features.py:build_dynamic_features(series, dynamic_cols, delay)`。

关键参数 `delay`：
- `delay=1`：表示“预测 t 时，最多用到 t-1 的真实 y”。
- `delay=2`：表示“预测 t 时，最多用到 t-2 的真实 y”（更贴近业务中数据落地延迟）。

特征项（含作用）：
- 滞后：`lag_1/2/3/7/14/21/28`  
  - 作用：提供短期惯性与周周期（lag_7）以及多周周期信息。
- 滚动均值：`roll_7/14/30`  
  - 作用：平滑噪声，构建“最近水平基线”。
- 滚动波动：`std_7/std_14`  
  - 作用：表达近期不确定性（波动越大越难预测）。
- 周趋势：`trend_7 = roll_7 - roll_7.shift(7)`  
  - 作用：周级别上升/下降趋势。
- 线性斜率：`slope_7/slope_14`  
  - 作用：更连续地表达“近期线性趋势强度”，增强趋势敏感性。
- 趋势加速度：`accel_7 = slope_7 - slope_7.shift(7)`  
  - 作用：趋势是否在加速（例如春运前的加速上行）。
- 短期变化：`recent_change_3d`  
  - 作用：近 3 天平均日变化速率（动量信号）。
- 偏离度：`delta_vs_roll7` / `delta_vs_lag7`  
  - 作用：是否偏离近期均值、是否相对上周同一天偏离（用于识别回归/突变）。

关于 NaN：
- 因为 `roll_30` 等需要足够历史，序列前 30 天会出现 NaN；训练时通过 `dropna()` 自动剔除不可用样本。

---

## 5. t-2 信息约束（delay=2）

本分支的所有建模都按 t-2 信息可得约束实现：
- 离线特征矩阵：动态特征通过 `shift(delay)` 构造，分解建模固定 `delay=2`（见 `src/gbdt/features.py`）。
- 滚动一步预测：对目标日 t 的动态特征仅使用到 `t-2` 的真实 y（见 `src/gbdt/forecasting.py:predict_tminus2_series`）。

---

## 7. 模型定义（XGBoost / GBDT）

实现：`src/gbdt/model.py`（`XGBRegressor`）。

关键配置：
- `tree_method="hist"`（CPU/GPU 通用的 histogram 算法）
- `device` 自动选择：如果本地 xgboost 编译带 CUDA 则用 `cuda`，否则 `cpu`
- 可用环境变量覆盖：`XGB_DEVICE=cpu|cuda`
- 默认 objective：`reg:absoluteerror`（直接优化 MAE）

---

## 7. 训练流程（无参数泄露）

训练入口：`python -m src.gbdt.train_decompose`。

年份拆分（默认约定）：
- 2023+2024：训练（用于拟合 baseline 与 uplift）
- 2025：测试/展示（只做最终评估与输出）

### 7.1 样本权重（强化春运窗口）

实现：`src/gbdt/features.py:make_sample_weight`
- 春运窗口（days_to_cny ∈ [-25,+15]）加权
- 核心窗口（days_to_cny ∈ [-3,+3]）进一步加权
- 权重对 `(w_window, w_core)` 在 `WEIGHT_GRID` 中网格搜索

### 7.2 分解训练与组合输出

实现：`src/gbdt/train_decompose.py:train_and_predict_decomposed`
- 训练 baseline_normal / baseline_cf
- 在 uplift 日上训练 uplift（以 residual 为标签）
- 输出 `pred_tminus2_decomposed` 以及可解释的中间项列

---

## 9. 可视化输出

入口：`python -m src.gbdt.visualize`。

核心图：
- `figs/forecast_2025_decomposed_full.png`：2025 全年曲线（含节假日竖线、春节窗口阴影）
- `figs/forecast_2025_decomposed_spring_window.png`：2025 春运窗口曲线（真实 vs 分解预测）

如果 `data/pred_2025_optimized.csv` 存在分解式字段（见下一节），图中也会自动叠加对应曲线。

---

## 10. （可选）分解式 t-2：Baseline + Uplift（用于解释/归因）

入口：`python -m src.gbdt.train_decompose`（实现：`src/gbdt/train_decompose.py`）。

核心思路（概念层面）：
- 先学一个“常规日/基础形态”的 baseline（t-2）
- 再在“节假日/春运窗口”子集上学习 uplift（残差项），把总预测写成：
  - `y ≈ baseline + uplift`
- 并提供 counterfactual baseline（`baseline_cf`）作为对照，用于分析“如果没有节假日/春运影响会怎样”

适用场景：
- 你希望不仅预测，还想更可解释地拆解“节假日/春运带来的增量”。

注意：
- 该路径为本分支主线实现：用于在 t-2 信息约束下以 baseline/uplift 的方式提升峰值拟合与可解释性。

---

## 10. 推理说明

本分支不提供单独推理 CLI；建议直接运行 `train_decompose` 产出 `data/pred_2025_optimized.csv`，再用 `visualize` 查看。

---

## 12. 无泄露检查清单（建议上线/复现实验前必看）

- 年份拆分：训练仅使用 2023+2024；2025 仅用于最终评估/展示
- 动态特征：必须使用 `delay>=1` 的 shift（见 `src/gbdt/features.py:build_dynamic_features`）
- t-2 推理：对目标日 t 构造动态特征时，历史截止点为 `t-delay`（见 `src/gbdt/forecasting.py:predict_tminus2_series`）

---

## 13. 评估指标（Metrics）

实现：`src/gbdt/metrics.py`。

输出指标：
- MAE：平均绝对误差（更符合“每天差多少”）
- RMSE：均方根误差（更强调大误差惩罚）
- MAPE：相对误差（实现中对分母做了下限保护，避免小值爆炸）

窗口指标：
- 春运窗口：`days_to_cny ∈ [-25, +15]`（与 `cny_window` 一致）

---

## 14. 复现步骤（推荐流程）

1) 安装依赖（示例）
```bash
pip install xgboost chinese_calendar pandas numpy matplotlib scikit-learn
```

2) 分解建模训练 + 生成 2025 预测
```bash
python -m src.gbdt.train_decompose
```

3) 仅画图（使用 `data/pred_2025_optimized.csv`）
```bash
python -m src.gbdt.visualize
```

---

## 15. 常见问题（FAQ）

### 15.1 如何保证无信息泄露？

本分支默认约定：
- 训练：2023+2024
- 测试/展示：2025

此外：
- 动态特征构造通过 `delay>=1` 的 shift，避免同日泄露（见 `src/gbdt/features.py`）
- t-2 模式构造特征时严格截止到 `t-delay`（见 `src/gbdt/forecasting.py`）

---

## 16. 附录：配置速查

见 `src/gbdt/config.py`：
- 输入/输出：`DATA_PATH`、`OUTPUT_PRED_PATH`
- 产物路径：`MODEL_PATH_TMINUS2_*`、`STATS_PATH`
- 春节日期：`CNY_DATES`
- 动态特征列：`DYNAMIC_COLS`
- 搜索次数：`RANDOM_SEARCH_ITERS`
- 春运窗口样本权重网格：`WEIGHT_GRID`
