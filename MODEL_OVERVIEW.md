# 模型介绍：节假日交通流量预测（GBDT / XGBoost）

本文档解释本仓库的交通流量预测模型“是什么、为什么这样设计、各模块怎么协作、输出怎么解释、边界条件与风险点是什么”。  
训练/运行指令请见 `GBDT_TRAINING_GUIDE.md`。

---

## 1. 任务定义

给定按天排列的时间序列数据（日期 `date`，目标 `y`），预测未来某一年（默认 2025）的每日交通流量。

本仓库主要关注两种“可用信息约束”下的预测：
- **recursive（递归预测）**：在预测期内使用模型的预测值作为滞后输入（更贴近“长期滚动预测”）。
- **t-2（t-minus-2）预测**：假设真实标签存在滞后可用性，到预测 t 时只能用到 `t-2` 及更早的真实观测；因此动态特征必须严格延迟构造。

---

## 2. 模型整体结构（标准管线）

标准管线入口：`src/gbdt/train.py`（并配套 `src/gbdt/infer.py` 推理）。

核心是一个 **XGBoost 回归模型**（`src/gbdt/model.py`）叠加两类工程增强：
1) **特征工程**（日期/节假日/春节窗口 + 动态滞后与滚动统计 + 训练期统计映射）
2) **后处理纠偏（calibration）**：用于修正 t-2 的系统性偏差（可选）

### 2.1 特征工程模块

#### 2.1.1 静态节假日特征（只依赖日期）
由 `HolidayFeatureEngine`（`src/holiday_feature.py`）从日期派生：
- 工作日/周末/法定节假日/调休工作日
- 距离上下一个假日的天数、假日类型、假日阶段、连休长度与进度等

并额外加入春节相对日特征（`src/gbdt/features.py:add_cny_features`）：
- `days_to_cny`：距离春节当天的天数
- `cny_window`：春节窗口（默认 [-25, +15]）
- `cny_pre/cny_post/cny_day`：窗口内细分

> 春节日期由 `src/gbdt/config.py:CNY_DATES` 提供。

#### 2.1.2 训练期统计映射（group stats）
目的：用“类别→统计量”的方式注入先验（例如周几均值/月均值/春节相对日均值）。

实现：
- `compute_group_stats(base, y, train_mask)`：在训练期上计算映射
- `apply_group_stats(base, stats)`：映射回每一天，生成如 `dow_mean/month_mean/holiday_type_mean/cny_offset_mean` 等特征

注意：**这些统计必须只用训练期 y 计算**，否则会把测试期信息泄露进特征。

#### 2.1.3 动态特征（滞后/滚动/趋势）
目的：提供短期状态（近期水平、波动、趋势）：
- `lag_1/2/3/7/14/21/28`
- `roll_7/14/30`、`std_7/14`
- `trend/slope/accel` 等

关键：支持 `delay`（`src/gbdt/features.py:build_dynamic_features`）：
- recursive 训练通常对应 `delay=1`
- t-2 训练对应 `delay=2`（保证对 t 只使用 `t-2` 及更早历史）

---

## 3. 为什么 t-2 容易“系统性偏低估”

你在结果里看到的现象（t-2 在峰值日/节假日附近偏低、整体偏低）通常由多因素叠加导致：
- **信息约束**：t-2 无法使用 `t-1` 的真实观测，短期突发上冲更难被提前捕捉。
- **损失函数偏好**：若用 MAE（L1），最优解更接近条件中位数，对“少量峰值 + 大量常态”的分布会更保守。
- **样本不均衡**：峰值日数量少，模型为了整体误差往往选择“常态更准、峰值吃亏”的折中。
- **分布漂移**：某些年份整体水平上移（例如 2025 均值/中位数高于 2023–2024），树模型对“外推抬升”能力有限，容易整体偏低。

---

## 4. t-2 纠偏（calibration）是什么、解决什么问题

入口：`src/gbdt/calibration.py`

纠偏的目标不是改变预测的“形状/波动”，而是修正 **可重复出现的系统性偏差（bias）**：
- 在验证集（例如 2024）上计算 `offset = mean(y_true - y_pred)`（按日类型分组）
- 推理时对预测加回 offset，得到 `pred_tminus2_calibrated`

这种方法在工业界常称为 bias correction / calibration / MOS（Model Output Statistics），尤其在存在分布漂移或峰值稀疏时非常实用。

风险与注意：
- offsets 必须在验证集上学（例如 2024），**不能用 2025 真实值调**，否则就是测试集调参。

---

## 5. 损失函数实验（`exp-loss`）的设计意图

分支 `exp-loss` 的核心想法：
- recursive 仍用 MAE（更稳）
- t-2 用 MSE（`reg:squarederror`）以更强地惩罚峰值误差，倾向减轻低估

本质上是“对峰值更敏感”的训练目标选择，通常会带来：
- 峰值日误差下降（但可能导致常态日波动更大）
- 与 calibration 结合时，整体 MAE/MAPE 更容易改善

---

## 6. 分解建模（`exp-decompose`）在做什么：baseline + uplift

分支 `exp-decompose`（入口 `src/gbdt/train_decompose.py`）引入了一个常见的结构化建模思路：

> **把“常态水平”与“节假日/春运增量”拆开建模**，最后再组合。

### 6.1 baseline_normal：常态基线
- 输入：偏周期/常态的特征 + t-2 动态特征
- 输出：直接预测 `y(t)`
- 训练样本：训练年全量
- 作用：给非节假日提供稳定的日常水平

### 6.2 baseline_cf：反事实基线（counterfactual）
- 输入：同上
- 输出：直接预测 `y(t)`
- 训练样本：训练年里“非 uplift 日”（不在法定假日且不在春运窗口）
- 作用：估计“如果没有节假日效应，当天应该是多少”

### 6.3 uplift：增量（残差）模型
- 先用 baseline_cf 得到 `pred_base_cf(t)`
- 定义残差标签：`residual(t) = y(t) - pred_base_cf(t)`
- 在训练年的 uplift 日上训练 uplift 模型预测 residual
- 为了让 uplift 更好地“看懂当前水平”，会把 baseline 预测作为额外特征输入（stacking 的一种简化形式）

### 6.4 组合输出（解释图里的曲线）

在分解可视化图中：
- `pred_tminus2_baseline_cf`：反事实基线（通常较低，代表“无假期效应”的水平）
- `pred_tminus2_decomposed`：最终组合预测，概念上近似：
  - 非 uplift 日：`baseline_normal`
  - uplift 日：`baseline_cf + uplift(residual)`
  - CNY 核心日额外做收缩/混合以避免极端样本稀疏导致的过冲

---

## 7. 数据泄露与评估污染：需要重点警惕的点

本仓库已明确规避的典型泄露：
- group stats 只用训练年 y 计算后再映射到 2025
- t-2 动态特征用 `delay=2` 保证不使用 `t-1` 或未来 y

仍需要注意/进一步工程化的点（尤其在分解建模中）：
- **stacking 的 in-sample 预测污染**：uplift 特征里使用了 baseline 模型对训练样本的预测；如果该预测是 in-sample，可能导致 uplift 训练过拟合。更严格做法是对 uplift 训练样本使用 OOF（按时间切块）生成 baseline 预测。
- **规则系数调参污染**：若根据 2025 表现反复调整混合/收缩系数，本质是对测试集调参。更严格做法是在 2024 上选择系数，2025 只做最终一次评估。

---

## 8. 输出字段解释（常用）

标准管线 `data/pred_2025_optimized.csv` 可能出现的列：
- `pred_recursive`：递归预测
- `pred_tminus2`：t-2 预测
- `pred_tminus2_calibrated`：t-2 + 纠偏（若启用）

分解建模分支可能输出：
- `pred_tminus2_baseline`：常态基线
- `pred_tminus2_baseline_cf`：反事实基线
- `pred_tminus2_uplift_cny / pred_tminus2_uplift_holiday`：增量项
- `pred_tminus2_decomposed`：最终组合预测

---

## 9. 如何阅读可视化图

图由 `src/gbdt/visualize.py` 生成：
- 全年图看整体偏差与节假日峰值拟合
- 春运窗口图看 `days_to_cny` 周期内是否跟住“节前上冲→除夕/初一低谷→节后回落”的形状
- `baseline_cf` 曲线偏低是预期现象：它代表“无假期效应的底座”，而不是最终预测

