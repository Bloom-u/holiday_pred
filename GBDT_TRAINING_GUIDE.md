# GBDT 交通流量预测：训练说明（含分解建模与纠偏）

本文档面向本仓库的 GBDT（XGBoost）流量预测代码，说明：
1) 数据与特征如何构建；2) 训练/推理如何运行；3) `t-2` 预测的两类改进：损失函数实验与分解建模；4) 可能的数据泄露点与规避建议。

> 代码入口主要在 `src/gbdt/train.py`（标准训练）与 `src/gbdt/train_decompose.py`（分解建模原型）。

---

## 1. 目录与产物

- 数据输入：`data/new_data.xlsx`
- 预测输出：`data/pred_2025_optimized.csv`
- 模型输出目录：`models/gbdt/`
  - `best_model_recursive.json`：递归预测（recursive）模型
  - `best_model_tminus2.json`：t-2 预测模型（标准管线）
  - `best_model.json`：兼容旧路径（同 recursive）
  - `group_stats.json`：训练期统计特征（group stats）
  - `calibration.json`：t-2 纠偏参数（若启用）
  - 分解建模额外产物（在 `exp-decompose` 分支中由 `train_decompose` 生成）：
    - `best_model_tminus2_baseline.json`：常态基线（baseline_normal）
    - `best_model_tminus2_baseline_cf.json`：反事实基线（baseline_cf）
    - `best_model_tminus2_uplift*.json`：增量模型（uplift）
- 图表输出目录：`figs/`
  - 标准管线：`forecast_2025_optimized_full.png`、`forecast_2025_optimized_spring_window.png`
  - 分解建模：`forecast_2025_decomposed_full.png`、`forecast_2025_decomposed_spring_window.png`

---

## 2. 数据格式

`data/new_data.xlsx` 约定：
- 第 1 列：日期（会被重命名为 `date`）
- 第 2 列：目标列（会被重命名为 `y`）

加载逻辑在 `src/gbdt/data.py`：
- 按日期排序后重命名为 `date/y`，并返回 DataFrame。

---

## 3. 特征工程

### 3.1 静态日历/节假日特征（只依赖日期）

在 `src/gbdt/features.py:build_base_features`：
- 调用 `HolidayFeatureEngine().create_features(...)` 生成节假日相关特征（工作日/周末/法定假日/调休、距离下个/上个假日、假期阶段等）
- 额外生成春节相对日特征（`days_to_cny/cny_window/cny_pre/cny_post/cny_day`），春节日期来自 `src/gbdt/config.py:CNY_DATES`
- 对年份做归一化：`year_normalized`

### 3.2 group stats（只用训练集 y 统计）

在 `src/gbdt/features.py`：
- `compute_group_stats(base, y, train_mask)`：对训练期样本按 `day_of_week/month/holiday_type/days_to_cny` 统计均值/标准差等
- `apply_group_stats(base, stats)`：把统计特征（如 `dow_mean/month_mean/cny_offset_mean`）映射回每一天

标准训练在 `src/gbdt/train.py` 中：
- 用 `train_mask_final = (2023, 2024)` 计算 stats，并保存到 `models/gbdt/group_stats.json`

### 3.3 动态特征（lag/roll/std/trend…）

在 `src/gbdt/features.py:build_dynamic_features`：
- 由 y 计算滞后、滚动均值/方差、斜率等动态特征
- 支持 `delay` 参数，用于“t-k 可用信息对齐”
  - `delay=1`：等价用到 `t-1`（递归/常规场景）
  - `delay=2`：只用到 `t-2`（t-2 场景）

构造总特征矩阵：
- `build_feature_matrix(base, y, DYNAMIC_COLS, delay=...) = concat([base, dynamic], axis=1)`

---

## 4. 标准训练管线（`src/gbdt/train.py`）

### 4.1 训练/验证/测试划分（按年份）

- 超参选择（validation）：用 2023 训练、在 2024 上验证（通过 `select_best_model`）
- 最终训练：用 2023+2024 训练
- 测试/评估与输出：2025

### 4.2 两个模型：recursive 与 t-2 分开训练

为了避免两种推理方式的特征分布不一致：
- recursive 模型：使用 `delay=1` 的动态特征训练
- t-2 模型：使用 `delay=2` 的动态特征训练

输出：
- `models/gbdt/best_model_recursive.json`
- `models/gbdt/best_model_tminus2.json`

### 4.3 t-2 的纠偏（calibration）

动机：t-2 在节假日/春运窗口容易出现系统性偏差（尤其低估）。

做法（`src/gbdt/calibration.py`）：
- 用 2023 训练一个 t-2 模型，在 2024 上生成 `pred_tminus2`
- 在 2024 上计算残差均值 `y_true - y_pred`，按日类型分组得到 offsets：
  - `normal_workday_cny_window`
  - `normal_workday_non_cny`
  - `statutory_holiday`
  - `overall`（兜底）
- 推理时对 `pred_tminus2` 做分组加回偏移，得到 `pred_tminus2_calibrated`

产物：
- `models/gbdt/calibration.json`
- `data/pred_2025_optimized.csv` 里新增列 `pred_tminus2_calibrated`

---

## 5. 损失函数实验（分支 `exp-loss`）

目的：t-2 模型更偏向拟合峰值（减少低估）。

做法：
- 把 t-2 模型的 objective 改为 `reg:squarederror`（MSE），recursive 保持 MAE（L1）
- 通过 `src/gbdt/config.py` 暴露 `OBJECTIVE_RECURSIVE/OBJECTIVE_TMINUS2`
- `src/gbdt/model.py` 允许从 params 注入 `objective/eval_metric`

运行方式同标准训练：
- `python -m src.gbdt.train`

---

## 6. 分解建模（分支 `exp-decompose`，`src/gbdt/train_decompose.py`）

### 6.1 目标：把“常态水平”和“节假日/春运增量”分开学

分解预测输出中，你会看到：
- `pred_tminus2_baseline`：常态基线（baseline_normal）
- `pred_tminus2_baseline_cf`：反事实基线（baseline_cf）
- `pred_tminus2_uplift_cny / pred_tminus2_uplift_holiday`：增量预测（uplift）
- `pred_tminus2_decomposed`：最终组合结果

### 6.2 baseline_normal（常态基线）

- 输入 X：`_baseline_cols(base) + DYNAMIC_COLS`（动态特征用 delay=2）
- 输出 y：`y(t)`
- 训练样本：2023+2024 全部天
- 得到：`pred_base_normal_all(t)`

### 6.3 baseline_cf（反事实基线）

- 输入 X：同上
- 输出 y：`y(t)`
- 训练样本：2023+2024 且 `~uplift_mask`（非法定节假日且不在春运窗口）
- 得到：`pred_base_cf_all(t)`

### 6.4 uplift（增量模型：学 residual）

- 定义 residual 标签：`residual(t) = y(t) - pred_base_cf_all(t)`
- uplift 训练样本：2023+2024 且 uplift_mask（`is_holiday==1` 或 `cny_window==1`）
- uplift 输入 X：`_uplift_cols(base) + [pred_base_normal, pred_base_cf] + DYNAMIC_COLS`
  - 其中两列基线预测会作为 `tminus2_base_pred_normal/tminus2_base_pred_cf` 注入
- 拆两个增量模型分别训练：
  - CNY 窗口：`uplift_cny`
  - 非 CNY 的法定节假日：`uplift_holiday`

### 6.5 组合（2025）

非 uplift 日：
- `pred = pred_base_normal`

uplift 日：
- 以 `pred_base_cf` 为底座
- 加上对应 uplift 增量（CNY 用 cny 模型，其他法定节假日用 holiday 模型）
- CNY 核心日（`days_to_cny in [-1,0]`）额外做收缩，避免样本极少导致过冲
- CNY 窗口在除夕/初一前后做“normal/cf 的分段混合”，补偿分布漂移导致 cf 底座过低

运行：
- `python -m src.gbdt.train_decompose`

可视化：
- `python -m src.gbdt.visualize`（会识别 `pred_tminus2_decomposed` 并输出 decomposed 图）

---

## 7. 推理（`src/gbdt/infer.py`）

标准推理：
- `python -m src.gbdt.infer --year 2025 --mode both`

模型路径：
- recursive：`--model-rec models/gbdt/best_model_recursive.json`
- t-2：`--model-tminus2 models/gbdt/best_model_tminus2.json`
- 兼容模式：`--model models/gbdt/best_model.json`（两种都用同一个）

校准：
- 默认若 `models/gbdt/calibration.json` 存在会输出 `pred_tminus2_calibrated`
- 关闭校准：`--no-calibration`

---

## 8. 数据泄露检查清单（强烈建议阅读）

### 8.1 已规避的泄露

- group stats：只用训练年 y 统计，再应用到 2025（无泄露）
- t-2 动态特征：`delay=2` 保证对 t 只用到 `t-2` 及更早（符合 t-2 信息约束）
- 校准 offsets：用 2024 验证集计算，再用于 2025（不使用 2025 真值）

### 8.2 仍需注意的“评估污染/过拟合风险”

1) 分解建模中 uplift 的 stacking 风险  
uplift 特征里包含 `pred_base_normal`（baseline_normal 对训练样本是 in-sample 预测），这会让 uplift 在训练期“看到了拟合痕迹”，导致训练效果偏乐观。

建议：
- 用 OOF（按时间切块）生成 baseline 预测作为 uplift 输入，避免 in-sample prediction leakage。

2) 规则系数调参  
如果对 2025 指标反复调混合系数/收缩强度，会造成“测试集调参”。

建议：
- 固定 2025 为最终测试集，只在 2024（或滚动验证）上选系数。

---

## 9. 复现命令汇总

- 标准训练（含 recursive/t-2/校准/出图）：
  - `python -m src.gbdt.train`

- 标准推理：
  - `python -m src.gbdt.infer --mode both --year 2025`

- 分解建模训练（`exp-decompose`）：
  - `python -m src.gbdt.train_decompose`

- 可视化（自动识别 CSV 列）：
  - `python -m src.gbdt.visualize`

