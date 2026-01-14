# GBDT 交通流量预测：分解建模训练说明（baseline + uplift）

本文档面向本仓库的 GBDT（XGBoost）流量预测代码，说明：
1) 数据与特征如何构建；2) 分解建模训练如何运行；3) 输出如何解释；4) 可能的数据泄露点与规避建议。

> 当前主入口为 `src/gbdt/train_decompose.py`。

---

## 1. 目录与产物

- 数据输入：`data/new_data.xlsx`
- 预测输出：`data/pred_2025_optimized.csv`
- 模型输出目录：`models/gbdt/`
  - （分解建模会生成多模型，见下方）
  - `group_stats.json`：训练期统计特征（group stats）
  - 分解建模额外产物（由 `train_decompose` 生成）：
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

在分解建模（`src/gbdt/train_decompose.py`）中：
- 用训练集（默认 2023+2024）计算 stats，并保存到 `models/gbdt/group_stats.json`

### 3.3 动态特征（lag/roll/std/trend…）

在 `src/gbdt/features.py:build_dynamic_features`：
- 由 y 计算滞后、滚动均值/方差、斜率等动态特征
- 支持 `delay` 参数，用于“t-k 可用信息对齐”
  - `delay=1`：等价用到 `t-1`（递归/常规场景）
  - `delay=2`：只用到 `t-2`（t-2 场景）

构造总特征矩阵：
- `build_feature_matrix(base, y, DYNAMIC_COLS, delay=...) = concat([base, dynamic], axis=1)`

---

## 4. 分解建模（`src/gbdt/train_decompose.py`）

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

## 5. 数据泄露检查清单（强烈建议阅读）

### 5.1 已规避的泄露

- group stats：只用训练年 y 统计，再应用到 2025（无泄露）
- t-2 动态特征：`delay=2` 保证对 t 只用到 `t-2` 及更早（符合 t-2 信息约束）
（本分解版本不包含系统性纠偏模块；如未来加回，请确保偏移量仅用验证集拟合）

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

## 6. 复现命令汇总

1) 分解建模训练（生成 `pred_tminus2_decomposed`）
```bash
python -m src.gbdt.train_decompose
```

2) 画图（读取 `data/pred_2025_optimized.csv`）
```bash
python -m src.gbdt.visualize
```

- 标准推理：
  - `python -m src.gbdt.infer --mode both --year 2025`

- 分解建模训练（`exp-decompose`）：
  - `python -m src.gbdt.train_decompose`

- 可视化（自动识别 CSV 列）：
  - `python -m src.gbdt.visualize`
