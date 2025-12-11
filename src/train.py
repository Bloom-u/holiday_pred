"""
优化的训练模块 V2
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from model import ImprovedTrafficModel, AdaptiveLoss


class ImprovedDataset(Dataset):
    """改进的数据集 - 使用高级特征"""
    def __init__(self, df, seq_len=14, pred_len=7, mode='train',
                 train_ratio=0.7, val_ratio=0.15):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode

        self.target_col = '机动车当量'

        # Lag特征（最重要）
        self.lag_cols = ['lag_1d', 'lag_7d', 'lag_14d', 'recent_change_3d']

        # 连续特征（增强版）
        self.continuous_cols = [
            # 原有特征
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'days_to_next_holiday', 'days_from_prev_holiday',
            'holiday_proximity',
            # 新增趋势特征
            'ma_7d', 'ma_14d', 'ma_30d', 'std_7d', 'trend_7d',
            # 新增上下文特征
            'is_before_holiday_1d', 'is_before_holiday_2d',
            'is_after_holiday_1d', 'is_after_holiday_2d',
            # 年度特征
            'year_normalized'
        ]

        # 二值特征（增强版）
        self.binary_cols = [
            'is_weekend', 'is_holiday', 'is_adjusted_workday',
            'is_high_traffic_holiday', 'is_low_traffic_holiday',
            'is_holiday_start', 'is_holiday_end', 'is_2022'
        ]

        # 节假日辅助特征（4维向量）
        self.holiday_feature_cols = [
            'holiday_day_num', 'total_holiday_length',
            'holiday_progress', 'holiday_traffic_ratio'
        ]

        # 确保所有特征都存在
        self.lag_cols = [c for c in self.lag_cols if c in df.columns]
        self.continuous_cols = [c for c in self.continuous_cols if c in df.columns]
        self.binary_cols = [c for c in self.binary_cols if c in df.columns]
        self.holiday_feature_cols = [c for c in self.holiday_feature_cols if c in df.columns]

        # 划分数据
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if mode == 'train':
            self.df = df.iloc[:train_end].reset_index(drop=True)
        elif mode == 'val':
            self.df = df.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.df = df.iloc[val_end:].reset_index(drop=True)

        self.num_samples = max(0, len(self.df) - seq_len - pred_len + 1)

        # 缩放器
        if mode == 'train':
            self.target_scaler = RobustScaler()
            self.lag_scaler = StandardScaler()
            self.cont_scaler = StandardScaler()

            train_df = df.iloc[:train_end]
            self.target_scaler.fit(train_df[[self.target_col]])
            self.lag_scaler.fit(train_df[self.lag_cols])
            self.cont_scaler.fit(train_df[self.continuous_cols])

            self._preprocess()
            self.sample_weights = self._compute_weights()
        else:
            self._preprocessed = False

        print(f"{mode:5s} 集: {self.num_samples} 样本")

    def set_scalers(self, target_scaler, lag_scaler, cont_scaler):
        self.target_scaler = target_scaler
        self.lag_scaler = lag_scaler
        self.cont_scaler = cont_scaler
        self._preprocess()

    def _preprocess(self):
        df = self.df

        # 目标
        self.target = torch.FloatTensor(
            self.target_scaler.transform(df[[self.target_col]].values).flatten()
        )

        # Lag特征
        self.lag = torch.FloatTensor(
            self.lag_scaler.transform(df[self.lag_cols].values)
        )

        # 连续特征
        self.cont = torch.FloatTensor(
            self.cont_scaler.transform(df[self.continuous_cols].values)
        )

        # 二值特征
        self.binary = torch.FloatTensor(df[self.binary_cols].values)

        # 节假日特征
        self.holiday_type = torch.LongTensor(df['holiday_type'].values)
        self.holiday_features = torch.FloatTensor(df[self.holiday_feature_cols].values)

        # 节假日标签（用于损失加权）
        self.is_high_traffic = torch.FloatTensor(df['is_high_traffic_holiday'].values)
        self.is_low_traffic = torch.FloatTensor(df['is_low_traffic_holiday'].values)

        self._preprocessed = True

    def _compute_weights(self):
        """计算样本权重 - 节假日大幅过采样"""
        weights = []
        for i in range(self.num_samples):
            y_start = i + self.seq_len
            y_end = y_start + self.pred_len

            # 检查预测窗口内是否有特殊节假日
            y_is_high = self.is_high_traffic[y_start:y_end].numpy()
            y_is_low = self.is_low_traffic[y_start:y_end].numpy()

            if y_is_low.sum() > 0:  # 春节
                weights.append(30.0)
            elif y_is_high.sum() > 0:  # 国庆/劳动节
                weights.append(25.0)
            elif self.holiday_type[y_start:y_end].numpy().max() > 1:  # 其他节假日
                weights.append(10.0)
            else:  # 平日
                weights.append(1.0)

        return weights

    def get_sampler(self):
        if self.mode == 'train':
            # 极端过采样
            return WeightedRandomSampler(
                self.sample_weights,
                num_samples=len(self.sample_weights) * 3,  # 3倍过采样
                replacement=True
            )
        return None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len

        # 判断预测窗口的节假日类型
        y_high = self.is_high_traffic[y_start:y_end].max()
        y_low = self.is_low_traffic[y_start:y_end].max()

        return {
            # 输入
            'x_lag': self.lag[x_start:x_end],
            'x_cont': self.cont[x_start:x_end],
            'x_binary': self.binary[x_start:x_end],
            'x_holiday_type': self.holiday_type[x_start:x_end],
            'x_holiday_features': self.holiday_features[x_start:x_end],

            # 输出辅助特征
            'y_cont': self.cont[y_start:y_end],
            'y_binary': self.binary[y_start:y_end],
            'y_holiday_type': self.holiday_type[y_start:y_end],
            'y_holiday_features': self.holiday_features[y_start:y_end],

            # 目标
            'y_target': self.target[y_start:y_end],

            # 损失权重标签
            'y_is_high_traffic': y_high,
            'y_is_low_traffic': y_low,
        }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        # 输入
        x_lag = batch['x_lag'].to(device)
        x_cont = batch['x_cont'].to(device)
        x_binary = batch['x_binary'].to(device)
        x_holiday_type = batch['x_holiday_type'].to(device)
        x_holiday_features = batch['x_holiday_features'].to(device)

        y_cont = batch['y_cont'].to(device)
        y_binary = batch['y_binary'].to(device)
        y_holiday_type = batch['y_holiday_type'].to(device)
        y_holiday_features = batch['y_holiday_features'].to(device)

        # 目标
        y_target = batch['y_target'].to(device)
        y_is_high = batch['y_is_high_traffic'].to(device)
        y_is_low = batch['y_is_low_traffic'].to(device)

        optimizer.zero_grad()

        pred = model(
            x_lag, x_cont, x_binary,
            y_cont, y_binary,
            x_holiday_type, x_holiday_features,
            y_holiday_type, y_holiday_features,
            y_is_high, y_is_low
        )

        loss = criterion(pred, y_target, y_is_high, y_is_low)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, scaler, device):
    model.eval()
    preds, targets = [], []
    holiday_types_all = []

    with torch.no_grad():
        for batch in loader:
            x_lag = batch['x_lag'].to(device)
            x_cont = batch['x_cont'].to(device)
            x_binary = batch['x_binary'].to(device)
            x_holiday_type = batch['x_holiday_type'].to(device)
            x_holiday_features = batch['x_holiday_features'].to(device)

            y_cont = batch['y_cont'].to(device)
            y_binary = batch['y_binary'].to(device)
            y_holiday_type = batch['y_holiday_type'].to(device)
            y_holiday_features = batch['y_holiday_features'].to(device)

            y_is_high = batch['y_is_high_traffic'].to(device)
            y_is_low = batch['y_is_low_traffic'].to(device)

            pred = model(
                x_lag, x_cont, x_binary,
                y_cont, y_binary,
                x_holiday_type, x_holiday_features,
                y_holiday_type, y_holiday_features,
                y_is_high, y_is_low
            )

            preds.append(pred.cpu().numpy())
            targets.append(batch['y_target'].numpy())
            holiday_types_all.append(batch['y_holiday_type'].numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    holiday_types = np.concatenate(holiday_types_all)

    # 反标准化
    pred_inv = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    target_inv = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)

    mae = mean_absolute_error(target_inv.flatten(), pred_inv.flatten())

    metrics = {'overall_mae': mae}

    # 节假日MAE
    holiday_mask = (holiday_types > 1).flatten()
    if holiday_mask.sum() > 0:
        metrics['holiday_mae'] = mean_absolute_error(
            target_inv.flatten()[holiday_mask],
            pred_inv.flatten()[holiday_mask]
        )

    # 春节MAE
    spring_mask = (holiday_types == 3).flatten()
    if spring_mask.sum() > 0:
        metrics['spring_mae'] = mean_absolute_error(
            target_inv.flatten()[spring_mask],
            pred_inv.flatten()[spring_mask]
        )

    # 国庆MAE
    national_mask = (holiday_types == 8).flatten()
    if national_mask.sum() > 0:
        metrics['national_mae'] = mean_absolute_error(
            target_inv.flatten()[national_mask],
            pred_inv.flatten()[national_mask]
        )

    return metrics, pred_inv, target_inv


def train_model_v2(df, save_dir='./models/model_v2_output',
                   seq_len=14, pred_len=7,
                   num_epochs=200, hidden_dim=64,
                   batch_size=32, patience=40, device=None):
    """训练V2模型"""

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备选择
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    print(f"设备: {device}")

    # 数据集
    train_ds = ImprovedDataset(df, seq_len, pred_len, mode='train')
    val_ds = ImprovedDataset(df, seq_len, pred_len, mode='val')
    val_ds.set_scalers(train_ds.target_scaler, train_ds.lag_scaler, train_ds.cont_scaler)
    test_ds = ImprovedDataset(df, seq_len, pred_len, mode='test')
    test_ds.set_scalers(train_ds.target_scaler, train_ds.lag_scaler, train_ds.cont_scaler)

    sampler = train_ds.get_sampler()
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Baseline
    baseline_preds, baseline_tgts = [], []
    for batch in test_loader:
        # 使用lag_1d作为基线
        last_lag = batch['x_lag'][:, -1, 0].numpy()  # lag_1d
        baseline_preds.append(np.repeat(last_lag.reshape(-1, 1), pred_len, axis=1))
        baseline_tgts.append(batch['y_target'].numpy())

    baseline_preds = np.concatenate(baseline_preds)
    baseline_tgts = np.concatenate(baseline_tgts)
    bp_inv = train_ds.target_scaler.inverse_transform(baseline_preds.reshape(-1, 1)).reshape(baseline_preds.shape)
    bt_inv = train_ds.target_scaler.inverse_transform(baseline_tgts.reshape(-1, 1)).reshape(baseline_tgts.shape)
    baseline_mae = mean_absolute_error(bt_inv.flatten(), bp_inv.flatten())
    print(f"Baseline MAE: {baseline_mae:.2f}")

    # 模型
    model = ImprovedTrafficModel(
        num_continuous=len(train_ds.continuous_cols),
        num_binary=len(train_ds.binary_cols),
        num_lag=len(train_ds.lag_cols),
        hidden_dim=hidden_dim,
        pred_len=pred_len,
        dropout=0.15
    ).to(device)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = AdaptiveLoss(base_weight=1.0, high_traffic_weight=15.0, low_traffic_weight=20.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    # 训练
    best_score = float('inf')
    patience_counter = 0
    model_path = os.path.join(save_dir, 'best_model.pth')

    print("\n开始训练...")
    print(f"{'Epoch':>6} {'Loss':>8} {'Val MAE':>10} {'Holiday':>10} {'National':>10}")
    print("-" * 55)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, train_ds.target_scaler, device)

        val_mae = val_metrics['overall_mae']
        holiday_mae = val_metrics.get('holiday_mae', 0)
        national_mae = val_metrics.get('national_mae', 0)

        scheduler.step(val_mae)

        # 综合评分（更重视节假日）
        score = val_mae * 0.3 + holiday_mae * 0.4 + national_mae * 0.3

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            marker = " ✓"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"{epoch+1:>6} {train_loss:>8.4f} {val_mae:>10.2f} {holiday_mae:>10.2f} {national_mae:>10.2f}{marker}")

        if patience_counter >= patience:
            print(f"\n早停于 epoch {epoch+1}")
            break

    # 测试
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_metrics, preds, tgts = evaluate(model, test_loader, train_ds.target_scaler, device)

    print(f"\n{'='*60}")
    print("测试集结果")
    print(f"{'='*60}")
    print(f"Baseline MAE:  {baseline_mae:.2f}")
    print(f"模型 MAE:      {test_metrics['overall_mae']:.2f}")
    print(f"改进:          {(1 - test_metrics['overall_mae']/baseline_mae)*100:.1f}%")
    if 'holiday_mae' in test_metrics:
        print(f"节假日 MAE:    {test_metrics['holiday_mae']:.2f}")
    if 'spring_mae' in test_metrics:
        print(f"春节 MAE:      {test_metrics['spring_mae']:.2f}")
    if 'national_mae' in test_metrics:
        print(f"国庆 MAE:      {test_metrics['national_mae']:.2f}")

    # 保存
    with open(f'{save_dir}/scalers.pkl', 'wb') as f:
        pickle.dump({
            'target_scaler': train_ds.target_scaler,
            'lag_scaler': train_ds.lag_scaler,
            'cont_scaler': train_ds.cont_scaler
        }, f)

    print(f"\n模型已保存至: {save_dir}")

    return {
        'model': model,
        'device': device,
        'test_metrics': test_metrics,
        'baseline_mae': baseline_mae,
        'dataset_config': {
            'lag_cols': train_ds.lag_cols,
            'continuous_cols': train_ds.continuous_cols,
            'binary_cols': train_ds.binary_cols,
            'holiday_feature_cols': train_ds.holiday_feature_cols
        }
    }
