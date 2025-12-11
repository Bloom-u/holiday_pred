"""
数据集模块 - 优化版
预处理所有数据，__getitem__ 只做切片
"""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler


def collate_fn(batch):
    """自定义 collate：高效处理嵌套字典"""
    x_target = torch.stack([b['x_target'] for b in batch])
    y_target = torch.stack([b['y_target'] for b in batch])
    x_cont = torch.stack([b['x_cont'] for b in batch])
    y_cont = torch.stack([b['y_cont'] for b in batch])
    x_binary = torch.stack([b['x_binary'] for b in batch])
    y_binary = torch.stack([b['y_binary'] for b in batch])
    is_holiday = torch.stack([b['is_holiday'] for b in batch])
    
    # 合并嵌套字典
    x_embed = {k: torch.stack([b['x_embed'][k] for b in batch]) for k in batch[0]['x_embed']}
    y_embed = {k: torch.stack([b['y_embed'][k] for b in batch]) for k in batch[0]['y_embed']}
    
    return {
        'x_target': x_target, 'y_target': y_target,
        'x_embed': x_embed, 'y_embed': y_embed,
        'x_cont': x_cont, 'y_cont': y_cont,
        'x_binary': x_binary, 'y_binary': y_binary,
        'is_holiday': is_holiday,
    }


class TrafficDatasetEnhanced(Dataset):
    """增强版数据集：预处理 + 向量化"""
    
    def __init__(self, df, seq_len=7, pred_len=7, stride=2, mode='train', 
                 train_ratio=0.7, val_ratio=0.15):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.mode = mode
        
        self.target_col = '机动车当量'
        self.embedding_cols = ['holiday_type', 'day_of_week', 'month']
        self.continuous_cols = [
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'days_to_next_holiday', 'days_from_prev_holiday', 
            'holiday_proximity', 'holiday_day_num', 'holiday_progress'
        ]
        self.binary_cols = ['is_weekend', 'is_holiday', 'is_adjusted_workday']
        
        # 过滤存在的列
        self.continuous_cols = [c for c in self.continuous_cols if c in df.columns]
        self.binary_cols = [c for c in self.binary_cols if c in df.columns]
        self.embedding_cols = [c for c in self.embedding_cols if c in df.columns]
        
        # 数据集划分
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if mode == 'train':
            self.df = df.iloc[:train_end]
        elif mode == 'val':
            self.df = df.iloc[train_end:val_end]
        else:
            self.df = df.iloc[val_end:]
        
        # 初始化缩放器（仅训练集）
        self.target_scaler = None
        self.cont_scaler = None
        
        if mode == 'train':
            self.target_scaler = RobustScaler()
            self.cont_scaler = StandardScaler()
            self.target_scaler.fit(df.iloc[:train_end][[self.target_col]])
            if self.continuous_cols:
                self.cont_scaler.fit(df.iloc[:train_end][self.continuous_cols])
            
            # 训练集直接预处理
            self._preprocess_all()
            self.sample_weights = self._compute_sample_weights_vectorized()
        else:
            # 验证集/测试集：先计算长度，等 set_scalers 后再预处理
            self._compute_num_samples()
            self._preprocessed = False
        
        self.embedding_dims = {
            'holiday_type': (11, 8),
            'day_of_week': (7, 4),
            'month': (12, 6),
        }
        
        print(f"{mode} 集: {len(self)} 样本 (Stride={self.stride})")
    
    def _compute_num_samples(self):
        """仅计算样本数量"""
        total_len = len(self.df) - self.seq_len - self.pred_len
        self.num_samples = max(0, total_len // self.stride + 1)
    
    def _preprocess_all(self):
        """一次性预处理所有数据为 Tensor"""
        df = self.df
        
        # 目标变量 - 预先缩放
        target_vals = df[self.target_col].values.reshape(-1, 1)
        self.target_scaled = torch.FloatTensor(
            self.target_scaler.transform(target_vals).flatten()
        )
        
        # 连续特征 - 预先缩放
        if self.continuous_cols:
            cont_vals = df[self.continuous_cols].values
            self.cont_scaled = torch.FloatTensor(self.cont_scaler.transform(cont_vals))
        else:
            self.cont_scaled = torch.zeros(len(df), 1)
        
        # 二值特征
        if self.binary_cols:
            self.binary_data = torch.FloatTensor(df[self.binary_cols].values)
        else:
            self.binary_data = torch.zeros(len(df), 1)
        
        # Embedding 特征
        self.embed_data = {
            col: torch.LongTensor(df[col].values) for col in self.embedding_cols
        }
        
        # 节假日标记
        self.holiday_types = df['holiday_type'].values
        
        # 计算样本数量
        self._compute_num_samples()
        self._preprocessed = True
    
    def _compute_sample_weights_vectorized(self):
        """向量化计算样本权重"""
        weights = np.ones(self.num_samples)
        target_q75 = np.quantile(self.target_scaled.numpy(), 0.75)
        
        for idx in range(self.num_samples):
            x_start = idx * self.stride
            y_start = x_start + self.seq_len
            y_end = y_start + self.pred_len
            
            if y_end > len(self.holiday_types):
                continue
            
            # 检查节假日
            has_holiday = (self.holiday_types[y_start:y_end] > 1).any()
            # 检查高流量
            is_high = self.target_scaled[y_start:y_end].mean().item() > target_q75
            
            if has_holiday:
                weights[idx] = 10.0
            elif is_high:
                weights[idx] = 2.0
        
        return weights.tolist()
    
    def get_sampler(self):
        if self.mode == 'train' and hasattr(self, 'sample_weights'):
            return WeightedRandomSampler(
                self.sample_weights, 
                num_samples=len(self.sample_weights) * 2,
                replacement=True
            )
        return None
    
    def set_scalers(self, target_scaler, cont_scaler):
        """设置外部缩放器并预处理数据"""
        self.target_scaler = target_scaler
        self.cont_scaler = cont_scaler
        self._preprocess_all()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 确保已预处理
        if not getattr(self, '_preprocessed', False):
            raise RuntimeError("Dataset not preprocessed. Call set_scalers() first for val/test sets.")
        
        x_start = idx * self.stride
        x_end = x_start + self.seq_len
        y_start, y_end = x_end, x_end + self.pred_len
        
        # 直接切片预处理好的 Tensor，无需任何计算
        x_target = self.target_scaled[x_start:x_end]
        y_target = self.target_scaled[y_start:y_end]
        
        x_embed = {col: self.embed_data[col][x_start:x_end] for col in self.embedding_cols}
        y_embed = {col: self.embed_data[col][y_start:y_end] for col in self.embedding_cols}
        
        x_cont = self.cont_scaled[x_start:x_end]
        y_cont = self.cont_scaled[y_start:y_end]
        
        x_binary = self.binary_data[x_start:x_end]
        y_binary = self.binary_data[y_start:y_end]
        
        is_holiday = (self.holiday_types[y_start:y_end] > 1).any()
        
        return {
            'x_target': x_target,
            'y_target': y_target,
            'x_embed': x_embed,
            'y_embed': y_embed,
            'x_cont': x_cont,
            'y_cont': y_cont,
            'x_binary': x_binary,
            'y_binary': y_binary,
            'is_holiday': torch.FloatTensor([float(is_holiday)]),
        }