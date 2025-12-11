"""
优化的交通流量预测模型 V2

核心改进：
1. 使用lag特征直接输入（避免过度依赖序列建模）
2. 更强的节假日特征提取
3. 改进的损失函数（节假日自适应权重）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPSLayerNorm(nn.Module):
    """MPS 兼容的 LayerNorm 实现"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = x.contiguous()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


class ImprovedEncoder(nn.Module):
    """改进的特征编码器 - 支持更多特征类型"""
    def __init__(self, num_continuous, num_binary, num_lag, hidden_dim):
        super().__init__()

        # 历史流量投影（lag特征，非常重要）
        self.lag_proj = nn.Sequential(
            nn.Linear(num_lag, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # 连续特征
        self.cont_proj = nn.Linear(num_continuous, hidden_dim // 4)

        # 二值特征
        self.binary_proj = nn.Linear(num_binary, hidden_dim // 4)

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            MPSLayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, lag, continuous, binary):
        # 确保张量连续性
        lag = lag.contiguous()
        continuous = continuous.contiguous()
        binary = binary.contiguous()

        l = self.lag_proj(lag)
        c = self.cont_proj(continuous)
        b = self.binary_proj(binary)

        combined = torch.cat([l, c, b], dim=-1)
        return self.fusion(combined)


class HolidayAwareAttention(nn.Module):
    """节假日感知的注意力机制"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.15):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = MPSLayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim)
        )
        self.norm2 = MPSLayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # 多头自注意力
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        attn_out = self.proj(out)

        # 残差连接
        x = self.norm1(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class AdaptivePredictor(nn.Module):
    """自适应预测头 - 根据节假日类型动态调整"""
    def __init__(self, hidden_dim, pred_len):
        super().__init__()

        # 通用基础预测器
        self.base_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len)
        )

        # 高流量节假日增强器
        self.high_traffic_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_len)
        )

        # 低流量节假日增强器（春节）
        self.low_traffic_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_len)
        )

        # 自适应权重门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, is_high_traffic, is_low_traffic):
        """
        x: [B, D] 聚合后的特征
        is_high_traffic: [B] 是否高流量节假日
        is_low_traffic: [B] 是否低流量节假日
        """
        # 基础预测
        base = self.base_predictor(x)

        # 专家预测
        high = self.high_traffic_enhancer(x)
        low = self.low_traffic_enhancer(x)

        # 动态门控权重
        weights = self.gate(x)  # [B, 3]

        # 加权组合
        output = (
            weights[:, 0:1] * base +
            weights[:, 1:2] * high +
            weights[:, 2:3] * low
        )

        return output


class ImprovedTrafficModel(nn.Module):
    """
    改进的交通流量预测模型 V2

    关键改进：
    1. 直接使用lag特征作为强基线
    2. 更深的编码器处理复杂特征
    3. 节假日自适应预测头
    """
    def __init__(self, num_continuous, num_binary, num_lag,
                 hidden_dim=64, num_heads=4, pred_len=7, dropout=0.15):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # 编码器
        self.encoder = ImprovedEncoder(num_continuous, num_binary, num_lag, hidden_dim)

        # 节假日嵌入（保持原有设计）
        self.holiday_type_embed = nn.Embedding(11, hidden_dim // 2)
        self.holiday_proj = nn.Linear(4, hidden_dim // 2)  # day_num, total_len, progress, traffic_ratio

        # 融合节假日特征
        self.holiday_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            MPSLayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 50, hidden_dim) * 0.02)

        # 时序建模层（加深）
        self.attention1 = HolidayAwareAttention(hidden_dim, num_heads, dropout)
        self.attention2 = HolidayAwareAttention(hidden_dim, num_heads, dropout)

        # 解码器
        self.decoder_attn = HolidayAwareAttention(hidden_dim, num_heads, dropout)

        # 预测头
        self.predictor = AdaptivePredictor(hidden_dim, pred_len)

    def forward(self, x_lag, x_cont, x_binary,
                y_cont, y_binary,
                x_holiday_type, x_holiday_features,
                y_holiday_type, y_holiday_features,
                y_is_high_traffic, y_is_low_traffic):
        """
        x_lag: [B, T, num_lag] - 历史流量特征
        x_cont: [B, T, num_cont] - 历史连续特征
        x_binary: [B, T, num_bin] - 历史二值特征
        y_cont, y_binary: 未来特征
        x/y_holiday_type: [B, T] - 节假日类型
        x/y_holiday_features: [B, T, 4] - (day_num, total_len, progress, traffic_ratio)
        y_is_high/low_traffic: [B] - 未来是否高/低流量节假日
        """
        B = x_lag.size(0)
        seq_len = x_lag.size(1)

        # 编码历史特征
        enc_feat = self.encoder(x_lag, x_cont, x_binary)  # [B, T, D]

        # 节假日特征
        x_htype_emb = self.holiday_type_embed(x_holiday_type.long())  # [B, T, D/2]
        x_hfeat_emb = self.holiday_proj(x_holiday_features)  # [B, T, D/2]
        x_holiday_emb = torch.cat([x_htype_emb, x_hfeat_emb], dim=-1)  # [B, T, D]

        # 融合
        enc_input = enc_feat + x_holiday_emb + self.pos_embed[:, :seq_len, :].contiguous()
        enc_input = self.holiday_fusion(enc_input)

        # 时序建模（双层注意力）
        enc_output = self.attention1(enc_input)
        enc_output = self.attention2(enc_output)

        # 编码器摘要
        enc_summary = enc_output.mean(dim=1)  # [B, D]

        # 解码器输入（未来节假日信息）
        y_htype_emb = self.holiday_type_embed(y_holiday_type.long())
        y_hfeat_emb = self.holiday_proj(y_holiday_features)
        y_holiday_emb = torch.cat([y_htype_emb, y_hfeat_emb], dim=-1)

        dec_input = y_holiday_emb + self.pos_embed[:, :self.pred_len, :].contiguous()
        dec_input = dec_input + enc_summary.unsqueeze(1)

        # 解码
        dec_output = self.decoder_attn(dec_input)
        dec_summary = dec_output.mean(dim=1)  # [B, D]

        # 最终特征
        final_feat = enc_summary + dec_summary

        # 自适应预测
        output = self.predictor(final_feat, y_is_high_traffic, y_is_low_traffic)

        return output


class AdaptiveLoss(nn.Module):
    """自适应损失函数 - 节假日自动加权"""
    def __init__(self, base_weight=1.0, high_traffic_weight=15.0, low_traffic_weight=20.0):
        super().__init__()
        self.base_weight = base_weight
        self.high_traffic_weight = high_traffic_weight
        self.low_traffic_weight = low_traffic_weight

    def forward(self, pred, target, is_high_traffic, is_low_traffic):
        """
        pred: [B, pred_len]
        target: [B, pred_len]
        is_high_traffic: [B] 是否高流量节假日（国庆/劳动节）
        is_low_traffic: [B] 是否低流量节假日（春节）
        """
        # Huber Loss（对异常值鲁棒）
        loss = F.smooth_l1_loss(pred, target, reduction='none')

        # 动态权重
        weights = torch.ones_like(loss) * self.base_weight

        # 高流量节假日加权
        is_high = is_high_traffic.unsqueeze(-1).expand_as(loss)
        weights = weights + is_high.float() * (self.high_traffic_weight - self.base_weight)

        # 低流量节假日加权
        is_low = is_low_traffic.unsqueeze(-1).expand_as(loss)
        weights = weights + is_low.float() * (self.low_traffic_weight - self.base_weight)

        return (loss * weights).mean()
