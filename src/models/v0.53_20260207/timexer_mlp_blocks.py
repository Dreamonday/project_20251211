#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimeXer-MLP 基础模块
版本: v0.5
日期: 20260107

实现TimeXer-MLP的核心组件：
- 5层残差模块（复用）
- 精简时间混合块（MLP）
- 精简特征混合块（MLP）
- 精简TSMixer块
- MLP交叉融合层（替代CrossAttentionLayer）
- MLP自增强层（替代SelfAttentionLayer）
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """
    5层残差模块
    
    结构：
    输入 -> Linear(D→D) + GELU -> ... (5层) -> 输出 + 输入（残差连接）
    
    特点：
    - 5层全连接，维度保持不变
    - 每层使用GELU激活
    - 最后一层输出与输入进行残差连接
    - 模块内部不使用Dropout
    """
    
    def __init__(self, dim: int, activation: str = "gelu"):
        """
        初始化5层残差模块
        
        Args:
            dim: 输入输出维度（保持不变）
            activation: 激活函数，"gelu"或"relu"
        """
        super(ResidualBlock, self).__init__()
        
        # 5层全连接
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(5)
        ])
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (..., dim)
        
        Returns:
            输出张量，形状为 (..., dim)
        """
        identity = x  # 保存输入用于残差连接
        
        # 5层前馈，每层都有激活
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        
        # 残差连接
        x = x + identity
        
        return x


class LightweightTimeMixingBlock(nn.Module):
    """
    精简时间混合块（共享版本）
    
    在时间维度上应用多层残差网络，用于捕捉复杂的时间依赖关系
    
    结构：
      seq_len → 256 (Linear + GELU) → res1(256) → res2(256) → seq_len (Linear + GELU + Dropout)
    """
    
    def __init__(
        self,
        seq_len: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化精简时间混合块
        
        Args:
            seq_len: 序列长度
            dropout: Dropout比率（仅用于下降路径）
            activation: 激活函数，"gelu"或"relu"
        """
        super(LightweightTimeMixingBlock, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ===== 平直路径 =====
        # seq_len → 256
        self.up1 = nn.Linear(seq_len, 256)
        self.res1 = ResidualBlock(256, activation)
        
        # 256 → 256 (平直，无维度变化)
        self.res2 = ResidualBlock(256, activation)
        
        # 256 → seq_len
        self.down2 = nn.Linear(256, seq_len)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # x: (batch, seq_len, n_features)
        # 转置以便在时间维度上应用MLP
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        
        # ===== 平直路径 =====
        # seq_len → 256
        x = self.up1(x)  # (batch, n_features, 256)
        x = self.activation(x)
        x = self.res1(x)  # ResidualBlock(256)
        
        # 256 → 256 (平直，无维度变化)
        x = self.res2(x)  # ResidualBlock(256)
        
        # 256 → seq_len
        x = self.down2(x)  # (batch, n_features, seq_len)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # 转置回来
        x = x.transpose(1, 2)  # (batch, seq_len, n_features)
        
        return x


class LightweightFeatureMixingBlock(nn.Module):
    """
    精简特征混合块
    
    在特征维度上应用多层残差网络，用于捕捉复杂的特征交互关系
    
    结构：
    上升路径（无dropout）：
      n_features → 64 (Linear + GELU) → res1(64) → 128 (Linear + GELU) → res2(128) 
      → 256 (Linear + GELU) → res3(256)
    
    下降路径（有dropout）：
      → 128 (Linear + GELU + Dropout) → res4(128) → 64 (Linear + GELU + Dropout) 
      → res5(64) → n_features (Linear + GELU + Dropout)
    """
    
    def __init__(
        self,
        n_features: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化精简特征混合块
        
        Args:
            n_features: 特征数量
            dropout: Dropout比率（仅用于下降路径）
            activation: 激活函数，"gelu"或"relu"
        """
        super(LightweightFeatureMixingBlock, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ===== 上升路径（无dropout） =====
        # n_features → 64
        self.up1 = nn.Linear(n_features, 64)
        self.res1 = ResidualBlock(64, activation)
        
        # 64 → 128
        self.up2 = nn.Linear(64, 128)
        self.res2 = ResidualBlock(128, activation)
        
        # 128 → 256
        self.up3 = nn.Linear(128, 256)
        self.res3 = ResidualBlock(256, activation)
        
        # ===== 下降路径（有dropout） =====
        # 256 → 128
        self.down1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.res4 = ResidualBlock(128, activation)
        
        # 128 → 64
        self.down2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.res5 = ResidualBlock(64, activation)
        
        # 64 → n_features
        self.down3 = nn.Linear(64, n_features)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # x: (batch, seq_len, n_features)
        # 直接在特征维度上应用MLP（最后一个维度）
        
        # ===== 上升路径（无dropout） =====
        # n_features → 64
        x = self.up1(x)  # (batch, seq_len, 64)
        x = self.activation(x)
        x = self.res1(x)  # ResidualBlock(64)
        
        # 64 → 128
        x = self.up2(x)  # (batch, seq_len, 128)
        x = self.activation(x)
        x = self.res2(x)  # ResidualBlock(128)
        
        # 128 → 256
        x = self.up3(x)  # (batch, seq_len, 256)
        x = self.activation(x)
        x = self.res3(x)  # ResidualBlock(256)
        
        # ===== 下降路径（有dropout） =====
        # 256 → 128
        x = self.down1(x)  # (batch, seq_len, 128)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.res4(x)  # ResidualBlock(128)
        
        # 128 → 64
        x = self.down2(x)  # (batch, seq_len, 64)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.res5(x)  # ResidualBlock(64)
        
        # 64 → n_features
        x = self.down3(x)  # (batch, seq_len, n_features)
        x = self.activation(x)
        x = self.dropout3(x)
        
        return x


class LightweightTSMixerBlock(nn.Module):
    """
    精简TSMixer块
    
    组合时间混合和特征混合，支持可选的LayerNorm和残差连接
    结构：
    1. LayerNorm（可选） -> Time Mixing -> Residual（可选）
    2. LayerNorm（可选） -> Feature Mixing -> Residual（可选）
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        shared_time_mixing: Optional[nn.Module],
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        use_residual: bool = True
    ):
        """
        初始化精简TSMixer块
        
        Args:
            seq_len: 序列长度
            n_features: 特征数量
            shared_time_mixing: 共享的时间混合块（如果为None则创建独立的）
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm（默认True）
            use_residual: 是否使用残差连接（默认True）
        """
        super(LightweightTSMixerBlock, self).__init__()
        
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        
        # LayerNorm（如果启用）
        if use_layernorm:
            self.norm1 = nn.LayerNorm(n_features)  # 时间混合前的归一化
            self.norm2 = nn.LayerNorm(n_features)  # 特征混合前的归一化
        
        # 时间混合（共享或独立）
        if shared_time_mixing is not None:
            self.time_mixing = shared_time_mixing
        else:
            # 使用MLP（默认）
            self.time_mixing = LightweightTimeMixingBlock(
                seq_len=seq_len,
                dropout=dropout,
                activation=activation
            )
        
        # 特征混合
        self.feature_mixing = LightweightFeatureMixingBlock(
            n_features=n_features,
            dropout=dropout,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # 时间混合
        if self.use_layernorm:
            residual = x if self.use_residual else None
            x = self.norm1(x)
            x = self.time_mixing(x)
            if self.use_residual:
                x = x + residual
        else:
            if self.use_residual:
                residual = x
                x = self.time_mixing(x)
                x = x + residual
            else:
                x = self.time_mixing(x)
        
        # 特征混合
        if self.use_layernorm:
            residual = x if self.use_residual else None
            x = self.norm2(x)
            x = self.feature_mixing(x)
            if self.use_residual:
                x = x + residual
        else:
            if self.use_residual:
                residual = x
                x = self.feature_mixing(x)
                x = x + residual
            else:
                x = self.feature_mixing(x)
        
        return x


class MLPCrossFusionLayer(nn.Module):
    """
    MLP交叉融合层（替代CrossAttentionLayer）
    
    使用MLP结构模拟交叉注意力机制，实现两个特征分支的融合
    
    结构：
    1. Query分支: d_model → ff_dim → ResBlock(ff_dim) → d_model
    2. Key/Value分支: d_model → ff_dim → ResBlock(ff_dim) → d_model
    3. 融合: Concat → Linear → Activation → ResBlock → Residual
    """
    
    def __init__(
        self,
        d_model: int,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True
    ):
        """
        初始化MLP交叉融合层
        
        Args:
            d_model: 模型维度
            ff_dim: 前馈网络隐藏维度（默认512）
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm
        """
        super(MLPCrossFusionLayer, self).__init__()
        
        self.d_model = d_model
        self.use_layernorm = use_layernorm
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Query分支处理
        self.query_proj_up = nn.Linear(d_model, ff_dim)
        self.query_res = ResidualBlock(ff_dim, activation)
        self.query_proj_down = nn.Linear(ff_dim, d_model)
        
        # Key/Value分支处理
        self.kv_proj_up = nn.Linear(d_model, ff_dim)
        self.kv_res = ResidualBlock(ff_dim, activation)
        self.kv_proj_down = nn.Linear(ff_dim, d_model)
        
        # 融合层
        fusion_dim = d_model * 2  # Concat后维度
        self.fusion_proj = nn.Linear(fusion_dim, d_model)
        self.fusion_res = ResidualBlock(d_model, activation)
        
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: Query张量，形状为 (batch, len_q, d_model)
            key: Key张量，形状为 (batch, len_k, d_model)
            value: Value张量，形状为 (batch, len_v, d_model)
            mask: 掩码（可选，保留接口兼容性，但不使用）
        
        Returns:
            融合输出，形状为 (batch, len_q, d_model)
        """
        batch_size, len_q, d_model = query.shape
        
        # Pre-Norm（如果启用）
        if self.use_layernorm:
            query_norm = self.norm1(query)
        else:
            query_norm = query
        
        # Query分支处理
        query_processed = self.query_proj_up(query_norm)  # (batch, len_q, ff_dim)
        query_processed = self.activation(query_processed)
        query_processed = self.query_res(query_processed)  # ResidualBlock
        query_processed = self.query_proj_down(query_processed)  # (batch, len_q, d_model)
        query_processed = self.dropout(query_processed)
        
        # Key/Value分支处理（使用key，value通常等于key）
        kv_processed = self.kv_proj_up(key)  # (batch, len_k, ff_dim)
        kv_processed = self.activation(kv_processed)
        kv_processed = self.kv_res(kv_processed)  # ResidualBlock
        kv_processed = self.kv_proj_down(kv_processed)  # (batch, len_k, d_model)
        kv_processed = self.dropout(kv_processed)
        
        # 如果len_q != len_k，需要对齐（通常len_q=len_k=1）
        if len_q != kv_processed.shape[1]:
            # 使用平均池化对齐（简单策略）
            kv_processed = kv_processed.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        
        # 融合：拼接两个分支
        fused = torch.cat([query_processed, kv_processed], dim=-1)  # (batch, len_q, 2*d_model)
        
        # 融合投影
        output = self.fusion_proj(fused)  # (batch, len_q, d_model)
        output = self.activation(output)
        output = self.fusion_res(output)  # ResidualBlock
        output = self.dropout(output)
        
        # 残差连接（与query相加）
        output = output + query
        
        # Post-Norm（如果启用）
        if self.use_layernorm:
            output = self.norm2(output)
        
        return output


class MLPSelfEnhancementLayer(nn.Module):
    """
    MLP自增强层（替代SelfAttentionLayer）
    
    使用MLP结构模拟自注意力机制，增强特征表示
    
    结构：
    1. LayerNorm（可选）
    2. FeatureMixingBlock风格U型结构:
       d_model → 2*d_model → 4*d_model → 2*d_model → d_model
       (每层带ResidualBlock)
    3. Residual: + 输入
    """
    
    def __init__(
        self,
        d_model: int,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True
    ):
        """
        初始化MLP自增强层
        
        Args:
            d_model: 模型维度
            ff_dim: 前馈网络隐藏维度（默认512，用于中间层）
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm
        """
        super(MLPSelfEnhancementLayer, self).__init__()
        
        self.d_model = d_model
        self.use_layernorm = use_layernorm
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # U型结构：d_model → 2*d_model → 4*d_model → 2*d_model → d_model
        # 上升路径
        self.up1 = nn.Linear(d_model, 2 * d_model)
        self.res1 = ResidualBlock(2 * d_model, activation)
        
        self.up2 = nn.Linear(2 * d_model, 4 * d_model)
        self.res2 = ResidualBlock(4 * d_model, activation)
        
        # 下降路径
        self.down1 = nn.Linear(4 * d_model, 2 * d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.res3 = ResidualBlock(2 * d_model, activation)
        
        self.down2 = nn.Linear(2 * d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.res4 = ResidualBlock(d_model, activation)
        
        # LayerNorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, len, d_model)
            mask: 掩码（可选，保留接口兼容性，但不使用）
        
        Returns:
            增强输出，形状为 (batch, len, d_model)
        """
        batch_size, len_seq, d_model = x.shape
        
        # Pre-Norm（如果启用）
        if self.use_layernorm:
            x_norm = self.norm1(x)
        else:
            x_norm = x
        
        # U型结构处理
        # 上升路径
        x_enhanced = self.up1(x_norm)  # (batch, len, 2*d_model)
        x_enhanced = self.activation(x_enhanced)
        x_enhanced = self.res1(x_enhanced)  # ResidualBlock
        
        x_enhanced = self.up2(x_enhanced)  # (batch, len, 4*d_model)
        x_enhanced = self.activation(x_enhanced)
        x_enhanced = self.res2(x_enhanced)  # ResidualBlock
        
        # 下降路径
        x_enhanced = self.down1(x_enhanced)  # (batch, len, 2*d_model)
        x_enhanced = self.activation(x_enhanced)
        x_enhanced = self.dropout1(x_enhanced)
        x_enhanced = self.res3(x_enhanced)  # ResidualBlock
        
        x_enhanced = self.down2(x_enhanced)  # (batch, len, d_model)
        x_enhanced = self.activation(x_enhanced)
        x_enhanced = self.dropout2(x_enhanced)
        x_enhanced = self.res4(x_enhanced)  # ResidualBlock
        
        # 残差连接
        output = x_enhanced + x
        
        # Post-Norm（如果启用）
        if self.use_layernorm:
            output = self.norm2(output)
        
        return output
