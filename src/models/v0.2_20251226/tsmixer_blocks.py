#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSMixer 基础模块（深度残差版本）
版本: v0.2
日期: 20251226

实现TSMixer的核心组件：
- 5层残差模块
- 时间混合块（多层残差架构）
- 特征混合块（多层残差架构）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TimeMixingBlock(nn.Module):
    """
    时间混合块（深度残差版本 - 平直结构）
    
    在时间维度上应用多层残差网络，用于捕捉复杂的时间依赖关系
    
    结构：
      seq_len → 512 (Linear + GELU) → res1(512) → res2(512) → res3(512) → seq_len (Linear + GELU + Dropout)
    """
    
    def __init__(
        self,
        seq_len: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化时间混合块
        
        Args:
            seq_len: 序列长度
            dropout: Dropout比率（仅用于下降路径）
            activation: 激活函数，"gelu"或"relu"
        """
        super(TimeMixingBlock, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ===== 平直路径 =====
        # seq_len → 512
        self.up1 = nn.Linear(seq_len, 512)
        self.res1 = ResidualBlock(512, activation)
        
        # 512 → 512 (平直，无维度变化)
        self.res2 = ResidualBlock(512, activation)
        
        # 512 → 512 (平直，无维度变化)
        self.res3 = ResidualBlock(512, activation)
        
        # 512 → seq_len
        self.down2 = nn.Linear(512, seq_len)
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
        # seq_len → 512
        x = self.up1(x)  # (batch, n_features, 512)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.res1(x)  # ResidualBlock(512)
        
        # 512 → 512 (平直，无维度变化)
        x = self.res2(x)  # ResidualBlock(512)
        
        # 512 → 512 (平直，无维度变化)
        x = self.res3(x)  # ResidualBlock(512)
        
        # 512 → seq_len
        x = self.down2(x)  # (batch, n_features, seq_len)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.dropout2(x)  # 保持dropout在下降路径
        
        # 转置回来
        x = x.transpose(1, 2)  # (batch, seq_len, n_features)
        
        return x


class FeatureMixingBlock(nn.Module):
    """
    特征混合块（深度残差版本 - 渐进式U型结构）
    
    在特征维度上应用多层残差网络，用于捕捉复杂的特征交互关系
    
    结构：
    上升路径（无dropout）：
      n_features → 128 (Linear + GELU) → res1(128) → 256 (Linear + GELU) → res2(256) 
      → 512 (Linear + GELU) → res3(512)
    
    下降路径（有dropout）：
      → 256 (Linear + GELU + Dropout) → res4(256) → 128 (Linear + GELU + Dropout) 
      → res5(128) → n_features (Linear + GELU + Dropout)
    """
    
    def __init__(
        self,
        n_features: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化特征混合块
        
        Args:
            n_features: 特征数量
            dropout: Dropout比率（仅用于下降路径）
            activation: 激活函数，"gelu"或"relu"
        """
        super(FeatureMixingBlock, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ===== 上升路径（无dropout） =====
        # n_features → 128
        self.up1 = nn.Linear(n_features, 128)
        self.res1 = ResidualBlock(128, activation)
        
        # 128 → 256
        self.up2 = nn.Linear(128, 256)
        self.res2 = ResidualBlock(256, activation)
        
        # 256 → 512
        self.up3 = nn.Linear(256, 512)
        self.res3 = ResidualBlock(512, activation)
        
        # ===== 下降路径（有dropout） =====
        # 512 → 256
        self.down1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.res4 = ResidualBlock(256, activation)
        
        # 256 → 128
        self.down2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.res5 = ResidualBlock(128, activation)
        
        # 128 → n_features
        self.down3 = nn.Linear(128, n_features)
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
        # n_features → 128
        x = self.up1(x)  # (batch, seq_len, 128)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.res1(x)  # ResidualBlock(128)
        
        # 128 → 256
        x = self.up2(x)  # (batch, seq_len, 256)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.res2(x)  # ResidualBlock(256)
        
        # 256 → 512
        x = self.up3(x)  # (batch, seq_len, 512)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.res3(x)  # ResidualBlock(512)
        
        # ===== 下降路径（有dropout） =====
        # 512 → 256
        x = self.down1(x)  # (batch, seq_len, 256)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.dropout1(x)
        x = self.res4(x)  # ResidualBlock(256)
        
        # 256 → 128
        x = self.down2(x)  # (batch, seq_len, 128)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.dropout2(x)
        x = self.res5(x)  # ResidualBlock(128)
        
        # 128 → n_features
        x = self.down3(x)  # (batch, seq_len, n_features)
        x = self.activation(x)  # ✓ 维度变化后有激活函数
        x = self.dropout3(x)
        
        return x


class TSMixerBlock(nn.Module):
    """
    TSMixer块（深度残差版本）
    
    组合时间混合和特征混合，支持可选的LayerNorm和残差连接
    结构：
    1. LayerNorm（可选） -> Time Mixing -> Residual（可选）
    2. LayerNorm（可选） -> Feature Mixing -> Residual（可选）
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        use_residual: bool = True
    ):
        """
        初始化TSMixer块
        
        Args:
            seq_len: 序列长度
            n_features: 特征数量
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm（默认True）
            use_residual: 是否使用残差连接（默认True）
        """
        super(TSMixerBlock, self).__init__()
        
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        
        # LayerNorm（如果启用）
        if use_layernorm:
            self.norm1 = nn.LayerNorm(n_features)  # 时间混合前的归一化
            self.norm2 = nn.LayerNorm(n_features)  # 特征混合前的归一化
        
        # 时间混合
        self.time_mixing = TimeMixingBlock(
            seq_len=seq_len,
            dropout=dropout,
            activation=activation
        )
        
        # 特征混合
        self.feature_mixing = FeatureMixingBlock(
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
            x = self.time_mixing(x)
        
        # 特征混合
        if self.use_layernorm:
            residual = x if self.use_residual else None
            x = self.norm2(x)
            x = self.feature_mixing(x)
            if self.use_residual:
                x = x + residual
        else:
            x = self.feature_mixing(x)
        
        return x

