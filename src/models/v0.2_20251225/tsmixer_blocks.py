#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSMixer 基础模块
版本: v0.2
日期: 20251225

实现TSMixer的核心组件：时间混合和特征混合块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TimeMixingBlock(nn.Module):
    """
    时间混合块 (Time Mixing Block)
    
    在时间维度上应用MLP，用于捕捉时间序列中的时间依赖关系
    输入: (batch, seq_len, n_features)
    处理: 对每个特征的时间序列独立应用MLP
    输出: (batch, seq_len, n_features)
    """
    
    def __init__(
        self,
        seq_len: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化时间混合块
        
        Args:
            seq_len: 序列长度
            ff_dim: 前馈网络隐藏维度
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
        """
        super(TimeMixingBlock, self).__init__()
        
        # 时间维度的MLP：seq_len -> ff_dim -> seq_len
        self.fc1 = nn.Linear(seq_len, ff_dim)
        self.fc2 = nn.Linear(ff_dim, seq_len)
        self.dropout = nn.Dropout(dropout)
        
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
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # x: (batch, seq_len, n_features)
        # 转置以便在时间维度上应用MLP
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        
        # 应用MLP
        x = self.fc1(x)  # (batch, n_features, ff_dim)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, n_features, seq_len)
        x = self.dropout(x)
        
        # 转置回来
        x = x.transpose(1, 2)  # (batch, seq_len, n_features)
        
        return x


class FeatureMixingBlock(nn.Module):
    """
    特征混合块 (Feature Mixing Block)
    
    在特征维度上应用MLP，用于捕捉不同特征之间的关系
    输入: (batch, seq_len, n_features)
    处理: 对每个时间步的特征向量应用MLP
    输出: (batch, seq_len, n_features)
    """
    
    def __init__(
        self,
        n_features: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化特征混合块
        
        Args:
            n_features: 特征数量
            ff_dim: 前馈网络隐藏维度
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
        """
        super(FeatureMixingBlock, self).__init__()
        
        # 特征维度的MLP：n_features -> ff_dim -> n_features
        self.fc1 = nn.Linear(n_features, ff_dim)
        self.fc2 = nn.Linear(ff_dim, n_features)
        self.dropout = nn.Dropout(dropout)
        
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
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # x: (batch, seq_len, n_features)
        # 直接在特征维度上应用MLP（最后一个维度）
        
        x = self.fc1(x)  # (batch, seq_len, ff_dim)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, seq_len, n_features)
        x = self.dropout(x)
        
        return x


class TSMixerBlock(nn.Module):
    """
    TSMixer块 (TSMixer Block)
    
    组合时间混合和特征混合，包含残差连接和LayerNorm
    结构：
    1. LayerNorm -> Time Mixing -> Residual
    2. LayerNorm -> Feature Mixing -> Residual
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer"
    ):
        """
        初始化TSMixer块
        
        Args:
            seq_len: 序列长度
            n_features: 特征数量
            ff_dim: 前馈网络隐藏维度
            dropout: Dropout比率
            activation: 激活函数
            norm_type: 归一化类型，目前只支持"layer"
        """
        super(TSMixerBlock, self).__init__()
        
        # 时间混合
        self.time_mixing = TimeMixingBlock(
            seq_len=seq_len,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )
        self.norm1 = nn.LayerNorm(n_features)
        
        # 特征混合
        self.feature_mixing = FeatureMixingBlock(
            n_features=n_features,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )
        self.norm2 = nn.LayerNorm(n_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # 时间混合 + 残差连接
        residual = x
        x = self.norm1(x)
        x = self.time_mixing(x)
        x = x + residual
        
        # 特征混合 + 残差连接
        residual = x
        x = self.norm2(x)
        x = self.feature_mixing(x)
        x = x + residual
        
        return x

