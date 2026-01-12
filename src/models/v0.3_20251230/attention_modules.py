#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力机制模块
版本: v0.3
日期: 20251230

提供位置编码、注意力计算等基础注意力模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    位置编码（支持可学习和固定Sinusoidal两种）
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        learnable: bool = True,
        dropout: float = 0.1
    ):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            learnable: 是否使用可学习的位置编码
            dropout: Dropout比率
        """
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable
        
        if learnable:
            # 可学习的位置编码
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            # 固定的Sinusoidal位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
        
        Returns:
            添加位置编码后的张量，形状为 (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # 添加位置编码
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


class SegmentPositionalEncoding(nn.Module):
    """
    分段位置编码
    
    为segment间和segment内分别提供位置编码
    """
    
    def __init__(
        self,
        d_model: int,
        max_n_segments: int = 20,
        max_seg_len: int = 100,
        learnable: bool = True,
        dropout: float = 0.1
    ):
        """
        初始化分段位置编码
        
        Args:
            d_model: 模型维度
            max_n_segments: 最大segment数量
            max_seg_len: 每个segment的最大长度
            learnable: 是否使用可学习的位置编码
            dropout: Dropout比率
        """
        super(SegmentPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable
        
        # Segment间的位置编码（标识第几个segment）
        if learnable:
            self.segment_pe = nn.Parameter(
                torch.randn(1, max_n_segments, 1, d_model // 2) * 0.02
            )
        else:
            segment_pe = self._create_sinusoidal_pe(max_n_segments, d_model // 2)
            segment_pe = segment_pe.unsqueeze(2)  # (1, max_n_segments, 1, d_model//2)
            self.register_buffer('segment_pe', segment_pe)
        
        # Segment内的位置编码（标识segment内第几个位置）
        if learnable:
            self.intra_segment_pe = nn.Parameter(
                torch.randn(1, 1, max_seg_len, d_model // 2) * 0.02
            )
        else:
            intra_pe = self._create_sinusoidal_pe(max_seg_len, d_model // 2)
            intra_pe = intra_pe.unsqueeze(0).unsqueeze(0)  # (1, 1, max_seg_len, d_model//2)
            self.register_buffer('intra_segment_pe', intra_pe)
    
    def _create_sinusoidal_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建Sinusoidal位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, n_segments, seg_len, d_model)
        
        Returns:
            添加分段位置编码后的张量，形状为 (batch, n_segments, seg_len, d_model)
        """
        batch_size, n_segments, seg_len, d_model = x.shape
        
        # 获取segment间位置编码
        seg_pe = self.segment_pe[:, :n_segments, :, :]  # (1, n_segments, 1, d_model//2)
        seg_pe = seg_pe.expand(batch_size, n_segments, seg_len, d_model // 2)
        
        # 获取segment内位置编码
        intra_pe = self.intra_segment_pe[:, :, :seg_len, :]  # (1, 1, seg_len, d_model//2)
        intra_pe = intra_pe.expand(batch_size, n_segments, seg_len, d_model // 2)
        
        # 拼接两种位置编码
        pe = torch.cat([seg_pe, intra_pe], dim=-1)  # (batch, n_segments, seg_len, d_model)
        
        # 添加到输入
        x = x + pe
        
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        初始化缩放点积注意力
        
        Args:
            dropout: 注意力权重的Dropout比率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: Query张量，形状为 (batch, n_heads, len_q, d_k)
            key: Key张量，形状为 (batch, n_heads, len_k, d_k)
            value: Value张量，形状为 (batch, n_heads, len_v, d_v)
            mask: 掩码张量（可选），形状为 (batch, 1, len_q, len_k)
        
        Returns:
            - 注意力输出，形状为 (batch, n_heads, len_q, d_v)
            - 注意力权重，形状为 (batch, n_heads, len_q, len_k)
        """
        d_k = query.size(-1)
        
        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: (batch, n_heads, len_q, len_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, len_q, len_k)
        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出: Attention @ V
        output = torch.matmul(attn_weights, value)  # (batch, n_heads, len_q, d_v)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力
    
    注：实际实现中可能直接使用 nn.MultiheadAttention
    这里提供一个自定义实现作为参考
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout比率
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        # 注意力计算
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: Query张量，形状为 (batch, len_q, d_model)
            key: Key张量，形状为 (batch, len_k, d_model)
            value: Value张量，形状为 (batch, len_v, d_model)
            mask: 掩码张量（可选），形状为 (batch, len_q, len_k) 或 (batch, 1, len_q, len_k)
        
        Returns:
            - 注意力输出，形状为 (batch, len_q, d_model)
            - 注意力权重，形状为 (batch, n_heads, len_q, len_k)
        """
        batch_size = query.size(0)
        
        # 线性投影并分离多头
        # (batch, len, d_model) -> (batch, len, n_heads, d_k) -> (batch, n_heads, len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 调整掩码维度（如果提供）
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (batch, 1, len_q, len_k)
        
        # 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch, n_heads, len_q, d_k)
        # attn_weights: (batch, n_heads, len_q, len_k)
        
        # 合并多头
        # (batch, n_heads, len_q, d_k) -> (batch, len_q, n_heads, d_k) -> (batch, len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights

