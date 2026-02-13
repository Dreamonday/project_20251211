#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力机制（从官方TimeXer提取）
版本: v0.6
日期: 20260206
"""

import torch
import torch.nn as nn
from math import sqrt


class FullAttention(nn.Module):
    """标准的全注意力机制"""
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Args:
            queries: [B, L, H, E]
            keys: [B, S, H, E]
            values: [B, S, H, D]
            attn_mask: 注意力掩码（可选）
            tau: De-stationary参数（可选，不使用）
            delta: De-stationary参数（可选，不使用）
        
        Returns:
            V: [B, L, H, D] 注意力输出
            attn: 注意力权重（如果output_attention=True）
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 计算注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 应用掩码（如果需要）
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, float('-inf'))

        # Softmax + Dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # 计算输出
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """注意力层包装器（包含Q/K/V投影）"""
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Args:
            queries: [B, L, d_model]
            keys: [B, S, d_model]
            values: [B, S, d_model]
        
        Returns:
            out: [B, L, d_model]
            attn: 注意力权重（可选）
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 线性投影并重塑为多头格式
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 注意力计算
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        
        # 合并多头并投影
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
