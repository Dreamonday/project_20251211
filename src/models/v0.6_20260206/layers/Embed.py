#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
嵌入层（从官方TimeXer提取并简化）
版本: v0.6
日期: 20260206
"""

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """位置编码（标准sin/cos编码）"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding_inverted(nn.Module):
    """
    反转数据嵌入（Variate-level表示）
    简化版：不依赖时间特征
    """
    
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # 将每个变量的完整时间序列投影到 d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, n_vars] 外生变量
        
        Returns:
            [B, n_vars, d_model]
        """
        # 转置: [B, seq_len, n_vars] → [B, n_vars, seq_len]
        x = x.permute(0, 2, 1)
        
        # 投影: [B, n_vars, seq_len] → [B, n_vars, d_model]
        x = self.value_embedding(x)
        
        return self.dropout(x)
