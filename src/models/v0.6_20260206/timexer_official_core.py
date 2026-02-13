#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
官方 TimeXer 核心组件（提取自官方仓库）
版本: v0.6
日期: 20260206

从 https://github.com/thuml/TimeXer 提取的核心模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import importlib.util


def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 导入依赖
layers_path = Path(__file__).parent / "layers"
try:
    attention_module = _load_module(layers_path / "SelfAttention_Family.py", "SelfAttention_Family")
    FullAttention = attention_module.FullAttention
    AttentionLayer = attention_module.AttentionLayer
    
    embed_module = _load_module(layers_path / "Embed.py", "Embed")
    PositionalEmbedding = embed_module.PositionalEmbedding
    DataEmbedding_inverted = embed_module.DataEmbedding_inverted
except Exception as e:
    raise ImportError(f"Failed to import layers: {e}") from e


class FlattenHead(nn.Module):
    """
    官方 TimeXer 输出头
    将 [bs, nvars, d_model, patch_num+1] 展平并投影到预测长度
    """
    
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Args:
            x: [bs, nvars, d_model, patch_num+1]
        
        Returns:
            [bs, nvars, target_window]
        """
        x = self.flatten(x)  # [bs, nvars, d_model*(patch_num+1)]
        x = self.linear(x)   # [bs, nvars, target_window]
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """
    官方 TimeXer 内生变量嵌入
    Patching + Global Token + Position Embedding
    """
    
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching配置
        self.patch_len = patch_len

        # Value嵌入：将每个patch投影到d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # 全局Token：可学习参数
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        
        # 位置编码
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, n_vars, seq_len] 内生变量
        
        Returns:
            embedded: [B*n_vars, patch_num+1, d_model] 嵌入后的表示
            n_vars: 变量数量
        """
        # 保存变量数量
        n_vars = x.shape[1]
        
        # 准备全局Token（复制batch维度）
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # Patching: 将时间序列切分成patches
        # unfold(dimension=-1, size=patch_len, step=patch_len)
        # [B, n_vars, seq_len] → [B, n_vars, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        
        # Reshape: [B, n_vars, patch_num, patch_len] → [B*n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # Value嵌入 + 位置编码
        # [B*n_vars, patch_num, patch_len] → [B*n_vars, patch_num, d_model]
        x = self.value_embedding(x) + self.position_embedding(x)
        
        # Reshape回: [B*n_vars, patch_num, d_model] → [B, n_vars, patch_num, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        
        # 拼接全局Token: [B, n_vars, patch_num, d_model] + [B, n_vars, 1, d_model]
        # → [B, n_vars, patch_num+1, d_model]
        x = torch.cat([x, glb], dim=2)
        
        # 最终Reshape: [B, n_vars, patch_num+1, d_model] → [B*n_vars, patch_num+1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """官方 TimeXer Encoder"""
    
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Args:
            x: [B*n_vars, patch_num+1, d_model] 内生嵌入
            cross: [B*n_exog, 1, d_model] 外生嵌入
        
        Returns:
            [B*n_vars, patch_num+1, d_model]
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        
        return x


class EncoderLayer(nn.Module):
    """
    官方 TimeXer EncoderLayer
    包含：Self-Attention + Cross-Attention（只对Global Token）+ FFN
    """
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # FFN: 使用Conv1D实现（逐位置前馈）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Args:
            x: [B*n_vars, patch_num+1, d_model] 内生嵌入（patches + global）
            cross: [B, n_exog, d_model] 外生嵌入
        
        Returns:
            [B*n_vars, patch_num+1, d_model]
        """
        # x: [B*n_vars, patch_num+1, d_model]
        # cross: [B, n_exog, d_model]
        n_vars = x.shape[0] // cross.shape[0]  # 推断内生变量数量
        B = cross.shape[0]
        D = cross.shape[-1]
        
        # ========== 步骤1: Patch-wise Self-Attention ==========
        # 所有内生tokens（包括global）之间做自注意力
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # ========== 步骤2: Variate-wise Cross-Attention（只对Global Token）==========
        # 提取Global Token（最后一个token）
        x_glb_ori = x[:, -1, :].unsqueeze(1)  # [B*n_vars, 1, d_model]
        
        # Reshape以恢复batch维度
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # [B, n_vars, d_model]
        
        # Cross-Attention: Global Token关注外生变量
        # Query: x_glb, Key/Value: cross
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])  # [B, n_vars, d_model]
        
        # Reshape回原格式
        x_glb_attn = torch.reshape(
            x_glb_attn,
            (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)  # [B*n_vars, 1, d_model]
        
        # 残差连接
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # 重新拼接：patches + 更新的global token
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # [B*n_vars, patch_num+1, d_model]

        # ========== 步骤3: Feed-Forward Network ==========
        # 转置以应用Conv1D
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 残差连接 + LayerNorm
        return self.norm3(x + y)
