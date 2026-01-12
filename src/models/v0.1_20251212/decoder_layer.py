#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decoder层实现
版本: v0.1
日期: 20251212

实现高度可配置的Transformer Decoder层
支持Pre-Norm/Post-Norm、可选因果掩码等配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    生成因果掩码（Causal Mask）
    确保位置i只能看到位置<=i的信息
    
    Args:
        seq_len: 序列长度
        device: 设备
    
    Returns:
        掩码张量，形状为 (seq_len, seq_len)
        True表示需要掩码的位置，False表示可见位置
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


class DecoderLayer(nn.Module):
    """
    Transformer Decoder层
    高度可配置，支持Pre-Norm/Post-Norm、可选因果掩码等
    支持自定义注意力嵌入维度（attn_embed_dim），可用于 iTransformer（注意力在特征维度）
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_causal_mask: bool = True,
        norm_type: str = "pre",
        attention_dropout: Optional[float] = None,
        ff_dropout: Optional[float] = None,
        use_bias: bool = True,
        attn_embed_dim: Optional[int] = None
    ):
        """
        初始化Decoder层
        
        Args:
            d_model: 模型隐藏维度（仅用于保持接口兼容）
            n_heads: 注意力头数
            d_ff: Feed-Forward网络隐藏维度
            dropout: 默认Dropout比率
            activation: 激活函数，"gelu"或"relu"
            use_causal_mask: 是否使用因果掩码
            norm_type: 归一化类型，"pre"（Pre-Norm）或"post"（Post-Norm）
            attention_dropout: 注意力层Dropout比率（如果为None，使用dropout）
            ff_dropout: Feed-Forward层Dropout比率（如果为None，使用dropout）
            use_bias: 是否在Linear层使用bias
            attn_embed_dim: 注意力嵌入维度（即最后一维的大小），如果为None则使用d_model
        """
        super(DecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.use_causal_mask = use_causal_mask
        self.norm_type = norm_type.lower()
        self.embed_dim = attn_embed_dim if attn_embed_dim is not None else d_model
        
        if self.norm_type not in ["pre", "post"]:
            raise ValueError(f"norm_type must be 'pre' or 'post', got {norm_type}")
        
        # 设置Dropout比率
        self.attention_dropout = attention_dropout if attention_dropout is not None else dropout
        self.ff_dropout = ff_dropout if ff_dropout is not None else dropout
        
        # 验证embed_dim是否能被head整除
        if self.embed_dim % n_heads != 0:
            raise ValueError(
                f"attn_embed_dim ({self.embed_dim}) must be divisible by n_heads ({n_heads})"
            )
        
        # 多头自注意力（注意力最后一维为embed_dim）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=n_heads,
            dropout=self.attention_dropout,
            bias=use_bias,
            batch_first=True
        )
        
        # Feed-Forward网络
        self.ff_linear1 = nn.Linear(self.embed_dim, d_ff, bias=use_bias)
        self.ff_linear2 = nn.Linear(d_ff, self.embed_dim, bias=use_bias)
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = F.gelu
        elif activation.lower() == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # Dropout层
        self.dropout1 = nn.Dropout(self.attention_dropout)
        self.dropout2 = nn.Dropout(self.ff_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, token_len, embed_dim)
            attn_mask: 注意力掩码（可选）
            key_padding_mask: Key填充掩码（可选）
        
        Returns:
            输出张量，形状为 (batch_size, token_len, embed_dim)
        """
        batch_size, token_len, embed_dim = x.shape
        
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Input embed dim ({embed_dim}) must equal attn_embed_dim ({self.embed_dim})"
            )
        
        # 生成因果掩码（如果需要）
        if self.use_causal_mask:
            causal_mask = generate_causal_mask(token_len, x.device)
            # 如果提供了attn_mask，需要合并
            if attn_mask is not None:
                # attn_mask和causal_mask都是True表示需要掩码
                attn_mask = attn_mask | causal_mask
            else:
                attn_mask = causal_mask
        
        # Pre-Norm或Post-Norm架构
        if self.norm_type == "pre":
            # Pre-Norm: Norm -> Attention -> Residual
            residual = x
            x = self.norm1(x)
            attn_output, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            x = residual + self.dropout1(attn_output)
            
            # Pre-Norm: Norm -> FF -> Residual
            residual = x
            x = self.norm2(x)
            # FF网络：embed_dim -> d_ff (有激活) -> embed_dim (添加激活)
            ff_intermediate = self.activation(self.ff_linear1(x))
            ff_output = self.activation(self.ff_linear2(self.dropout2(ff_intermediate)))
            x = residual + ff_output
        
        else:  # Post-Norm
            # Post-Norm: Attention -> Norm -> Residual
            residual = x
            attn_output, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            x = self.norm1(residual + self.dropout1(attn_output))
            
            # Post-Norm: FF -> Norm -> Residual
            residual = x
            # FF网络：embed_dim -> d_ff (有激活) -> embed_dim (添加激活)
            ff_intermediate = self.activation(self.ff_linear1(x))
            ff_output = self.activation(self.ff_linear2(self.dropout2(ff_intermediate)))
            x = self.norm2(residual + ff_output)
        
        return x
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入张量
            attn_mask: 注意力掩码
            key_padding_mask: Key填充掩码
        
        Returns:
            注意力权重，形状为 (batch_size, n_heads, token_len, token_len)
        """
        batch_size, token_len, _ = x.shape
        
        # 生成因果掩码（如果需要）
        if self.use_causal_mask:
            causal_mask = generate_causal_mask(token_len, x.device)
            if attn_mask is not None:
                attn_mask = attn_mask | causal_mask
            else:
                attn_mask = causal_mask
        
        if self.norm_type == "pre":
            x = self.norm1(x)
        
        # 使用self_attn获取注意力权重
        # 注意：MultiheadAttention的forward返回(attn_output, attn_weights)
        _, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False  # 返回每个头的权重
        )
        
        return attn_weights
