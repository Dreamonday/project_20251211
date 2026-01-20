#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Masked LayerNorm
版本: v0.42
日期: 20260118

实现支持mask的LayerNorm，在计算均值和方差时排除空白数据
"""

import torch
import torch.nn as nn
from typing import Optional


class MaskedLayerNorm(nn.Module):
    """
    支持mask的LayerNorm
    
    在计算均值和方差时，只对有效数据（mask=True）进行计算，
    避免空白数据（mask=False）影响归一化统计量
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        初始化MaskedLayerNorm
        
        Args:
            normalized_shape: 归一化维度（通常是特征维度）
            eps: 防止除零的小常数
            elementwise_affine: 是否使用可学习的affine参数（gamma和beta）
        """
        super(MaskedLayerNorm, self).__init__()
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # 可学习的affine参数
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (..., normalized_shape)
            mask: 掩码张量，形状与x相同，True表示有效数据，False表示空白数据
                  如果为None，则使用标准LayerNorm
        
        Returns:
            归一化后的张量，形状与输入相同
        """
        if mask is None:
            # 如果没有mask，使用标准LayerNorm
            return self._standard_layernorm(x)
        else:
            # 使用masked layernorm
            return self._masked_layernorm(x, mask)
    
    def _standard_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        """标准LayerNorm（无mask）"""
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用affine变换
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
        
        return x_normalized
    
    def _masked_layernorm(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked LayerNorm（考虑mask）"""
        # x: (..., normalized_shape)
        # mask: (..., normalized_shape), bool类型
        
        # 确保mask是float类型以便计算
        mask_float = mask.float()
        
        # 计算每个样本有效数据的数量
        # 在最后一个维度上求和
        valid_count = mask_float.sum(dim=-1, keepdim=True).clamp(min=1)  # 防止除零
        
        # 计算masked mean
        # 将无效数据（mask=False）置为0，然后求和除以有效数据数量
        masked_sum = (x * mask_float).sum(dim=-1, keepdim=True)
        mean = masked_sum / valid_count
        
        # 计算masked variance
        # var = E[(x - mean)^2]，只对有效数据计算
        diff = x - mean  # 广播
        masked_var_sum = ((diff ** 2) * mask_float).sum(dim=-1, keepdim=True)
        var = masked_var_sum / valid_count
        
        # 归一化（所有数据，包括无效数据）
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 注意：不再将无效数据位置归零
        # 保留异常值（如-1000的归一化结果），让下游层学习识别和忽略它们
        # 只在最终聚合时通过mask排除，避免与zscore后的0值混淆
        # x_normalized = x_normalized * mask_float  # 已删除
        
        # 应用affine变换
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
        
        return x_normalized
    
    def extra_repr(self) -> str:
        """返回额外的表示信息"""
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
