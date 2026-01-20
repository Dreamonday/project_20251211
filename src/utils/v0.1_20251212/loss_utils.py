#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失函数工具
版本: v0.1
日期: 20251212

提供自定义损失函数类
"""

import torch
import torch.nn as nn


class MAPELoss(nn.Module):
    """
    平均绝对百分比误差损失函数 (Mean Absolute Percentage Error)
    
    公式: MAPE = mean(|pred - target| / (|target| + epsilon))
    
    特点:
    - 对相对误差敏感，适合不同量级的数据
    - 当target接近0时，使用epsilon避免除零
    - 支持裁剪相对误差上限，避免极端值主导训练
    
    示例:
        >>> criterion = MAPELoss(reduction='mean', epsilon=1e-8, max_relative_error=5.0)
        >>> pred = torch.tensor([10.0, 20.0, 30.0])
        >>> target = torch.tensor([9.0, 21.0, 29.0])
        >>> loss = criterion(pred, target)
    """
    def __init__(self, reduction='mean', epsilon=1e-8, max_relative_error=None):
        """
        初始化MAPE损失函数
        
        Args:
            reduction: 损失缩减方式，可选 'mean', 'sum', 'none'
            epsilon: 防止除零的小值，默认1e-8
            max_relative_error: 相对误差的上限（可选）
                               例如5.0表示限制单个样本的相对误差最大为5.0（500%）
                               None表示不裁剪，使用原始MAPE
        """
        super(MAPELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.max_relative_error = max_relative_error
    
    def forward(self, pred, target):
        """
        前向传播
        
        Args:
            pred: 预测值，形状为 (batch_size, ...)
            target: 真实值，形状与pred相同
        
        Returns:
            损失值（标量或与输入相同形状）
        """
        # 计算绝对误差
        abs_error = torch.abs(pred - target)
        
        # 计算分母（避免除零）
        denominator = torch.abs(target) + self.epsilon
        
        # 计算相对误差
        relative_error = abs_error / denominator
        
        # 裁剪相对误差（如果设置了max_relative_error）
        if self.max_relative_error is not None:
            relative_error = torch.clamp(relative_error, max=self.max_relative_error)
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return torch.mean(relative_error)
        elif self.reduction == 'sum':
            return torch.sum(relative_error)
        elif self.reduction == 'none':
            return relative_error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

