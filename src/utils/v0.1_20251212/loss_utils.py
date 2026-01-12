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
    
    示例:
        >>> criterion = MAPELoss(reduction='mean', epsilon=1e-8)
        >>> pred = torch.tensor([10.0, 20.0, 30.0])
        >>> target = torch.tensor([9.0, 21.0, 29.0])
        >>> loss = criterion(pred, target)
    """
    def __init__(self, reduction='mean', epsilon=1e-8):
        """
        初始化MAPE损失函数
        
        Args:
            reduction: 损失缩减方式，可选 'mean', 'sum', 'none'
            epsilon: 防止除零的小值，默认1e-8
        """
        super(MAPELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
    
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
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return torch.mean(relative_error)
        elif self.reduction == 'sum':
            return torch.sum(relative_error)
        elif self.reduction == 'none':
            return relative_error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

