#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分段工具函数
版本: v0.3
日期: 20251230

提供序列分段、特征分组等工具函数
"""

import torch
import torch.nn as nn
from typing import Tuple


def segment_sequence(
    x: torch.Tensor,
    n_segments: int
) -> torch.Tensor:
    """
    将序列分段
    
    Args:
        x: 输入张量，形状为 (batch, seq_len, d_model)
        n_segments: 分段数量
    
    Returns:
        分段后的张量，形状为 (batch, n_segments, seg_len, d_model)
    
    Raises:
        ValueError: 如果 seq_len 不能被 n_segments 整除
    """
    batch_size, seq_len, d_model = x.shape
    
    if seq_len % n_segments != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by n_segments ({n_segments})"
        )
    
    seg_len = seq_len // n_segments
    
    # Reshape: (batch, seq_len, d_model) -> (batch, n_segments, seg_len, d_model)
    x_segmented = x.view(batch_size, n_segments, seg_len, d_model)
    
    return x_segmented


def merge_segments(x: torch.Tensor) -> torch.Tensor:
    """
    合并分段
    
    Args:
        x: 分段张量，形状为 (batch, n_segments, seg_len, d_model)
    
    Returns:
        合并后的张量，形状为 (batch, seq_len, d_model)
    """
    batch_size, n_segments, seg_len, d_model = x.shape
    seq_len = n_segments * seg_len
    
    # Reshape: (batch, n_segments, seg_len, d_model) -> (batch, seq_len, d_model)
    # 使用reshape而不是view，因为可能不连续
    x_merged = x.reshape(batch_size, seq_len, d_model)
    
    return x_merged


def group_features(
    x: torch.Tensor,
    n_groups: int
) -> torch.Tensor:
    """
    特征分组
    
    Args:
        x: 输入张量，形状为 (batch, seq_len, n_features)
        n_groups: 分组数量
    
    Returns:
        分组后的张量，形状为 (batch, seq_len, n_groups, group_size)
    
    Raises:
        ValueError: 如果 n_features 不能被 n_groups 整除
    """
    batch_size, seq_len, n_features = x.shape
    
    if n_features % n_groups != 0:
        raise ValueError(
            f"n_features ({n_features}) must be divisible by n_groups ({n_groups})"
        )
    
    group_size = n_features // n_groups
    
    # Reshape: (batch, seq_len, n_features) -> (batch, seq_len, n_groups, group_size)
    x_grouped = x.view(batch_size, seq_len, n_groups, group_size)
    
    return x_grouped


def ungroup_features(x: torch.Tensor) -> torch.Tensor:
    """
    特征解组
    
    Args:
        x: 分组张量，形状为 (batch, seq_len, n_groups, group_size)
    
    Returns:
        解组后的张量，形状为 (batch, seq_len, n_features)
    """
    batch_size, seq_len, n_groups, group_size = x.shape
    n_features = n_groups * group_size
    
    # Reshape: (batch, seq_len, n_groups, group_size) -> (batch, seq_len, n_features)
    # 使用reshape而不是view，因为可能不连续
    x_ungrouped = x.reshape(batch_size, seq_len, n_features)
    
    return x_ungrouped


def gather_by_topk(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    根据TopK索引提取数据
    
    Args:
        x: 输入张量，形状为 (batch, n_items, ...)
        topk_indices: TopK索引，形状为 (batch, topk_k)
        dim: 提取的维度（默认为1）
    
    Returns:
        提取后的张量，形状为 (batch, topk_k, ...)
    """
    batch_size = x.size(0)
    topk_k = topk_indices.size(1)
    
    # 扩展索引以匹配x的所有维度
    # 例如: x为(batch, n_items, d), 索引为(batch, topk_k)
    # 需要扩展为(batch, topk_k, d)
    index_shape = list(x.shape)
    index_shape[dim] = topk_k
    
    # 扩展 topk_indices 的维度
    expanded_indices = topk_indices
    for _ in range(len(x.shape) - 2):
        expanded_indices = expanded_indices.unsqueeze(-1)
    
    # 扩展到所有维度
    expanded_indices = expanded_indices.expand(index_shape)
    
    # 使用gather提取
    selected = torch.gather(x, dim, expanded_indices)
    
    return selected


def scatter_by_topk(
    x_full: torch.Tensor,
    x_selected: torch.Tensor,
    topk_indices: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    将TopK计算结果放回原位置
    
    Args:
        x_full: 原始完整张量，形状为 (batch, n_items, ...)
        x_selected: 选中的TopK结果，形状为 (batch, topk_k, ...)
        topk_indices: TopK索引，形状为 (batch, topk_k)
        dim: 散射的维度（默认为1）
    
    Returns:
        更新后的张量，形状为 (batch, n_items, ...)
        未选中的位置保持原值
    """
    batch_size = x_full.size(0)
    topk_k = topk_indices.size(1)
    
    # 创建输出张量（复制原始数据）
    output = x_full.clone()
    
    # 扩展索引
    expanded_indices = topk_indices
    for _ in range(len(x_full.shape) - 2):
        expanded_indices = expanded_indices.unsqueeze(-1)
    
    # 扩展到所有维度
    index_shape = list(x_full.shape)
    index_shape[dim] = topk_k
    expanded_indices = expanded_indices.expand(index_shape)
    
    # 使用scatter更新
    output.scatter_(dim, expanded_indices, x_selected)
    
    return output


def compute_segment_repr(
    x_segmented: torch.Tensor,
    method: str = "mean"
) -> torch.Tensor:
    """
    计算每个segment的代表向量
    
    Args:
        x_segmented: 分段张量，形状为 (batch, n_segments, seg_len, d_model)
        method: 聚合方法，"mean"（平均）或 "max"（最大）
    
    Returns:
        segment代表向量，形状为 (batch, n_segments, d_model)
    """
    if method == "mean":
        # 对seg_len维度取平均
        segment_repr = x_segmented.mean(dim=2)  # (batch, n_segments, d_model)
    elif method == "max":
        # 对seg_len维度取最大
        segment_repr = x_segmented.max(dim=2)[0]  # (batch, n_segments, d_model)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return segment_repr


def compute_group_repr(
    x_grouped: torch.Tensor,
    method: str = "mean"
) -> torch.Tensor:
    """
    计算每个特征组的代表向量
    
    Args:
        x_grouped: 分组张量，形状为 (batch, seq_len, n_groups, group_size)
        method: 聚合方法，"mean"（平均）或 "max"（最大）
    
    Returns:
        group代表向量，形状为 (batch, seq_len, n_groups)
    """
    if method == "mean":
        # 对group_size维度取平均
        group_repr = x_grouped.mean(dim=-1)  # (batch, seq_len, n_groups)
    elif method == "max":
        # 对group_size维度取最大
        group_repr = x_grouped.max(dim=-1)[0]  # (batch, seq_len, n_groups)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return group_repr

