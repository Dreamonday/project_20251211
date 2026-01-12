#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crossformer 核心组件
版本: v0.3
日期: 20251230

实现Crossformer的核心组件：
- 5层残差模块（复用）
- Router Attention（Stage-1）
- Cross-Dimension Attention（Stage-2）
- Two-Stage Temporal Attention
- Two-Stage Feature Attention
- Hierarchical Aggregation
- CrossformerBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
import importlib.util

# 导入RoPE
try:
    from .rope import RotaryPositionEmbedding
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    rope_path = Path(__file__).parent / "rope.py"
    if rope_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("rope", rope_path)
        rope_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rope_module)
        RotaryPositionEmbedding = rope_module.RotaryPositionEmbedding
    else:
        raise ImportError(f"Cannot find rope.py at {rope_path}")


# 动态导入segment_utils
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 导入工具函数
models_path = Path(__file__).parent
try:
    segment_utils = _load_module(models_path / "segment_utils.py", "segment_utils")
    segment_sequence = segment_utils.segment_sequence
    merge_segments = segment_utils.merge_segments
    compute_segment_repr = segment_utils.compute_segment_repr
    gather_by_topk = segment_utils.gather_by_topk
    scatter_by_topk = segment_utils.scatter_by_topk
except Exception as e:
    raise ImportError(f"Failed to import segment_utils: {e}") from e


class ResidualBlock(nn.Module):
    """
    5层残差模块（与TSMixer相同）
    
    结构：
    输入 -> Linear(D→D) + GELU -> ... (5层) -> 输出 + 输入（残差连接）-> Dropout（可选）
    """
    
    def __init__(self, dim: int, activation: str = "gelu", dropout: float = None):
        """
        初始化5层残差模块
        
        Args:
            dim: 输入输出维度（保持不变）
            activation: 激活函数，"gelu"或"relu"
            dropout: Dropout比率（可选，如果为None则不使用Dropout）
        """
        super(ResidualBlock, self).__init__()
        
        # 5层全连接
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(5)
        ])
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (..., dim)
        
        Returns:
            输出张量，形状为 (..., dim)
        """
        identity = x
        
        # 5层前馈，每层都有激活
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        
        # 残差连接
        x = x + identity
        
        # Dropout（如果启用）
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class RouterAttention(nn.Module):
    """
    Router Attention（Stage-1）
    
    使用残差网络的深度注意力机制，用于评估每个item的重要性
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化Router Attention
        
        Args:
            d_model: 模型维度
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
        """
        super(RouterAttention, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 根据d_model确定结构（渐进式维度变化）
        if d_model >= 512:
            # 时间注意力Router（d_model=512或更大）
            # 512 → 512: ResidualBlock(512, dropout) → 256 → 256: ResidualBlock(256) → 128 → 128: ResidualBlock(128) → 64 → 64: ResidualBlock(64) → 32 → 32: ResidualBlock(32) → 1
            self.is_feature_router = False
            
            # 512维度的残差模块（最后一个有dropout）
            self.res1 = ResidualBlock(d_model, activation, dropout)
            self.proj1 = nn.Linear(d_model, 256)
            
            # 256维度的残差模块
            self.res2 = ResidualBlock(256, activation)
            self.proj2 = nn.Linear(256, 128)
            
            # 128维度的残差模块
            self.res3 = ResidualBlock(128, activation)
            self.proj3 = nn.Linear(128, 64)
            
            # 64维度的残差模块
            self.res4 = ResidualBlock(64, activation)
            self.proj4 = nn.Linear(64, 32)
            
            # 32维度的残差模块
            self.res5 = ResidualBlock(32, activation)
            self.proj5 = nn.Linear(32, 1)
        else:
            # 特征注意力Router（group_size通常较小，如64）
            # 使用渐进式降维：d_model → ... → 1
            self.is_feature_router = True
            
            # 构建渐进式降维路径
            dims = [d_model]
            temp_dim = d_model
            while temp_dim > 1:
                next_dim = max(temp_dim // 2, 1)
                if next_dim == temp_dim:
                    break
                dims.append(next_dim)
                temp_dim = next_dim
            
            # 确保最后是1
            if dims[-1] != 1:
                dims.append(1)
            
            # 去重
            dims = list(dict.fromkeys(dims))
            
            # 构建模块
            self.residuals = nn.ModuleList()
            self.projections = nn.ModuleList()
            
            for i in range(len(dims)):
                current = dims[i]
                
                # 第一个维度：添加残差模块
                if i == 0:
                    is_last_residual = (len(dims) == 2)
                    self.residuals.append(ResidualBlock(
                        current,
                        activation,
                        dropout if is_last_residual else None
                    ))
                
                # 如果不是最后一个维度，添加线性层
                if i < len(dims) - 1:
                    next_dim = dims[i + 1]
                    self.projections.append(nn.Linear(current, next_dim))
                    
                    # 如果不是最后一个维度变化，添加残差模块
                    if i < len(dims) - 2:
                        is_last_residual = (i == len(dims) - 3)
                        self.residuals.append(ResidualBlock(
                            next_dim,
                            activation,
                            dropout if is_last_residual else None
                        ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, n_items, d_model)
        
        Returns:
            重要性分数，形状为 (batch, n_items)
        """
        # 判断是时间注意力还是特征注意力
        if self.is_feature_router:
            # 特征注意力Router：渐进式降维
            # 使用动态构建的模块
            res_idx = 0
            proj_idx = 0
            
            # 第一个残差模块
            if len(self.residuals) > 0:
                x = self.residuals[res_idx](x)
                res_idx += 1
            
            # 渐进式降维
            for i in range(len(self.projections)):
                x = self.projections[proj_idx](x)
                proj_idx += 1
                
                # 如果不是最后一个投影，添加激活函数和残差模块
                if i < len(self.projections) - 1:
                    x = self.activation(x)
                    if res_idx < len(self.residuals):
                        x = self.residuals[res_idx](x)
                        res_idx += 1
                # 最后一个投影不需要激活（输出分数）
        else:
            # 时间注意力Router：512 → 512: ResidualBlock(512, dropout) → 256 → 256: ResidualBlock(256) → 128 → 128: ResidualBlock(128) → 64 → 64: ResidualBlock(64) → 32 → 32: ResidualBlock(32) → 1
            # 512维度的残差模块（最后一个有dropout）
            x = self.res1(x)  # (batch, n_items, 512)
            
            # 512 → 256
            x = self.proj1(x)  # (batch, n_items, 256)
            x = self.activation(x)
            
            # 256维度的残差模块
            x = self.res2(x)  # (batch, n_items, 256)
            
            # 256 → 128
            x = self.proj2(x)  # (batch, n_items, 128)
            x = self.activation(x)
            
            # 128维度的残差模块
            x = self.res3(x)  # (batch, n_items, 128)
            
            # 128 → 64
            x = self.proj3(x)  # (batch, n_items, 64)
            x = self.activation(x)
            
            # 64维度的残差模块
            x = self.res4(x)  # (batch, n_items, 64)
            
            # 64 → 32
            x = self.proj4(x)  # (batch, n_items, 32)
            x = self.activation(x)
            
            # 32维度的残差模块
            x = self.res5(x)  # (batch, n_items, 32)
            
            # 32 → 1（最后一步不需要激活）
            x = self.proj5(x)  # (batch, n_items, 1)
        
        # 计算重要性分数
        scores = x.squeeze(-1)  # (batch, n_items)
        
        return scores


class CrossDimensionAttention(nn.Module):
    """
    Cross-Dimension Attention（Stage-2）
    
    标准Multi-head Attention，用于计算item间的交互
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        use_rope: bool = True
    ):
        """
        初始化Cross-Dimension Attention
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout比率
            max_seq_len: 最大序列长度（用于RoPE）
            use_rope: 是否使用RoPE位置编码（默认True）
        """
        super(CrossDimensionAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert (d_model // n_heads) % 2 == 0, "d_k must be even for RoPE"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rope = use_rope
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        # RoPE位置编码
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                d_model=self.d_k,
                max_seq_len=max_seq_len,
                base=10000
            )
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, n_items, d_model)
            position_ids: 位置索引，形状为 (batch, n_items) 或 (1, n_items)
                         如果为None，则使用默认位置 [0, 1, 2, ..., n_items-1]
            mask: 掩码（可选）
        
        Returns:
            注意力输出，形状为 (batch, n_items, d_model)
        """
        batch_size, n_items, d_model = x.shape
        
        # 线性投影并分离多头
        Q = self.W_q(x).view(batch_size, n_items, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, n_items, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, n_items, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, n_items, d_k)
        
        # 应用RoPE位置编码
        if self.use_rope:
            Q = self.rope(Q, position_ids)
            K = self.rope(K, position_ids)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores: (batch, n_heads, n_items, n_items)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, n_heads, n_items, d_k)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, n_items, d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output


class TwoStageTemporalAttention(nn.Module):
    """
    Two-Stage Temporal Attention
    
    跨segment的时间注意力，使用Two-Stage机制降低计算量
    """
    
    def __init__(
        self,
        d_model: int,
        n_segments: int,
        n_heads: int,
        topk_ratio: float = 0.5,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化Two-Stage Temporal Attention
        
        Args:
            d_model: 模型维度
            n_segments: segment数量
            n_heads: 注意力头数
            topk_ratio: Router保留的比例（0-1）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
        """
        super(TwoStageTemporalAttention, self).__init__()
        
        self.d_model = d_model
        self.n_segments = n_segments
        self.topk_k = max(1, int(n_segments * topk_ratio))
        
        # Stage-1: Router
        self.router = RouterAttention(d_model, dropout, activation)
        
        # Stage-2: Cross-Dimension Attention
        self.cross_attn = CrossDimensionAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            dropout=dropout,
            max_seq_len=1000,  # RoPE最大序列长度设为1000
            use_rope=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
        
        Returns:
            输出张量，形状为 (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 分段
        x_seg = segment_sequence(x, self.n_segments)
        # x_seg: (batch, n_segments, seg_len, d_model)
        
        # 2. 计算每个segment的代表向量
        seg_repr = compute_segment_repr(x_seg, method="mean")
        # seg_repr: (batch, n_segments, d_model)
        
        # ===== Stage-1: Router =====
        # 计算重要性分数
        importance_scores = self.router(seg_repr)
        # importance_scores: (batch, n_segments)
        
        # 选择TopK个重要的segments
        topk_scores, topk_indices = torch.topk(
            importance_scores, k=self.topk_k, dim=1
        )
        # topk_indices: (batch, topk_k)
        
        # ===== Stage-2: Cross-Segment Attention =====
        # 提取重要的segments
        selected_seg_repr = gather_by_topk(seg_repr, topk_indices, dim=1)
        # selected_seg_repr: (batch, topk_k, d_model)
        
        # 获取选中segments的位置索引（segment索引）
        selected_position_ids = topk_indices  # (batch, topk_k)
        
        # 对选中的segments做cross-attention，传入segment位置索引
        attn_output = self.cross_attn(selected_seg_repr, position_ids=selected_position_ids)
        # attn_output: (batch, topk_k, d_model)
        
        # 将结果放回原位置
        seg_repr_updated = scatter_by_topk(seg_repr, attn_output, topk_indices, dim=1)
        # seg_repr_updated: (batch, n_segments, d_model)
        
        # 3. 将segment代表向量广播回原序列
        # 每个segment内的所有时间步共享相同的更新
        seg_repr_expanded = seg_repr_updated.unsqueeze(2).expand_as(x_seg)
        # seg_repr_expanded: (batch, n_segments, seg_len, d_model)
        
        # 4. 合并回原序列
        output = merge_segments(seg_repr_expanded)
        # output: (batch, seq_len, d_model)
        
        return output


class TwoStageFeatureAttention(nn.Module):
    """
    Two-Stage Feature Attention
    
    跨特征组的注意力，使用Two-Stage机制降低计算量
    
    注意：这里我们将d_model视为"特征"维度
    """
    
    def __init__(
        self,
        d_model: int,
        n_feature_groups: int,
        n_heads: int,
        topk_ratio: float = 0.5,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化Two-Stage Feature Attention
        
        Args:
            d_model: 模型维度（会被分为n_feature_groups组）
            n_feature_groups: 特征分组数
            n_heads: 注意力头数
            topk_ratio: Router保留的比例（0-1）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
        """
        super(TwoStageFeatureAttention, self).__init__()
        
        assert d_model % n_feature_groups == 0, \
            f"d_model ({d_model}) must be divisible by n_feature_groups ({n_feature_groups})"
        
        self.d_model = d_model
        self.n_feature_groups = n_feature_groups
        self.group_size = d_model // n_feature_groups
        self.topk_k = max(1, int(n_feature_groups * topk_ratio))
        
        # Stage-1: Router
        self.router = RouterAttention(self.group_size, dropout, activation)
        
        # Stage-2: Cross-Dimension Attention
        self.cross_attn = CrossDimensionAttention(
            d_model=self.group_size,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=1000,  # RoPE最大序列长度设为1000
            use_rope=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
        
        Returns:
            输出张量，形状为 (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 特征分组
        # (batch, seq_len, d_model) -> (batch, seq_len, n_groups, group_size)
        x_grouped = x.view(batch_size, seq_len, self.n_feature_groups, self.group_size)
        
        # 2. 对每个时间步，计算每个特征组的代表向量（已经是group_size维度）
        # Reshape以便Router处理
        # (batch, seq_len, n_groups, group_size) -> (batch*seq_len, n_groups, group_size)
        x_grouped_flat = x_grouped.view(batch_size * seq_len, self.n_feature_groups, self.group_size)
        
        # ===== Stage-1: Router =====
        # 计算重要性分数
        importance_scores = self.router(x_grouped_flat)
        # importance_scores: (batch*seq_len, n_groups)
        
        # 选择TopK个重要的feature groups
        topk_scores, topk_indices = torch.topk(
            importance_scores, k=self.topk_k, dim=1
        )
        # topk_indices: (batch*seq_len, topk_k)
        
        # ===== Stage-2: Cross-Group Attention =====
        # 提取重要的feature groups
        selected_groups = gather_by_topk(x_grouped_flat, topk_indices, dim=1)
        # selected_groups: (batch*seq_len, topk_k, group_size)
        
        # 获取选中groups的位置索引（feature group索引）
        selected_position_ids = topk_indices  # (batch*seq_len, topk_k)
        
        # 对选中的groups做cross-attention，传入group位置索引
        attn_output = self.cross_attn(selected_groups, position_ids=selected_position_ids)
        # attn_output: (batch*seq_len, topk_k, group_size)
        
        # 将结果放回原位置
        x_grouped_flat_updated = scatter_by_topk(
            x_grouped_flat, attn_output, topk_indices, dim=1
        )
        # x_grouped_flat_updated: (batch*seq_len, n_groups, group_size)
        
        # 3. Reshape回原形状
        output = x_grouped_flat_updated.view(batch_size, seq_len, d_model)
        
        return output


class HierarchicalAggregation(nn.Module):
    """
    层次化时间聚合
    
    使用多尺度segment表示 + attention加权融合
    """
    
    def __init__(
        self,
        d_model: int,
        n_segments: int,
        dropout: float = 0.1
    ):
        """
        初始化层次化聚合
        
        Args:
            d_model: 模型维度
            n_segments: segment数量
            dropout: Dropout比率
        """
        super(HierarchicalAggregation, self).__init__()
        
        self.d_model = d_model
        self.n_segments = n_segments
        
        # Attention权重计算
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
        
        Returns:
            聚合后的向量，形状为 (batch, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 分段
        x_seg = segment_sequence(x, self.n_segments)
        # x_seg: (batch, n_segments, seg_len, d_model)
        
        # 2. 计算每个segment的代表向量
        seg_repr = compute_segment_repr(x_seg, method="mean")
        # seg_repr: (batch, n_segments, d_model)
        
        # 3. 计算attention权重
        attn_scores = self.attn_mlp(seg_repr).squeeze(-1)
        # attn_scores: (batch, n_segments)
        
        attn_weights = F.softmax(attn_scores, dim=1)
        # attn_weights: (batch, n_segments)
        
        # 4. 加权融合
        # (batch, n_segments, d_model) * (batch, n_segments, 1) -> (batch, n_segments, d_model)
        weighted_seg = seg_repr * attn_weights.unsqueeze(-1)
        
        # 5. 求和得到全局表示
        global_repr = weighted_seg.sum(dim=1)
        # global_repr: (batch, d_model)
        
        return global_repr


class CrossformerBlock(nn.Module):
    """
    Crossformer Block
    
    包含Two-Stage时间注意力和Two-Stage特征注意力
    使用Pre-LayerNorm和残差连接
    """
    
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        n_segments: int,
        n_feature_groups: int,
        n_heads: int,
        topk_ratio: float = 0.5,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = False
    ):
        """
        初始化Crossformer Block
        
        Args:
            d_model: 模型维度
            seq_len: 序列长度
            n_segments: 时间分段数
            n_feature_groups: 特征分组数
            n_heads: 注意力头数
            topk_ratio: Router保留比例
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm（默认True）
        """
        super(CrossformerBlock, self).__init__()
        
        self.use_layernorm = use_layernorm
        
        # Two-Stage时间注意力
        self.temporal_attn = TwoStageTemporalAttention(
            d_model=d_model,
            n_segments=n_segments,
            n_heads=n_heads,
            topk_ratio=topk_ratio,
            dropout=dropout,
            activation=activation
        )
        
        # Two-Stage特征注意力
        self.feature_attn = TwoStageFeatureAttention(
            d_model=d_model,
            n_feature_groups=n_feature_groups,
            n_heads=n_heads,
            topk_ratio=topk_ratio,
            dropout=dropout,
            activation=activation
        )
        
        # LayerNorm（保留层，但可通过use_layernorm控制是否使用）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
        
        Returns:
            输出张量，形状为 (batch, seq_len, d_model)
        """
        # Pre-Norm + Two-Stage Temporal Attention + 残差
        if self.use_layernorm:
            x = x + self.temporal_attn(self.norm1(x))
        else:
            x = x + self.temporal_attn(x)
        
        # Pre-Norm + Two-Stage Feature Attention + 残差
        if self.use_layernorm:
            x = x + self.feature_attn(self.norm2(x))
        else:
            x = x + self.feature_attn(x)
        
        return x

