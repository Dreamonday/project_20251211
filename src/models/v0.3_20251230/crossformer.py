#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crossformer 主模型
版本: v0.3
日期: 20251230

实现Crossformer模型，用于多变量时间序列预测
采用Two-Stage Attention机制，实现跨维度交互
完全兼容项目的训练框架（与iTransformer、TSMixer接口一致）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from pathlib import Path
import importlib.util


# 动态导入模块
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 导入Crossformer组件
models_path = Path(__file__).parent
try:
    blocks_module = _load_module(models_path / "crossformer_blocks.py", "crossformer_blocks")
    CrossformerBlock = blocks_module.CrossformerBlock
    HierarchicalAggregation = blocks_module.HierarchicalAggregation
    ResidualBlock = blocks_module.ResidualBlock
    # 导入RoPE
    rope_module = _load_module(models_path / "rope.py", "rope")
    RotaryPositionEmbedding = rope_module.RotaryPositionEmbedding
except Exception as e:
    raise ImportError(f"Failed to import Crossformer blocks: {e}") from e


def build_progressive_projection(
    input_dim: int,
    output_dim: int,
    activation: nn.Module,
    dropout: float,
    residual_block_class: type
) -> nn.Module:
    """
    构建渐进式维度变化投影
    
    结构：每个维度变化都有激活函数，维度变化过程中有残差模块参与
    例如：64 → 512
    - 64 → 64: ResidualBlock(64, dropout)  # 第一个残差模块，最后一个有dropout
    - 64 → 128: Linear + Activation
    - 128 → 128: ResidualBlock(128)
    - 128 → 256: Linear + Activation
    - 256 → 256: ResidualBlock(256)
    - 256 → 512: Linear + Activation
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        activation: 激活函数模块
        dropout: Dropout比率
        residual_block_class: 残差模块类
    
    Returns:
        nn.Sequential模块
    """
    modules = []
    
    # 如果输入输出维度相同，只使用残差模块（带dropout）
    if input_dim == output_dim:
        modules.append(residual_block_class(
            input_dim,
            activation="gelu" if isinstance(activation, nn.GELU) else "relu",
            dropout=dropout
        ))
        return nn.Sequential(*modules)
    
    # 确定中间维度（渐进式增长或减小）
    current_dim = input_dim
    target_dim = output_dim
    
    # 计算中间维度路径
    # 如果目标维度更大，逐步增长：64 -> 128 -> 256 -> 512
    # 如果目标维度更小，逐步减小：512 -> 256 -> 128 -> 64
    dims = [current_dim]
    if target_dim > current_dim:
        # 增长路径
        temp_dim = current_dim
        while temp_dim < target_dim:
            # 每次至少翻倍，但不超过目标维度
            next_dim = min(temp_dim * 2, target_dim)
            if next_dim == temp_dim:
                break
            dims.append(next_dim)
            temp_dim = next_dim
    else:
        # 减小路径
        temp_dim = current_dim
        while temp_dim > target_dim:
            # 每次至少减半，但不小于目标维度
            next_dim = max(temp_dim // 2, target_dim)
            if next_dim == temp_dim:
                break
            dims.append(next_dim)
            temp_dim = next_dim
    
    # 确保包含目标维度
    if dims[-1] != target_dim:
        dims.append(target_dim)
    
    # 去重并保持顺序
    dims = list(dict.fromkeys(dims))
    
    # 确定激活函数字符串
    act_str = "gelu" if isinstance(activation, nn.GELU) else "relu"
    
    # 构建模块序列
    for i in range(len(dims)):
        current = dims[i]
        
        # 第一个维度：添加残差模块
        if i == 0:
            # 判断是否是最后一个残差模块（如果只有一个维度变化，就是最后一个）
            is_last_residual = (len(dims) == 2)
            modules.append(residual_block_class(
                current,
                activation=act_str,
                dropout=dropout if is_last_residual else None
            ))
        
        # 如果不是最后一个维度，添加线性层和激活函数
        if i < len(dims) - 1:
            next_dim = dims[i + 1]
            modules.append(nn.Linear(current, next_dim))
            modules.append(activation)
            
            # 如果不是最后一个维度变化，添加残差模块
            if i < len(dims) - 2:
                # 判断是否是最后一个残差模块
                is_last_residual = (i == len(dims) - 3)
                modules.append(residual_block_class(
                    next_dim,
                    activation=act_str,
                    dropout=dropout if is_last_residual else None
                ))
    
    return nn.Sequential(*modules)


class Crossformer(nn.Module):
    """
    Crossformer模型
    
    用于多变量时间序列预测的深度神经网络
    基于Two-Stage Attention的跨维度交互
    
    模型结构：
    1. 输入: (batch, seq_len, n_features)
    2. 输入投影: n_features -> d_model
    3. Crossformer块 × n_blocks (Two-Stage时间注意力 + Two-Stage特征注意力)
    4. 层次化聚合: (batch, seq_len, d_model) -> (batch, d_model)
    5. 输出投影: 多层残差降维 (d_model -> 64 -> 32 -> 16 -> prediction_len)
    6. 输出: (batch, prediction_len)
    
    完全兼容训练脚本，接口与iTransformer、TSMixer一致
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        prediction_len: int = 1,
        d_model: int = 512,
        n_blocks: int = 4,
        n_heads: int = 8,
        # Crossformer特有参数
        n_segments: int = 50,
        n_feature_groups: int = 8,
        router_topk_ratio: float = 0.5,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = False,  # 是否在CrossformerBlock中使用LayerNorm（默认关闭）
        # 为了兼容训练脚本的配置加载，接受额外的配置参数
        ff_dim: int = 2048,  # 保留以兼容，但不使用
        norm_type: str = "layer",  # 保留以兼容，但不使用
        temporal_aggregation_config: Optional[Dict] = None,
        output_projection_config: Optional[Dict] = None,
        use_input_rope: bool = True,  # 是否在输入投影后使用RoPE位置编码
        rope_max_seq_len: int = 1000,  # RoPE最大序列长度
        n_rope_heads: int = 8,  # 输入级RoPE的头数（默认8，与n_heads一致）
        rope_alpha: float = 0.1  # RoPE位置编码的缩放因子（可学习，初始值0.1）
    ):
        """
        初始化Crossformer模型
        
        Args:
            seq_len: 输入序列长度
            n_features: 特征数量
            prediction_len: 预测长度（输出维度）
            d_model: 模型隐藏维度
            n_blocks: Crossformer块的数量（默认4）
            n_heads: 注意力头数（默认8）
            n_segments: 时间分段数（默认50，seq_len=500时每段10步）
            n_feature_groups: 特征分组数（默认8，d_model必须能整除）
            router_topk_ratio: Router保留比例（默认0.5，保留50%）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            use_layernorm: 是否在CrossformerBlock中使用LayerNorm（默认True）
            ff_dim: 前馈网络隐藏维度（保留以兼容，实际不使用）
            norm_type: 归一化类型（保留以兼容，实际使用LayerNorm）
            temporal_aggregation_config: 时间聚合配置（可选）
            output_projection_config: 输出投影配置（可选）
            use_input_rope: 是否在输入投影后使用RoPE位置编码（默认True）
            rope_max_seq_len: RoPE最大序列长度（默认1000）
            n_rope_heads: 输入级RoPE的头数（默认8，与n_heads一致）
            rope_alpha: RoPE位置编码的缩放因子（默认0.1，可学习参数）
        """
        super(Crossformer, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.prediction_len = prediction_len
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_segments = n_segments
        self.n_feature_groups = n_feature_groups
        self.use_input_rope = use_input_rope
        self.n_rope_heads = n_rope_heads
        
        # 验证参数
        if seq_len % n_segments != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by n_segments ({n_segments})"
            )
        if d_model % n_feature_groups != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_feature_groups ({n_feature_groups})"
            )
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model ({d_model}) must be even for RoPE"
            )
        if d_model % n_rope_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_rope_heads ({n_rope_heads})"
            )
        
        # 默认配置
        temporal_aggregation_config = temporal_aggregation_config or {}
        output_projection_config = output_projection_config or {}
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ========== 输入投影（渐进式维度变化） ==========
        # n_features -> d_model（渐进式：每个维度变化都有激活函数和残差模块）
        self.input_proj = build_progressive_projection(
            input_dim=n_features,
            output_dim=d_model,
            activation=self.activation,
            dropout=dropout,
            residual_block_class=ResidualBlock
        )
        
        # ========== 输入级RoPE位置编码（多头融合方案） ==========
        # 在输入投影后，对整个序列的每个时间步应用多头RoPE位置编码
        if use_input_rope:
            # 计算每个头的维度
            d_k = d_model // n_rope_heads
            
            # 验证d_k必须是偶数（RoPE要求）
            if d_k % 2 != 0:
                raise ValueError(
                    f"d_k ({d_k} = d_model/{n_rope_heads}) must be even for RoPE"
                )
            
            # 创建多头RoPE（每个头独立应用RoPE）
            self.input_rope = RotaryPositionEmbedding(
                d_model=d_k,  # 每个头的维度
                max_seq_len=rope_max_seq_len,
                base=10000
            )
            
            # 可学习的缩放因子（初始值较小，让模型逐步学习位置编码的重要性）
            self.rope_alpha = nn.Parameter(torch.tensor(rope_alpha, dtype=torch.float32))
        
        # ========== Crossformer块堆叠 ==========
        self.blocks = nn.ModuleList([
            CrossformerBlock(
                d_model=d_model,
                seq_len=seq_len,
                n_segments=n_segments,
                n_feature_groups=n_feature_groups,
                n_heads=n_heads,
                topk_ratio=router_topk_ratio,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm
            )
            for _ in range(n_blocks)
        ])
        
        # ========== 层次化时间聚合 ==========
        agg_type = temporal_aggregation_config.get('type', 'hierarchical')
        if agg_type == 'hierarchical':
            self.aggregation = HierarchicalAggregation(
                d_model=d_model,
                n_segments=n_segments,
                dropout=dropout
            )
        else:
            # 简单平均池化
            self.aggregation = None
        
        # ========== 输出投影层（渐进式多层残差降维） ==========
        # 结构：d_model → 64 → 32 → 16 → prediction_len
        # 每个维度变化都有激活函数，维度变化过程中有残差模块参与
        
        # d_model维度的残差模块（最后一个有dropout）
        self.output_res1 = ResidualBlock(d_model, activation, dropout)
        
        # d_model → 64
        self.output_proj1 = nn.Linear(d_model, 64)
        
        # 64维度的残差模块
        self.output_res2 = ResidualBlock(64, activation)
        
        # 64 → 32
        self.output_proj2 = nn.Linear(64, 32)
        
        # 32维度的残差模块
        self.output_res3 = ResidualBlock(32, activation)
        
        # 32 → 16
        self.output_proj3 = nn.Linear(32, 16)
        
        # 16维度的残差模块
        self.output_res4 = ResidualBlock(16, activation)
        
        # 16 → prediction_len（最后一步不需要激活）
        self.output_proj4 = nn.Linear(16, prediction_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, prediction_len)
            
        注意：为了兼容训练脚本，当prediction_len=1时，输出形状为(batch, 1)
        """
        # x: (batch, seq_len, n_features)
        
        # ========== 输入投影 ==========
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # ========== 输入级RoPE位置编码（多头融合方案） ==========
        # 对整个序列的每个时间步应用多头RoPE位置编码
        if self.use_input_rope:
            batch_size, seq_len, d_model = x.shape
            
            # 1. 将输入分成多个头
            # (batch, seq_len, d_model) -> (batch, seq_len, n_rope_heads, d_k)
            d_k = d_model // self.n_rope_heads
            x_multihead = x.view(batch_size, seq_len, self.n_rope_heads, d_k)
            
            # 2. 转换为RoPE期望的形状：(batch, n_rope_heads, seq_len, d_k)
            x_multihead = x_multihead.transpose(1, 2)  # (batch, n_rope_heads, seq_len, d_k)
            
            # 3. 对每个头应用RoPE（使用连续位置编码：0, 1, 2, ..., seq_len-1）
            # position_ids=None表示使用默认连续位置 [0, 1, 2, ..., seq_len-1]
            x_rope_multihead = self.input_rope(x_multihead)  # (batch, n_rope_heads, seq_len, d_k)
            
            # 4. 转换回原形状并融合多头
            # (batch, n_rope_heads, seq_len, d_k) -> (batch, seq_len, n_rope_heads, d_k)
            x_rope_multihead = x_rope_multihead.transpose(1, 2).contiguous()
            # (batch, seq_len, n_rope_heads, d_k) -> (batch, seq_len, d_model)
            x_rope = x_rope_multihead.view(batch_size, seq_len, d_model)
            
            # 5. 通过残差连接添加位置编码（使用可学习的缩放因子）
            # 保留原始输入信息，位置编码作为增量信息
            x = x + self.rope_alpha * x_rope  # (batch, seq_len, d_model)
        
        # ========== Crossformer块堆叠 ==========
        for block in self.blocks:
            x = block(x)  # (batch, seq_len, d_model)
        
        # ========== 层次化时间聚合 ==========
        if self.aggregation is not None:
            x = self.aggregation(x)  # (batch, d_model)
        else:
            # 简单平均池化
            x = x.mean(dim=1)  # (batch, d_model)
        
        # ========== 输出投影（渐进式多层残差降维） ==========
        # d_model维度的残差模块（最后一个有dropout）
        x = self.output_res1(x)  # (batch, d_model)
        
        # d_model → 64
        x = self.output_proj1(x)  # (batch, 64)
        x = self.activation(x)
        
        # 64维度的残差模块
        x = self.output_res2(x)  # (batch, 64)
        
        # 64 → 32
        x = self.output_proj2(x)  # (batch, 32)
        x = self.activation(x)
        
        # 32维度的残差模块
        x = self.output_res3(x)  # (batch, 32)
        
        # 32 → 16
        x = self.output_proj3(x)  # (batch, 16)
        x = self.activation(x)
        
        # 16维度的残差模块
        x = self.output_res4(x)  # (batch, 16)
        
        # 16 → prediction_len（最后一步不需要激活）
        x = self.output_proj4(x)  # (batch, prediction_len)
        
        return x
    
    def get_num_parameters(self) -> int:
        """
        返回模型的可训练参数数量
        
        此方法是为了兼容训练脚本
        
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== 配置加载辅助函数 ==========
def create_crossformer_from_config(config: Dict) -> Crossformer:
    """
    从配置字典创建Crossformer模型
    
    此函数用于从YAML配置文件创建模型，保持与训练脚本的兼容性
    
    Args:
        config: 模型配置字典
    
    Returns:
        Crossformer模型实例
    """
    model_cfg = config.get('model', config)  # 兼容嵌套和非嵌套配置
    
    return Crossformer(
        seq_len=model_cfg.get('seq_len', 500),
        n_features=model_cfg.get('n_features', 64),
        prediction_len=model_cfg.get('prediction_len', 1),
        d_model=model_cfg.get('d_model', 512),
        n_blocks=model_cfg.get('n_blocks', 4),
        n_heads=model_cfg.get('n_heads', 8),
        n_segments=model_cfg.get('n_segments', 50),
        n_feature_groups=model_cfg.get('n_feature_groups', 8),
        router_topk_ratio=model_cfg.get('router_topk_ratio', 0.5),
        dropout=model_cfg.get('dropout', 0.1),
        activation=model_cfg.get('activation', 'gelu'),
        use_layernorm=model_cfg.get('use_layernorm', False),
        ff_dim=model_cfg.get('ff_dim', 2048),
        norm_type=model_cfg.get('norm_type', 'layer'),
        temporal_aggregation_config=model_cfg.get('temporal_aggregation', {}),
        output_projection_config=model_cfg.get('output_projection', {}),
        use_input_rope=model_cfg.get('use_input_rope', True),
        rope_max_seq_len=model_cfg.get('rope_max_seq_len', 1000),
        n_rope_heads=model_cfg.get('n_rope_heads', 8),
        rope_alpha=model_cfg.get('rope_alpha', 0.1)
    )

