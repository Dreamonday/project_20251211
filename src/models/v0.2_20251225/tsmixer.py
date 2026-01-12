#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSMixer 主模型
版本: v0.2
日期: 20251225

实现TSMixer模型，用于多变量时间序列预测
完全兼容项目的训练框架（与iTransformer接口一致）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from pathlib import Path
import importlib.util


# 动态导入TSMixerBlock
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 导入TSMixerBlock
models_path = Path(__file__).parent
try:
    blocks_module = _load_module(models_path / "tsmixer_blocks.py", "tsmixer_blocks")
    TSMixerBlock = blocks_module.TSMixerBlock
except Exception as e:
    raise ImportError(f"Failed to import TSMixerBlock: {e}") from e


class TSMixer(nn.Module):
    """
    TSMixer模型
    
    用于多变量时间序列预测的轻量级神经网络
    基于MLP的时间和特征混合，无需注意力机制
    
    模型结构：
    1. 输入: (batch, seq_len, n_features)
    2. TSMixer块 × n_blocks (时间混合 + 特征混合)
    3. 时间聚合层 (seq_len -> 1)
    4. 输出层 (n_features -> prediction_len)
    5. 输出: (batch, prediction_len)
    
    完全兼容训练脚本，接口与iTransformer一致
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        prediction_len: int = 1,
        n_blocks: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
        # 为了兼容训练脚本的配置加载，接受额外的配置参数
        temporal_aggregation_config: Optional[Dict] = None,
        output_projection_config: Optional[Dict] = None
    ):
        """
        初始化TSMixer模型
        
        Args:
            seq_len: 输入序列长度
            n_features: 特征数量
            prediction_len: 预测长度（输出维度）
            n_blocks: TSMixer块的数量
            ff_dim: 前馈网络隐藏维度
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            norm_type: 归一化类型，"layer"
            temporal_aggregation_config: 时间聚合配置（可选）
            output_projection_config: 输出投影配置（可选）
        """
        super(TSMixer, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.prediction_len = prediction_len
        self.n_blocks = n_blocks
        
        # 默认配置
        temporal_aggregation_config = temporal_aggregation_config or {}
        output_projection_config = output_projection_config or {}
        
        # ========== TSMixer块堆叠 ==========
        self.mixer_blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                n_features=n_features,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type
            )
            for _ in range(n_blocks)
        ])
        
        # ========== 时间维度聚合 ==========
        # 将时间序列聚合为单个向量
        # 方案1：简单平均池化（默认）
        # 方案2：可学习的注意力聚合（配置启用）
        
        use_attention_pooling = temporal_aggregation_config.get('use_attention', False)
        
        if use_attention_pooling:
            # 可学习的注意力聚合
            self.temporal_aggregation = nn.Sequential(
                nn.Linear(seq_len, seq_len // 2),
                nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(seq_len // 2, 1),
                nn.Softmax(dim=1)
            )
            self.use_attention_pooling = True
        else:
            # 简单的平均池化（更稳定，推荐用于初始训练）
            self.temporal_aggregation = None
            self.use_attention_pooling = False
        
        # ========== 输出投影层 ==========
        # 将特征维度投影到预测维度
        # 使用多层MLP，提供更强的表达能力
        
        output_hidden_dim = output_projection_config.get('hidden_dim', n_features // 2)
        
        self.output_projection = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, output_hidden_dim),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden_dim, prediction_len)
        )
        
        # 最终LayerNorm
        self.final_norm = nn.LayerNorm(n_features)
    
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
        
        # ========== TSMixer块堆叠 ==========
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)  # (batch, seq_len, n_features)
        
        # 最终归一化
        x = self.final_norm(x)  # (batch, seq_len, n_features)
        
        # ========== 时间维度聚合 ==========
        if self.use_attention_pooling:
            # 可学习的注意力聚合
            # x: (batch, seq_len, n_features)
            # 转置: (batch, n_features, seq_len)
            x_transposed = x.transpose(1, 2)
            
            # 计算注意力权重: (batch, n_features, 1)
            attention_weights = self.temporal_aggregation(x_transposed)
            
            # 加权求和: (batch, n_features, seq_len) * (batch, n_features, 1) -> (batch, n_features)
            x = (x_transposed * attention_weights).sum(dim=2)
        else:
            # 简单的平均池化
            x = x.mean(dim=1)  # (batch, n_features)
        
        # ========== 输出投影 ==========
        x = self.output_projection(x)  # (batch, prediction_len)
        
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
def create_tsmixer_from_config(config: Dict) -> TSMixer:
    """
    从配置字典创建TSMixer模型
    
    此函数用于从YAML配置文件创建模型，保持与训练脚本的兼容性
    
    Args:
        config: 模型配置字典
    
    Returns:
        TSMixer模型实例
    """
    model_cfg = config.get('model', config)  # 兼容嵌套和非嵌套配置
    
    return TSMixer(
        seq_len=model_cfg.get('seq_len', 100),
        n_features=model_cfg.get('n_features', 40),
        prediction_len=model_cfg.get('prediction_len', 1),
        n_blocks=model_cfg.get('n_blocks', 8),
        ff_dim=model_cfg.get('ff_dim', 2048),
        dropout=model_cfg.get('dropout', 0.1),
        activation=model_cfg.get('activation', 'gelu'),
        norm_type=model_cfg.get('norm_type', 'layer'),
        temporal_aggregation_config=model_cfg.get('temporal_aggregation', {}),
        output_projection_config=model_cfg.get('output_projection', {})
    )

