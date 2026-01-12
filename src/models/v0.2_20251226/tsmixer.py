#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSMixer 主模型（深度残差版本）
版本: v0.2
日期: 20251226

实现TSMixer模型，用于多变量时间序列预测
采用深度残差架构，支持可选的LayerNorm和残差连接
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


# 导入TSMixerBlock和ResidualBlock
models_path = Path(__file__).parent
try:
    blocks_module = _load_module(models_path / "tsmixer_blocks.py", "tsmixer_blocks")
    TSMixerBlock = blocks_module.TSMixerBlock
    ResidualBlock = blocks_module.ResidualBlock
except Exception as e:
    raise ImportError(f"Failed to import TSMixerBlock: {e}") from e


class TSMixer(nn.Module):
    """
    TSMixer模型（深度残差版本）
    
    用于多变量时间序列预测的深度神经网络
    基于MLP的时间和特征混合，采用多层残差架构
    
    模型结构：
    1. 输入: (batch, seq_len, n_features)
    2. TSMixer块 × n_blocks (时间混合 + 特征混合，可选LayerNorm和残差连接)
    3. LayerNorm（可选）-> 时间聚合层 (平均池化)
    4. 输出投影层 (多层残差降维)
    5. 输出: (batch, prediction_len)
    
    完全兼容训练脚本，接口与iTransformer一致
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        prediction_len: int = 1,
        n_blocks: int = 4,
        ff_dim: int = 2048,  # 保留参数以兼容配置，但不使用
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",  # 保留参数以兼容配置
        use_layernorm: bool = True,  # 是否启用LayerNorm
        use_residual: bool = True,  # 是否启用残差连接
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
            n_blocks: TSMixer块的数量（默认4）
            ff_dim: 前馈网络隐藏维度（保留以兼容，实际架构固定）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            norm_type: 归一化类型（保留以兼容）
            use_layernorm: 是否启用LayerNorm（默认True）
            use_residual: 是否启用残差连接（默认True）
            temporal_aggregation_config: 时间聚合配置（可选）
            output_projection_config: 输出投影配置（可选）
        """
        super(TSMixer, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.prediction_len = prediction_len
        self.n_blocks = n_blocks
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.norm_type = norm_type
        self.dropout = dropout
        self.activation_name = activation
        
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
        
        # ========== TSMixer块堆叠 ==========
        self.mixer_blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                n_features=n_features,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
                use_residual=use_residual
            )
            for _ in range(n_blocks)
        ])
        
        # ========== 时间维度聚合 ==========
        # 使用简单的平均池化
        # x.mean(dim=1): (batch, seq_len, n_features) -> (batch, n_features)
        
        # LayerNorm（如果启用，在时间聚合前）
        if use_layernorm:
            self.final_norm = nn.LayerNorm(n_features)
        
        # ========== 输出投影层（多层残差降维） ==========
        # 结构：n_features → 64维残差 → 32维残差 → 1
        # 
        # n_features维度的5层残差模块
        self.output_res1 = ResidualBlock(n_features, activation)
        
        # n_features → 64
        self.output_proj1 = nn.Linear(n_features, 64)
        self.output_dropout1 = nn.Dropout(dropout)
        
        # 64维度的5层残差模块
        self.output_res2 = ResidualBlock(64, activation)
        
        # 64 → 32
        self.output_proj2 = nn.Linear(64, 32)
        self.output_dropout2 = nn.Dropout(dropout)
        
        # 32维度的5层残差模块
        self.output_res3 = ResidualBlock(32, activation)
        
        # 32 → 16
        self.output_proj3 = nn.Linear(32, 16)
        self.output_dropout3 = nn.Dropout(dropout)
        
        # 16 → prediction_len
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
        
        # ========== TSMixer块堆叠 ==========
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)  # (batch, seq_len, n_features)
        
        # ========== 时间维度聚合前的LayerNorm（如果启用） ==========
        if self.use_layernorm:
            x = self.final_norm(x)  # (batch, seq_len, n_features)
        
        # ========== 时间维度聚合（平均池化） ==========
        x = x.mean(dim=1)  # (batch, n_features)
        
        # ========== 输出投影（多层残差降维） ==========
        # n_features维度的残差模块
        x = self.output_res1(x)  # (batch, n_features)
        
        # n_features → 64
        x = self.output_proj1(x)  # (batch, 64)
        x = self.activation(x)
        x = self.output_dropout1(x)
        
        # 64维度的残差模块
        x = self.output_res2(x)  # (batch, 64)
        
        # 64 → 32
        x = self.output_proj2(x)  # (batch, 32)
        x = self.activation(x)
        x = self.output_dropout2(x)
        
        # 32维度的残差模块
        x = self.output_res3(x)  # (batch, 32)
        
        # 32 → 16
        x = self.output_proj3(x)  # (batch, 16)
        x = self.activation(x)
        x = self.output_dropout3(x)
        
        # 16 → prediction_len
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
    
    def get_model_info(self) -> Dict:
        """
        返回模型的详细信息（用于保存到配置文件）
        
        Returns:
            包含模型详细信息的字典
        """
        return {
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'prediction_len': self.prediction_len,
            'n_blocks': self.n_blocks,
            'dropout': self.dropout,
            'activation': self.activation_name,
            'norm_type': self.norm_type,
            'use_layernorm': self.use_layernorm,
            'use_residual': self.use_residual,
            'num_parameters': self.get_num_parameters(),
            'architecture': {
                'mixer_blocks': {
                    'count': self.n_blocks,
                    'time_mixing': {
                        'path': 'seq_len → 512 → 1024 → 512 → seq_len',
                        'residual_dims': [512, 1024, 512]
                    },
                    'feature_mixing': {
                        'path': 'n_features → 512 → 1024 → 512 → n_features',
                        'residual_dims': [512, 1024, 512]
                    },
                    'layernorm': 'enabled' if self.use_layernorm else 'disabled',
                    'residual_connection': 'enabled' if self.use_residual else 'disabled'
                },
                'temporal_aggregation': {
                    'method': 'mean_pooling',
                    'layernorm_before': 'enabled' if self.use_layernorm else 'disabled'
                },
                'output_projection': {
                    'path': f'{self.n_features} → 64 → 32 → 16 → {self.prediction_len}',
                    'residual_dims': [self.n_features, 64, 32],
                    'layernorm': 'disabled'
                }
            }
        }


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
        n_features=model_cfg.get('n_features', 64),
        prediction_len=model_cfg.get('prediction_len', 1),
        n_blocks=model_cfg.get('n_blocks', 4),
        ff_dim=model_cfg.get('ff_dim', 2048),
        dropout=model_cfg.get('dropout', 0.1),
        activation=model_cfg.get('activation', 'gelu'),
        norm_type=model_cfg.get('norm_type', 'layer'),
        use_layernorm=model_cfg.get('use_layernorm', True),
        use_residual=model_cfg.get('use_residual', True),
        temporal_aggregation_config=model_cfg.get('temporal_aggregation', {}),
        output_projection_config=model_cfg.get('output_projection', {})
    )

