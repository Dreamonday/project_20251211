#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimeXer-MLP 主模型（精细化LayerNorm控制）
版本: v0.52
日期: 20260128

双分支架构：内生分支 + 宏观分支，MLP融合
v0.52新增：精细化LayerNorm控制，可分别控制不同位置的LayerNorm启用状态
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union
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


# 导入TimeXerMLPBlock和相关组件
models_path = Path(__file__).parent
try:
    blocks_module = _load_module(models_path / "timexer_mlp_blocks.py", "timexer_mlp_blocks")
    LightweightTSMixerBlock = blocks_module.LightweightTSMixerBlock
    LightweightTimeMixingBlock = blocks_module.LightweightTimeMixingBlock
    ResidualBlock = blocks_module.ResidualBlock
    MLPCrossFusionLayer = blocks_module.MLPCrossFusionLayer
    MLPSelfEnhancementLayer = blocks_module.MLPSelfEnhancementLayer
except Exception as e:
    raise ImportError(f"Failed to import TimeXerMLP blocks: {e}") from e


class TimeXerMLP(nn.Module):
    """
    TimeXer-MLP模型（精细化LayerNorm控制）
    
    双分支架构：内生分支 + 宏观分支，通过MLP融合
    v0.52新增：支持精细化LayerNorm控制，可分别控制5个不同位置的LayerNorm
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        endogenous_features: int = 44,
        exogenous_features: int = 20,
        prediction_len: int = 1,
        endogenous_indices: Optional[List[int]] = None,
        exogenous_indices: Optional[List[int]] = None,
        endogenous_blocks: int = 3,
        endogenous_hidden_dim: int = 256,
        exogenous_blocks: int = 2,
        exogenous_hidden_dim: int = 256,
        shared_time_mixing: bool = True,
        mlp_fusion_ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: Union[bool, Dict[str, bool]] = True,
        use_residual: bool = True,
        missing_value: float = -1000.0,  # 缺失值标记
        n_blocks: int = None,  # 兼容参数
        ff_dim: int = None,  # 兼容参数
        norm_type: str = "layer",  # 兼容参数
        temporal_aggregation_config: Optional[Dict] = None,
        output_projection_config: Optional[Dict] = None
    ):
        super(TimeXerMLP, self).__init__()
        
        # ========== v0.52关键：精细化LayerNorm配置 ==========
        # 支持布尔值（向后兼容）或字典（精细化控制）
        if isinstance(use_layernorm, bool):
            # 兼容模式：所有位置使用相同配置
            self.layernorm_config = {
                'endogenous_mixer': use_layernorm,
                'exogenous_mixer': use_layernorm,
                'endogenous_output': use_layernorm,
                'exogenous_output': use_layernorm,
                'mlp_fusion': use_layernorm,
            }
            self.use_layernorm = use_layernorm  # 保留以兼容
        elif isinstance(use_layernorm, dict):
            # 精细化模式：分别控制每个位置
            default_config = {
                'endogenous_mixer': True,
                'exogenous_mixer': True,
                'endogenous_output': True,
                'exogenous_output': True,
                'mlp_fusion': True,
            }
            self.layernorm_config = {**default_config, **use_layernorm}
            # use_layernorm 设置为字典模式标记
            self.use_layernorm = self.layernorm_config
        else:
            raise ValueError(f"use_layernorm must be bool or dict, got {type(use_layernorm)}")
        
        # 特征分离配置
        if endogenous_indices is not None and exogenous_indices is not None:
            self.register_buffer('endogenous_indices', torch.tensor(endogenous_indices, dtype=torch.long))
            self.register_buffer('exogenous_indices', torch.tensor(exogenous_indices, dtype=torch.long))
            self.use_indices = True
            endogenous_features = len(endogenous_indices)
            exogenous_features = len(exogenous_indices)
            
            # 验证索引
            all_indices = set(endogenous_indices) | set(exogenous_indices)
            if len(all_indices) != len(endogenous_indices) + len(exogenous_indices):
                raise ValueError(f"内生索引和宏观索引有重叠")
            if max(all_indices) >= n_features:
                raise ValueError(f"索引超出范围: 最大索引={max(all_indices)}, 总特征数={n_features}")
            if min(all_indices) < 0:
                raise ValueError(f"索引不能为负数: 最小索引={min(all_indices)}")
            if len(all_indices) != n_features:
                raise ValueError(f"索引数量({len(all_indices)}) != 总特征数({n_features})")
        else:
            self.endogenous_indices = None
            self.exogenous_indices = None
            self.use_indices = False
            assert endogenous_features + exogenous_features == n_features, \
                f"内生特征({endogenous_features}) + 宏观特征({exogenous_features}) != 总特征({n_features})"
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.endogenous_features = endogenous_features
        self.exogenous_features = exogenous_features
        self.prediction_len = prediction_len
        self.endogenous_blocks = endogenous_blocks
        self.exogenous_blocks = exogenous_blocks
        self.endogenous_hidden_dim = endogenous_hidden_dim
        self.exogenous_hidden_dim = exogenous_hidden_dim
        self.dropout = dropout
        self.activation_name = activation
        self.use_residual = use_residual
        self.norm_type = norm_type
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ========== 学习型Missing Embedding ==========
        # 可学习的缺失值表示，形状：(1, 1, n_features)
        self.missing_embedding = nn.Parameter(
            torch.randn(1, 1, n_features) * 0.01
        )
        self.missing_value_flag = missing_value
        
        # ========== 共享时间混合层 ==========
        self.time_mixing_dim = 48
        self.shared_time_mixing = None
        if shared_time_mixing:
            self.shared_time_mixing = LightweightTimeMixingBlock(
                seq_len=seq_len,
                dropout=dropout,
                activation=activation
            )
        
        # ========== 时间混合投影层 ==========
        self.endogenous_time_proj_up = nn.Linear(endogenous_features, self.time_mixing_dim)
        self.endogenous_time_proj_down = nn.Linear(self.time_mixing_dim, endogenous_features)
        self.exogenous_time_proj_up = nn.Linear(exogenous_features, self.time_mixing_dim)
        self.exogenous_time_proj_down = nn.Linear(self.time_mixing_dim, exogenous_features)
        
        # ========== 内生分支 ==========
        endogenous_mixer_n_features = self.time_mixing_dim if shared_time_mixing else endogenous_features
        self.endogenous_mixer_blocks = nn.ModuleList([
            LightweightTSMixerBlock(
                seq_len=seq_len,
                n_features=endogenous_mixer_n_features,
                shared_time_mixing=self.shared_time_mixing,
                dropout=dropout,
                activation=activation,
                use_layernorm=self.layernorm_config['endogenous_mixer'],  # v0.52: 精细化控制
                use_residual=use_residual
            )
            for _ in range(endogenous_blocks)
        ])
        
        # v0.52: 精细化控制内生输出LayerNorm
        if self.layernorm_config['endogenous_output']:
            self.endogenous_norm = nn.LayerNorm(endogenous_features)
        else:
            self.endogenous_norm = None
        
        self.endogenous_proj = nn.Linear(endogenous_features, endogenous_hidden_dim)
        
        # ========== 宏观分支 ==========
        exogenous_mixer_n_features = self.time_mixing_dim if shared_time_mixing else exogenous_features
        self.exogenous_mixer_blocks = nn.ModuleList([
            LightweightTSMixerBlock(
                seq_len=seq_len,
                n_features=exogenous_mixer_n_features,
                shared_time_mixing=self.shared_time_mixing,
                dropout=dropout,
                activation=activation,
                use_layernorm=self.layernorm_config['exogenous_mixer'],  # v0.52: 精细化控制
                use_residual=use_residual
            )
            for _ in range(exogenous_blocks)
        ])
        
        # v0.52: 精细化控制宏观输出LayerNorm
        if self.layernorm_config['exogenous_output']:
            self.exogenous_norm = nn.LayerNorm(exogenous_features)
        else:
            self.exogenous_norm = None
        
        self.exogenous_proj = nn.Linear(exogenous_features, exogenous_hidden_dim)
        
        # ========== MLP融合层 ==========
        # v0.52: 精细化控制融合层LayerNorm
        self.mlp_cross_fusion1 = MLPCrossFusionLayer(
            d_model=endogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=self.layernorm_config['mlp_fusion']
        )
        
        self.exogenous_to_endogenous_proj = nn.Linear(exogenous_hidden_dim, endogenous_hidden_dim)
        self.exogenous_to_endogenous_res = ResidualBlock(endogenous_hidden_dim, activation)
        
        self.mlp_self_enhancement = MLPSelfEnhancementLayer(
            d_model=endogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=self.layernorm_config['mlp_fusion']
        )
        
        self.mlp_cross_fusion2 = MLPCrossFusionLayer(
            d_model=exogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=self.layernorm_config['mlp_fusion']
        )
        
        self.endogenous_to_exogenous_proj = nn.Linear(endogenous_hidden_dim, exogenous_hidden_dim)
        self.endogenous_to_exogenous_res = ResidualBlock(exogenous_hidden_dim, activation)
        
        # ========== 融合策略 ==========
        fusion_dim = endogenous_hidden_dim + exogenous_hidden_dim
        self.fusion_proj = nn.Linear(fusion_dim, endogenous_hidden_dim)
        
        # ========== 输出投影 ==========
        self.output_res1 = ResidualBlock(endogenous_hidden_dim, activation)
        self.output_proj1 = nn.Linear(endogenous_hidden_dim, 64)
        self.output_dropout1 = nn.Dropout(dropout)
        
        self.output_res2 = ResidualBlock(64, activation)
        self.output_proj2 = nn.Linear(64, 32)
        self.output_dropout2 = nn.Dropout(dropout)
        
        self.output_res3 = ResidualBlock(32, activation)
        self.output_proj3 = nn.Linear(32, prediction_len)
        
        if prediction_len == 32:
            self.output_proj3_res = ResidualBlock(prediction_len, activation)
        else:
            self.output_proj3_res = None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播（学习型Missing Embedding）
        
        Args:
            x: 输入 (batch, seq_len, n_features)，可能包含-1000缺失值
            mask: 可选掩码（兼容参数，v0.51自动检测）
        
        Returns:
            输出 (batch, prediction_len)
        """
        # ========== 关键：立即处理缺失值，必须在所有操作之前 ==========
        # 精确检测-1000标记（不使用容错空间）
        if mask is None:
            mask = (x != self.missing_value_flag)  # True=有效, False=缺失(-1000)
        
        # 替换缺失值为可学习embedding
        missing_emb = self.missing_embedding.expand_as(x)
        x = torch.where(mask, x, missing_emb)
        
        # ========== 分离内生和宏观特征 ==========
        if self.use_indices:
            endogenous_x = x[:, :, self.endogenous_indices]
            exogenous_x = x[:, :, self.exogenous_indices]
        else:
            endogenous_x = x[:, :, :self.endogenous_features]
            exogenous_x = x[:, :, self.endogenous_features:]
        
        # ========== 内生分支 ==========
        endogenous_x = self.endogenous_time_proj_up(endogenous_x)
        
        for mixer_block in self.endogenous_mixer_blocks:
            endogenous_x = mixer_block(endogenous_x)
        
        endogenous_x = self.endogenous_time_proj_down(endogenous_x)
        
        # v0.52: 精细化控制
        if self.endogenous_norm is not None:
            endogenous_x = self.endogenous_norm(endogenous_x)
        
        endogenous_x = endogenous_x.mean(dim=1)
        endogenous_x = self.endogenous_proj(endogenous_x)
        endogenous_x = self.activation(endogenous_x)
        
        # ========== 宏观分支 ==========
        exogenous_x = self.exogenous_time_proj_up(exogenous_x)
        
        for mixer_block in self.exogenous_mixer_blocks:
            exogenous_x = mixer_block(exogenous_x)
        
        exogenous_x = self.exogenous_time_proj_down(exogenous_x)
        
        # v0.52: 精细化控制
        if self.exogenous_norm is not None:
            exogenous_x = self.exogenous_norm(exogenous_x)
        
        exogenous_x = exogenous_x.mean(dim=1)
        exogenous_x = self.exogenous_proj(exogenous_x)
        exogenous_x = self.activation(exogenous_x)
        
        # ========== MLP融合层 ==========
        # 第一层：交叉融合
        endogenous_for_fusion1 = self.exogenous_to_endogenous_proj(endogenous_x)
        endogenous_for_fusion1 = self.activation(endogenous_for_fusion1)
        endogenous_for_fusion1 = self.exogenous_to_endogenous_res(endogenous_for_fusion1)
        endogenous_for_fusion1 = endogenous_for_fusion1.unsqueeze(1)
        
        exogenous_for_fusion1 = self.exogenous_to_endogenous_proj(exogenous_x)
        exogenous_for_fusion1 = self.activation(exogenous_for_fusion1)
        exogenous_for_fusion1 = self.exogenous_to_endogenous_res(exogenous_for_fusion1)
        exogenous_for_fusion1 = exogenous_for_fusion1.unsqueeze(1)
        
        enhanced_endogenous = self.mlp_cross_fusion1(
            query=endogenous_for_fusion1,
            key=exogenous_for_fusion1,
            value=exogenous_for_fusion1
        )
        enhanced_endogenous = enhanced_endogenous.squeeze(1)
        
        # 第二层：自增强
        enhanced_endogenous = enhanced_endogenous.unsqueeze(1)
        enhanced_endogenous = self.mlp_self_enhancement(enhanced_endogenous)
        enhanced_endogenous = enhanced_endogenous.squeeze(1)
        
        # 第三层：交叉融合
        exogenous_for_fusion2 = self.endogenous_to_exogenous_proj(exogenous_x)
        exogenous_for_fusion2 = self.activation(exogenous_for_fusion2)
        exogenous_for_fusion2 = self.endogenous_to_exogenous_res(exogenous_for_fusion2)
        exogenous_for_fusion2 = exogenous_for_fusion2.unsqueeze(1)
        
        endogenous_for_fusion2 = self.endogenous_to_exogenous_proj(enhanced_endogenous)
        endogenous_for_fusion2 = self.activation(endogenous_for_fusion2)
        endogenous_for_fusion2 = self.endogenous_to_exogenous_res(endogenous_for_fusion2)
        endogenous_for_fusion2 = endogenous_for_fusion2.unsqueeze(1)
        
        enhanced_exogenous = self.mlp_cross_fusion2(
            query=exogenous_for_fusion2,
            key=endogenous_for_fusion2,
            value=endogenous_for_fusion2
        )
        enhanced_exogenous = enhanced_exogenous.squeeze(1)
        
        # ========== 融合策略 ==========
        fused_features = torch.cat([enhanced_endogenous, enhanced_exogenous], dim=1)
        fused_features = self.fusion_proj(fused_features)
        fused_features = self.activation(fused_features)
        
        # ========== 输出投影 ==========
        x = self.output_res1(fused_features)
        x = self.output_proj1(x)
        x = self.activation(x)
        x = self.output_dropout1(x)
        
        x = self.output_res2(x)
        x = self.output_proj2(x)
        x = self.activation(x)
        x = self.output_dropout2(x)
        
        x = self.output_res3(x)
        x = self.output_proj3(x)
        
        if self.output_proj3_res is not None:
            x = self.output_proj3_res(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """返回模型详细信息"""
        return {
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'endogenous_features': self.endogenous_features,
            'exogenous_features': self.exogenous_features,
            'prediction_len': self.prediction_len,
            'endogenous_blocks': self.endogenous_blocks,
            'exogenous_blocks': self.exogenous_blocks,
            'endogenous_hidden_dim': self.endogenous_hidden_dim,
            'exogenous_hidden_dim': self.exogenous_hidden_dim,
            'dropout': self.dropout,
            'activation': self.activation_name,
            'norm_type': self.norm_type,
            'layernorm_config': self.layernorm_config,  # v0.52: 精细化配置
            'use_residual': self.use_residual,
            'num_parameters': self.get_num_parameters(),
            'missing_embedding_enabled': True,
            'missing_value_flag': self.missing_value_flag,
            'architecture': {
                'shared_time_mixing': self.shared_time_mixing is not None,
                'time_mixing_type': 'mlp',
                'endogenous_branch': {
                    'n_blocks': self.endogenous_blocks,
                    'hidden_dim': self.endogenous_hidden_dim
                },
                'exogenous_branch': {
                    'n_blocks': self.exogenous_blocks,
                    'hidden_dim': self.exogenous_hidden_dim
                },
                'fusion': {
                    'type': 'mlp',
                    'cross_fusion_layers': 2,
                    'self_enhancement_layers': 1
                }
            }
        }
