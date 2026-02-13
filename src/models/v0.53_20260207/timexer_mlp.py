#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimeXer-MLP 主模型（学习型Missing Embedding + Instance Normalization）
版本: v0.53
日期: 20260207

双分支架构：内生分支 + 宏观分支，MLP融合
v0.53新增：Instance Normalization + 反归一化（基于v0.51）
- 在模型内部进行归一化（每个样本独立）
- 输出前进行反归一化到原始尺度
- 损失计算在原始尺度上进行
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
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
    TimeXer-MLP模型（学习型Missing Embedding + Instance Normalization）
    
    双分支架构：内生分支 + 宏观分支，通过MLP融合
    v0.53新增：Instance Normalization + 反归一化（方案B：masked 位置不参与）
    - 自动检测并替换-1000缺失值为可学习embedding
    - 均值和标准差仅基于有效位置（mask=True）计算，归一化仅作用于有效位置，masked 位置保持 missing_emb
    - 输出前反归一化到原始尺度
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
        use_layernorm: bool = True,
        use_residual: bool = True,
        n_blocks: int = None,  # 兼容参数
        ff_dim: int = None,  # 兼容参数
        norm_type: str = "layer",  # 兼容参数
        temporal_aggregation_config: Optional[Dict] = None,
        output_projection_config: Optional[Dict] = None,
        # v0.53新增参数
        use_norm: bool = True,  # 是否使用Instance Normalization
        norm_feature_indices: Optional[List[int]] = None,  # 需要归一化的特征索引（None=全部）
        output_feature_index: int = 2  # 输出对应的特征索引（用于反归一化）
    ):
        super(TimeXerMLP, self).__init__()
        
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
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.norm_type = norm_type
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ========== v0.51：学习型Missing Embedding ==========
        # 可学习的缺失值表示，形状：(1, 1, n_features)
        self.missing_embedding = nn.Parameter(
            torch.randn(1, 1, n_features) * 0.01
        )
        self.missing_value_flag = -1000.0
        
        # ========== v0.53新增：Instance Normalization 配置 ==========
        self.use_norm = use_norm
        self.output_feature_index = output_feature_index
        
        # 验证输出特征索引
        assert 0 <= output_feature_index < n_features, \
            f"output_feature_index ({output_feature_index}) 超出范围 [0, {n_features})"
        
        # 创建归一化特征掩码
        if norm_feature_indices is not None:
            # 只对指定索引的特征进行归一化
            norm_mask = torch.zeros(n_features, dtype=torch.bool)
            for idx in norm_feature_indices:
                assert 0 <= idx < n_features, \
                    f"norm_feature_indices中的索引 {idx} 超出范围 [0, {n_features})"
                norm_mask[idx] = True
            self.register_buffer('norm_mask', norm_mask)
        else:
            # 默认对所有特征归一化
            self.register_buffer('norm_mask', torch.ones(n_features, dtype=torch.bool))
        
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
                use_layernorm=use_layernorm,
                use_residual=use_residual
            )
            for _ in range(endogenous_blocks)
        ])
        
        if use_layernorm:
            self.endogenous_norm = nn.LayerNorm(endogenous_features)
        
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
                use_layernorm=use_layernorm,
                use_residual=use_residual
            )
            for _ in range(exogenous_blocks)
        ])
        
        if use_layernorm:
            self.exogenous_norm = nn.LayerNorm(exogenous_features)
        
        self.exogenous_proj = nn.Linear(exogenous_features, exogenous_hidden_dim)
        
        # ========== MLP融合层 ==========
        self.mlp_cross_fusion1 = MLPCrossFusionLayer(
            d_model=endogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm
        )
        
        self.exogenous_to_endogenous_proj = nn.Linear(exogenous_hidden_dim, endogenous_hidden_dim)
        self.exogenous_to_endogenous_res = ResidualBlock(endogenous_hidden_dim, activation)
        
        self.mlp_self_enhancement = MLPSelfEnhancementLayer(
            d_model=endogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm
        )
        
        self.mlp_cross_fusion2 = MLPCrossFusionLayer(
            d_model=exogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm
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
        前向传播（学习型Missing Embedding + Instance Normalization）
        
        Args:
            x: 输入 (batch, seq_len, n_features)，可能包含-1000缺失值
            mask: 可选掩码（兼容参数，自动检测）
        
        Returns:
            输出 (batch, prediction_len)，已反归一化到原始尺度
        """
        # ========== Step 1: 处理缺失值（必须在所有操作之前）==========
        # 精确检测-1000标记（不使用容错空间）
        if mask is None:
            mask = (x != self.missing_value_flag)  # True=有效, False=缺失(-1000)
        
        # 替换缺失值为可学习embedding
        missing_emb = self.missing_embedding.expand_as(x)
        x = torch.where(mask, x, missing_emb)
        
        # ========== Step 2: v0.53 - Instance Normalization（可选，方案B：仅有效位置参与统计与归一化）==========
        if self.use_norm:
            B, T, F = x.shape
            # 初始化：先复制原始数据
            x_normed = x.clone()
            # 初始化均值和标准差（所有特征）
            means = torch.zeros(B, 1, F, device=x.device)
            stdev = torch.ones(B, 1, F, device=x.device)
            
            # 只对指定特征进行归一化
            norm_indices = self.norm_mask.nonzero(as_tuple=True)[0]
            if len(norm_indices) > 0:
                # 提取需要归一化的特征子集及对应 mask
                x_norm_subset = x[:, :, norm_indices]  # [B, T, num_norm_features]
                mask_subset = mask[:, :, norm_indices].float()  # [B, T, num_norm_features]，True=有效
                
                # 仅对有效位置（mask=True）计算均值和标准差（masked statistics）
                count = mask_subset.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, num_norm_features]
                sum_x = (x_norm_subset * mask_subset).sum(dim=1, keepdim=True)
                means_subset = (sum_x / count).detach()  # [B, 1, num_norm_features]
                x_centered = x_norm_subset - means_subset
                sum_sq = ((x_centered ** 2) * mask_subset).sum(dim=1, keepdim=True)
                var_subset = sum_sq / count
                stdev_subset = torch.sqrt(var_subset + 1e-5).detach()  # [B, 1, num_norm_features]
                
                # 方案B：仅对有效位置做归一化，masked 位置保持原值（missing_emb）
                x_normed_subset = (x_norm_subset - means_subset) / stdev_subset
                x_normed_subset = torch.where(
                    mask_subset.to(torch.bool), x_normed_subset, x_norm_subset
                )
                
                # 将归一化后的值写回
                x_normed[:, :, norm_indices] = x_normed_subset
                
                # 保存均值和标准差（用于反归一化）
                means[:, :, norm_indices] = means_subset
                stdev[:, :, norm_indices] = stdev_subset
            
            x = x_normed
        else:
            means = None
            stdev = None
        
        # ========== Step 3: 分离内生和宏观特征 ==========
        if self.use_indices:
            endogenous_x = x[:, :, self.endogenous_indices]
            exogenous_x = x[:, :, self.exogenous_indices]
        else:
            endogenous_x = x[:, :, :self.endogenous_features]
            exogenous_x = x[:, :, self.endogenous_features:]
        
        # ========== Step 4: 内生分支 ==========
        endogenous_x = self.endogenous_time_proj_up(endogenous_x)
        
        for mixer_block in self.endogenous_mixer_blocks:
            endogenous_x = mixer_block(endogenous_x)
        
        endogenous_x = self.endogenous_time_proj_down(endogenous_x)
        
        if self.use_layernorm:
            endogenous_x = self.endogenous_norm(endogenous_x)
        
        endogenous_x = endogenous_x.mean(dim=1)
        endogenous_x = self.endogenous_proj(endogenous_x)
        endogenous_x = self.activation(endogenous_x)
        
        # ========== Step 5: 宏观分支 ==========
        exogenous_x = self.exogenous_time_proj_up(exogenous_x)
        
        for mixer_block in self.exogenous_mixer_blocks:
            exogenous_x = mixer_block(exogenous_x)
        
        exogenous_x = self.exogenous_time_proj_down(exogenous_x)
        
        if self.use_layernorm:
            exogenous_x = self.exogenous_norm(exogenous_x)
        
        exogenous_x = exogenous_x.mean(dim=1)
        exogenous_x = self.exogenous_proj(exogenous_x)
        exogenous_x = self.activation(exogenous_x)
        
        # ========== Step 6: MLP融合层 ==========
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
        
        # ========== Step 7: 融合策略 ==========
        fused_features = torch.cat([enhanced_endogenous, enhanced_exogenous], dim=1)
        fused_features = self.fusion_proj(fused_features)
        fused_features = self.activation(fused_features)
        
        # ========== Step 8: 输出投影 ==========
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
        
        # ========== Step 9: v0.53新增 - 反归一化（将输出还原到原始尺度）==========
        if self.use_norm and means is not None:
            # 使用输出特征索引对应的均值和标准差
            output_mean = means[:, :, self.output_feature_index]  # [B, 1]
            output_std = stdev[:, :, self.output_feature_index]   # [B, 1]
            
            # 反归一化: x_original = x_normalized * std + mean
            # 广播: [B, 1] → [B, prediction_len]
            x = x * output_std + output_mean
        
        return x
    
    def get_num_parameters(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """返回模型详细信息"""
        # 获取归一化特征索引
        norm_indices = self.norm_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        
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
            'use_layernorm': self.use_layernorm,
            'use_residual': self.use_residual,
            'num_parameters': self.get_num_parameters(),
            'missing_embedding_enabled': True,
            'missing_value_flag': self.missing_value_flag,
            # v0.53新增信息
            'use_norm': self.use_norm,
            'norm_feature_indices': norm_indices,
            'num_norm_features': len(norm_indices),
            'output_feature_index': self.output_feature_index,
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
