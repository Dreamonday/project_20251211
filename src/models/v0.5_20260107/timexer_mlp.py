#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimeXer-MLP 主模型
版本: v0.5
日期: 20260107

实现TimeXer-MLP模型，用于多变量时间序列预测
采用双分支架构：内生分支 + 宏观分支，通过MLP融合（替代Attention）
完全兼容项目的训练框架（与TSMixer接口一致）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from pathlib import Path
import importlib.util


# 动态导入TimeXerMLPBlock
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
    TimeXer-MLP模型
    
    用于多变量时间序列预测的深度神经网络
    采用双分支架构：内生分支处理公司内部数据，宏观分支处理宏观指标
    使用MLP结构替代Attention机制进行融合
    
    模型结构：
    1. 输入: (batch, seq_len, n_features) = (batch, 500, 64)
    2. 分离: 内生 [batch, 500, 44] + 宏观 [batch, 500, 20]
    3. 共享时间混合层（MLP，处理500时间步）
    4. 内生分支: 3个LightweightTSMixerBlock → 时间聚合 → [batch, 256]
    5. 宏观分支: 2个LightweightTSMixerBlock → 时间聚合 → [batch, 256]
    6. MLP融合层（3层：交叉融合 → 自增强 → 交叉融合）
    7. 输出投影: 多层残差降维 → [batch, prediction_len]
    
    完全兼容训练脚本，接口与TSMixer一致
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        endogenous_features: int = 44,
        exogenous_features: int = 20,
        prediction_len: int = 1,
        # 特征分离配置（使用索引列表，优先级高于数量）
        endogenous_indices: Optional[List[int]] = None,
        exogenous_indices: Optional[List[int]] = None,
        # 内生分支配置
        endogenous_blocks: int = 3,
        endogenous_hidden_dim: int = 256,
        # 宏观分支配置
        exogenous_blocks: int = 2,
        exogenous_hidden_dim: int = 256,
        # 共享时间混合配置
        shared_time_mixing: bool = True,
        # MLP融合配置
        mlp_fusion_ff_dim: int = 512,
        # 通用配置
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        use_residual: bool = True,
        # 为了兼容训练脚本的配置加载，接受额外的配置参数
        n_blocks: int = None,  # 保留以兼容，但不使用
        ff_dim: int = None,  # 保留以兼容，但不使用
        norm_type: str = "layer",  # 保留以兼容
        temporal_aggregation_config: Optional[Dict] = None,
        output_projection_config: Optional[Dict] = None
    ):
        """
        初始化TimeXer-MLP模型
        
        Args:
            seq_len: 输入序列长度
            n_features: 总特征数量（内生+宏观）
            endogenous_features: 内生特征数量（默认44，如果提供了endogenous_indices则忽略）
            exogenous_features: 宏观特征数量（默认20，如果提供了exogenous_indices则忽略）
            prediction_len: 预测长度（输出维度）
            endogenous_indices: 内生特征位置索引列表（可选，优先级高于endogenous_features）
            exogenous_indices: 宏观特征位置索引列表（可选，优先级高于exogenous_features）
            endogenous_blocks: 内生分支TSMixer块数量（默认3）
            endogenous_hidden_dim: 内生分支隐藏维度（默认256）
            exogenous_blocks: 宏观分支TSMixer块数量（默认2）
            exogenous_hidden_dim: 宏观分支隐藏维度（默认256）
            shared_time_mixing: 是否共享时间混合层（默认True）
            mlp_fusion_ff_dim: MLP融合层FFN维度（默认512）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            use_layernorm: 是否启用LayerNorm（默认True）
            use_residual: 是否启用残差连接（默认True）
            n_blocks: TSMixer块数量（保留以兼容，不使用）
            ff_dim: 前馈网络隐藏维度（保留以兼容，不使用）
            norm_type: 归一化类型（保留以兼容）
            temporal_aggregation_config: 时间聚合配置（可选）
            output_projection_config: 输出投影配置（可选）
        """
        super(TimeXerMLP, self).__init__()
        
        # 如果提供了索引列表，使用索引列表；否则使用数量
        if endogenous_indices is not None and exogenous_indices is not None:
            # 使用索引列表方式
            # 使用register_buffer注册索引，PyTorch会自动处理设备移动
            self.register_buffer('endogenous_indices', torch.tensor(endogenous_indices, dtype=torch.long))
            self.register_buffer('exogenous_indices', torch.tensor(exogenous_indices, dtype=torch.long))
            self.use_indices = True
            
            # 从索引列表计算特征数量
            endogenous_features = len(endogenous_indices)
            exogenous_features = len(exogenous_indices)
            
            # 验证索引范围
            all_indices = set(endogenous_indices) | set(exogenous_indices)
            if len(all_indices) != len(endogenous_indices) + len(exogenous_indices):
                raise ValueError(f"内生索引和宏观索引有重叠")
            if max(all_indices) >= n_features:
                raise ValueError(f"索引超出范围: 最大索引={max(all_indices)}, 总特征数={n_features}")
            if min(all_indices) < 0:
                raise ValueError(f"索引不能为负数: 最小索引={min(all_indices)}")
            
            # 验证索引数量
            if len(all_indices) != n_features:
                raise ValueError(f"索引数量({len(all_indices)}) != 总特征数({n_features})")
        else:
            # 使用数量方式（向后兼容）
            self.endogenous_indices = None
            self.exogenous_indices = None
            self.use_indices = False
            
            # 验证特征数量
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
        
        # ========== 共享时间混合层（MLP） ==========
        # 统一使用48维
        self.time_mixing_dim = 48
        self.shared_time_mixing = None
        if shared_time_mixing:
            # 使用MLP时间混合
            self.shared_time_mixing = LightweightTimeMixingBlock(
                seq_len=seq_len,
                dropout=dropout,
                activation=activation
            )
        
        # ========== 时间混合投影层（统一投影到48维） ==========
        # 内生分支投影：44 → 48 → 44
        self.endogenous_time_proj_up = nn.Linear(endogenous_features, self.time_mixing_dim)
        self.endogenous_time_proj_down = nn.Linear(self.time_mixing_dim, endogenous_features)
        
        # 宏观分支投影：20 → 48 → 20
        self.exogenous_time_proj_up = nn.Linear(exogenous_features, self.time_mixing_dim)
        self.exogenous_time_proj_down = nn.Linear(self.time_mixing_dim, exogenous_features)
        
        # ========== 内生分支 ==========
        # 如果使用共享时间混合层，特征混合层也使用统一维度（48）
        # 否则使用原始特征维度（44）
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
        
        # 内生分支时间聚合前的LayerNorm
        if use_layernorm:
            self.endogenous_norm = nn.LayerNorm(endogenous_features)
        
        # 内生分支投影到hidden_dim
        self.endogenous_proj = nn.Linear(endogenous_features, endogenous_hidden_dim)
        
        # ========== 宏观分支 ==========
        # 如果使用共享时间混合层，特征混合层也使用统一维度（48）
        # 否则使用原始特征维度（20）
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
        
        # 宏观分支时间聚合前的LayerNorm
        if use_layernorm:
            self.exogenous_norm = nn.LayerNorm(exogenous_features)
        
        # 宏观分支投影到hidden_dim
        self.exogenous_proj = nn.Linear(exogenous_features, exogenous_hidden_dim)
        
        # ========== MLP融合层（替代Attention） ==========
        # 第一层：内生(Query) × 宏观(Key/Value) 交叉融合
        self.mlp_cross_fusion1 = MLPCrossFusionLayer(
            d_model=endogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm
        )
        
        # 需要将宏观特征投影到内生维度（带激活和残差）
        self.exogenous_to_endogenous_proj = nn.Linear(exogenous_hidden_dim, endogenous_hidden_dim)
        self.exogenous_to_endogenous_res = ResidualBlock(endogenous_hidden_dim, activation)
        
        # 第二层：自增强（替代Self-Attention）
        self.mlp_self_enhancement = MLPSelfEnhancementLayer(
            d_model=endogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm
        )
        
        # 第三层：宏观(Query) × 增强内生(Key/Value) 交叉融合
        self.mlp_cross_fusion2 = MLPCrossFusionLayer(
            d_model=exogenous_hidden_dim,
            ff_dim=mlp_fusion_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm
        )
        
        # 需要将增强内生特征投影到宏观维度（带激活和残差）
        self.endogenous_to_exogenous_proj = nn.Linear(endogenous_hidden_dim, exogenous_hidden_dim)
        self.endogenous_to_exogenous_res = ResidualBlock(exogenous_hidden_dim, activation)
        
        # ========== 融合策略 ==========
        # 拼接两个增强特征
        fusion_dim = endogenous_hidden_dim + exogenous_hidden_dim
        self.fusion_proj = nn.Linear(fusion_dim, endogenous_hidden_dim)  # 融合到内生维度
        
        # ========== 输出投影（多层残差降维） ==========
        # 结构：endogenous_hidden_dim → 64维残差 → 32维残差 → prediction_len
        
        # endogenous_hidden_dim维度的5层残差模块
        self.output_res1 = ResidualBlock(endogenous_hidden_dim, activation)
        
        # endogenous_hidden_dim → 64
        self.output_proj1 = nn.Linear(endogenous_hidden_dim, 64)
        self.output_dropout1 = nn.Dropout(dropout)
        
        # 64维度的5层残差模块
        self.output_res2 = ResidualBlock(64, activation)
        
        # 64 → 32
        self.output_proj2 = nn.Linear(64, 32)
        self.output_dropout2 = nn.Dropout(dropout)
        
        # 32维度的5层残差模块
        self.output_res3 = ResidualBlock(32, activation)
        
        # 32 → prediction_len（带激活和残差，如果维度相同）
        self.output_proj3 = nn.Linear(32, prediction_len)
        if prediction_len == 32:
            self.output_proj3_res = ResidualBlock(prediction_len, activation)
        else:
            self.output_proj3_res = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, prediction_len)
            
        注意：为了兼容训练脚本，当prediction_len=1时，输出形状为(batch, 1)
        """
        # x: (batch, seq_len, n_features) = (batch, 500, 64)
        
        # ========== 分离内生和宏观特征 ==========
        if self.use_indices:
            # 使用索引列表方式分离特征
            # register_buffer会自动处理设备移动，无需手动处理
            endogenous_x = x[:, :, self.endogenous_indices]  # (batch, 500, endogenous_features)
            exogenous_x = x[:, :, self.exogenous_indices]  # (batch, 500, exogenous_features)
        else:
            # 使用数量方式分离特征（向后兼容）
            endogenous_x = x[:, :, :self.endogenous_features]  # (batch, 500, 44)
            exogenous_x = x[:, :, self.endogenous_features:]  # (batch, 500, 20)
        
        # ========== 内生分支 ==========
        # 投影到统一时间混合维度（44 → 48）
        endogenous_x = self.endogenous_time_proj_up(endogenous_x)  # (batch, 500, 48)
        
        for mixer_block in self.endogenous_mixer_blocks:
            endogenous_x = mixer_block(endogenous_x)  # (batch, 500, 48)
        
        # 投影回原始维度（48 → 44）
        endogenous_x = self.endogenous_time_proj_down(endogenous_x)  # (batch, 500, 44)
        
        # 时间聚合前的LayerNorm
        if self.use_layernorm:
            endogenous_x = self.endogenous_norm(endogenous_x)
        
        # 时间聚合（平均池化）
        endogenous_x = endogenous_x.mean(dim=1)  # (batch, 44)
        
        # 投影到hidden_dim
        endogenous_x = self.endogenous_proj(endogenous_x)  # (batch, 256)
        endogenous_x = self.activation(endogenous_x)
        
        # ========== 宏观分支 ==========
        # 投影到统一时间混合维度（20 → 48）
        exogenous_x = self.exogenous_time_proj_up(exogenous_x)  # (batch, 500, 48)
        
        for mixer_block in self.exogenous_mixer_blocks:
            exogenous_x = mixer_block(exogenous_x)  # (batch, 500, 48)
        
        # 投影回原始维度（48 → 20）
        exogenous_x = self.exogenous_time_proj_down(exogenous_x)  # (batch, 500, 20)
        
        # 时间聚合前的LayerNorm
        if self.use_layernorm:
            exogenous_x = self.exogenous_norm(exogenous_x)
        
        # 时间聚合（平均池化）
        exogenous_x = exogenous_x.mean(dim=1)  # (batch, 20)
        
        # 投影到hidden_dim
        exogenous_x = self.exogenous_proj(exogenous_x)  # (batch, 256)
        exogenous_x = self.activation(exogenous_x)
        
        # ========== MLP融合层（替代Attention） ==========
        # 第一层：内生(Query) × 宏观(Key/Value) 交叉融合
        # 统一处理：两个特征都经过投影+激活+残差
        endogenous_for_fusion1 = self.exogenous_to_endogenous_proj(endogenous_x)  # (batch, 256)
        endogenous_for_fusion1 = self.activation(endogenous_for_fusion1)  # 激活
        endogenous_for_fusion1 = self.exogenous_to_endogenous_res(endogenous_for_fusion1)  # 残差
        endogenous_for_fusion1 = endogenous_for_fusion1.unsqueeze(1)  # (batch, 1, 256)
        
        exogenous_for_fusion1 = self.exogenous_to_endogenous_proj(exogenous_x)  # (batch, 256)
        exogenous_for_fusion1 = self.activation(exogenous_for_fusion1)  # 激活
        exogenous_for_fusion1 = self.exogenous_to_endogenous_res(exogenous_for_fusion1)  # 残差
        exogenous_for_fusion1 = exogenous_for_fusion1.unsqueeze(1)  # (batch, 1, 256)
        
        enhanced_endogenous = self.mlp_cross_fusion1(
            query=endogenous_for_fusion1,
            key=exogenous_for_fusion1,
            value=exogenous_for_fusion1
        )  # (batch, 1, 256)
        enhanced_endogenous = enhanced_endogenous.squeeze(1)  # (batch, 256)
        
        # 第二层：自增强（替代Self-Attention）
        enhanced_endogenous = enhanced_endogenous.unsqueeze(1)  # (batch, 1, 256)
        enhanced_endogenous = self.mlp_self_enhancement(enhanced_endogenous)  # (batch, 1, 256)
        enhanced_endogenous = enhanced_endogenous.squeeze(1)  # (batch, 256)
        
        # 第三层：宏观(Query) × 增强内生(Key/Value) 交叉融合
        # 统一处理：两个特征都经过投影+激活+残差
        exogenous_for_fusion2 = self.endogenous_to_exogenous_proj(exogenous_x)  # (batch, 256)
        exogenous_for_fusion2 = self.activation(exogenous_for_fusion2)  # 激活
        exogenous_for_fusion2 = self.endogenous_to_exogenous_res(exogenous_for_fusion2)  # 残差
        exogenous_for_fusion2 = exogenous_for_fusion2.unsqueeze(1)  # (batch, 1, 256)
        
        endogenous_for_fusion2 = self.endogenous_to_exogenous_proj(enhanced_endogenous)  # (batch, 256)
        endogenous_for_fusion2 = self.activation(endogenous_for_fusion2)  # 激活
        endogenous_for_fusion2 = self.endogenous_to_exogenous_res(endogenous_for_fusion2)  # 残差
        endogenous_for_fusion2 = endogenous_for_fusion2.unsqueeze(1)  # (batch, 1, 256)
        
        enhanced_exogenous = self.mlp_cross_fusion2(
            query=exogenous_for_fusion2,
            key=endogenous_for_fusion2,
            value=endogenous_for_fusion2
        )  # (batch, 1, 256)
        enhanced_exogenous = enhanced_exogenous.squeeze(1)  # (batch, 256)
        
        # ========== 融合策略 ==========
        # 拼接两个增强特征
        fused_features = torch.cat([enhanced_endogenous, enhanced_exogenous], dim=1)  # (batch, 512)
        
        # 投影到融合维度
        fused_features = self.fusion_proj(fused_features)  # (batch, 256)
        fused_features = self.activation(fused_features)
        
        # ========== 输出投影（多层残差降维） ==========
        # endogenous_hidden_dim维度的残差模块
        x = self.output_res1(fused_features)  # (batch, 256)
        
        # endogenous_hidden_dim → 64
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
        
        # 32 → prediction_len（不使用激活函数，直接输出）
        x = self.output_proj3(x)  # (batch, prediction_len)
        # 注意：最终输出层不使用激活函数，避免将预测值裁剪为0
        # x = self.activation(x)  # 已移除激活
        if self.output_proj3_res is not None:
            x = self.output_proj3_res(x)  # 残差（仅当prediction_len==32时）
        
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
