#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
官方 TimeXer 适配器
版本: v0.6
日期: 20260206

适配现有数据流到官方TimeXer架构：
- 输入格式: [batch, seq_len, n_features] + mask
- 输出格式: [batch, pred_len]
- 内生变量: 索引1（第2个特征）
- 外生变量: 其他所有特征
- patch_len: 25
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


# 导入官方核心组件
models_path = Path(__file__).parent
try:
    core_module = _load_module(models_path / "timexer_official_core.py", "timexer_official_core")
    EnEmbedding = core_module.EnEmbedding
    Encoder = core_module.Encoder
    EncoderLayer = core_module.EncoderLayer
    FlattenHead = core_module.FlattenHead
    AttentionLayer = core_module.AttentionLayer
    FullAttention = core_module.FullAttention
    DataEmbedding_inverted = core_module.DataEmbedding_inverted
except Exception as e:
    raise ImportError(f"Failed to import official TimeXer core: {e}") from e


class TimeXerOfficialAdapter(nn.Module):
    """
    官方 TimeXer 适配器
    
    将你的数据格式适配到官方TimeXer架构，保持数据流不变
    
    关键特性：
    1. 保留学习型Missing Embedding
    2. 使用官方的Patch-Level内生表示
    3. 使用官方的Variate-Level外生表示
    4. 使用官方的Global Token机制
    5. 内生变量：索引1（可配置）
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        endogenous_index: int = 1,  # 默认第2个特征（索引1）
        prediction_len: int = 1,
        patch_len: int = 25,
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
        norm_feature_indices: Optional[List[int]] = None,  # 需要归一化的特征索引列表
        missing_value_flag: float = -1000.0  # 缺失值标记
    ):
        """
        初始化官方TimeXer适配器
        
        Args:
            seq_len: 输入序列长度
            n_features: 总特征数量
            endogenous_index: 内生变量在特征中的索引位置（默认1，即第2个特征）
            prediction_len: 预测长度
            patch_len: Patching大小（默认25）
            d_model: 模型嵌入维度
            n_heads: 注意力头数
            e_layers: Encoder层数
            d_ff: FFN隐藏维度
            dropout: Dropout比率
            activation: 激活函数
            use_norm: 是否使用Instance Normalization
            norm_feature_indices: 需要归一化的特征索引列表（默认None，归一化所有特征）
            missing_value_flag: 缺失值标记（默认-1000.0）
        """
        super(TimeXerOfficialAdapter, self).__init__()
        
        # 验证参数
        assert seq_len % patch_len == 0, f"seq_len ({seq_len}) 必须能被 patch_len ({patch_len}) 整除"
        assert endogenous_index >= 0 and endogenous_index < n_features, \
            f"endogenous_index ({endogenous_index}) 超出范围 [0, {n_features})"
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.endogenous_index = endogenous_index
        self.prediction_len = prediction_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.use_norm = use_norm
        self.n_vars = 1  # 只有1个内生变量
        
        # 归一化特征掩码
        if norm_feature_indices is not None:
            # 创建布尔掩码：只对指定索引的特征进行归一化
            norm_mask = torch.zeros(n_features, dtype=torch.bool)
            for idx in norm_feature_indices:
                assert 0 <= idx < n_features, f"norm_feature_indices中的索引 {idx} 超出范围 [0, {n_features})"
                norm_mask[idx] = True
            self.register_buffer('norm_mask', norm_mask)
        else:
            # 默认对所有特征归一化
            self.register_buffer('norm_mask', torch.ones(n_features, dtype=torch.bool))
        
        # ========== 学习型Missing Embedding（保留你的特性）==========
        self.missing_embedding = nn.Parameter(
            torch.randn(1, 1, n_features) * 0.01
        )
        self.missing_value_flag = missing_value_flag
        
        # ========== 官方TimeXer核心组件 ==========
        
        # 1. 内生变量嵌入（Patch-Level + Global Token）
        self.en_embedding = EnEmbedding(
            n_vars=self.n_vars,  # 1个内生变量
            d_model=d_model,
            patch_len=patch_len,
            dropout=dropout
        )
        
        # 2. 外生变量嵌入（Variate-Level）
        self.ex_embedding = DataEmbedding_inverted(
            c_in=seq_len,  # 每个变量的完整时间序列长度
            d_model=d_model,
            dropout=dropout
        )
        
        # 3. Encoder（官方核心）
        encoder_layers = []
        for _ in range(e_layers):
            encoder_layer = EncoderLayer(
                self_attention=AttentionLayer(
                    FullAttention(
                        mask_flag=False,
                        factor=5,
                        attention_dropout=dropout,
                        output_attention=False
                    ),
                    d_model=d_model,
                    n_heads=n_heads
                ),
                cross_attention=AttentionLayer(
                    FullAttention(
                        mask_flag=False,
                        factor=5,
                        attention_dropout=dropout,
                        output_attention=False
                    ),
                    d_model=d_model,
                    n_heads=n_heads
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            encoder_layers.append(encoder_layer)
        
        self.encoder = Encoder(
            layers=encoder_layers,
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # 4. 输出头
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            n_vars=self.n_vars,
            nf=self.head_nf,
            target_window=prediction_len,
            head_dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播（适配接口）
        
        Args:
            x: [batch, seq_len, n_features] = [B, 500, 64]
            mask: [batch, seq_len, n_features] (可选)
        
        Returns:
            [batch, prediction_len] = [B, 1]
        """
        B, T, F = x.shape
        
        # ========== Step 1: 处理缺失值（学习型Missing Embedding）==========
        if mask is None:
            mask = (x != self.missing_value_flag)
        else:
            # 确保mask是bool类型
            mask = mask.bool()
        
        missing_emb = self.missing_embedding.expand_as(x)
        x = torch.where(mask, x, missing_emb)
        
        # ========== Step 2: 可选的Instance Normalization（选择性归一化）==========
        if self.use_norm:
            # 初始化：先复制原始数据
            x_normed = x.clone()
            # 初始化均值和标准差（所有特征）
            means = torch.zeros(B, 1, F, device=x.device)
            stdev = torch.ones(B, 1, F, device=x.device)
            
            # 只对指定特征进行归一化
            norm_indices = self.norm_mask.nonzero(as_tuple=True)[0]
            if len(norm_indices) > 0:
                # 提取需要归一化的特征子集
                x_norm_subset = x[:, :, norm_indices]  # [B, T, num_norm_features]
                
                # 计算均值和标准差（在时间维度上）
                means_subset = x_norm_subset.mean(1, keepdim=True).detach()  # [B, 1, num_norm_features]
                x_centered = x_norm_subset - means_subset
                stdev_subset = torch.sqrt(
                    torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
                )  # [B, 1, num_norm_features]
                
                # 归一化
                x_normed_subset = x_centered / stdev_subset
                
                # 将归一化后的值写回
                x_normed[:, :, norm_indices] = x_normed_subset
                
                # 保存均值和标准差（用于反归一化）
                means[:, :, norm_indices] = means_subset
                stdev[:, :, norm_indices] = stdev_subset
        else:
            x_normed = x
            means = None
            stdev = None
        
        # ========== Step 3: 分离内生和外生变量 ==========
        # 内生：索引1（第2个特征）
        endogenous = x_normed[:, :, self.endogenous_index:self.endogenous_index+1]  # [B, T, 1]
        
        # 外生：其他所有特征（拼接索引1前后的列）
        if self.endogenous_index == 0:
            # 如果内生是第1个，外生是后面所有
            exogenous = x_normed[:, :, 1:]
        elif self.endogenous_index == F - 1:
            # 如果内生是最后1个，外生是前面所有
            exogenous = x_normed[:, :, :-1]
        else:
            # 内生在中间，拼接前后
            exogenous = torch.cat([
                x_normed[:, :, :self.endogenous_index],
                x_normed[:, :, self.endogenous_index+1:]
            ], dim=-1)
        # exogenous: [B, T, F-1]
        
        # ========== Step 4: 转换为官方TimeXer格式 ==========
        # 内生：[B, T, 1] → [B, 1, T]（变量维度在前）
        endogenous = endogenous.permute(0, 2, 1)
        
        # 外生：保持 [B, T, F-1]
        
        # ========== Step 5: 嵌入层 ==========
        # 内生嵌入（Patch-Level + Global Token）
        en_embed, n_vars = self.en_embedding(endogenous)
        # en_embed: [B*1, patch_num+1, d_model]
        
        # 外生嵌入（Variate-Level）
        ex_embed = self.ex_embedding(exogenous)
        # ex_embed: [B, F-1, d_model]
        # 注意：这里不需要reshape，EncoderLayer会处理
        
        # ========== Step 6: Encoder（官方核心）==========
        enc_out = self.encoder(en_embed, ex_embed)
        # enc_out: [B*1, patch_num+1, d_model]
        
        # ========== Step 7: Reshape为输出头格式 ==========
        # [B*1, patch_num+1, d_model] → [B, 1, patch_num+1, d_model]
        enc_out = enc_out.reshape(B, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        
        # 转置: [B, 1, patch_num+1, d_model] → [B, 1, d_model, patch_num+1]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # ========== Step 8: 输出头 ==========
        dec_out = self.head(enc_out)
        # dec_out: [B, 1, prediction_len]
        
        # 转置: [B, 1, prediction_len] → [B, prediction_len, 1]
        dec_out = dec_out.permute(0, 2, 1)
        
        # Squeeze变量维度: [B, prediction_len, 1] → [B, prediction_len]
        dec_out = dec_out.squeeze(-1)
        
        # ========== Step 9: 反归一化 ==========
        if self.use_norm and means is not None:
            # 只对内生变量反归一化
            endogenous_mean = means[:, :, self.endogenous_index]  # [B, 1]
            endogenous_std = stdev[:, :, self.endogenous_index]   # [B, 1]
            
            # 广播: [B, 1] → [B, prediction_len]
            dec_out = dec_out * endogenous_std
            dec_out = dec_out + endogenous_mean
        
        return dec_out
    
    def get_num_parameters(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """返回模型详细信息"""
        # 获取归一化特征索引
        norm_indices = self.norm_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        
        return {
            'model_type': 'TimeXer-Official-Adapter',
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'endogenous_index': self.endogenous_index,
            'exogenous_features': self.n_features - 1,
            'prediction_len': self.prediction_len,
            'patch_len': self.patch_len,
            'patch_num': self.patch_num,
            'd_model': self.head.linear.in_features // (self.patch_num + 1),
            'e_layers': len(self.encoder.layers),
            'dropout': self.missing_embedding.data.std().item(),  # 近似
            'use_norm': self.use_norm,
            'norm_feature_indices': norm_indices,
            'num_norm_features': len(norm_indices),
            'num_parameters': self.get_num_parameters(),
            'missing_embedding_enabled': True,
            'missing_value_flag': self.missing_value_flag,
            'architecture': {
                'type': 'official_timexer',
                'endogenous_representation': 'patch_level',
                'exogenous_representation': 'variate_level',
                'global_token': True,
                'encoder_layers': len(self.encoder.layers)
            }
        }
