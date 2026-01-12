#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析传统Transformer Attention vs Two-Stage Attention的参数量对比
版本: v0.3
日期: 20251230
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入模型
import importlib.util
models_path = Path("/data/project_20251211/src/models/v0.3_20251230")

def _load_module(module_path, module_name):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

crossformer_blocks_module = _load_module(models_path / "crossformer_blocks.py", "crossformer_blocks")
RouterAttention = crossformer_blocks_module.RouterAttention
CrossDimensionAttention = crossformer_blocks_module.CrossDimensionAttention


def count_params(module):
    """计算模块参数量"""
    return sum(p.numel() for p in module.parameters())


class FullTemporalAttention(nn.Module):
    """
    传统的时间注意力（不使用Two-Stage，所有segment都参与）
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(FullTemporalAttention, self).__init__()
        self.cross_attn = CrossDimensionAttention(d_model, n_heads, dropout)
    
    def forward(self, seg_repr):
        # seg_repr: (batch, n_segments, d_model)
        # 直接对所有segment做attention
        return self.cross_attn(seg_repr)


class FullFeatureAttention(nn.Module):
    """
    传统的特征注意力（不使用Two-Stage，所有group都参与）
    """
    def __init__(self, group_size: int, n_heads: int, dropout: float = 0.1):
        super(FullFeatureAttention, self).__init__()
        self.cross_attn = CrossDimensionAttention(group_size, n_heads, dropout)
    
    def forward(self, x_grouped_flat):
        # x_grouped_flat: (batch*seq_len, n_feature_groups, group_size)
        # 直接对所有group做attention
        return self.cross_attn(x_grouped_flat)


def analyze_full_vs_twostage():
    """分析传统Attention vs Two-Stage Attention的参数量"""
    print("=" * 80)
    print("传统Transformer Attention vs Two-Stage Attention参数量对比")
    print("=" * 80)
    
    d_model = 512
    n_heads = 8
    n_segments = 10
    n_feature_groups = 8
    group_size = d_model // n_feature_groups
    topk_ratio = 0.8
    topk_k_temporal = max(1, int(n_segments * topk_ratio))
    topk_k_feature = max(1, int(n_feature_groups * topk_ratio))
    
    print(f"\n配置:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_segments: {n_segments}")
    print(f"  n_feature_groups: {n_feature_groups}")
    print(f"  group_size: {group_size}")
    print(f"  topk_ratio: {topk_ratio}")
    print(f"  topk_k (时间): {topk_k_temporal}")
    print(f"  topk_k (特征): {topk_k_feature}")
    
    # 1. 时间注意力对比
    print("\n" + "=" * 80)
    print("1. 时间注意力参数量对比")
    print("=" * 80)
    
    # Two-Stage版本
    router_temporal = RouterAttention(d_model, dropout=0.1)
    router_temporal_params = count_params(router_temporal)
    
    cross_attn_temporal = CrossDimensionAttention(d_model, n_heads, dropout=0.1)
    cross_attn_temporal_params = count_params(cross_attn_temporal)
    
    twostage_temporal_params = router_temporal_params + cross_attn_temporal_params
    
    # 传统版本（不使用Router）
    full_temporal = FullTemporalAttention(d_model, n_heads, dropout=0.1)
    full_temporal_params = count_params(full_temporal)
    
    print(f"\nTwo-Stage时间注意力:")
    print(f"  Router参数量: {router_temporal_params:,}")
    print(f"  Cross-Dimension Attention参数量: {cross_attn_temporal_params:,}")
    print(f"  总计: {twostage_temporal_params:,}")
    print(f"  处理segment数: {topk_k_temporal} (从{n_segments}个中选)")
    
    print(f"\n传统时间注意力:")
    print(f"  Cross-Dimension Attention参数量: {full_temporal_params:,}")
    print(f"  总计: {full_temporal_params:,}")
    print(f"  处理segment数: {n_segments} (全部)")
    
    print(f"\n差异: {full_temporal_params - twostage_temporal_params:+,}")
    print(f"  传统版本参数量 {'减少' if full_temporal_params < twostage_temporal_params else '增加'} {abs(full_temporal_params - twostage_temporal_params):,}")
    
    # 2. 特征注意力对比
    print("\n" + "=" * 80)
    print("2. 特征注意力参数量对比")
    print("=" * 80)
    
    # Two-Stage版本
    router_feature = RouterAttention(group_size, dropout=0.1)
    router_feature_params = count_params(router_feature)
    
    cross_attn_feature = CrossDimensionAttention(group_size, n_heads, dropout=0.1)
    cross_attn_feature_params = count_params(cross_attn_feature)
    
    twostage_feature_params = router_feature_params + cross_attn_feature_params
    
    # 传统版本（不使用Router）
    full_feature = FullFeatureAttention(group_size, n_heads, dropout=0.1)
    full_feature_params = count_params(full_feature)
    
    print(f"\nTwo-Stage特征注意力:")
    print(f"  Router参数量: {router_feature_params:,}")
    print(f"  Cross-Dimension Attention参数量: {cross_attn_feature_params:,}")
    print(f"  总计: {twostage_feature_params:,}")
    print(f"  处理group数: {topk_k_feature} (从{n_feature_groups}个中选)")
    
    print(f"\n传统特征注意力:")
    print(f"  Cross-Dimension Attention参数量: {full_feature_params:,}")
    print(f"  总计: {full_feature_params:,}")
    print(f"  处理group数: {n_feature_groups} (全部)")
    
    print(f"\n差异: {full_feature_params - twostage_feature_params:+,}")
    print(f"  传统版本参数量 {'减少' if full_feature_params < twostage_feature_params else '增加'} {abs(full_feature_params - twostage_feature_params):,}")
    
    # 3. 总对比
    print("\n" + "=" * 80)
    print("3. 总体参数量对比（每个CrossformerBlock）")
    print("=" * 80)
    
    twostage_total = twostage_temporal_params + twostage_feature_params
    full_total = full_temporal_params + full_feature_params
    
    print(f"\nTwo-Stage Attention (每个Block):")
    print(f"  时间注意力: {twostage_temporal_params:,}")
    print(f"  特征注意力: {twostage_feature_params:,}")
    print(f"  总计: {twostage_total:,}")
    
    print(f"\n传统Attention (每个Block):")
    print(f"  时间注意力: {full_temporal_params:,}")
    print(f"  特征注意力: {full_feature_params:,}")
    print(f"  总计: {full_total:,}")
    
    print(f"\n差异: {full_total - twostage_total:+,}")
    print(f"  传统版本参数量 {'减少' if full_total < twostage_total else '增加'} {abs(full_total - twostage_total):,}")
    print(f"  变化比例: {(full_total - twostage_total) / twostage_total * 100:+.2f}%")
    
    # 4. 详细分析
    print("\n" + "=" * 80)
    print("详细分析")
    print("=" * 80)
    
    print("""
1. 参数量变化:
   ✓ 传统Attention参数量会减少！
   ✓ 原因：
      - 去掉了Router（Stage-1）
      - 只保留Cross-Dimension Attention（Stage-2）
      - Router的参数量会被移除
   
   ⚠️ 但注意：
      - Cross-Dimension Attention的参数量不变
      - 因为参数量只与d_model（或group_size）和n_heads有关
      - 与处理的segment/group数量无关

2. 参数量对比:
   
   时间注意力:
   - Two-Stage: Router + Cross-Attn = Router(d_model) + Cross-Attn(d_model)
   - 传统: Cross-Attn(d_model)
   - 差异: -Router参数量
   
   特征注意力:
   - Two-Stage: Router + Cross-Attn = Router(group_size) + Cross-Attn(group_size)
   - 传统: Cross-Attn(group_size)
   - 差异: -Router参数量

3. 计算量变化:
   ⚠️ 虽然参数量减少，但计算量会增加！
   
   时间注意力:
   - Two-Stage: 处理topk_k个segment
   - 传统: 处理n_segments个segment
   - 计算量: O(n_segments²) vs O(topk_k²)
   
   特征注意力:
   - Two-Stage: 处理topk_k个group
   - 传统: 处理n_feature_groups个group
   - 计算量: O(n_feature_groups²) vs O(topk_k²)

4. 能力对比:
   
   传统Attention:
   ✓ 所有segment/group都参与交互
   ✓ 可能捕捉更全面的信息
   ✓ 但计算量大，可能包含噪声
   
   Two-Stage Attention:
   ✓ 只处理重要的segment/group
   ✓ 计算量小，更高效
   ✓ 但可能遗漏一些信息

5. 总结:
   ❌ 参数量不会增加，反而会减少（去掉Router）
   ✓ 参数量减少: -Router参数量
   ⚠️ 但计算量会增加: O(n²) vs O(topk²)
   ✓ 传统Attention参数量更少，但计算量更大
    """)


if __name__ == '__main__':
    analyze_full_vs_twostage()
