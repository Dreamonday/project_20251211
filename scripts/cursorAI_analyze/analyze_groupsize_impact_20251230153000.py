#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析group_size对模型的影响
版本: v0.3
日期: 20251230
"""

import torch
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

crossformer_module = _load_module(models_path / "crossformer.py", "crossformer")
Crossformer = crossformer_module.Crossformer
crossformer_blocks_module = _load_module(models_path / "crossformer_blocks.py", "crossformer_blocks")
RouterAttention = crossformer_blocks_module.RouterAttention
CrossDimensionAttention = crossformer_blocks_module.CrossDimensionAttention


def count_params(module):
    """计算模块参数量"""
    return sum(p.numel() for p in module.parameters())


def analyze_groupsize_impact():
    """分析group_size对模型的影响"""
    print("=" * 80)
    print("group_size对模型的影响分析")
    print("=" * 80)
    
    base_d_model = 512
    base_n_feature_groups = 8
    base_group_size = base_d_model // base_n_feature_groups
    
    print(f"\n基准配置:")
    print(f"  d_model: {base_d_model}")
    print(f"  n_feature_groups: {base_n_feature_groups}")
    print(f"  group_size: {base_group_size} (d_model / n_feature_groups)")
    
    # 分析不同group_size配置
    print(f"\n不同group_size配置对比:")
    print("-" * 80)
    print(f"{'方案':<20} {'d_model':<15} {'n_feature_groups':<20} {'group_size':<15} {'Router参数量':<20} {'Attention参数量':<25}")
    print("-" * 80)
    
    configs = [
        ("基准配置", 512, 8, 64),
        ("增加d_model", 1024, 8, 128),
        ("减少n_feature_groups", 512, 4, 128),
        ("大幅增加d_model", 1024, 4, 256),
        ("中等增加", 768, 4, 192),
    ]
    
    for name, d_model, n_feature_groups, group_size in configs:
        # 计算Router参数量
        router = RouterAttention(group_size, dropout=0.1)
        router_params = count_params(router)
        
        # 计算Cross-Dimension Attention参数量（假设n_heads=8，但需要调整）
        n_heads = 8
        if group_size % n_heads != 0:
            # 调整n_heads使其能整除group_size
            n_heads = group_size // (group_size // n_heads) if group_size >= n_heads else 1
        
        cross_attn = CrossDimensionAttention(group_size, n_heads, dropout=0.1)
        cross_attn_params = count_params(cross_attn)
        
        print(f"{name:<20} {d_model:<15} {n_feature_groups:<20} {group_size:<15} {router_params:<20,} {cross_attn_params:<25,}")
    
    # 详细分析影响
    print("\n" + "=" * 80)
    print("增加group_size的影响分析")
    print("=" * 80)
    
    print("""
1. 如何增加group_size:
   ✓ 方法1: 增加d_model（保持n_feature_groups不变）
      - 例如: d_model=512→1024, n_feature_groups=8, group_size=64→128
      - 优点: 同时增加时间注意力和特征注意力的能力
      - 缺点: 大幅增加参数量和计算量
   
   ✓ 方法2: 减少n_feature_groups（保持d_model不变）
      - 例如: d_model=512, n_feature_groups=8→4, group_size=64→128
      - 优点: 只增加特征注意力的参数量，时间注意力不变
      - 缺点: 特征分组变少，可能影响特征交互的粒度
   
   ✓ 方法3: 同时调整d_model和n_feature_groups
      - 例如: d_model=512→768, n_feature_groups=8→4, group_size=64→192
      - 优点: 更灵活地控制group_size
      - 缺点: 需要平衡多个因素

2. 参数量变化:
   ✓ 特征注意力Router参数量:
      - 公式: group_size × (group_size//2) + (group_size//2) + (group_size//2) × 1 + 1
      - group_size=64: 2,113 参数
      - group_size=128: 8,321 参数 (约4倍)
      - group_size=256: 33,025 参数 (约16倍)
   
   ✓ 特征注意力Cross-Dimension Attention参数量:
      - Q/K/V/输出投影各: group_size × group_size + group_size
      - group_size=64: 16,640 参数 (4 × 4,160)
      - group_size=128: 66,048 参数 (4 × 16,512) (约4倍)
      - group_size=256: 263,168 参数 (4 × 65,792) (约16倍)
   
   ⚠️ 参数量增长为平方级（group_size²）

3. 对模型能力的影响:
   
   a) 更"近视"（局部性增强）:
      ✓ group_size越大，每个特征组包含的特征维度越多
      ✓ 模型可能更关注组内的局部特征交互
      ✓ 跨组的全局交互相对减少（因为分组变少）
      ✓ 类似于CNN中更大的卷积核，感受野更大但更局部
   
   b) 表达能力变化:
      ✓ 每个特征组内的表达能力增强（因为group_size更大）
      ✓ 但特征组数量减少，可能影响不同特征组之间的交互
      ✓ 需要平衡局部表达能力和全局交互能力
   
   c) 计算量变化:
      ⚠️ Router计算量: 线性增长（group_size）
      ⚠️ Attention计算量: 平方增长（group_size²）
      ⚠️ 总体计算量大幅增加

4. 是否会让模型更"近视":
   ✓ 是的，增加group_size会让模型更"近视"（局部性增强）
   ✓ 原因:
      - group_size越大，每个特征组包含的特征维度越多
      - 模型更关注组内的局部特征交互
      - 跨组的全局交互相对减少（因为分组变少）
      - 类似于增大卷积核，感受野更大但更关注局部模式
   
   ⚠️ 但这不是绝对的:
      - 如果同时增加d_model，时间注意力也会增强，可能平衡局部性
      - 如果只减少n_feature_groups，确实会更"近视"
      - 需要根据任务特点选择合适的配置

5. 推荐策略:
   ✓ 如果希望保持全局交互能力:
      - 增加d_model，保持或适当增加n_feature_groups
      - 例如: d_model=512→768, n_feature_groups=8→12, group_size=64→64
   
   ✓ 如果希望增强局部表达能力:
      - 减少n_feature_groups，增加group_size
      - 例如: d_model=512, n_feature_groups=8→4, group_size=64→128
   
   ✓ 如果希望平衡:
      - 同时调整d_model和n_feature_groups
      - 例如: d_model=512→768, n_feature_groups=8→6, group_size=64→128
    """)


if __name__ == '__main__':
    analyze_groupsize_impact()
