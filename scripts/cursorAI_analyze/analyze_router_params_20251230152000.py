#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析Router参数量与n_segments和topk_ratio的关系
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

crossformer_module = _load_module(models_path / "crossformer_blocks.py", "crossformer_blocks")
RouterAttention = crossformer_module.RouterAttention
TwoStageTemporalAttention = crossformer_module.TwoStageTemporalAttention
TwoStageFeatureAttention = crossformer_module.TwoStageFeatureAttention


def count_params(module):
    """计算模块参数量"""
    return sum(p.numel() for p in module.parameters())


def analyze_router_params():
    """分析Router参数量"""
    print("=" * 80)
    print("Router参数量分析")
    print("=" * 80)
    
    d_model = 512
    group_size = 64  # d_model / n_feature_groups = 512 / 8
    
    print(f"\n基础配置:")
    print(f"  d_model: {d_model}")
    print(f"  group_size: {group_size} (d_model / n_feature_groups)")
    
    # 分析时间注意力的Router
    print(f"\n1. 时间注意力Router (TwoStageTemporalAttention)")
    print("-" * 80)
    
    router_temporal = RouterAttention(d_model, dropout=0.1)
    router_temporal_params = count_params(router_temporal)
    
    print(f"  Router参数量: {router_temporal_params:,}")
    print(f"  MLP结构: d_model ({d_model}) → d_model//2 ({d_model//2}) → 1")
    print(f"  参数量组成:")
    print(f"    - Linear({d_model}, {d_model//2}): {d_model * (d_model//2) + (d_model//2):,}")
    print(f"    - Linear({d_model//2}, 1): {(d_model//2) * 1 + 1:,}")
    print(f"    总计: {router_temporal_params:,}")
    
    # 测试不同n_segments和topk_ratio
    print(f"\n  测试不同n_segments和topk_ratio:")
    print(f"  {'n_segments':<15} {'topk_ratio':<15} {'Router参数量':<20} {'选择segment数':<20}")
    print(f"  {'-'*15} {'-'*15} {'-'*20} {'-'*20}")
    
    for n_segments in [5, 10, 20, 50]:
        for topk_ratio in [0.3, 0.5, 0.7, 1.0]:
            topk_k = max(1, int(n_segments * topk_ratio))
            # Router参数量不变
            print(f"  {n_segments:<15} {topk_ratio:<15} {router_temporal_params:<20,} {topk_k:<20}")
    
    # 分析特征注意力的Router
    print(f"\n2. 特征注意力Router (TwoStageFeatureAttention)")
    print("-" * 80)
    
    router_feature = RouterAttention(group_size, dropout=0.1)
    router_feature_params = count_params(router_feature)
    
    print(f"  Router参数量: {router_feature_params:,}")
    print(f"  MLP结构: group_size ({group_size}) → group_size//2 ({group_size//2}) → 1")
    print(f"  参数量组成:")
    print(f"    - Linear({group_size}, {group_size//2}): {group_size * (group_size//2) + (group_size//2):,}")
    print(f"    - Linear({group_size//2}, 1): {(group_size//2) * 1 + 1:,}")
    print(f"    总计: {router_feature_params:,}")
    
    # 测试不同n_feature_groups和topk_ratio
    print(f"\n  测试不同n_feature_groups和topk_ratio:")
    print(f"  {'n_feature_groups':<20} {'topk_ratio':<15} {'Router参数量':<20} {'选择group数':<20}")
    print(f"  {'-'*20} {'-'*15} {'-'*20} {'-'*20}")
    
    for n_feature_groups in [4, 8, 16]:
        group_size_test = d_model // n_feature_groups
        router_feature_test = RouterAttention(group_size_test, dropout=0.1)
        router_feature_test_params = count_params(router_feature_test)
        for topk_ratio in [0.3, 0.5, 0.7, 1.0]:
            topk_k = max(1, int(n_feature_groups * topk_ratio))
            print(f"  {n_feature_groups:<20} {topk_ratio:<15} {router_feature_test_params:<20,} {topk_k:<20}")
    
    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
1. Router参数量与n_segments的关系:
   ❌ 增加n_segments不会增加Router参数量
   ✓ Router参数量只与d_model（或group_size）有关
   ✓ Router的MLP结构是固定的: d_model → d_model//2 → 1
   ⚠️ 但增加n_segments会增加计算量（需要评估更多segment）

2. Router参数量与topk_ratio的关系:
   ❌ 增加topk_ratio（保留更多segment）不会增加Router参数量
   ✓ topk_ratio只影响选择多少个segment进行attention
   ✓ Router参数量是固定的，不随topk_ratio变化
   ⚠️ 但增加topk_ratio会增加计算量（需要处理更多segment）

3. 参数量变化的情况:
   ✓ 只有改变d_model或group_size才会改变Router参数量
   ✓ 时间注意力Router参数量 = f(d_model)
   ✓ 特征注意力Router参数量 = f(group_size) = f(d_model / n_feature_groups)

4. 计算量变化的情况:
   ⚠️ 增加n_segments: 增加Router的计算量（需要评估更多segment）
   ⚠️ 增加topk_ratio: 增加Cross-Dimension Attention的计算量（需要处理更多segment）
   ⚠️ 增加n_feature_groups: 增加特征注意力的计算量（但group_size会减小）
    """)


if __name__ == '__main__':
    analyze_router_params()
