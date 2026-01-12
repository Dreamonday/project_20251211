#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析时间维度segment长度对模型的影响
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


def count_params(module):
    """计算模块参数量"""
    return sum(p.numel() for p in module.parameters())


def analyze_segment_length_impact():
    """分析segment长度对模型的影响"""
    print("=" * 80)
    print("时间维度segment长度对模型的影响分析")
    print("=" * 80)
    
    seq_len = 500
    d_model = 512
    
    print(f"\n基准配置:")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  公式: seg_len = seq_len / n_segments")
    
    # 分析不同n_segments配置
    print(f"\n不同n_segments配置对比:")
    print("-" * 80)
    print(f"{'方案':<25} {'n_segments':<15} {'seg_len':<15} {'Router参数量':<20} {'topk_k (0.8)':<20}")
    print("-" * 80)
    
    configs = [
        ("更多segment（细粒度）", 20, 25),
        ("基准配置", 10, 50),
        ("较少segment（粗粒度）", 5, 100),
        ("很少segment（很粗）", 4, 125),
    ]
    
    for name, n_segments, seg_len in configs:
        # 计算Router参数量（只与d_model有关）
        router = RouterAttention(d_model, dropout=0.1)
        router_params = count_params(router)
        
        # topk_k计算
        topk_ratio = 0.8
        topk_k = max(1, int(n_segments * topk_ratio))
        
        print(f"{name:<25} {n_segments:<15} {seg_len:<15} {router_params:<20,} {topk_k:<20}")
    
    # 详细分析影响
    print("\n" + "=" * 80)
    print("增加segment长度（减少n_segments）的影响分析")
    print("=" * 80)
    
    print("""
1. 概念澄清:
   ✓ 时间维度没有"group_size"的概念
   ✓ 对应概念是"segment长度"（seg_len）
   ✓ seg_len = seq_len / n_segments
   ✓ 减少n_segments → 增加seg_len → 每个segment包含更多时间步

2. 你的逻辑是否正确:
   ✓ 是的，逻辑基本正确！
   ✓ 如果减少n_segments（增加seg_len）：
      - 每个segment包含更多时间步
      - 模型在每个segment内能看到更多的时间信息
      - 类似于"看得更远"或"看得更多"
   
   ⚠️ 但需要注意：
      - 这不是"group_size"的概念，而是segment的长度
      - 与特征维度的group_size是不同层面的概念
      - 特征维度：group_size = d_model / n_feature_groups（特征维度分组）
      - 时间维度：seg_len = seq_len / n_segments（时间步分组）

3. 参数量变化:
   ✓ Router参数量: 不变（只与d_model有关）
      - 无论n_segments是多少，Router参数量都是固定的
      - 因为Router评估的是segment代表向量（d_model维）
   
   ✓ Cross-Dimension Attention参数量: 不变
      - Attention的参数量只与d_model和n_heads有关
      - 不随n_segments变化
   
   ⚠️ 但计算量会变化:
      - 减少n_segments: 减少Router需要评估的segment数量
      - 但每个segment包含更多时间步，计算segment代表向量时需要处理更多数据

4. 对模型能力的影响:
   
   a) "看得更多"（时间范围扩大）:
      ✓ seg_len越大，每个segment包含的时间步越多
      ✓ 模型在每个segment内能看到更长的时间范围
      ✓ 类似于增大时间窗口，能看到更长期的时间依赖
      ✓ 你的理解是正确的！
   
   b) 时间粒度变化:
      ✓ 减少n_segments → 更粗的时间粒度
         - 每个segment代表更长的时间段
         - 可能更适合捕捉长期趋势
         - 但可能丢失短期细节
      
      ✓ 增加n_segments → 更细的时间粒度
         - 每个segment代表更短的时间段
         - 可能更适合捕捉短期波动
         - 但需要更多segment才能覆盖完整序列
   
   c) 与特征维度group_size的对比:
      ✓ 特征维度group_size增加 → 更"近视"（局部性增强）
         - 关注特征组内的局部交互
         - 跨组全局交互减少
      
      ✓ 时间维度seg_len增加 → 更"远视"（时间范围扩大）
         - 每个segment看到更长的时间范围
         - 可能更好地捕捉长期依赖
         - 但segment数量减少，跨segment交互减少

5. 是否会让模型更"远视":
   ✓ 是的，增加seg_len会让模型在时间维度上更"远视"
   ✓ 原因:
      - 每个segment包含更多时间步
      - 模型在每个segment内能看到更长的时间范围
      - 可能更好地捕捉长期时间依赖
   
   ⚠️ 但这不是绝对的:
      - segment数量减少，跨segment的交互机会减少
      - 需要在长期依赖和跨segment交互之间平衡
      - 如果seg_len太大，可能丢失短期细节

6. 推荐策略:
   ✓ 如果希望捕捉长期依赖:
      - 减少n_segments，增加seg_len
      - 例如: n_segments=10→5, seg_len=50→100
      - 配合较大的topk_ratio（如0.8）保留更多segment
   
   ✓ 如果希望捕捉短期波动:
      - 增加n_segments，减少seg_len
      - 例如: n_segments=10→20, seg_len=50→25
      - 配合较小的topk_ratio进行筛选
   
   ✓ 如果希望平衡:
      - 保持适中的n_segments
      - 例如: n_segments=10, seg_len=50
      - 通过topk_ratio控制保留的segment数量

7. 与特征维度group_size的对比总结:
   
   特征维度group_size增加:
   - 更"近视"（局部性增强）
   - 关注特征组内的局部交互
   - 参数量平方增长
   
   时间维度seg_len增加:
   - 更"远视"（时间范围扩大）
   - 每个segment看到更长的时间范围
   - 参数量不变，但计算量可能变化
    """)


if __name__ == '__main__':
    analyze_segment_length_impact()
