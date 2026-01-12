#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析n_heads对模型参数量的影响
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
CrossDimensionAttention = crossformer_blocks_module.CrossDimensionAttention


def count_params(module):
    """计算模块参数量"""
    return sum(p.numel() for p in module.parameters())


def analyze_nheads_impact():
    """分析n_heads对参数量的影响"""
    print("=" * 80)
    print("n_heads对模型参数量的影响分析")
    print("=" * 80)
    
    d_model = 512
    
    print(f"\n基准配置:")
    print(f"  d_model: {d_model}")
    print(f"  公式: d_k = d_model / n_heads")
    
    # 分析不同n_heads配置下的Attention参数量
    print(f"\n不同n_heads配置下的Cross-Dimension Attention参数量:")
    print("-" * 80)
    print(f"{'n_heads':<15} {'d_k':<15} {'Q投影':<20} {'K投影':<20} {'V投影':<20} {'输出投影':<20} {'总计':<20}")
    print("-" * 80)
    
    n_heads_list = [4, 8, 16, 32]
    
    for n_heads in n_heads_list:
        if d_model % n_heads != 0:
            print(f"{n_heads:<15} {'N/A':<15} {'d_model不能被n_heads整除':<80}")
            continue
        
        d_k = d_model // n_heads
        
        # 创建Cross-Dimension Attention
        cross_attn = CrossDimensionAttention(d_model, n_heads, dropout=0.1)
        total_params = count_params(cross_attn)
        
        # 分解参数量
        q_params = count_params(cross_attn.W_q)
        k_params = count_params(cross_attn.W_k)
        v_params = count_params(cross_attn.W_v)
        o_params = count_params(cross_attn.W_o)
        
        print(f"{n_heads:<15} {d_k:<15} {q_params:<20,} {k_params:<20,} {v_params:<20,} {o_params:<20,} {total_params:<20,}")
    
    # 测试完整模型
    print("\n" + "=" * 80)
    print("完整模型参数量对比（n_heads变化）")
    print("=" * 80)
    
    base_config = {
        'seq_len': 500,
        'n_features': 64,
        'prediction_len': 1,
        'd_model': 512,
        'n_blocks': 4,
        'n_segments': 10,
        'n_feature_groups': 8,
        'router_topk_ratio': 0.8,
        'dropout': 0.1,
        'activation': 'gelu'
    }
    
    print(f"\n{'n_heads':<15} {'总参数量':<20} {'与基准差异':<20} {'存储大小(MB)':<20}")
    print("-" * 80)
    
    base_params = None
    for n_heads in [4, 8, 16, 32]:
        if d_model % n_heads != 0:
            continue
        
        try:
            model = Crossformer(
                n_heads=n_heads,
                **base_config
            )
            params = count_params(model)
            size_mb = params * 4 / 1024 / 1024
            
            if base_params is None:
                base_params = params
                diff = 0
            else:
                diff = params - base_params
            
            print(f"{n_heads:<15} {params:<20,} {diff:+,} {size_mb:<20.2f}")
        except Exception as e:
            print(f"{n_heads:<15} {'错误':<20} {str(e):<60}")
    
    # 详细分析
    print("\n" + "=" * 80)
    print("详细分析")
    print("=" * 80)
    
    print("""
1. n_heads对参数量的影响:
   ✓ 参数量不变！
   ✓ 原因：
      - Q/K/V投影: nn.Linear(d_model, d_model)
        * 参数量 = d_model × d_model + d_model
        * 与n_heads无关
      
      - 输出投影: nn.Linear(d_model, d_model)
        * 参数量 = d_model × d_model + d_model
        * 与n_heads无关
      
      - n_heads只是将d_model分成n_heads个头
        * 每个头的维度 d_k = d_model / n_heads
        * 但总参数量不变（只是计算方式不同）

2. 参数量计算公式:
   Cross-Dimension Attention参数量:
   = Q投影 + K投影 + V投影 + 输出投影
   = 4 × (d_model × d_model + d_model)
   = 4 × (d_model² + d_model)
   
   例如 d_model=512:
   = 4 × (512² + 512)
   = 4 × (262,144 + 512)
   = 4 × 262,656
   = 1,050,624 参数
   
   无论n_heads=4, 8, 16, 32，参数量都是相同的！

3. n_heads的作用:
   ✓ 改变注意力的计算方式，但不改变参数量
   ✓ n_heads越大：
      - 每个头的维度 d_k 越小
      - 可以学习更多样化的注意力模式
      - 但总参数量不变
   
   ✓ 类似于"并行计算"的概念：
      - 将注意力计算分成n_heads个并行头
      - 每个头关注不同的特征子空间
      - 但总参数量保持不变

4. 对模型能力的影响:
   ✓ n_heads增加 → 更多样化的注意力模式
      - 可以同时关注不同类型的特征关系
      - 每个头可能学习不同的注意力模式
   
   ⚠️ 但需要注意：
      - d_k = d_model / n_heads 会变小
      - 如果n_heads太大，d_k太小，可能影响表达能力
      - 通常建议 d_k ≥ 32 或 64

5. 推荐配置:
   ✓ 保持 d_k = d_model / n_heads ≥ 32
   ✓ 对于 d_model=512:
      - n_heads=8: d_k=64 ✓ (推荐)
      - n_heads=16: d_k=32 ✓ (可以)
      - n_heads=32: d_k=16 ⚠️ (可能太小)
   
   ✓ 如果d_model=768:
      - n_heads=12: d_k=64 ✓ (推荐)
      - n_heads=16: d_k=48 ✓ (可以)
      - n_heads=24: d_k=32 ✓ (可以)

6. 总结:
   ❌ n_heads增加不会增加参数量
   ✓ 参数量只与d_model有关
   ✓ n_heads只影响注意力的计算方式
   ✓ 但n_heads会影响模型的学习能力和表达能力
    """)


if __name__ == '__main__':
    analyze_nheads_impact()
