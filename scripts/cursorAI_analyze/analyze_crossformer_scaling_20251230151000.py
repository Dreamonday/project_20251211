#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析Crossformer模型参数量扩展方案
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


def calculate_params(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def test_config(name, seq_len, n_features, prediction_len, d_model, n_blocks, 
                n_heads, n_segments, n_feature_groups, dropout, activation):
    """测试配置"""
    try:
        model = Crossformer(
            seq_len=seq_len,
            n_features=n_features,
            prediction_len=prediction_len,
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_segments=n_segments,
            n_feature_groups=n_feature_groups,
            router_topk_ratio=0.5,
            dropout=dropout,
            activation=activation
        )
        params = calculate_params(model)
        size_mb = params * 4 / 1024 / 1024
        
        # 测试前向传播
        x = torch.randn(2, seq_len, n_features)
        with torch.no_grad():
            y = model(x)
        
        return {
            'name': name,
            'params': params,
            'size_mb': size_mb,
            'valid': True,
            'config': {
                'd_model': d_model,
                'n_blocks': n_blocks,
                'n_heads': n_heads
            }
        }
    except Exception as e:
        return {
            'name': name,
            'valid': False,
            'error': str(e)
        }


def main():
    """主函数"""
    print("=" * 80)
    print("Crossformer 模型参数量扩展方案分析")
    print("=" * 80)
    
    # 基准配置
    base_config = {
        'seq_len': 500,
        'n_features': 64,
        'prediction_len': 1,
        'd_model': 512,
        'n_blocks': 4,
        'n_heads': 8,
        'n_segments': 10,
        'n_feature_groups': 8,
        'dropout': 0.1,
        'activation': 'gelu'
    }
    
    # 测试基准配置
    base_result = test_config("基准配置", **base_config)
    base_params = base_result['params']
    target_params = base_params * 3
    
    print(f"\n基准配置:")
    print(f"  参数量: {base_params:,}")
    print(f"  存储大小: {base_result['size_mb']:.2f} MB")
    print(f"\n目标参数量 (3倍): {target_params:,}")
    print("=" * 80)
    
    # 方案1: 增加d_model
    print("\n方案1: 增加d_model (隐藏维度)")
    print("-" * 80)
    configs_1 = [
        ('d_model=768', {**base_config, 'd_model': 768, 'n_heads': 12}),
        ('d_model=1024', {**base_config, 'd_model': 1024, 'n_heads': 16}),
        ('d_model=1280', {**base_config, 'd_model': 1280, 'n_heads': 16}),
    ]
    
    for name, config in configs_1:
        result = test_config(name, **config)
        if result['valid']:
            ratio = result['params'] / base_params
            diff = abs(result['params'] - target_params) / target_params * 100
            print(f"  {name}:")
            print(f"    参数量: {result['params']:,} ({ratio:.2f}x)")
            print(f"    存储大小: {result['size_mb']:.2f} MB")
            print(f"    与目标差异: {diff:.1f}%")
            print(f"    配置: d_model={config['d_model']}, n_heads={config['n_heads']}")
    
    # 方案2: 增加n_blocks
    print("\n方案2: 增加n_blocks (块数量)")
    print("-" * 80)
    configs_2 = [
        ('n_blocks=6', {**base_config, 'n_blocks': 6}),
        ('n_blocks=8', {**base_config, 'n_blocks': 8}),
        ('n_blocks=10', {**base_config, 'n_blocks': 10}),
    ]
    
    for name, config in configs_2:
        result = test_config(name, **config)
        if result['valid']:
            ratio = result['params'] / base_params
            diff = abs(result['params'] - target_params) / target_params * 100
            print(f"  {name}:")
            print(f"    参数量: {result['params']:,} ({ratio:.2f}x)")
            print(f"    存储大小: {result['size_mb']:.2f} MB")
            print(f"    与目标差异: {diff:.1f}%")
    
    # 方案3: 组合方案
    print("\n方案3: 组合方案 (d_model + n_blocks)")
    print("-" * 80)
    configs_3 = [
        ('d_model=768, n_blocks=6', {**base_config, 'd_model': 768, 'n_blocks': 6, 'n_heads': 12}),
        ('d_model=768, n_blocks=8', {**base_config, 'd_model': 768, 'n_blocks': 8, 'n_heads': 12}),
        ('d_model=1024, n_blocks=4', {**base_config, 'd_model': 1024, 'n_blocks': 4, 'n_heads': 16}),
        ('d_model=1024, n_blocks=5', {**base_config, 'd_model': 1024, 'n_blocks': 5, 'n_heads': 16}),
    ]
    
    best_config = None
    best_diff = float('inf')
    
    for name, config in configs_3:
        result = test_config(name, **config)
        if result['valid']:
            ratio = result['params'] / base_params
            diff = abs(result['params'] - target_params) / target_params * 100
            print(f"  {name}:")
            print(f"    参数量: {result['params']:,} ({ratio:.2f}x)")
            print(f"    存储大小: {result['size_mb']:.2f} MB")
            print(f"    与目标差异: {diff:.1f}%")
            if diff < best_diff:
                best_diff = diff
                best_config = (name, config, result)
    
    # 推荐方案
    print("\n" + "=" * 80)
    print("推荐方案")
    print("=" * 80)
    if best_config:
        name, config, result = best_config
        print(f"\n最接近目标的配置: {name}")
        print(f"  参数量: {result['params']:,} ({result['params']/base_params:.2f}x)")
        print(f"  存储大小: {result['size_mb']:.2f} MB")
        print(f"  与目标差异: {best_diff:.1f}%")
        print(f"\n配置参数:")
        print(f"  d_model: {config['d_model']}")
        print(f"  n_blocks: {config['n_blocks']}")
        print(f"  n_heads: {config['n_heads']}")
        print(f"  n_segments: {config['n_segments']}")
        print(f"  n_feature_groups: {config['n_feature_groups']}")
    
    print("\n" + "=" * 80)
    print("能力提升分析")
    print("=" * 80)
    print("""
1. 增加d_model (隐藏维度):
   ✓ 最直接有效的方式
   ✓ 增加模型的表达能力，可以捕捉更复杂的特征交互
   ✓ 注意力机制的计算能力增强
   ✓ 参数量增长为平方级（d_model²）
   ⚠️ 计算量和显存占用也会大幅增加

2. 增加n_blocks (块数量):
   ✓ 增加模型深度，可以学习更复杂的时序模式
   ✓ 通过多层堆叠，可以捕捉不同层次的抽象
   ✓ 参数量增长为线性
   ⚠️ 训练可能更困难，需要更好的正则化

3. 组合方案 (d_model + n_blocks):
   ✓ 平衡表达能力和深度
   ✓ 可以更灵活地控制参数量
   ✓ 通常能获得更好的性能
   ⚠️ 需要更多计算资源

推荐策略:
- 优先增加d_model到768或1024（如果显存允许）
- 然后适当增加n_blocks到6-8层
- 保持n_heads与d_model的比例（d_model/n_heads ≈ 64）
    """)


if __name__ == '__main__':
    main()
