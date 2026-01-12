#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算Crossformer模型参数量
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


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def calculate_model_size(model: torch.nn.Module) -> tuple[int, int]:
    """
    计算模型的参数量和存储大小
    
    Returns:
        (参数量, 存储大小_bytes)
    """
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    
    # 计算存储大小（假设使用float32，每个参数4字节）
    size_bytes = num_params * 4
    
    return num_params, size_bytes


def main():
    """主函数"""
    print("=" * 80)
    print("Crossformer 模型参数量计算")
    print("=" * 80)
    
    # 模型配置
    seq_len = 500
    n_features = 64
    prediction_len = 1
    d_model = 512
    n_blocks = 4
    n_heads = 8
    n_segments = 10
    n_feature_groups = 8
    dropout = 0.1
    activation = "gelu"
    
    print(f"\n模型配置:")
    print(f"  输入形状: [batch_size, {seq_len}, {n_features}]")
    print(f"  输出形状: [batch_size, {prediction_len}]")
    print(f"  d_model: {d_model}")
    print(f"  n_blocks: {n_blocks}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_segments: {n_segments}")
    print(f"  n_feature_groups: {n_feature_groups}")
    
    # 创建模型
    print(f"\n创建模型...")
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
    
    # 计算参数量
    num_params, size_bytes = calculate_model_size(model)
    
    # 使用模型自带的方法验证
    model_params = model.get_num_parameters()
    
    print(f"\n参数量统计:")
    print(f"  总参数量: {num_params:,}")
    print(f"  模型方法返回: {model_params:,}")
    print(f"  差异: {abs(num_params - model_params):,}")
    
    print(f"\n存储大小:")
    print(f"  模型参数: {format_size(size_bytes)}")
    print(f"  约 {size_bytes / 1024 / 1024:.2f} MB")
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    batch_size = 32
    x = torch.randn(batch_size, seq_len, n_features)
    with torch.no_grad():
        y = model(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")
    print(f"  前向传播成功!")
    
    # 详细参数统计（按模块）
    print(f"\n详细参数统计（按模块）:")
    total = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count
        if param_count > 1000:  # 只显示大于1000的参数模块
            print(f"  {name}: {param_count:,} ({param.shape})")
    
    print(f"\n总计验证: {total:,}")
    print("=" * 80)


if __name__ == '__main__':
    main()
