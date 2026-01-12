#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较两个模型的参数量和存储大小
版本: 20251226120000

计算 iTransformer (v0.1TB) 和 TSMixer (v0.2) 的模型大小
输入: [128, 500, 64]
输出: [128, 1]
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import importlib.util

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def calculate_model_size(model: nn.Module) -> tuple[int, int]:
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


def create_itransformer_model(seq_len: int = 500, n_features: int = 64):
    """创建 iTransformer 模型"""
    models_path = project_root / "src" / "models" / "v0.1_20251212"
    itransformer_module = _load_module(models_path / "itransformer_decoder.py", "itransformer_decoder")
    iTransformerDecoder = itransformer_module.iTransformerDecoder
    
    # 使用配置文件中的默认参数
    model = iTransformerDecoder(
        input_features=n_features,
        seq_len=seq_len,
        d_model=512,
        n_layers=6,
        n_heads=4,
        d_ff=2048,
        dropout=0.1,
        activation="gelu",
        decoder_config={
            "use_causal_mask": False,
            "norm_type": "pre",
            "attention_dropout": 0.1,
            "ff_dropout": 0.1,
            "use_bias": True,
            "attn_embed_dim": seq_len  # iTransformer注意力在特征维度
        },
        input_resnet_config={
            "enabled": True,
            "dims": [128, 256, 512],
            "dropout": 0.1,
            "activation": "gelu",
            "use_bias": True
        },
        time_aggregation_resnet_config={
            "enabled": True,
            "dims": [50],
            "dropout": 0.1,
            "activation": "gelu",
            "use_bias": True
        },
        output_resnet_config={
            "enabled": True,
            "dims": [256, 128, 64],
            "dropout": 0.1,
            "activation": "gelu",
            "use_bias": True
        },
        final_output_config={
            "output_dim": 1,
            "activation": "gelu",
            "use_bias": True
        }
    )
    
    return model


def create_tsmixer_model(seq_len: int = 500, n_features: int = 64):
    """创建 TSMixer 模型"""
    models_path = project_root / "src" / "models" / "v0.2_20251226"
    tsmixer_module = _load_module(models_path / "tsmixer.py", "tsmixer")
    TSMixer = tsmixer_module.TSMixer
    
    # 使用配置文件中的默认参数
    model = TSMixer(
        seq_len=seq_len,
        n_features=n_features,
        prediction_len=1,
        n_blocks=2,  # 配置文件中的默认值
        ff_dim=2048,  # 保留以兼容，实际不使用
        dropout=0.1,
        activation="gelu",
        norm_type="layer",
        use_layernorm=False,  # 配置文件中的默认值
        use_residual=True,
        temporal_aggregation_config={},
        output_projection_config={}
    )
    
    return model


def main():
    """主函数"""
    print("=" * 80)
    print("模型参数量和存储大小比较")
    print("=" * 80)
    print(f"输入形状: [128, 500, 64] (batch_size=128, seq_len=500, features=64)")
    print(f"输出形状: [128, 1] (batch_size=128, output_dim=1)")
    print("=" * 80)
    print()
    
    # 创建模型
    print("正在创建模型...")
    print()
    
    # iTransformer 模型
    print("1. iTransformer (v0.1TB_20251225)")
    print("-" * 80)
    try:
        itransformer_model = create_itransformer_model(seq_len=500, n_features=64)
        itransformer_params, itransformer_size = calculate_model_size(itransformer_model)
        print(f"   参数量: {itransformer_params:,}")
        print(f"   存储大小: {format_size(itransformer_size)} ({itransformer_size:,} bytes)")
        
        # 验证输入输出形状（可选，如果失败不影响参数量计算）
        try:
            dummy_input = torch.randn(128, 500, 64)
            with torch.no_grad():
                output = itransformer_model(dummy_input)
            print(f"   输入形状验证: {dummy_input.shape} -> {output.shape}")
            if output.shape == (128, 1):
                print("   ✓ 输出形状正确")
            else:
                print(f"   ⚠ 输出形状: {output.shape} (期望 (128, 1))")
        except Exception as e:
            print(f"   ⚠ 前向传播验证失败（不影响参数量计算）: {e}")
    except Exception as e:
        print(f"   ✗ 创建模型失败: {e}")
        import traceback
        traceback.print_exc()
        itransformer_params, itransformer_size = None, None
    
    print()
    
    # TSMixer 模型
    print("2. TSMixer (v0.2_20251226)")
    print("-" * 80)
    try:
        tsmixer_model = create_tsmixer_model(seq_len=500, n_features=64)
        tsmixer_params, tsmixer_size = calculate_model_size(tsmixer_model)
        print(f"   参数量: {tsmixer_params:,}")
        print(f"   存储大小: {format_size(tsmixer_size)} ({tsmixer_size:,} bytes)")
        
        # 验证输入输出形状（可选）
        try:
            dummy_input = torch.randn(128, 500, 64)
            with torch.no_grad():
                output = tsmixer_model(dummy_input)
            print(f"   输入形状验证: {dummy_input.shape} -> {output.shape}")
            if output.shape == (128, 1):
                print("   ✓ 输出形状正确")
            else:
                print(f"   ⚠ 输出形状: {output.shape} (期望 (128, 1))")
        except Exception as e:
            print(f"   ⚠ 前向传播验证失败（不影响参数量计算）: {e}")
    except Exception as e:
        print(f"   ✗ 创建模型失败: {e}")
        import traceback
        traceback.print_exc()
        tsmixer_params, tsmixer_size = None, None
    
    print()
    print("=" * 80)
    print("比较结果")
    print("=" * 80)
    
    if itransformer_params is not None and tsmixer_params is not None:
        print(f"iTransformer 参数量: {itransformer_params:,}")
        print(f"TSMixer 参数量:      {tsmixer_params:,}")
        print()
        
        if itransformer_params > tsmixer_params:
            ratio = itransformer_params / tsmixer_params
            print(f"iTransformer 比 TSMixer 大约 {ratio:.2f} 倍")
        else:
            ratio = tsmixer_params / itransformer_params
            print(f"TSMixer 比 iTransformer 大约 {ratio:.2f} 倍")
        
        print()
        print(f"iTransformer 存储大小: {format_size(itransformer_size)}")
        print(f"TSMixer 存储大小:      {format_size(tsmixer_size)}")
        print()
        
        diff_params = abs(itransformer_params - tsmixer_params)
        diff_size = abs(itransformer_size - tsmixer_size)
        print(f"参数量差异: {diff_params:,}")
        print(f"存储大小差异: {format_size(diff_size)}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
