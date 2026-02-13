#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试推理问题
版本: v1.0
时间: 20260211183500

目的：找出为什么推理时预测值是负数
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 检查点路径
CHECKPOINT_PATH = "/data/project_20251211/experiments/timexer_v0.43_20260207232015_20260119170929_500120/checkpoints/best_model.pth"
DATA_PATH = "/data/project_20251211/data/processed/preprocess_data_v1.0_20260119170929_500120"

def main():
    print("="*80)
    print("调试推理问题")
    print("="*80)
    
    # 1. 加载checkpoint，检查norm_mask
    print("\n1. 检查checkpoint中的norm_mask:")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    norm_mask = checkpoint['model_state_dict'].get('norm_mask')
    if norm_mask is not None:
        print(f"   norm_mask存在: {norm_mask.shape}, 全为True: {norm_mask.all()}")
    else:
        print("   norm_mask不存在")
    
    # 检查是否有norm相关的统计量
    print("\n检查checkpoint中是否有归一化统计量:")
    for key in ['norm_mean', 'norm_std', 'norm_scale', 'norm_bias']:
        if key in checkpoint['model_state_dict']:
            print(f"   ✓ {key}: {checkpoint['model_state_dict'][key].shape}")
        elif key in checkpoint:
            print(f"   ✓ {key} (顶层): {checkpoint[key]}")
        else:
            print(f"   ✗ {key}: 不存在")
    
    # 2. 加载训练数据，检查标签的范围
    print("\n2. 检查训练数据中标签的范围:")
    data_dir = Path(DATA_PATH)
    train_files = sorted(data_dir.glob("train_*.pt"))
    if train_files:
        train_data = torch.load(train_files[-1], map_location='cpu', weights_only=False)
        train_y = train_data['y']
        print(f"   训练标签形状: {train_y.shape}")
        print(f"   训练标签范围: [{train_y.min().item():.4f}, {train_y.max().item():.4f}]")
        print(f"   训练标签均值: {train_y.mean().item():.4f}")
        print(f"   训练标签标准差: {train_y.std().item():.4f}")
        
        # 检查X的范围（收盘价在第2维）
        train_X = train_data['X']
        close_idx = 2  # 假设收盘价在索引2
        train_close = train_X[:, -1, close_idx]  # 最后一个时间步的收盘价
        print(f"\n   训练输入最后收盘价范围: [{train_close.min().item():.4f}, {train_close.max().item():.4f}]")
        print(f"   训练输入最后收盘价均值: {train_close.mean().item():.4f}")
    
    # 3. 加载验证数据
    print("\n3. 检查验证数据中标签的范围:")
    val_files = sorted(data_dir.glob("val_*.pt"))
    if val_files:
        val_data = torch.load(val_files[-1], map_location='cpu', weights_only=False)
        val_y = val_data['y']
        print(f"   验证标签形状: {val_y.shape}")
        print(f"   验证标签范围: [{val_y.min().item():.4f}, {val_y.max().item():.4f}]")
        print(f"   验证标签均值: {val_y.mean().item():.4f}")
        print(f"   验证标签标准差: {val_y.std().item():.4f}")
        
        # 检查X的范围
        val_X = val_data['X']
        close_idx = 2
        val_close = val_X[:, -1, close_idx]
        print(f"\n   验证输入最后收盘价范围: [{val_close.min().item():.4f}, {val_close.max().item():.4f}]")
        print(f"   验证输入最后收盘价均值: {val_close.mean().item():.4f}")
    
    # 4. 加载模型并测试预测
    print("\n4. 测试模型预测:")
    print("   使用一个简单的测试样本...")
    
    # 创建一个测试输入（使用训练数据的第一个样本）
    test_X = train_X[0:1]  # (1, 500, 64)
    test_y = train_y[0:1]  # (1, 1)
    
    print(f"   测试输入形状: {test_X.shape}")
    print(f"   测试标签值: {test_y.item():.4f}")
    print(f"   测试输入最后收盘价: {test_X[0, -1, close_idx].item():.4f}")
    
    # 5. 检查模型配置
    print("\n5. 检查模型配置是否包含归一化参数:")
    import yaml
    
    config_path = Path(CHECKPOINT_PATH).parent.parent / "configs" / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    if 'use_norm' in model_config['model']:
        print(f"   ✓ use_norm: {model_config['model']['use_norm']}")
        print(f"   ✓ output_feature_index: {model_config['model'].get('output_feature_index', 'N/A')}")
    else:
        print("   ✗ 配置中没有use_norm参数")
    
    # 6. 分析问题
    print("\n"+"="*80)
    print("问题分析:")
    print("="*80)
    
    if norm_mask is not None and norm_mask.all():
        print("\n✓ checkpoint中有norm_mask (全为True)")
        print("  → 说明训练时尝试启用了归一化")
    
    if 'norm_mean' not in checkpoint['model_state_dict'] and 'norm_mean' not in checkpoint:
        print("\n✗ checkpoint中没有norm_mean和norm_std")
        print("  → 说明归一化统计量没有被保存")
    
    # 检查标签是否可能被归一化
    if train_y.min() >= 0 and train_y.max() < 10:
        print("\n⚠️  训练标签范围在 [0, 10) 之间")
        print("  → 标签可能已经被归一化或对数变换")
    elif train_y.min() > 10:
        print("\n✓ 训练标签范围正常（原始收盘价范围）")
        print("  → 标签未被归一化")
    
    print("\n推测的问题:")
    print("1. 训练时配置了Instance Normalization，创建了norm_mask")
    print("2. 但是归一化的统计量（norm_mean, norm_std）没有被正确保存到checkpoint")
    print("3. 模型在forward时可能对输入进行了归一化，但输出没有反归一化")
    print("4. 推理时模型输出的是归一化空间的值（接近0均值），而不是原始尺度")
    
    print("\n建议:")
    print("1. 检查v0.45的模型代码，确认Instance Normalization的实现")
    print("2. 如果模型确实使用了归一化，需要找到训练时的归一化统计量")
    print("3. 或者使用没有Instance Normalization的v0.43模型代码重新训练")


if __name__ == '__main__':
    main()
