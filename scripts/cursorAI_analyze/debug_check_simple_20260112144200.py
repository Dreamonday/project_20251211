#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单调试脚本：只检查数据统计
日期: 20260112144200
"""

import torch
import numpy as np
from pathlib import Path

# 数据路径
PREPROCESSED_DIR = "/data/project_20251211/data/processed/preprocess_data_v0.5_20260112120624_500120_more_continueTV"

print("=" * 80)
print("检查预处理数据")
print("=" * 80)

# 加载训练数据
train_pt = Path(PREPROCESSED_DIR) / "train_v0.5.pt"
print(f"\n加载训练数据: {train_pt}")
train_data = torch.load(train_pt, map_location='cpu', weights_only=False)

print("\n训练数据的键:")
print(train_data.keys())

# 检查X和y的形状和统计信息
print("\n训练数据的X:")
print(f"  形状: {train_data['X'].shape}")
print(f"  数据类型: {train_data['X'].dtype}")
print(f"  最小值: {train_data['X'].min().item():.6f}")
print(f"  最大值: {train_data['X'].max().item():.6f}")
print(f"  平均值: {train_data['X'].mean().item():.6f}")
print(f"  标准差: {train_data['X'].std().item():.6f}")

# 检查是否有NaN或Inf
print(f"  包含NaN: {torch.isnan(train_data['X']).any().item()}")
print(f"  包含Inf: {torch.isinf(train_data['X']).any().item()}")

print("\n训练数据的y:")
print(f"  形状: {train_data['y'].shape}")
print(f"  数据类型: {train_data['y'].dtype}")
print(f"  最小值: {train_data['y'].min().item():.6f}")
print(f"  最大值: {train_data['y'].max().item():.6f}")
print(f"  平均值: {train_data['y'].mean().item():.6f}")
print(f"  标准差: {train_data['y'].std().item():.6f}")

# 检查是否有NaN或Inf
print(f"  包含NaN: {torch.isnan(train_data['y']).any().item()}")
print(f"  包含Inf: {torch.isinf(train_data['y']).any().item()}")

# 打印前20个y值
print("\n前20个目标值(y):")
for i in range(min(20, len(train_data['y']))):
    print(f"  样本{i:3d}: {train_data['y'][i].item():10.2f}")

# 检查是否有标准化信息
print("\n")
if 'scaler' in train_data:
    print("✓ 包含scaler信息")
elif 'scaler_X' in train_data or 'scaler_y' in train_data:
    print("✓ 包含scaler_X或scaler_y信息")
else:
    print("✗ 不包含scaler信息")

# 检查metadata中是否有scaler相关信息
print("\nmetadata中的键:")
for key in train_data['metadata'].keys():
    print(f"  - {key}")
    
if 'scaler_info' in train_data['metadata']:
    print("\n✓ metadata中包含scaler_info")
    print(train_data['metadata']['scaler_info'])

print("\n" + "=" * 80)
print("数据检查完成")
print("=" * 80)
