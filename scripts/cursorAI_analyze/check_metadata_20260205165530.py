#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查预处理数据metadata内容的临时脚本
日期: 20260205165530
"""

import torch
from pathlib import Path

preprocessed_dir = Path("/data/project_20251211/data/processed/preprocess_data_v1.0_20260119170929_500120")

print("=" * 80)
print("检查预处理数据的metadata内容")
print("=" * 80)

# 加载训练集
train_files = sorted(preprocessed_dir.glob("train_*.pt"))
if train_files:
    print(f"\n找到训练集文件: {train_files[-1].name}")
    train_data = torch.load(train_files[-1], map_location='cpu', weights_only=False)
    
    print("\n" + "=" * 80)
    print("训练集 metadata 所有键")
    print("=" * 80)
    print(list(train_data['metadata'].keys()))
    
    print("\n" + "=" * 80)
    print("训练集 metadata 完整内容")
    print("=" * 80)
    for key, value in train_data['metadata'].items():
        print(f"\n{key}:")
        print(f"  类型: {type(value)}")
        if isinstance(value, (list, tuple)) and len(value) > 10:
            print(f"  长度: {len(value)}")
            print(f"  前10个元素: {value[:10]}")
        else:
            print(f"  值: {value}")
    
    print("\n" + "=" * 80)
    print("检查 feature_names 字段")
    print("=" * 80)
    if 'feature_names' in train_data['metadata']:
        feature_names = train_data['metadata']['feature_names']
        print(f"✓ feature_names 存在！")
        print(f"  类型: {type(feature_names)}")
        print(f"  长度: {len(feature_names)}")
        print(f"\n所有特征名:")
        for idx, name in enumerate(feature_names):
            print(f"  [{idx}] {name}")
        
        # 检查是否包含"收盘"
        print("\n" + "=" * 80)
        print("查找包含'收盘'的特征")
        print("=" * 80)
        found = False
        for idx, name in enumerate(feature_names):
            if '收盘' in str(name):
                print(f"✓ 找到: [{idx}] {name}")
                found = True
        if not found:
            print("✗ 未找到包含'收盘'的特征")
    else:
        print("✗ feature_names 字段不存在！")
        print("\n可能需要通过其他方式确定收盘价索引")
else:
    print("✗ 未找到训练集文件")

# 检查验证集
print("\n" + "=" * 80)
print("检查验证集")
print("=" * 80)
val_files = sorted(preprocessed_dir.glob("val_*.pt"))
if val_files:
    print(f"找到验证集文件: {val_files[-1].name}")
    val_data = torch.load(val_files[-1], map_location='cpu', weights_only=False)
    
    print("\n验证集 metadata 所有键:")
    print(list(val_data['metadata'].keys()))
    
    if 'feature_names' in val_data['metadata']:
        print("✓ 验证集也有 feature_names 字段")
    else:
        print("✗ 验证集没有 feature_names 字段")
else:
    print("✗ 未找到验证集文件")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)
