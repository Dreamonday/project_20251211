#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看预处理数据的脚本
"""
import torch
import json
from pathlib import Path

# 加载预处理数据
pt_file = Path("/data/project_20251211/data/processed/preprocess_data_20251216122731/train_v0.1_20251212.pt")

print("=" * 80)
print("加载预处理数据...")
print("=" * 80)

data = torch.load(pt_file, weights_only=False)

print("\n数据文件包含的键:")
for key in data.keys():
    print(f"  - {key}")

print("\n" + "=" * 80)
print("X (输入特征) 信息:")
print("=" * 80)
X = data['X']
print(f"  类型: {type(X)}")
print(f"  形状: {X.shape}")
print(f"  数据类型: {X.dtype}")
print(f"  占用内存: {X.element_size() * X.nelement() / (1024**2):.2f} MB")
print(f"\n  解释:")
print(f"    - 样本数量: {X.shape[0]}")
print(f"    - 序列长度: {X.shape[1]}")
print(f"    - 特征维度: {X.shape[2]}")

print(f"\n  前3个样本的第1个时间步的前5个特征:")
print(X[:3, 0, :5])

print("\n" + "=" * 80)
print("y (目标值) 信息:")
print("=" * 80)
y = data['y']
print(f"  类型: {type(y)}")
print(f"  形状: {y.shape}")
print(f"  数据类型: {y.dtype}")
print(f"  占用内存: {y.element_size() * y.nelement() / (1024**2):.2f} MB")
print(f"\n  前10个目标值:")
print(y[:10].squeeze())

print("\n" + "=" * 80)
print("metadata (元数据) 信息:")
print("=" * 80)
metadata = data['metadata']
for key, value in metadata.items():
    if isinstance(value, list) and len(value) > 5:
        print(f"  {key}: [列表，共{len(value)}项，前5项: {value[:5]}...]")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("feature_stats (特征统计量) 示例:")
print("=" * 80)
stats = data['feature_stats']
if stats:
    # 显示前3个特征的统计量
    feature_names = list(stats.keys())[:3]
    for fname in feature_names:
        print(f"\n  特征: {fname}")
        print(f"    - 均值: {stats[fname]['mean']:.6f}")
        print(f"    - 标准差: {stats[fname]['std']:.6f}")
        print(f"    - 最小值: {stats[fname]['min']:.6f}")
        print(f"    - 最大值: {stats[fname]['max']:.6f}")

print("\n" + "=" * 80)
print("总结:")
print("=" * 80)
print(f"✅ 预处理数据包含 {X.shape[0]} 个样本")
print(f"✅ 每个样本是 {X.shape[1]} 个时间步 × {X.shape[2]} 个特征的张量")
print(f"✅ 特征已经{'标准化' if metadata.get('normalize') else '未标准化'}")
print(f"✅ 所有数据已转为torch.Tensor，可直接训练")
print(f"✅ 总文件大小: {pt_file.stat().st_size / (1024**2):.2f} MB")

