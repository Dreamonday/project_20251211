#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时调试脚本：检查预处理数据和模型输出
日期: 20260112143800
"""

import torch
import numpy as np
from pathlib import Path

# 数据路径
PREPROCESSED_DIR = "/data/project_20251211/data/processed/preprocess_data_v0.5_20260112120624_500120_more_continueTV"
MODEL_DIR = "/data/project_20251211/experiments/timexer_mlp_v0.5_20260112122205_20260112120624_500120_more_continueTV"

print("=" * 80)
print("检查预处理数据")
print("=" * 80)

# 加载训练数据
train_pt = Path(PREPROCESSED_DIR) / "train_v0.5.pt"
print(f"\n加载训练数据: {train_pt}")
train_data = torch.load(train_pt, map_location='cpu', weights_only=False)

print("\n训练数据的键:")
print(train_data.keys())

print("\n训练数据的metadata:")
for key, value in train_data['metadata'].items():
    print(f"  {key}: {value}")

# 检查X和y的形状和统计信息
print("\n训练数据的X:")
print(f"  形状: {train_data['X'].shape}")
print(f"  数据类型: {train_data['X'].dtype}")
print(f"  最小值: {train_data['X'].min().item():.6f}")
print(f"  最大值: {train_data['X'].max().item():.6f}")
print(f"  平均值: {train_data['X'].mean().item():.6f}")
print(f"  标准差: {train_data['X'].std().item():.6f}")

print("\n训练数据的y:")
print(f"  形状: {train_data['y'].shape}")
print(f"  数据类型: {train_data['y'].dtype}")
print(f"  最小值: {train_data['y'].min().item():.6f}")
print(f"  最大值: {train_data['y'].max().item():.6f}")
print(f"  平均值: {train_data['y'].mean().item():.6f}")
print(f"  标准差: {train_data['y'].std().item():.6f}")

# 打印前10个y值
print("\n前10个目标值:")
for i in range(min(10, len(train_data['y']))):
    print(f"  样本{i}: {train_data['y'][i].item():.2f}")

# 检查是否有标准化信息
if 'scaler' in train_data:
    print("\n包含scaler信息")
    print(train_data['scaler'])
else:
    print("\n不包含scaler信息")

print("\n" + "=" * 80)
print("检查模型输出（前16个样本）")
print("=" * 80)

# 加载模型
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import importlib.util
import yaml

def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

model_dir = Path(MODEL_DIR)
config_path = model_dir / "configs" / "model_config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model_config = config['model']

# 加载模型
models_path = Path(__file__).parent.parent.parent / "src" / "models" / "v0.5_20260107"
timexer_mlp_module = _load_module(models_path / "timexer_mlp.py", "timexer_mlp")
ModelClass = timexer_mlp_module.TimeXerMLP

model = ModelClass(
    seq_len=model_config['seq_len'],
    n_features=model_config['n_features'],
    endogenous_features=model_config.get('endogenous_features', 44),
    exogenous_features=model_config.get('exogenous_features', 20),
    prediction_len=model_config.get('prediction_len', 1),
    endogenous_indices=model_config.get('endogenous_indices'),
    exogenous_indices=model_config.get('exogenous_indices'),
    endogenous_blocks=model_config.get('endogenous_blocks', 3),
    endogenous_hidden_dim=model_config.get('endogenous_hidden_dim', 256),
    exogenous_blocks=model_config.get('exogenous_blocks', 2),
    exogenous_hidden_dim=model_config.get('exogenous_hidden_dim', 256),
    shared_time_mixing=model_config.get('shared_time_mixing', True),
    mlp_fusion_ff_dim=model_config.get('mlp_fusion_ff_dim', 512),
    dropout=model_config.get('dropout', 0.1),
    activation=model_config.get('activation', 'gelu'),
    use_layernorm=model_config.get('use_layernorm', True),
    use_residual=model_config.get('use_residual', True),
    norm_type=model_config.get('norm_type', 'layer'),
    n_blocks=model_config.get('n_blocks', None),
    ff_dim=model_config.get('ff_dim', None),
    temporal_aggregation_config=model_config.get('temporal_aggregation', {}),
    output_projection_config=model_config.get('output_projection', {})
)

# 加载权重
checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("\n模型加载完成")

# 推理前16个样本
batch_X = train_data['X'][:16]
batch_y = train_data['y'][:16]

with torch.no_grad():
    predictions = model(batch_X)

print("\n前16个样本的推理结果:")
print(f"{'样本':<8} {'预测值':<15} {'真实值':<15} {'误差':<15}")
print("-" * 60)
for i in range(16):
    pred = predictions[i].item()
    true = batch_y[i].item()
    error = abs(pred - true) / abs(true) * 100 if true != 0 else float('inf')
    print(f"{i:<8} {pred:<15.6f} {true:<15.2f} {error:<15.2f}%")

print("\n预测值统计:")
print(f"  最小值: {predictions.min().item():.6f}")
print(f"  最大值: {predictions.max().item():.6f}")
print(f"  平均值: {predictions.mean().item():.6f}")
print(f"  标准差: {predictions.std().item():.6f}")
