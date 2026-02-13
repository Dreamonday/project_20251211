#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主训练脚本 - TimeXer-Official适配器版本
版本: v0.6
日期: 20260206
时间戳: 110256

整合所有模块，作为训练入口，支持TensorBoard可视化
使用官方TimeXer架构（Patch-Level内生 + Variate-Level外生 + Global Token）

v0.6新增：
- 使用官方TimeXer核心架构
- 单一内生变量（索引2，第3个特征-收盘）
- patch_len=25
- 保留学习型Missing Embedding
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import argparse
import sys
import importlib.util
from datetime import datetime
import pandas as pd
import json
import os
import re


# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ===== 默认训练数据路径（如需调整请修改此处） =====
DEFAULT_PREPROCESSED_DIR = project_root / "data" / "processed" / "preprocess_data_v1.0_20260119170929_500120"

# ===== 数据设备配置（如需调整请修改此处） =====
# 选项: 'cpu' 或 'cuda'
# - 'cpu': 数据存储在CPU内存中，训练时通过pin_memory传输到GPU（推荐，支持多进程加载）
# - 'cuda': 数据直接存储在GPU显存中，训练时无需传输（更快，但需要显存充足，且num_workers必须为0）
# 注意: 使用内存映射模式（mmap_mode=True）时，数据自动使用CPU，此配置被忽略
DATA_DEVICE = 'cpu'  # 修改此处: 'cpu' 或 'cuda'


# 动态导入模块
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入所需模块（使用v0.6_20260206版本的模型，v0.3_20260118版本的trainer，v0.3_20260119版本的数据(支持mmap)，v0.1版本的utils）
models_path = project_root / "src" / "models" / "v0.6_20260206"
data_path = project_root / "src" / "data" / "v0.3_20260119"
training_path = project_root / "src" / "training" / "v0.3_20260118"
utils_path = project_root / "src" / "utils" / "v0.1_20251212"

# 导入模型
timexer_module = _load_module(models_path / "timexer_official_adapter.py", "timexer_official_adapter")
TimeXerOfficialAdapter = timexer_module.TimeXerOfficialAdapter

# 导入数据集
stock_dataset_module = _load_module(data_path / "stock_dataset.py", "stock_dataset")
StockDataset = stock_dataset_module.StockDataset

# 导入预处理数据集
preprocessed_dataset_module = _load_module(data_path / "preprocessed_dataset.py", "preprocessed_dataset")
PreprocessedStockDataset = preprocessed_dataset_module.PreprocessedStockDataset

# 导入训练器
trainer_module = _load_module(training_path / "trainer.py", "trainer")
Trainer = trainer_module.Trainer
EarlyStopping = trainer_module.EarlyStopping

# 导入工具函数
feature_utils_module = _load_module(utils_path / "feature_utils.py", "feature_utils")
compute_feature_stats = feature_utils_module.compute_feature_stats
save_feature_stats = feature_utils_module.save_feature_stats
get_feature_columns = feature_utils_module.get_feature_columns

data_utils_module = _load_module(utils_path / "data_utils.py", "data_utils")
load_parquet_file = data_utils_module.load_parquet_file
get_data_by_rows = data_utils_module.get_data_by_rows

loss_utils_module = _load_module(utils_path / "loss_utils.py", "loss_utils")
MAPELoss = loss_utils_module.MAPELoss
SMAPELoss = loss_utils_module.SMAPELoss


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, optimizer_config: dict) -> torch.optim.Optimizer:
    """创建优化器"""
    opt_type = optimizer_config['type'].lower()
    lr = float(optimizer_config['lr'])
    weight_decay = float(optimizer_config.get('weight_decay', 0.0))
    betas = optimizer_config.get('betas', [0.9, 0.999])
    # 确保 betas 是浮点数列表
    if isinstance(betas, list):
        betas = tuple(float(b) for b in betas)
    
    if opt_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'sgd':
        momentum = float(optimizer_config.get('momentum', 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict, num_epochs: int):
    """创建学习率调度器"""
    sched_type = scheduler_config['type'].lower()
    
    if sched_type == 'none':
        return None
    elif sched_type == 'cosine':
        T_max = int(scheduler_config.get('T_max', num_epochs))
        eta_min = float(scheduler_config.get('eta_min', 0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif sched_type == 'step':
        step_size = int(scheduler_config.get('step_size', 30))
        gamma = float(scheduler_config.get('gamma', 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'plateau':
        factor = float(scheduler_config.get('factor', 0.5))
        patience = int(scheduler_config.get('patience', 10))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def create_criterion(loss_config: dict) -> nn.Module:
    """创建损失函数"""
    loss_type = loss_config['type'].lower()
    reduction = loss_config.get('reduction', 'mean')
    
    if loss_type == 'mse':
        return nn.MSELoss(reduction=reduction)
    elif loss_type == 'mae':
        return nn.L1Loss(reduction=reduction)
    elif loss_type == 'huber':
        delta = float(loss_config.get('delta', 1.0))
        return nn.HuberLoss(reduction=reduction, delta=delta)
    elif loss_type == 'mape':
        epsilon = float(loss_config.get('epsilon', 1e-8))
        max_relative_error = loss_config.get('max_relative_error', None)
        if max_relative_error is not None:
            max_relative_error = float(max_relative_error)
        return MAPELoss(reduction=reduction, epsilon=epsilon, max_relative_error=max_relative_error)
    elif loss_type == 'smape':
        epsilon = float(loss_config.get('epsilon', 1e-8))
        max_relative_error = loss_config.get('max_relative_error', None)
        if max_relative_error is not None:
            max_relative_error = float(max_relative_error)
        return SMAPELoss(reduction=reduction, epsilon=epsilon, max_relative_error=max_relative_error)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: 'mse', 'mae', 'huber', 'mape', 'smape'")


def find_preprocessed_files(preprocessed_dir: Path) -> tuple[Path, Path, Path]:
    """
    自动检测预处理目录中的训练集、验证集和测试集文件
    
    Args:
        preprocessed_dir: 预处理数据目录
        
    Returns:
        (train_pt_file, val_pt_file, test_pt_file): 训练集、验证集和测试集文件路径
        
    Raises:
        FileNotFoundError: 如果找不到对应的文件
    """
    # 查找 train_*.pt 文件
    train_files = list(preprocessed_dir.glob("train_*.pt"))
    if not train_files:
        available_files = list(preprocessed_dir.glob("*.pt"))
        raise FileNotFoundError(
            f"在预处理目录中未找到训练集文件 (train_*.pt)\n"
            f"预处理目录: {preprocessed_dir}\n"
            f"目录中的 .pt 文件: {[f.name for f in available_files]}\n"
            f"请先运行预处理脚本生成训练集文件"
        )
    if len(train_files) > 1:
        print(f"  警告: 找到多个训练集文件，使用最新的: {train_files[0].name}")
        train_files = sorted(train_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    # 查找 val_*.pt 文件
    val_files = list(preprocessed_dir.glob("val_*.pt"))
    if not val_files:
        available_files = list(preprocessed_dir.glob("*.pt"))
        raise FileNotFoundError(
            f"在预处理目录中未找到验证集文件 (val_*.pt)\n"
            f"预处理目录: {preprocessed_dir}\n"
            f"目录中的 .pt 文件: {[f.name for f in available_files]}\n"
            f"请先运行预处理脚本生成验证集文件"
        )
    if len(val_files) > 1:
        print(f"  警告: 找到多个验证集文件，使用最新的: {val_files[0].name}")
        val_files = sorted(val_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    # 查找 test_*.pt 文件（可选）
    test_files = list(preprocessed_dir.glob("test_*.pt"))
    if not test_files:
        print(f"  警告: 未找到测试集文件 (test_*.pt)")
        test_pt_file = None
    else:
        if len(test_files) > 1:
            print(f"  警告: 找到多个测试集文件，使用最新的: {test_files[0].name}")
            test_files = sorted(test_files, key=lambda x: x.stat().st_mtime, reverse=True)
        test_pt_file = test_files[0]
    
    train_pt_file = train_files[0]
    val_pt_file = val_files[0]
    
    print(f"✓ 检测到训练集文件: {train_pt_file.name}")
    print(f"✓ 检测到验证集文件: {val_pt_file.name}")
    if test_pt_file:
        print(f"✓ 检测到测试集文件: {test_pt_file.name}")
    
    return train_pt_file, val_pt_file, test_pt_file


def compute_train_stats(train_dataset: StockDataset, dataset_config: dict) -> dict:
    """计算训练集的特征统计量"""
    print("计算训练集特征统计量...")
    
    # 获取特征列
    exclude_columns = dataset_config['features']['exclude_columns']
    
    # 从训练集中采样一些数据来计算统计量
    sample_size = min(1000, len(train_dataset))
    indices = torch.randperm(len(train_dataset))[:sample_size].tolist()
    
    all_features = []
    for idx in indices:
        sample = train_dataset.index_df.iloc[idx]
        source_file = sample['source_file']
        
        # 读取数据文件
        df = load_parquet_file(source_file, use_cache=True)
        
        # 提取输入数据
        input_start = sample['input_row_start']
        input_end = sample['input_row_end']
        input_df = get_data_by_rows(df, input_start, input_end)
        
        # 获取特征列
        feature_columns = get_feature_columns(input_df, exclude_columns)
        features = input_df[feature_columns]
        all_features.append(features)
    
    # 合并所有特征
    all_features_df = pd.concat(all_features, ignore_index=True)
    
    # 计算统计量
    stats = compute_feature_stats(all_features_df)
    
    print(f"✓ 完成统计量计算（样本数: {sample_size}）")
    return stats


def main():
    parser = argparse.ArgumentParser(description='训练TimeXer-Official模型')
    parser.add_argument('--model-config', type=str, 
                        default=str(project_root / 'configs' / 'v0.6_20260206' / 'timexer_official_config.yaml'),
                        help='模型配置文件路径')
    parser.add_argument('--train-config', type=str,
                        default=str(project_root / 'configs' / 'v0.6_20260206' / 'training_config.yaml'),
                        help='训练配置文件路径')
    parser.add_argument('--data-dir', type=str,
                        default=str(DEFAULT_PREPROCESSED_DIR),
                        help='预处理数据目录路径')
    parser.add_argument('--output-dir', type=str,
                        default=str(project_root / 'experiments'),
                        help='输出目录路径')
    parser.add_argument('--experiment-name', type=str,
                        default=None,
                        help='实验名称（如果不指定，将自动生成）')
    parser.add_argument('--resume', type=str,
                        default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--mmap-mode', action='store_true',
                        help='是否使用内存映射模式加载数据（推荐用于大数据集）')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)，覆盖配置文件')
    
    args = parser.parse_args()
    
    # 加载配置
    print("=" * 80)
    print("加载配置文件...")
    print("=" * 80)
    model_config = load_config(args.model_config)
    training_config = load_config(args.train_config)
    
    # 打印配置信息
    print(f"\n模型配置文件: {args.model_config}")
    print(f"训练配置文件: {args.train_config}")
    print(f"数据目录: {args.data_dir}")
    print(f"内存映射模式: {'开启' if args.mmap_mode else '关闭'}")
    
    # 设备配置（命令行参数优先级最高）
    if args.device:
        device_str = args.device
    else:
        device_str = training_config['training']['device']
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建实验目录
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        data_dir_name = Path(args.data_dir).name
        experiment_name = f"timexer_v0.6_{timestamp}_{data_dir_name}"
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"实验目录: {output_dir}")
    
    # 保存配置文件副本
    config_dir = output_dir / 'configs'
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / 'model_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)
    with open(config_dir / 'training_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 配置文件已保存到: {config_dir}")
    
    # ========== 数据加载 ==========
    print("\n" + "=" * 80)
    print("加载数据集...")
    print("=" * 80)
    
    data_dir = Path(args.data_dir)
    
    # 获取数据加载配置（优先级：命令行 > 配置文件 > 默认值）
    data_loading_config = training_config['training'].get('data_loading', {})
    
    # mmap_mode: 命令行参数优先，否则使用配置文件，默认False
    if args.mmap_mode:
        mmap_mode = True
        print("✓ 使用命令行参数: mmap_mode=True")
    else:
        mmap_mode = data_loading_config.get('mmap_mode', False)
        print(f"✓ 使用配置文件: mmap_mode={mmap_mode}")
    
    precompute_mask = data_loading_config.get('precompute_mask', True)
    load_to_memory = data_loading_config.get('load_to_memory', False)
    blank_value = data_loading_config.get('blank_value', -1000.0)
    return_mask = data_loading_config.get('return_mask', True)
    
    print(f"  - mmap_mode: {mmap_mode}")
    print(f"  - precompute_mask: {precompute_mask}")
    print(f"  - load_to_memory: {load_to_memory}")
    print(f"  - blank_value: {blank_value}")
    
    # 自动查找预处理文件
    print(f"\n预处理数据目录: {data_dir}")
    train_pt_file, val_pt_file, test_pt_file = find_preprocessed_files(data_dir)
    
    # 加载训练集
    train_dataset = PreprocessedStockDataset(
        pt_file_path=str(train_pt_file),
        load_to_memory=load_to_memory,
        device=None,
        blank_value=blank_value,
        return_mask=return_mask,
        mmap_mode=mmap_mode,
        precompute_mask=precompute_mask
    )
    
    # 加载验证集
    val_dataset = PreprocessedStockDataset(
        pt_file_path=str(val_pt_file),
        load_to_memory=load_to_memory,
        device=None,
        blank_value=blank_value,
        return_mask=return_mask,
        mmap_mode=mmap_mode,
        precompute_mask=precompute_mask
    )
    
    # 加载测试集（如果存在）
    if test_pt_file:
        test_dataset = PreprocessedStockDataset(
            pt_file_path=str(test_pt_file),
            load_to_memory=load_to_memory,
            device=None,
            blank_value=blank_value,
            return_mask=return_mask,
            mmap_mode=mmap_mode,
            precompute_mask=False  # 测试集不预计算mask以节省启动时间
        )
    else:
        test_dataset = None
        print("  ⚠ 未找到测试集，跳过测试集加载")
    
    print(f"✓ 训练集样本数: {len(train_dataset)}")
    print(f"✓ 验证集样本数: {len(val_dataset)}")
    if test_dataset:
        print(f"✓ 测试集样本数: {len(test_dataset)}")
    
    # 获取数据维度（从第一个样本推断）
    sample_data = train_dataset[0]
    if len(sample_data) == 3:  # 返回了 (X, y, mask)
        sample_X, sample_y, sample_mask = sample_data
    elif len(sample_data) == 2:  # 返回了 (X, y)
        sample_X, sample_y = sample_data
    else:
        raise ValueError(f"Dataset返回了意外数量的值: {len(sample_data)}")
    
    if isinstance(sample_X, dict):
        seq_len, n_features = sample_X['X'].shape
    else:
        seq_len, n_features = sample_X.shape
    
    print(f"✓ 输入维度: seq_len={seq_len}, n_features={n_features}")
    
    # 更新模型配置中的维度（如果不匹配）
    if model_config['model']['seq_len'] != seq_len:
        print(f"  ⚠ 更新seq_len: {model_config['model']['seq_len']} → {seq_len}")
        model_config['model']['seq_len'] = seq_len
    if model_config['model']['n_features'] != n_features:
        print(f"  ⚠ 更新n_features: {model_config['model']['n_features']} → {n_features}")
        model_config['model']['n_features'] = n_features
    
    # ========== 模型初始化 ==========
    print("\n" + "=" * 80)
    print("初始化模型...")
    print("=" * 80)
    
    model_params = model_config['model']
    
    # 创建TimeXer-Official适配器
    model = TimeXerOfficialAdapter(
        seq_len=model_params['seq_len'],
        n_features=model_params['n_features'],
        endogenous_index=model_params.get('endogenous_index', 2),
        prediction_len=model_params['prediction_len'],
        patch_len=model_params.get('patch_len', 25),
        d_model=model_params.get('d_model', 64),
        n_heads=model_params.get('n_heads', 8),
        e_layers=model_params.get('e_layers', 2),
        d_ff=model_params.get('d_ff', 256),
        dropout=model_params.get('dropout', 0.1),
        activation=model_params.get('activation', 'gelu'),
        use_norm=model_params.get('use_norm', True),
        norm_feature_indices=model_params.get('norm_feature_indices', None),  # 归一化特征索引列表
        missing_value_flag=blank_value  # 使用training_config中的blank_value作为缺失值标记
    )
    
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"✓ 模型类型: {model_info['model_type']}")
    print(f"✓ 内生变量索引: {model_info['endogenous_index']} (第{model_info['endogenous_index']+1}个特征)")
    print(f"✓ 外生变量数量: {model_info['exogenous_features']}")
    print(f"✓ Patch配置: patch_len={model_info['patch_len']}, patch_num={model_info['patch_num']}")
    print(f"✓ 模型维度: d_model={model_info['d_model']}, e_layers={model_info['e_layers']}")
    print(f"✓ 可训练参数数量: {model_info['num_parameters']:,}")
    print(f"✓ Missing Embedding: {'启用' if model_info['missing_embedding_enabled'] else '禁用'}")
    print(f"✓ Instance Norm: {'启用' if model_info['use_norm'] else '禁用'}")
    if model_info['use_norm']:
        print(f"  - 归一化特征数量: {model_info['num_norm_features']}/{model_info['n_features']}")
        print(f"  - 归一化特征索引: {model_info['norm_feature_indices']}")
    
    # ========== 创建DataLoader ==========
    print("\n" + "=" * 80)
    print("创建DataLoader...")
    print("=" * 80)
    
    train_config = training_config['training']
    
    # 根据mmap模式调整num_workers
    num_workers = train_config['num_workers']
    if mmap_mode and num_workers > 0:
        print(f"  ⚠ mmap模式下，num_workers调整为0（避免多进程共享内存问题）")
        num_workers = 0
    
    # 根据数据设备调整pin_memory
    pin_memory = train_config.get('pin_memory', True)
    if DATA_DEVICE == 'cuda':
        print(f"  ⚠ 数据已在GPU，pin_memory设置为False")
        pin_memory = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_config.get('prefetch_factor', 2) if num_workers > 0 else None,
        persistent_workers=train_config.get('persistent_workers', False) if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_config.get('prefetch_factor', 2) if num_workers > 0 else None,
        persistent_workers=train_config.get('persistent_workers', False) if num_workers > 0 else False
    )
    
    # 创建测试集DataLoader（如果存在）
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        test_loader = None
    
    print(f"✓ DataLoader创建完成")
    print(f"  - batch_size: {train_config['batch_size']}")
    print(f"  - num_workers: {num_workers}")
    print(f"  - pin_memory: {pin_memory}")
    
    # ========== 创建优化器和调度器 ==========
    print("\n" + "=" * 80)
    print("创建优化器和调度器...")
    print("=" * 80)
    
    optimizer = create_optimizer(model, train_config['optimizer'])
    scheduler = create_scheduler(optimizer, train_config['scheduler'], train_config['num_epochs'])
    
    print(f"✓ 优化器: {train_config['optimizer']['type']}")
    print(f"✓ 学习率: {train_config['optimizer']['lr']}")
    if scheduler:
        print(f"✓ 调度器: {train_config['scheduler']['type']}")
    
    # ========== 创建损失函数 ==========
    criterion = create_criterion(train_config['loss'])
    print(f"✓ 损失函数: {train_config['loss']['type']}")
    
    # ========== 创建早停 ==========
    early_stopping = None
    if train_config['early_stopping']['enabled']:
        # 早停使用的指标和模式（如果没配置，使用best_metric的配置）
        early_stop_metric = train_config['early_stopping'].get('metric', train_config.get('best_metric', 'loss'))
        early_stop_mode = train_config['early_stopping'].get('mode', train_config.get('best_metric_mode', 'min'))
        
        early_stopping = EarlyStopping(
            patience=train_config['early_stopping']['patience'],
            min_delta=train_config['early_stopping']['min_delta'],
            mode=early_stop_mode
        )
        print(f"✓ 早停: patience={early_stopping.patience}, mode={early_stopping.mode}, metric={early_stop_metric}")
    
    # ========== 创建训练器 ==========
    print("\n" + "=" * 80)
    print("创建训练器...")
    print("=" * 80)
    
    # 创建必要的子目录
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    tb_config = training_config.get('tensorboard', {})
    tb_log_dir = tb_config.get('log_dir', 'tensorboard')
    tensorboard_dir = output_dir / tb_log_dir if tb_config.get('enabled', True) else None
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        save_dir=str(output_dir / "checkpoints"),
        save_best=train_config.get('save_best', True),
        best_metric=train_config.get('best_metric', 'loss'),
        best_metric_mode=train_config.get('best_metric_mode', 'min'),
        early_stopping=early_stopping,
        mixed_precision=train_config.get('mixed_precision', False),
        val_interval=train_config.get('val_interval', 1),
        save_interval=train_config.get('save_interval', 10),
        log_file=str(logs_dir / "training_log.jsonl"),
        history_file=str(logs_dir / "training_history.json"),
        # TensorBoard参数
        tensorboard_enabled=tb_config.get('enabled', True),
        tensorboard_dir=str(tensorboard_dir) if tensorboard_dir else None,
        tb_log_interval=tb_config.get('log_interval', 50),
        tb_histogram_interval=tb_config.get('histogram_interval', 500),
        tb_log_histograms=tb_config.get('log_histograms', True)
    )
    
    print(f"✓ 训练器创建完成")
    if tensorboard_dir:
        print(f"✓ TensorBoard日志: {tensorboard_dir}")
    
    # ========== 开始训练 ==========
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    trainer.train(num_epochs=train_config['num_epochs'])
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"最佳模型已保存到: {output_dir}")
    
    # 打印最终结果
    print("\n最终结果:")
    print(f"  - 最佳epoch: {trainer.best_epoch}")
    print(f"  - 最佳{trainer.best_metric}: {trainer.best_val_metric:.6f}")


if __name__ == '__main__':
    main()
