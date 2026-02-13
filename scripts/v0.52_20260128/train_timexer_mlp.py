#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主训练脚本 - TimeXer-MLP v0.52（精细化LayerNorm控制）
版本: v0.52
日期: 20260128

使用TimeXer-MLP双分支模型，集成TensorBoard，支持精细化LayerNorm控制
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


# ===== 默认训练数据路径 =====
DEFAULT_PREPROCESSED_DIR = project_root / "data" / "processed" / "preprocess_data_v0.51_20260120092547"

# ===== 数据设备配置 =====
DATA_DEVICE = 'cpu'  # 'cpu' 或 'cuda'


def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入模块（使用v0.52模型，v0.2 trainer，v0.1数据和utils）
models_path = project_root / "src" / "models" / "v0.52_20260128"
data_path = project_root / "src" / "data" / "v0.1_20251212"
training_path = project_root / "src" / "training" / "v0.2_20251226"
utils_path = project_root / "src" / "utils" / "v0.1_20251212"

# 导入模型
timexer_mlp_module = _load_module(models_path / "timexer_mlp.py", "timexer_mlp")
TimeXerMLP = timexer_mlp_module.TimeXerMLP

# 导入数据集
stock_dataset_module = _load_module(data_path / "stock_dataset.py", "stock_dataset")
StockDataset = stock_dataset_module.StockDataset

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
        return MAPELoss(reduction=reduction, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_train_stats(train_dataset: StockDataset, dataset_config: dict) -> dict:
    """计算训练集特征统计量"""
    print("计算训练集特征统计量...")
    
    exclude_columns = dataset_config['features']['exclude_columns']
    sample_size = min(1000, len(train_dataset))
    indices = torch.randperm(len(train_dataset))[:sample_size].tolist()
    
    all_features = []
    for idx in indices:
        sample = train_dataset.index_df.iloc[idx]
        source_file = sample['source_file']
        df = load_parquet_file(source_file, use_cache=True)
        input_start = sample['input_row_start']
        input_end = sample['input_row_end']
        input_df = get_data_by_rows(df, input_start, input_end)
        feature_columns = get_feature_columns(input_df, exclude_columns)
        features = input_df[feature_columns]
        all_features.append(features)
    
    all_features_df = pd.concat(all_features, ignore_index=True)
    stats = compute_feature_stats(all_features_df, feature_columns, method="standard")
    
    print(f"特征统计量计算完成，共 {len(stats)} 个特征")
    return stats


def get_latest_code_modify_time(config_version: str) -> str:
    """获取代码最新修改时间"""
    files_to_check = []
    
    train_script = Path(__file__)
    files_to_check.append(train_script)
    
    models_path = project_root / "src" / "models" / "v0.52_20260128"
    files_to_check.append(models_path / "timexer_mlp.py")
    files_to_check.append(models_path / "timexer_mlp_blocks.py")
    
    data_path = project_root / "src" / "data" / "v0.1_20251212"
    files_to_check.append(data_path / "stock_dataset.py")
    files_to_check.append(data_path / "preprocessed_dataset.py")
    
    training_path = project_root / "src" / "training" / "v0.2_20251226"
    files_to_check.append(training_path / "trainer.py")
    
    utils_path = project_root / "src" / "utils" / "v0.1_20251212"
    files_to_check.append(utils_path / "feature_utils.py")
    files_to_check.append(utils_path / "data_utils.py")
    files_to_check.append(utils_path / "loss_utils.py")
    
    config_dir = project_root / "configs" / config_version
    config_files = [config_dir / "training_config.yaml"]
    for config_name in ["dataset_config.yaml", "timexer_mlp_config.yaml"]:
        config_file = config_dir / config_name
        if config_file.exists():
            config_files.append(config_file)
        else:
            old_config_file = project_root / "configs" / "v0.1_20251212" / config_name
            if old_config_file.exists():
                config_files.append(old_config_file)
    
    files_to_check.extend(config_files)
    
    latest_mtime = 0
    for file_path in files_to_check:
        if file_path.exists():
            mtime = os.path.getmtime(file_path)
            latest_mtime = max(latest_mtime, mtime)
    
    dt = datetime.fromtimestamp(latest_mtime)
    timestamp = dt.strftime('%Y%m%d%H%M%S')
    
    return timestamp


def find_preprocessed_files(preprocessed_dir: Path) -> tuple[Path, Path]:
    """自动检测预处理目录中的训练集和验证集文件"""
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
        print(f"警告: 找到多个训练集文件，使用最新的: {train_files[0].name}")
        train_files = sorted(train_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
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
        print(f"警告: 找到多个验证集文件，使用最新的: {val_files[0].name}")
        val_files = sorted(val_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    train_pt_file = train_files[0]
    val_pt_file = val_files[0]
    
    print(f"检测到训练集文件: {train_pt_file.name}")
    print(f"检测到验证集文件: {val_pt_file.name}")
    
    return train_pt_file, val_pt_file


def extract_version_number(config_version: str) -> str:
    """从config_version中提取版本号部分"""
    parts = config_version.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return config_version


def get_data_generation_time(preprocessed_dir: Path = None, dataset_config: dict = None) -> str:
    """获取数据生成时间戳及后缀"""
    if preprocessed_dir is not None:
        dir_name = preprocessed_dir.name
        match = re.search(r'preprocess_data_v\d+\.\d+_(.+)', dir_name)
        if match:
            return match.group(1)
        
        summary_files = list(preprocessed_dir.glob("preprocess_summary_*.json"))
        if summary_files:
            try:
                with open(summary_files[0], 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    created_at = summary.get('created_at', '')
                    if created_at:
                        dt = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                        return dt.strftime('%Y%m%d%H%M%S')
            except Exception:
                pass
    
    if dataset_config is not None:
        index_dir = dataset_config.get('index_dir', '')
        if index_dir:
            index_path = Path(index_dir)
            dir_name = index_path.name
            match = re.search(r'(\d{8})_(\d{6})(.*)', dir_name)
            if match:
                date_part = match.group(1)
                time_part = match.group(2)
                suffix = match.group(3)
                return f"{date_part}{time_part}{suffix}"
            
            match = re.search(r'(\d{14})(.*)', dir_name)
            if match:
                timestamp = match.group(1)
                suffix = match.group(2)
                return f"{timestamp}{suffix}"
    
    return datetime.now().strftime('%Y%m%d%H%M%S')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练TimeXer-MLP模型 v0.52（精细化LayerNorm控制）')
    parser.add_argument('--config_version', type=str, default='v0.52_20260128',
                        help='配置版本号')
    parser.add_argument('--model_config', type=str, default=None,
                        help='模型配置文件路径')
    parser.add_argument('--training_config', type=str, default=None,
                        help='训练配置文件路径')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='数据集配置文件路径')
    parser.add_argument('--use_preprocessed', action='store_true', default=True,
                        help='使用预处理数据')
    parser.add_argument('--no_preprocessed', action='store_true',
                        help='不使用预处理数据')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                        help='预处理数据目录路径')
    
    args = parser.parse_args()
    
    if args.no_preprocessed:
        args.use_preprocessed = False
    
    config_dir = project_root / "configs" / args.config_version
    fallback_config_dir = project_root / "configs" / "v0.1_20251212"
    
    # 模型配置
    if args.model_config:
        model_config_path = args.model_config
    elif (config_dir / "timexer_mlp_config.yaml").exists():
        model_config_path = str(config_dir / "timexer_mlp_config.yaml")
    else:
        model_config_path = str(fallback_config_dir / "tsmixer_config.yaml")
    
    # 训练配置
    if args.training_config:
        training_config_path = args.training_config
    elif (config_dir / "training_config.yaml").exists():
        training_config_path = str(config_dir / "training_config.yaml")
    else:
        training_config_path = str(fallback_config_dir / "training_config.yaml")
    
    # 数据集配置
    if args.dataset_config:
        dataset_config_path = args.dataset_config
    elif (config_dir / "dataset_config.yaml").exists():
        dataset_config_path = str(config_dir / "dataset_config.yaml")
    else:
        dataset_config_path = str(fallback_config_dir / "dataset_config.yaml")
    
    print("=" * 80)
    print("TimeXer-MLP v0.52 训练脚本（精细化LayerNorm控制）")
    print("=" * 80)
    print(f"模型配置: {model_config_path}")
    print(f"训练配置: {training_config_path}")
    print(f"数据集配置: {dataset_config_path}")
    print("=" * 80)
    
    # 加载配置
    print("\n加载配置文件...")
    model_config = load_config(model_config_path)
    training_config = load_config(training_config_path)
    dataset_config = load_config(dataset_config_path)
    
    model_cfg = model_config['model']
    train_cfg = training_config['training']
    dataset_cfg = dataset_config['dataset']
    
    tb_config = training_config.get('tensorboard', {})
    tb_enabled = tb_config.get('enabled', True)
    
    config_version = args.config_version
    print("\n获取代码最新修改时间...")
    code_timestamp = get_latest_code_modify_time(config_version)
    print(f"代码最新修改时间: {code_timestamp}")
    
    # 初始化数据集
    print("\n初始化数据集...")
    
    preprocessed_dir = DEFAULT_PREPROCESSED_DIR
    if args.use_preprocessed:
        print("使用预处理数据")
        
        if args.preprocessed_dir:
            preprocessed_dir = project_root / args.preprocessed_dir
            print(f"使用命令行指定的预处理目录: {preprocessed_dir}")
        else:
            print(f"使用默认预处理目录: {preprocessed_dir}")
        
        if not preprocessed_dir.exists():
            raise FileNotFoundError(
                f"预处理数据目录不存在: {preprocessed_dir}\n"
                f"请检查路径是否正确，或先运行预处理脚本生成数据"
            )
        
        print("\n获取数据生成时间...")
        data_timestamp = get_data_generation_time(preprocessed_dir=preprocessed_dir, dataset_config=None)
        print(f"数据生成时间: {data_timestamp}")
        
        print("\n自动检测预处理文件...")
        train_pt_file, val_pt_file = find_preprocessed_files(preprocessed_dir)
        
        dataset_device = None if DATA_DEVICE == 'cpu' else DATA_DEVICE
        train_dataset = PreprocessedStockDataset(
            pt_file_path=str(train_pt_file),
            load_to_memory=True,
            device=dataset_device
        )
        
        val_dataset = PreprocessedStockDataset(
            pt_file_path=str(val_pt_file),
            load_to_memory=True,
            device=dataset_device
        )
        
        feature_stats = train_dataset.get_feature_stats()
    
    else:
        print("使用原始数据")
        
        print("\n获取数据生成时间...")
        data_timestamp = get_data_generation_time(preprocessed_dir=None, dataset_config=dataset_cfg)
        print(f"数据生成时间: {data_timestamp}")
        
        index_dir = Path(dataset_cfg['index_dir'])
        train_index_file = index_dir / dataset_cfg['train_index_file']
        val_index_file = index_dir / dataset_cfg['val_index_file']
        
        exclude_columns = dataset_cfg['features']['exclude_columns']
        target_column = dataset_cfg['features']['target_column']
        normalize = dataset_cfg['features']['normalize']
        normalize_method = dataset_cfg['features']['normalize_method']
        cache_enabled = dataset_cfg['cache_enabled']
        cache_size = dataset_cfg['cache_size']
        
        train_dataset = StockDataset(
            index_file=str(train_index_file),
            exclude_columns=exclude_columns,
            target_column=target_column,
            normalize=False,
            normalize_method=normalize_method,
            feature_stats=None,
            stats_file=None,
            cache_enabled=cache_enabled,
            cache_size=cache_size
        )
        
        feature_stats = None
        stats_file = None
        if normalize:
            feature_stats = compute_train_stats(train_dataset, dataset_cfg)
            
            train_dataset = StockDataset(
                index_file=str(train_index_file),
                exclude_columns=exclude_columns,
                target_column=target_column,
                normalize=normalize,
                normalize_method=normalize_method,
                feature_stats=feature_stats,
                stats_file=None,
                cache_enabled=cache_enabled,
                cache_size=cache_size
            )
        
        val_dataset = StockDataset(
            index_file=str(val_index_file),
            exclude_columns=exclude_columns,
            target_column=target_column,
            normalize=normalize,
            normalize_method=normalize_method,
            feature_stats=feature_stats,
            stats_file=stats_file,
            cache_enabled=cache_enabled,
            cache_size=cache_size
        )
    
    # 创建输出目录
    version_number = extract_version_number(config_version)
    output_dir = project_root / "experiments" / f"timexer_mlp_{version_number}_{code_timestamp}_{data_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 保存特征统计量
    if feature_stats is not None:
        stats_file = output_dir / "feature_stats.json"
        save_feature_stats(feature_stats, stats_file)
        print(f"特征统计量已保存到: {stats_file}")
    
    # 获取实际特征数量和序列长度
    num_features = train_dataset.get_num_features()
    seq_len = train_dataset.get_seq_len()
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"特征数量: {num_features}")
    print(f"序列长度: {seq_len}")
    
    # 更新模型配置
    if model_cfg.get('n_features', 0) != num_features:
        print(f"\n警告: 配置中的特征数量 ({model_cfg.get('n_features', 0)}) 与实际特征数量 ({num_features}) 不同")
        print(f"更新模型配置中的特征数量为: {num_features}")
        model_cfg['n_features'] = num_features
    
    if model_cfg.get('seq_len', 0) != seq_len:
        print(f"\n警告: 配置中的序列长度 ({model_cfg.get('seq_len', 0)}) 与实际序列长度 ({seq_len}) 不同")
        print(f"更新模型配置中的序列长度为: {seq_len}")
        model_cfg['seq_len'] = seq_len
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    num_workers = 0 if DATA_DEVICE == 'cuda' else train_cfg['num_workers']
    pin_memory = False if DATA_DEVICE == 'cuda' else train_cfg['pin_memory']
    
    if DATA_DEVICE == 'cuda':
        print(f"  数据设备: GPU (num_workers=0, pin_memory=False)")
    else:
        print(f"  数据设备: CPU (num_workers={num_workers}, pin_memory={pin_memory})")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_cfg.get('prefetch_factor', 2) if num_workers > 0 else None,
        persistent_workers=train_cfg.get('persistent_workers', True) if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_cfg.get('prefetch_factor', 2) if num_workers > 0 else None,
        persistent_workers=train_cfg.get('persistent_workers', True) if num_workers > 0 else False
    )
    
    # 初始化模型
    print("\n初始化模型...")
    
    endogenous_indices = model_cfg.get('endogenous_indices', None)
    exogenous_indices = model_cfg.get('exogenous_indices', None)
    
    if endogenous_indices is not None and exogenous_indices is not None:
        print(f"使用索引列表方式分离特征:")
        print(f"  内生数据位置索引: {endogenous_indices[:5]}...{endogenous_indices[-5:]} (共{len(endogenous_indices)}个)")
        print(f"  宏观数据位置索引: {exogenous_indices[:5]}...{exogenous_indices[-5:]} (共{len(exogenous_indices)}个)")
    else:
        print(f"使用特征数量方式分离特征:")
        print(f"  内生特征数量: {model_cfg.get('endogenous_features', 44)}")
        print(f"  宏观特征数量: {model_cfg.get('exogenous_features', 20)}")
    
    # v0.52: 支持精细化LayerNorm配置
    use_layernorm_cfg = model_cfg.get('use_layernorm', True)
    
    model = TimeXerMLP(
        seq_len=model_cfg['seq_len'],
        n_features=model_cfg['n_features'],
        endogenous_features=model_cfg.get('endogenous_features', 44),
        exogenous_features=model_cfg.get('exogenous_features', 20),
        prediction_len=model_cfg.get('prediction_len', 1),
        endogenous_indices=endogenous_indices,
        exogenous_indices=exogenous_indices,
        endogenous_blocks=model_cfg.get('endogenous_blocks', 3),
        endogenous_hidden_dim=model_cfg.get('endogenous_hidden_dim', 256),
        exogenous_blocks=model_cfg.get('exogenous_blocks', 2),
        exogenous_hidden_dim=model_cfg.get('exogenous_hidden_dim', 256),
        shared_time_mixing=model_cfg.get('shared_time_mixing', True),
        mlp_fusion_ff_dim=model_cfg.get('mlp_fusion_ff_dim', 512),
        dropout=model_cfg.get('dropout', 0.1),
        activation=model_cfg.get('activation', 'gelu'),
        use_layernorm=use_layernorm_cfg,  # v0.52: 支持布尔或字典
        use_residual=model_cfg.get('use_residual', True),
        missing_value=model_cfg.get('missing_value', -1000.0),
        norm_type=model_cfg.get('norm_type', 'layer'),
        n_blocks=model_cfg.get('n_blocks', None),
        ff_dim=model_cfg.get('ff_dim', None),
        temporal_aggregation_config=model_cfg.get('temporal_aggregation', {}),
        output_projection_config=model_cfg.get('output_projection', {})
    )
    print(f"模型参数数量: {model.get_num_parameters():,}")
    
    # v0.52: 显示LayerNorm配置详情
    if isinstance(use_layernorm_cfg, bool):
        print(f"LayerNorm: {'全部启用' if use_layernorm_cfg else '全部禁用'}")
    elif isinstance(use_layernorm_cfg, dict):
        print(f"LayerNorm（精细化控制）:")
        print(f"  内生Mixer: {'启用' if use_layernorm_cfg.get('endogenous_mixer', True) else '禁用'}")
        print(f"  宏观Mixer: {'启用' if use_layernorm_cfg.get('exogenous_mixer', True) else '禁用'}")
        print(f"  内生输出: {'启用' if use_layernorm_cfg.get('endogenous_output', True) else '禁用'}")
        print(f"  宏观输出: {'启用' if use_layernorm_cfg.get('exogenous_output', True) else '禁用'}")
        print(f"  MLP融合: {'启用' if use_layernorm_cfg.get('mlp_fusion', True) else '禁用'}")
    
    print(f"残差连接: {'启用' if model.use_residual else '禁用'}")
    missing_val = model_cfg.get('missing_value', -1000.0)
    print(f"学习型Missing Embedding: 启用 (缺失值标记={missing_val})")
    
    # 创建优化器
    print("\n创建优化器...")
    optimizer = create_optimizer(model, train_cfg['optimizer'])
    
    # 创建学习率调度器
    print("创建学习率调度器...")
    scheduler = create_scheduler(optimizer, train_cfg['scheduler'], train_cfg['num_epochs'])
    
    # 创建损失函数
    print("创建损失函数...")
    criterion = create_criterion(train_cfg['loss'])
    
    # 创建早停机制
    early_stopping = None
    if train_cfg['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=train_cfg['early_stopping']['patience'],
            min_delta=train_cfg['early_stopping']['min_delta'],
            mode="min"
        )
    
    # 创建子目录
    configs_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard目录
    tensorboard_dir = None
    if tb_enabled:
        tb_log_dir = tb_config.get('log_dir', 'tensorboard')
        run_name = output_dir.name
        tensorboard_dir = output_dir / tb_log_dir / run_name
        print(f"\nTensorBoard已启用")
        print(f"  日志目录: {tensorboard_dir}")
        print(f"  运行名称: {run_name}")
        print(f"  记录间隔: 每 {tb_config.get('log_interval', 50)} batch")
        print(f"  直方图: {'启用' if tb_config.get('log_histograms', True) else '禁用'}")
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=train_cfg['device'],
        save_dir=str(output_dir / "checkpoints"),
        save_best=train_cfg['save_best'],
        early_stopping=early_stopping,
        mixed_precision=train_cfg['mixed_precision'],
        val_interval=train_cfg['val_interval'],
        save_interval=train_cfg['save_interval'],
        log_file=str(logs_dir / "training_log.jsonl"),
        history_file=str(logs_dir / "training_history.json"),
        tensorboard_enabled=tb_enabled,
        tensorboard_dir=str(tensorboard_dir) if tensorboard_dir else None,
        tb_log_interval=tb_config.get('log_interval', 50),
        tb_histogram_interval=tb_config.get('histogram_interval', 500),
        tb_log_histograms=tb_config.get('log_histograms', True)
    )
    
    # 保存配置
    print("\n保存配置...")
    
    model_info = model.get_model_info()
    
    detailed_model_config = {
        'metadata': model_config.get('metadata', {}),
        'model': {
            **model_config.get('model', {}),
            **model_info
        }
    }
    
    with open(configs_dir / "model_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(detailed_model_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    with open(configs_dir / "training_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(training_config, f, allow_unicode=True)
    with open(configs_dir / "dataset_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True)
    
    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    if tb_enabled:
        tb_parent_dir = output_dir / tb_config.get('log_dir', 'tensorboard')
        print(f"\n启动TensorBoard查看训练:")
        print(f"  单次运行: tensorboard --logdir={tensorboard_dir}")
        print(f"  对比多次: tensorboard --logdir={tb_parent_dir.parent.parent / 'tensorboard'}")
        print("=" * 80)
    
    history = trainer.train(train_cfg['num_epochs'])
    
    with open(logs_dir / "training_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练完成！所有结果已保存到: {output_dir}")
    if tb_enabled:
        tb_parent_dir = output_dir / tb_config.get('log_dir', 'tensorboard')
        print(f"\n查看TensorBoard:")
        print(f"  单次运行: tensorboard --logdir={tensorboard_dir}")
        print(f"  对比多次: tensorboard --logdir={tb_parent_dir.parent.parent / 'tensorboard'}")


if __name__ == '__main__':
    main()
