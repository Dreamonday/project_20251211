#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按公司分组的模型推理脚本
版本: v0.3
日期: 20260119145214

功能:
1. 支持多种模型类型（iTransformer、CrossFormer、TimeXer、TSMixer）
2. 支持TimeXer新版本：v0.41_20260116、v0.42_20260118、v0.43_20260119
3. 加载训练好的模型和对应的数据
4. 对所有训练集和验证集样本进行推理
5. 按公司分组，每个公司生成独立的Excel和Parquet文件
6. 计算绝对相对误差和相对误差（有正负值）
7. 计算每个公司训练集、验证集的统计指标（平均值和标准差）
8. 生成公司排名文件（按不同误差指标排名）

v0.3更新：
- 添加对TimeXer v0.41（细粒度LayerNorm控制）的支持
- 添加对TimeXer v0.42（完整mask支持）的支持
- 添加对TimeXer v0.43（学习型Missing Embedding）的支持
- 改进版本检测逻辑，从目录名精确识别版本号

使用方法:
    修改代码最前面的配置区域，设置MODEL_DIR和PREPROCESSED_DATA_DIR
    然后运行: python inference_by_company_v0.3_20260119145214.py
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util
from datetime import datetime
from tqdm import tqdm
import yaml
import json
import re

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ===== 模型和数据配置（修改此处） =====
MODEL_DIR = "/data/project_20251211/experiments/timexer_v0.43_20260120094726_20260120092547"
PREPROCESSED_DATA_DIR = "/data/project_20251211/data/processed/preprocess_data_v0.51_20260120092547"
DEVICE = 'cuda'  # 或 'cpu'
BATCH_SIZE = 256  # 推理批次大小


# 动态导入模块
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_model_type(model_dir: Path) -> tuple:
    """
    检测模型类型和版本
    
    Returns:
        (model_type, model_version): 模型类型和版本
    """
    config_path = model_dir / "configs" / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_name = config.get('model', {}).get('name', '').lower()
    dir_name = model_dir.name.lower()
    
    # 优先从目录名精确识别TimeXer版本
    if 'timexer' in model_name or 'timexer' in dir_name:
        # 尝试从目录名提取精确版本号
        # 匹配模式：v0.XX_YYYYMMDD
        version_pattern = r'v0\.\d+_\d{8}'
        match = re.search(version_pattern, dir_name)
        if match:
            version_str = match.group(0)
            # 提取具体版本号（使用前缀匹配，避免日期差异导致的识别失败）
            if version_str.startswith('v0.43_'):
                return 'timexer', 'v0.43_20260119'
            elif version_str.startswith('v0.42_'):
                return 'timexer', 'v0.42_20260118'
            elif version_str.startswith('v0.41_'):
                return 'timexer', 'v0.41_20260116'
            elif version_str.startswith('v0.5_'):
                return 'timexer_mlp', 'v0.5_20260107'
            elif version_str.startswith('v0.4_'):
                return 'timexer', 'v0.4_20260106'
        
        # 如果没有匹配到，使用默认版本
        if 'timexer_mlp' in model_name:
            return 'timexer_mlp', 'v0.5_20260107'
        else:
            return 'timexer', 'v0.4_20260106'
    
    # 从模型名称识别其他类型
    if 'itransformer' in model_name or 'itransformer_decoder' in model_name:
        return 'itransformer', 'v0.1_20251212'
    elif 'crossformer' in model_name:
        return 'crossformer', 'v0.3_20251230'
    elif 'tsmixer' in model_name:
        return 'tsmixer', 'v0.2_20251226'
    else:
        # 尝试从文件夹名称识别
        if 'itransformer' in dir_name:
            return 'itransformer', 'v0.1_20251212'
        elif 'crossformer' in dir_name:
            return 'crossformer', 'v0.3_20251230'
        elif 'tsmixer' in dir_name:
            return 'tsmixer', 'v0.2_20251226'
        else:
            raise ValueError(f"无法识别模型类型: {model_name} (目录: {dir_name})")


def load_model_dynamically(model_dir: Path, device: str = 'cuda'):
    """
    动态加载模型
    
    Args:
        model_dir: 模型目录路径
        device: 计算设备
        
    Returns:
        加载好的模型
    """
    print(f"\n加载模型: {model_dir}")
    
    # 检测模型类型
    model_type, model_version = detect_model_type(model_dir)
    print(f"检测到模型类型: {model_type}, 版本: {model_version}")
    
    # 加载模型配置
    config_path = model_dir / "configs" / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # 加载checkpoint
    checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 根据模型类型加载对应的模型类
    models_path = project_root / "src" / "models" / model_version
    
    if model_type == 'itransformer':
        itransformer_module = _load_module(models_path / "itransformer_decoder.py", "itransformer_decoder")
        ModelClass = itransformer_module.iTransformerDecoder
        
        model = ModelClass(
            input_features=model_config['input_features'],
            seq_len=model_config['seq_len'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            decoder_config=model_config.get('decoder', {}),
            input_resnet_config=model_config.get('input_resnet', {}),
            output_resnet_config=model_config.get('output_resnet', {}),
            final_output_config=model_config.get('final_output', {})
        )
    
    elif model_type == 'crossformer':
        crossformer_module = _load_module(models_path / "crossformer.py", "crossformer")
        ModelClass = crossformer_module.Crossformer
        
        model = ModelClass(
            seq_len=model_config['seq_len'],
            n_features=model_config['n_features'],
            d_model=model_config['d_model'],
            n_blocks=model_config['n_blocks'],
            n_heads=model_config['n_heads'],
            n_segments=model_config['n_segments'],
            n_feature_groups=model_config['n_feature_groups'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            prediction_len=model_config.get('prediction_len', 1),
            router_topk_ratio=model_config.get('router_topk_ratio', 0.5),
            positional_encoding=model_config.get('positional_encoding', {}),
            temporal_aggregation=model_config.get('temporal_aggregation', {}),
            output_projection=model_config.get('output_projection', {})
        )
    
    elif model_type == 'timexer_mlp':
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
    
    elif model_type == 'timexer':
        timexer_module = _load_module(models_path / "timexer.py", "timexer")
        ModelClass = timexer_module.TimeXer
        
        # 基础参数（所有版本通用）
        model_params = {
            'seq_len': model_config['seq_len'],
            'n_features': model_config['n_features'],
            'prediction_len': model_config.get('prediction_len', 1),
            'endogenous_features': model_config.get('endogenous_features', 44),
            'exogenous_features': model_config.get('exogenous_features', 20),
            'endogenous_indices': model_config.get('endogenous_indices'),
            'exogenous_indices': model_config.get('exogenous_indices'),
            'endogenous_blocks': model_config.get('endogenous_blocks', 3),
            'endogenous_hidden_dim': model_config.get('endogenous_hidden_dim', 256),
            'exogenous_blocks': model_config.get('exogenous_blocks', 2),
            'exogenous_hidden_dim': model_config.get('exogenous_hidden_dim', 256),
            'shared_time_mixing': model_config.get('shared_time_mixing', True),
            'time_mixing_type': model_config.get('time_mixing_type', 'attention'),
            'time_attn_n_heads': model_config.get('time_attn_n_heads', 8),
            'use_rope': model_config.get('use_rope', True),
            'cross_attn_n_heads': model_config.get('cross_attn_n_heads', 8),
            'cross_attn_ff_dim': model_config.get('cross_attn_ff_dim', 1024),
            'dropout': model_config.get('dropout', 0.1),
            'activation': model_config.get('activation', 'gelu'),
            'use_layernorm': model_config.get('use_layernorm', True),
            'use_residual': model_config.get('use_residual', True),
            'temporal_aggregation_config': model_config.get('temporal_aggregation', {}),
            'output_projection_config': model_config.get('output_projection', {})
        }
        
        # v0.41 及以上版本特有参数：细粒度LayerNorm控制
        if model_version in ['v0.41_20260116', 'v0.42_20260118', 'v0.43_20260119']:
            model_params['use_layernorm_in_tsmixer'] = model_config.get('use_layernorm_in_tsmixer')
            model_params['use_layernorm_in_attention'] = model_config.get('use_layernorm_in_attention')
            model_params['use_layernorm_before_pooling'] = model_config.get('use_layernorm_before_pooling')
        
        # 注意：v0.42的mask支持和v0.43的Missing Embedding都是内部自动处理，不需要外部参数
        
        model = ModelClass(**model_params)
    
    elif model_type == 'tsmixer':
        tsmixer_module = _load_module(models_path / "tsmixer.py", "tsmixer")
        ModelClass = tsmixer_module.TSMixer
        
        model = ModelClass(
            seq_len=model_config['seq_len'],
            n_features=model_config['n_features'],
            prediction_len=model_config.get('prediction_len', 1),
            n_blocks=model_config.get('n_blocks', 4),
            ff_dim=model_config.get('ff_dim', 2048),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'gelu'),
            norm_type=model_config.get('norm_type', 'layer'),
            use_residual=model_config.get('use_residual', True),
            temporal_aggregation_config=model_config.get('temporal_aggregation', {}),
            output_projection_config=model_config.get('output_projection', {})
        )
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型参数数量: {model.get_num_parameters():,}")
    print(f"模型加载完成，使用设备: {device}")
    
    return model


def load_preprocessed_data(preprocessed_dir: Path):
    """
    加载预处理的数据
    支持只有训练集或只有验证集的情况
    
    Returns:
        (train_data, val_data): 训练集和验证集数据，如果不存在则返回None
    """
    print(f"\n加载预处理数据: {preprocessed_dir}")
    
    train_data = None
    val_data = None
    
    # 自动查找训练集文件
    train_files = sorted(preprocessed_dir.glob("train_*.pt"))
    if train_files:
        # 如果有多个文件，选择最后一个（通常是最新版本）
        train_pt = train_files[-1]
        print(f"训练集: {train_pt.name}")
        train_data = torch.load(train_pt, map_location='cpu', weights_only=False)
        print(f"训练集样本数: {train_data['metadata']['num_samples']}")
    else:
        print("警告: 未找到训练集文件 (train_*.pt)，将跳过训练集推理")
    
    # 自动查找验证集文件
    val_files = sorted(preprocessed_dir.glob("val_*.pt"))
    if val_files:
        # 如果有多个文件，选择最后一个（通常是最新版本）
        val_pt = val_files[-1]
        print(f"验证集: {val_pt.name}")
        val_data = torch.load(val_pt, map_location='cpu', weights_only=False)
        print(f"验证集样本数: {val_data['metadata']['num_samples']}")
    else:
        print("警告: 未找到验证集文件 (val_*.pt)，将跳过验证集推理")
    
    # 检查是否至少有一个数据集
    if train_data is None and val_data is None:
        raise FileNotFoundError(
            f"在 {preprocessed_dir} 中未找到任何数据文件 (train_*.pt 或 val_*.pt)\n"
            f"请确保至少存在训练集或验证集文件"
        )
    
    return train_data, val_data


def load_index_files(train_data_metadata: dict = None, val_data_metadata: dict = None):
    """
    加载索引文件
    支持训练集或验证集索引缺失的情况
    
    Args:
        train_data_metadata: 训练集元数据，如果为None则跳过训练集索引加载
        val_data_metadata: 验证集元数据，如果为None则跳过验证集索引加载
    
    Returns:
        (train_index_df, val_index_df): 训练集和验证集索引DataFrame，如果不存在则返回None
    """
    print(f"\n加载索引文件:")
    
    train_index_df = None
    val_index_df = None
    
    if train_data_metadata is not None:
        train_index_path = Path(train_data_metadata['index_file'])
        print(f"训练集索引: {train_index_path}")
        if train_index_path.exists():
            train_index_df = pd.read_parquet(train_index_path)
            print(f"训练集索引记录数: {len(train_index_df)}")
        else:
            print(f"警告: 训练集索引文件不存在: {train_index_path}")
    
    if val_data_metadata is not None:
        val_index_path = Path(val_data_metadata['index_file'])
        print(f"验证集索引: {val_index_path}")
        if val_index_path.exists():
            val_index_df = pd.read_parquet(val_index_path)
            print(f"验证集索引记录数: {len(val_index_df)}")
        else:
            print(f"警告: 验证集索引文件不存在: {val_index_path}")
    
    return train_index_df, val_index_df


def inference_all_samples(
    model: nn.Module,
    train_data: dict = None,
    val_data: dict = None,
    train_index_df: pd.DataFrame = None,
    val_index_df: pd.DataFrame = None,
    device: str = 'cuda',
    batch_size: int = 64
):
    """
    对所有样本进行推理
    支持只有训练集或只有验证集的情况
    
    Args:
        model: 模型
        train_data: 训练集数据，如果为None则跳过训练集推理
        val_data: 验证集数据，如果为None则跳过验证集推理
        train_index_df: 训练集索引DataFrame，如果为None则跳过训练集推理
        val_index_df: 验证集索引DataFrame，如果为None则跳过验证集推理
        device: 计算设备
        batch_size: 批次大小
    
    Returns:
        results: 包含所有推理结果的列表
    """
    print("\n" + "=" * 80)
    print("开始推理")
    print("=" * 80)
    
    # 存储结果
    results = []
    
    model.eval()
    
    # 推理训练集（如果存在）
    if train_data is not None and train_index_df is not None:
        print(f"\n训练集样本数: {len(train_data['X'])}")
        
        # 提取数据张量
        train_X = train_data['X']
        train_y = train_data['y']
        
        # 推理训练集
        print("\n推理训练集...")
        train_predictions = []
        train_indices = []
        
        with torch.no_grad():
            # 批量推理
            for i in tqdm(range(0, len(train_X), batch_size), desc="训练集推理"):
                batch_X = train_X[i:i+batch_size].to(device)
                batch_pred = model(batch_X)
                train_predictions.extend(batch_pred.cpu().numpy().flatten().tolist())
                train_indices.extend(range(i, min(i+batch_size, len(train_X))))
        
        # 合并训练集结果
        print("\n合并训练集结果...")
        for idx, pred_value in zip(train_indices, train_predictions):
            if idx < len(train_index_df):
                sample_info = train_index_df.iloc[idx]
                true_value = train_y[idx].item()
                
                # 计算绝对相对误差
                abs_relative_error = abs(pred_value - true_value) / abs(true_value) * 100 if true_value != 0 else float('inf')
                
                # 计算相对误差（有正负值）
                relative_error_signed = (pred_value - true_value) / true_value * 100 if true_value != 0 else float('inf')
                
                # 处理日期字段（可能是datetime对象）
                target_date = sample_info.get('target_date', '')
                if pd.notna(target_date) and isinstance(target_date, pd.Timestamp):
                    target_date = target_date.strftime('%Y-%m-%d')
                elif pd.isna(target_date):
                    target_date = ''
                
                results.append({
                    'split': 'train',
                    'sample_id': sample_info['sample_id'],
                    'company_id': sample_info.get('company_id', ''),
                    'company_name': sample_info.get('company_name', ''),
                    'stock_code': sample_info.get('stock_code', ''),
                    'target_date': target_date,
                    'pred_value': pred_value,
                    'true_value': true_value,
                    'abs_relative_error': abs_relative_error,
                    'relative_error_signed': relative_error_signed
                })
    else:
        print("\n跳过训练集推理（数据或索引不存在）")
    
    # 推理验证集（如果存在）
    if val_data is not None and val_index_df is not None:
        print(f"\n验证集样本数: {len(val_data['X'])}")
        
        # 提取数据张量
        val_X = val_data['X']
        val_y = val_data['y']
        
        # 推理验证集
        print("\n推理验证集...")
        val_predictions = []
        val_indices = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(val_X), batch_size), desc="验证集推理"):
                batch_X = val_X[i:i+batch_size].to(device)
                batch_pred = model(batch_X)
                val_predictions.extend(batch_pred.cpu().numpy().flatten().tolist())
                val_indices.extend(range(i, min(i+batch_size, len(val_X))))
        
        # 合并验证集结果
        print("\n合并验证集结果...")
        for idx, pred_value in zip(val_indices, val_predictions):
            if idx < len(val_index_df):
                sample_info = val_index_df.iloc[idx]
                true_value = val_y[idx].item()
                
                # 计算绝对相对误差
                abs_relative_error = abs(pred_value - true_value) / abs(true_value) * 100 if true_value != 0 else float('inf')
                
                # 计算相对误差（有正负值）
                relative_error_signed = (pred_value - true_value) / true_value * 100 if true_value != 0 else float('inf')
                
                # 处理日期字段（可能是datetime对象）
                target_date = sample_info.get('target_date', '')
                if pd.notna(target_date) and isinstance(target_date, pd.Timestamp):
                    target_date = target_date.strftime('%Y-%m-%d')
                elif pd.isna(target_date):
                    target_date = ''
                
                results.append({
                    'split': 'val',
                    'sample_id': sample_info['sample_id'],
                    'company_id': sample_info.get('company_id', ''),
                    'company_name': sample_info.get('company_name', ''),
                    'stock_code': sample_info.get('stock_code', ''),
                    'target_date': target_date,
                    'pred_value': pred_value,
                    'true_value': true_value,
                    'abs_relative_error': abs_relative_error,
                    'relative_error_signed': relative_error_signed
                })
    else:
        print("\n跳过验证集推理（数据或索引不存在）")
    
    print(f"\n推理完成！共处理 {len(results)} 个样本")
    
    return results


def group_by_company_and_save(results: list, output_dir: Path):
    """
    按公司分组并保存结果
    
    Args:
        results: 推理结果列表
        output_dir: 输出目录
    """
    print("\n" + "=" * 80)
    print("按公司分组并保存结果")
    print("=" * 80)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 按公司分组
    company_groups = results_df.groupby(['company_id', 'company_name', 'stock_code'])
    
    print(f"\n共 {len(company_groups)} 个公司")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储所有公司的统计信息
    all_stats = []
    
    # 处理每个公司
    for (company_id, company_name, stock_code), group_df in tqdm(company_groups, desc="保存公司数据"):
        # 分离训练集和验证集
        train_df = group_df[group_df['split'] == 'train'].copy()
        val_df = group_df[group_df['split'] == 'val'].copy()
        
        # 准备输出DataFrame
        # 按日期排序（如果有日期）
        if 'target_date' in train_df.columns and len(train_df) > 0:
            train_df = train_df.sort_values('target_date').reset_index(drop=True)
        if 'target_date' in val_df.columns and len(val_df) > 0:
            val_df = val_df.sort_values('target_date').reset_index(drop=True)
        
        output_data = []
        
        # 处理训练集和验证集数据，对齐到同一行（如果可能）
        max_len = max(len(train_df), len(val_df))
        
        for i in range(max_len):
            row = {}
            
            # 训练集数据
            if i < len(train_df):
                train_row = train_df.iloc[i]
                row['训练_数据时间'] = train_row['target_date']
                row['训练_预测收盘价'] = train_row['pred_value']
                row['训练_真实收盘价'] = train_row['true_value']
                row['训练_绝对相对误差'] = train_row['abs_relative_error']
                row['训练_相对误差'] = train_row['relative_error_signed']
            else:
                row['训练_数据时间'] = None
                row['训练_预测收盘价'] = None
                row['训练_真实收盘价'] = None
                row['训练_绝对相对误差'] = None
                row['训练_相对误差'] = None
            
            # 验证集数据
            if i < len(val_df):
                val_row = val_df.iloc[i]
                row['验证_数据时间'] = val_row['target_date']
                row['验证_预测收盘价'] = val_row['pred_value']
                row['验证_真实收盘价'] = val_row['true_value']
                row['验证_绝对相对误差'] = val_row['abs_relative_error']
                row['验证_相对误差'] = val_row['relative_error_signed']
            else:
                row['验证_数据时间'] = None
                row['验证_预测收盘价'] = None
                row['验证_真实收盘价'] = None
                row['验证_绝对相对误差'] = None
                row['验证_相对误差'] = None
            
            output_data.append(row)
        
        # 创建DataFrame
        output_df = pd.DataFrame(output_data)
        
        # 计算统计指标
        stats_row = {
            'company_id': company_id,
            'company_name': company_name,
            'stock_code': stock_code
        }
        
        # 训练集统计
        if len(train_df) > 0:
            # 过滤掉inf和NaN值
            train_abs_rel_errors = train_df['abs_relative_error'].replace([float('inf'), -float('inf')], np.nan)
            train_rel_errors = train_df['relative_error_signed'].replace([float('inf'), -float('inf')], np.nan)
            
            train_abs_rel_errors_clean = train_abs_rel_errors.dropna()
            train_rel_errors_clean = train_rel_errors.dropna()
            
            if len(train_abs_rel_errors_clean) > 0:
                stats_row['训练_绝对相对误差_平均值'] = train_abs_rel_errors_clean.mean()
                stats_row['训练_绝对相对误差_标准差'] = train_abs_rel_errors_clean.std()
            else:
                stats_row['训练_绝对相对误差_平均值'] = np.nan
                stats_row['训练_绝对相对误差_标准差'] = np.nan
            
            if len(train_rel_errors_clean) > 0:
                stats_row['训练_相对误差_平均值'] = train_rel_errors_clean.mean()
                stats_row['训练_相对误差_标准差'] = train_rel_errors_clean.std()
            else:
                stats_row['训练_相对误差_平均值'] = np.nan
                stats_row['训练_相对误差_标准差'] = np.nan
        else:
            stats_row['训练_绝对相对误差_平均值'] = np.nan
            stats_row['训练_绝对相对误差_标准差'] = np.nan
            stats_row['训练_相对误差_平均值'] = np.nan
            stats_row['训练_相对误差_标准差'] = np.nan
        
        # 验证集统计
        if len(val_df) > 0:
            # 过滤掉inf和NaN值
            val_abs_rel_errors = val_df['abs_relative_error'].replace([float('inf'), -float('inf')], np.nan)
            val_rel_errors = val_df['relative_error_signed'].replace([float('inf'), -float('inf')], np.nan)
            
            val_abs_rel_errors_clean = val_abs_rel_errors.dropna()
            val_rel_errors_clean = val_rel_errors.dropna()
            
            if len(val_abs_rel_errors_clean) > 0:
                stats_row['验证_绝对相对误差_平均值'] = val_abs_rel_errors_clean.mean()
                stats_row['验证_绝对相对误差_标准差'] = val_abs_rel_errors_clean.std()
            else:
                stats_row['验证_绝对相对误差_平均值'] = np.nan
                stats_row['验证_绝对相对误差_标准差'] = np.nan
            
            if len(val_rel_errors_clean) > 0:
                stats_row['验证_相对误差_平均值'] = val_rel_errors_clean.mean()
                stats_row['验证_相对误差_标准差'] = val_rel_errors_clean.std()
            else:
                stats_row['验证_相对误差_平均值'] = np.nan
                stats_row['验证_相对误差_标准差'] = np.nan
        else:
            stats_row['验证_绝对相对误差_平均值'] = np.nan
            stats_row['验证_绝对相对误差_标准差'] = np.nan
            stats_row['验证_相对误差_平均值'] = np.nan
            stats_row['验证_相对误差_标准差'] = np.nan
        
        all_stats.append(stats_row)
        
        # 文件命名：{company_id}_{company_name}_{stock_code}
        # 清理文件名中的特殊字符
        safe_company_name = re.sub(r'[<>:"/\\|?*]', '_', str(company_name))
        safe_stock_code = re.sub(r'[<>:"/\\|?*]', '_', str(stock_code))
        filename_base = f"{company_id}_{safe_company_name}_{safe_stock_code}"
        
        # 创建统计DataFrame
        stats_df = pd.DataFrame([stats_row])
        
        # 保存Excel
        excel_path = output_dir / f"{filename_base}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 第一个sheet：推理结果
            output_df.to_excel(writer, sheet_name='推理结果', index=False)
            
            # 第二个sheet：统计汇总
            stats_df.to_excel(writer, sheet_name='统计汇总', index=False)
            
            # 自动调整列宽
            for sheet_name in ['推理结果', '统计汇总']:
                worksheet = writer.sheets[sheet_name]
                df_to_adjust = output_df if sheet_name == '推理结果' else stats_df
                for idx, col in enumerate(df_to_adjust.columns):
                    max_length = max(
                        df_to_adjust[col].astype(str).map(len).max() if len(df_to_adjust) > 0 else len(col),
                        len(col)
                    ) + 2
                    col_letter = chr(65 + idx) if idx < 26 else chr(65 + idx // 26 - 1) + chr(65 + idx % 26)
                    worksheet.column_dimensions[col_letter].width = min(max_length, 50)
        
        # 将统计信息追加到parquet文件
        # 先保存原始数据
        parquet_path = output_dir / f"{filename_base}.parquet"
        
        # 使用字典直接构造stats_row_df，只包含有实际值的列
        # 这样可以避免创建大量NA列，让concat自动处理列对齐
        stats_dict = {
            '统计标记': '统计汇总',
            'company_id': company_id,
            'company_name': company_name,
            'stock_code': stock_code
        }
        
        # 添加统计列（排除已经添加的基本信息列）
        stats_dict.update({k: v for k, v in stats_row.items() 
                          if k not in ['company_id', 'company_name', 'stock_code']})
        
        # 创建统计行DataFrame
        stats_row_df = pd.DataFrame([stats_dict])
        
        # 合并原始数据和统计信息，让pandas自动处理列对齐和缺失值
        output_df_with_stats = pd.concat([output_df, stats_row_df], ignore_index=True, sort=False)
        
        # 所有列保持pd.NA，在Excel和Parquet中正常显示为空
        output_df_with_stats.to_parquet(parquet_path, index=False)
    
    print(f"\n所有文件已保存到: {output_dir}")
    
    # 生成排名文件
    generate_ranking_files(all_stats, output_dir)


def generate_ranking_files(all_stats: list, output_dir: Path):
    """
    生成公司排名文件
    
    Args:
        all_stats: 所有公司的统计信息列表
        output_dir: 输出目录
    """
    print("\n" + "=" * 80)
    print("生成公司排名文件")
    print("=" * 80)
    
    # 转换为DataFrame
    stats_df = pd.DataFrame(all_stats)
    
    # 定义排名指标配置
    ranking_configs = [
        {
            'name': '训练集_绝对相对误差平均值',
            'value_col': '训练_绝对相对误差_平均值',
            'sheet_name': '训练集_绝对相对误差平均值'
        },
        {
            'name': '训练集_相对误差平均值',
            'value_col': '训练_相对误差_平均值',
            'sheet_name': '训练集_相对误差平均值'
        },
        {
            'name': '训练集_绝对相对误差+2标准差',
            'value_col': '训练_绝对相对误差_平均值',
            'std_col': '训练_绝对相对误差_标准差',
            'sheet_name': '训练集_绝对相对误差+2标准差',
            'calc_func': lambda row: row['训练_绝对相对误差_平均值'] + 2 * row['训练_绝对相对误差_标准差'] if pd.notna(row['训练_绝对相对误差_平均值']) and pd.notna(row['训练_绝对相对误差_标准差']) else np.nan
        },
        {
            'name': '训练集_相对误差绝对值+2标准差',
            'value_col': '训练_相对误差_平均值',
            'std_col': '训练_相对误差_标准差',
            'sheet_name': '训练集_相对误差绝对值+2标准差',
            'calc_func': lambda row: abs(row['训练_相对误差_平均值']) + 2 * row['训练_相对误差_标准差'] if pd.notna(row['训练_相对误差_平均值']) and pd.notna(row['训练_相对误差_标准差']) else np.nan
        },
        {
            'name': '验证集_绝对相对误差平均值',
            'value_col': '验证_绝对相对误差_平均值',
            'sheet_name': '验证集_绝对相对误差平均值'
        },
        {
            'name': '验证集_相对误差平均值',
            'value_col': '验证_相对误差_平均值',
            'sheet_name': '验证集_相对误差平均值'
        },
        {
            'name': '验证集_绝对相对误差+2标准差',
            'value_col': '验证_绝对相对误差_平均值',
            'std_col': '验证_绝对相对误差_标准差',
            'sheet_name': '验证集_绝对相对误差+2标准差',
            'calc_func': lambda row: row['验证_绝对相对误差_平均值'] + 2 * row['验证_绝对相对误差_标准差'] if pd.notna(row['验证_绝对相对误差_平均值']) and pd.notna(row['验证_绝对相对误差_标准差']) else np.nan
        },
        {
            'name': '验证集_相对误差绝对值+2标准差',
            'value_col': '验证_相对误差_平均值',
            'std_col': '验证_相对误差_标准差',
            'sheet_name': '验证集_相对误差绝对值+2标准差',
            'calc_func': lambda row: abs(row['验证_相对误差_平均值']) + 2 * row['验证_相对误差_标准差'] if pd.notna(row['验证_相对误差_平均值']) and pd.notna(row['验证_相对误差_标准差']) else np.nan
        }
    ]
    
    # 准备Excel和Parquet数据
    excel_sheets = {}
    parquet_data = []
    
    for config in ranking_configs:
        # 计算指标值
        if 'calc_func' in config:
            # 需要计算的指标
            ranking_df = stats_df.copy()
            ranking_df['指标值'] = ranking_df.apply(config['calc_func'], axis=1)
        else:
            # 直接使用平均值
            ranking_df = stats_df.copy()
            ranking_df['指标值'] = ranking_df[config['value_col']]
        
        # 添加排名方式列
        ranking_df['排名方式'] = config['name']
        
        # 选择需要的列
        result_df = ranking_df[['排名方式', 'company_id', 'company_name', 'stock_code', '指标值']].copy()
        
        # 排序：NaN值排在最后
        result_df['排序键'] = result_df['指标值'].fillna(float('inf'))
        result_df = result_df.sort_values('排序键', ascending=True).reset_index(drop=True)
        result_df = result_df.drop('排序键', axis=1)
        
        # 添加排名（NaN值排名为NaN）
        result_df['排名'] = result_df['指标值'].notna().cumsum()
        result_df.loc[result_df['指标值'].isna(), '排名'] = np.nan
        
        # 重新排列列顺序
        result_df = result_df[['排名方式', '排名', 'company_id', 'company_name', 'stock_code', '指标值']]
        
        # 保存到Excel sheet
        excel_sheets[config['sheet_name']] = result_df[['排名', 'company_id', 'company_name', 'stock_code', '指标值']]
        
        # 添加到Parquet数据
        parquet_data.append(result_df)
    
    # 保存Excel文件
    excel_path = output_dir / "不同误差下的公司排名.xlsx"
    print(f"\n保存Excel文件: {excel_path}")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for sheet_name, df in excel_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 自动调整列宽
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).map(len).max() if len(df) > 0 else len(col),
                    len(col)
                ) + 2
                col_letter = chr(65 + idx) if idx < 26 else chr(65 + idx // 26 - 1) + chr(65 + idx % 26)
                worksheet.column_dimensions[col_letter].width = min(max_length, 50)
    
    # 保存Parquet文件
    parquet_path = output_dir / "不同误差下的公司排名.parquet"
    print(f"保存Parquet文件: {parquet_path}")
    all_ranking_df = pd.concat(parquet_data, ignore_index=True)
    all_ranking_df.to_parquet(parquet_path, index=False)
    
    print(f"\n排名文件已保存到: {output_dir}")


def main():
    """主函数"""
    # 路径处理
    model_dir = Path(MODEL_DIR)
    preprocessed_dir = Path(PREPROCESSED_DATA_DIR)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"预处理数据目录不存在: {preprocessed_dir}")
    
    # 检查设备
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    else:
        device = DEVICE
    
    # 确定输出目录（使用模型文件夹名称 + 运行时间）
    model_dir_name = model_dir.name
    # 获取当前运行时间，格式为YYYYMMDDHHMMSS
    run_time = datetime.now().strftime('%Y%m%d%H%M%S')
    # 在模型文件夹名称后追加运行时间
    output_dir_name = f"{model_dir_name}_{run_time}"
    output_base_dir = project_root / "tests" / "inference_results"
    output_dir = output_base_dir / output_dir_name
    
    print("=" * 80)
    print("按公司分组的模型推理脚本 v0.3")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"预处理数据目录: {preprocessed_dir}")
    print(f"输出目录: {output_dir}")
    print(f"计算设备: {device}")
    print("=" * 80)
    
    # 加载模型
    model = load_model_dynamically(model_dir, device=device)
    
    # 加载预处理数据
    train_data, val_data = load_preprocessed_data(preprocessed_dir)
    
    # 加载索引文件
    train_index_df, val_index_df = load_index_files(
        train_data['metadata'] if train_data is not None else None,
        val_data['metadata'] if val_data is not None else None
    )
    
    # 推理所有样本
    results = inference_all_samples(
        model=model,
        train_data=train_data,
        val_data=val_data,
        train_index_df=train_index_df,
        val_index_df=val_index_df,
        device=device,
        batch_size=BATCH_SIZE
    )
    
    # 按公司分组并保存
    group_by_company_and_save(results, output_dir)
    
    print(f"\n所有任务完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
