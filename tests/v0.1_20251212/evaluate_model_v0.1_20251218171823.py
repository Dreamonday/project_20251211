#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 评估训练集和验证集上每个公司最后一条记录的预测效果
版本: v0.1
日期: 20251218171823

功能:
1. 加载训练好的模型
2. 分别处理训练集和验证集
3. 找到每个公司在训练集和验证集中时间最靠后的记录
4. 进行预测并计算多种损失指标
5. 输出Excel表格

使用方法:
    python evaluate_model_v0.1_20251218171823.py \
        --model_dir /path/to/model/dir \
        --preprocessed_dir /path/to/preprocessed/data \
        --output_dir /path/to/output
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import importlib.util
from datetime import datetime
from tqdm import tqdm
import json
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 导入模型
models_path = project_root / "src" / "models" / "v0.1_20251212"
itransformer_module = _load_module(models_path / "itransformer_decoder.py", "itransformer_decoder")
iTransformerDecoder = itransformer_module.iTransformerDecoder

# 导入工具函数
utils_path = project_root / "src" / "utils" / "v0.1_20251212"
data_utils_module = _load_module(utils_path / "data_utils.py", "data_utils")
load_parquet_file = data_utils_module.load_parquet_file
get_data_by_single_row = data_utils_module.get_data_by_single_row


def load_model(checkpoint_path: Path, model_dir: Path, device: str = 'cuda'):
    """加载训练好的模型"""
    print(f"\n加载模型: {checkpoint_path}")
    
    # 加载模型配置
    config_path = model_dir / "configs" / "model_config.yaml"
    print(f"加载模型配置: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 创建模型
    model = iTransformerDecoder(
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
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型参数数量: {model.get_num_parameters():,}")
    print(f"模型加载完成，使用设备: {device}")
    
    return model


def load_preprocessed_data(preprocessed_dir: Path):
    """加载预处理的数据"""
    print(f"\n加载预处理数据: {preprocessed_dir}")
    
    # 自动查找训练集文件
    train_files = sorted(preprocessed_dir.glob("train_*.pt"))
    if not train_files:
        raise FileNotFoundError(f"在 {preprocessed_dir} 中未找到 train_*.pt 文件")
    
    # 自动查找验证集文件
    val_files = sorted(preprocessed_dir.glob("val_*.pt"))
    if not val_files:
        raise FileNotFoundError(f"在 {preprocessed_dir} 中未找到 val_*.pt 文件")
    
    # 如果有多个文件，选择最后一个（通常是最新版本）
    train_pt = train_files[-1]
    val_pt = val_files[-1]
    
    # 提示信息
    if len(train_files) > 1:
        print(f"找到 {len(train_files)} 个训练集文件，使用: {train_pt.name}")
    else:
        print(f"训练集: {train_pt.name}")
    
    if len(val_files) > 1:
        print(f"找到 {len(val_files)} 个验证集文件，使用: {val_pt.name}")
    else:
        print(f"验证集: {val_pt.name}")
    
    # 加载数据
    train_data = torch.load(train_pt, map_location='cpu', weights_only=False)
    val_data = torch.load(val_pt, map_location='cpu', weights_only=False)
    
    print(f"训练集样本数: {train_data['metadata']['num_samples']}")
    print(f"验证集样本数: {val_data['metadata']['num_samples']}")
    
    return train_data, val_data


def load_index_files(train_data_metadata: dict, val_data_metadata: dict):
    """加载索引文件"""
    train_index_path = Path(train_data_metadata['index_file'])
    val_index_path = Path(val_data_metadata['index_file'])
    
    print(f"\n加载索引文件:")
    print(f"训练集索引: {train_index_path}")
    print(f"验证集索引: {val_index_path}")
    
    train_index_df = pd.read_parquet(train_index_path)
    val_index_df = pd.read_parquet(val_index_path)
    
    print(f"训练集索引记录数: {len(train_index_df)}")
    print(f"验证集索引记录数: {len(val_index_df)}")
    
    return train_index_df, val_index_df


def find_last_samples_per_company(index_df: pd.DataFrame):
    """
    找到每个公司在数据集中时间最靠后的样本
    
    Args:
        index_df: 索引DataFrame，包含 sample_id, source_file, target_row 等列
        
    Returns:
        dict: {company_file: {sample_id, target_row, sample_idx}}
    """
    # 按公司分组，找每组中 target_row 最大的记录
    last_samples = {}
    
    for company_file, group in index_df.groupby('source_file'):
        # 找到 target_row 最大的记录
        max_idx = group['target_row'].idxmax()
        max_row = group.loc[max_idx]
        
        # 直接使用DataFrame的索引（max_idx就是在整个DataFrame中的位置）
        sample_idx = max_idx
        
        last_samples[company_file] = {
            'sample_id': max_row['sample_id'],
            'target_row': max_row['target_row'],
            'sample_idx': sample_idx,
            'input_row_start': max_row['input_row_start'],
            'input_row_end': max_row['input_row_end']
        }
    
    return last_samples


def extract_company_code(source_file: str) -> str:
    """从source_file路径中提取公司代码"""
    # 例如: /data/project_20251211/data/raw/processed_data_20251212/600000.parquet
    # 提取: 600000
    path = Path(source_file)
    return path.stem  # 文件名（不含扩展名）


def get_date_from_parquet(source_file: str, target_row: int) -> str:
    """
    从原始parquet文件中提取指定行的日期
    
    Args:
        source_file: parquet文件路径
        target_row: 目标行号
        
    Returns:
        日期字符串
    """
    try:
        df = load_parquet_file(source_file, use_cache=False)
        
        # 尝试不同的日期列名
        date_columns = ['日期', 'date', 'trade_date', '交易日期', 'Date']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            return "未知"
        
        # 获取日期
        date_value = df.iloc[target_row][date_col]
        
        # 转换为字符串
        if pd.isna(date_value):
            return "未知"
        
        # 如果是Timestamp，转换为字符串
        if isinstance(date_value, pd.Timestamp):
            return date_value.strftime('%Y-%m-%d')
        else:
            return str(date_value)
    
    except Exception as e:
        print(f"\n警告: 无法从 {source_file} 提取日期: {str(e)}")
        return "未知"


def compute_losses(y_true: float, y_pred: float, delta: float = 1.0) -> dict:
    """
    计算各种损失指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        delta: Huber损失的delta参数
        
    Returns:
        包含各种损失的字典
    """
    abs_error = abs(y_true - y_pred)
    relative_error = (abs_error / abs(y_true) * 100) if y_true != 0 else float('inf')
    
    mse = (y_true - y_pred) ** 2
    mae = abs_error
    rmse = np.sqrt(mse)
    mape = relative_error
    
    # Huber Loss
    if abs_error <= delta:
        huber = 0.5 * (abs_error ** 2)
    else:
        huber = delta * (abs_error - 0.5 * delta)
    
    return {
        '绝对误差': abs_error,
        '相对误差(%)': relative_error,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Huber': huber
    }


def evaluate_model(
    model: nn.Module,
    train_data: dict,
    val_data: dict,
    train_index_df: pd.DataFrame,
    val_index_df: pd.DataFrame,
    device: str = 'cuda'
):
    """
    评估模型在每个公司最后一条记录上的表现
    
    Returns:
        DataFrame: 包含所有评估结果
    """
    print("\n" + "=" * 80)
    print("开始评估模型")
    print("=" * 80)
    
    # 找到每个公司的最后一条样本
    print("\n查找每个公司的最后一条记录...")
    train_last_samples = find_last_samples_per_company(train_index_df)
    val_last_samples = find_last_samples_per_company(val_index_df)
    
    print(f"训练集中的公司数: {len(train_last_samples)}")
    print(f"验证集中的公司数: {len(val_last_samples)}")
    
    # 获取所有公司（训练集和验证集的并集）
    all_companies = set(train_last_samples.keys()) | set(val_last_samples.keys())
    print(f"总公司数: {len(all_companies)}")
    
    # 提取数据张量
    train_X = train_data['X']
    train_y = train_data['y']
    val_X = val_data['X']
    val_y = val_data['y']
    
    # 存储结果
    results = []
    
    print("\n开始逐公司评估...")
    
    with torch.no_grad():
        for company_file in tqdm(sorted(all_companies), desc="评估进度"):
            company_code = extract_company_code(company_file)
            result_row = {'公司代码': company_code}
            
            # 处理训练集
            if company_file in train_last_samples:
                sample_info = train_last_samples[company_file]
                sample_idx = sample_info['sample_idx']
                
                # 提取输入和真实值
                X = train_X[sample_idx].unsqueeze(0).to(device)  # [1, seq_len, features]
                y_true = train_y[sample_idx].item()
                
                # 预测
                y_pred = model(X).item()
                
                # 获取日期
                date_str = get_date_from_parquet(company_file, sample_info['target_row'])
                
                # 计算损失
                losses = compute_losses(y_true, y_pred)
                
                # 添加到结果
                result_row['训练_真实值'] = y_true
                result_row['训练_时间'] = date_str
                result_row['训练_预测值'] = y_pred
                result_row['训练_绝对误差'] = losses['绝对误差']
                result_row['训练_相对误差(%)'] = losses['相对误差(%)']
                result_row['训练_MSE'] = losses['MSE']
                result_row['训练_MAE'] = losses['MAE']
                result_row['训练_RMSE'] = losses['RMSE']
                result_row['训练_MAPE'] = losses['MAPE']
                result_row['训练_Huber'] = losses['Huber']
            else:
                # 如果训练集中没有该公司，填充空值
                for col in ['训练_真实值', '训练_时间', '训练_预测值', '训练_绝对误差', 
                           '训练_相对误差(%)', '训练_MSE', '训练_MAE', '训练_RMSE', 
                           '训练_MAPE', '训练_Huber']:
                    result_row[col] = None
            
            # 处理验证集
            if company_file in val_last_samples:
                sample_info = val_last_samples[company_file]
                sample_idx = sample_info['sample_idx']
                
                # 提取输入和真实值
                X = val_X[sample_idx].unsqueeze(0).to(device)
                y_true = val_y[sample_idx].item()
                
                # 预测
                y_pred = model(X).item()
                
                # 获取日期
                date_str = get_date_from_parquet(company_file, sample_info['target_row'])
                
                # 计算损失
                losses = compute_losses(y_true, y_pred)
                
                # 添加到结果
                result_row['验证_真实值'] = y_true
                result_row['验证_时间'] = date_str
                result_row['验证_预测值'] = y_pred
                result_row['验证_绝对误差'] = losses['绝对误差']
                result_row['验证_相对误差(%)'] = losses['相对误差(%)']
                result_row['验证_MSE'] = losses['MSE']
                result_row['验证_MAE'] = losses['MAE']
                result_row['验证_RMSE'] = losses['RMSE']
                result_row['验证_MAPE'] = losses['MAPE']
                result_row['验证_Huber'] = losses['Huber']
            else:
                # 如果验证集中没有该公司，填充空值
                for col in ['验证_真实值', '验证_时间', '验证_预测值', '验证_绝对误差',
                           '验证_相对误差(%)', '验证_MSE', '验证_MAE', '验证_RMSE',
                           '验证_MAPE', '验证_Huber']:
                    result_row[col] = None
            
            results.append(result_row)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 按公司代码排序
    results_df = results_df.sort_values('公司代码').reset_index(drop=True)
    
    print(f"\n评估完成！共评估 {len(results_df)} 个公司")
    
    return results_df


def save_results(results_df: pd.DataFrame, output_path: Path):
    """保存结果到Excel"""
    print(f"\n保存结果到: {output_path}")
    
    # 保存为Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='评估结果', index=False)
        
        # 自动调整列宽
        worksheet = writer.sheets['评估结果']
        for idx, col in enumerate(results_df.columns):
            max_length = max(
                results_df[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
    
    print(f"结果已保存！")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("评估统计")
    print("=" * 80)
    
    # 训练集统计
    train_valid = results_df['训练_真实值'].notna().sum()
    if train_valid > 0:
        print(f"\n训练集统计（{train_valid} 个公司）:")
        print(f"  平均绝对误差: {results_df['训练_绝对误差'].mean():.4f}")
        print(f"  平均相对误差: {results_df['训练_相对误差(%)'].mean():.2f}%")
        print(f"  RMSE: {results_df['训练_RMSE'].mean():.4f}")
        print(f"  MAPE: {results_df['训练_MAPE'].mean():.2f}%")
    
    # 验证集统计
    val_valid = results_df['验证_真实值'].notna().sum()
    if val_valid > 0:
        print(f"\n验证集统计（{val_valid} 个公司）:")
        print(f"  平均绝对误差: {results_df['验证_绝对误差'].mean():.4f}")
        print(f"  平均相对误差: {results_df['验证_相对误差(%)'].mean():.2f}%")
        print(f"  RMSE: {results_df['验证_RMSE'].mean():.4f}")
        print(f"  MAPE: {results_df['验证_MAPE'].mean():.2f}%")
    
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估模型在每个公司最后一条记录上的表现')
    parser.add_argument('--model_dir', type=str, 
                       default='/data/project_20251211/experiments/itransformer_v0.1_20251225181558_20251225173341',
                       help='模型目录路径')
    parser.add_argument('--preprocessed_dir', type=str,
                       default='/data/project_20251211/data/processed/preprocess_data_v0.5_20251225173341',
                       help='预处理数据目录路径')
    parser.add_argument('--output_dir', type=str,
                       default='/data/project_20251211/tests/val_data',
                       help='输出目录路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 路径处理
    model_dir = Path(args.model_dir)
    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    else:
        device = args.device
    
    print("=" * 80)
    print("模型评估脚本 v0.1")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"预处理数据目录: {preprocessed_dir}")
    print(f"输出目录: {output_dir}")
    print(f"计算设备: {device}")
    print("=" * 80)
    
    # 加载模型
    checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
    model = load_model(checkpoint_path, model_dir, device=device)
    
    # 加载预处理数据
    train_data, val_data = load_preprocessed_data(preprocessed_dir)
    
    # 加载索引文件
    train_index_df, val_index_df = load_index_files(
        train_data['metadata'],
        val_data['metadata']
    )
    
    # 评估模型
    results_df = evaluate_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        train_index_df=train_index_df,
        val_index_df=val_index_df,
        device=device
    )
    
    # 保存结果
    # Excel文件名使用模型目录名，并在最后添加存储时间戳
    model_dir_name = model_dir.name
    save_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_file = output_dir / f"{model_dir_name}_{save_timestamp}.xlsx"
    save_results(results_df, output_file)
    
    print(f"\n所有任务完成！输出文件: {output_file}")


if __name__ == '__main__':
    main()
