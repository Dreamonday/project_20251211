#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch数据文件(.pt)转换为Excel文件工具
版本: v0.2
日期: 20251225170308

功能:
1. 读取预处理的.pt文件（train_v0.4.pt和val_v0.4.pt等）
2. 根据指定范围提取样本数据（如10000-10010表示第10000到10010个样本）
3. 将3D tensor数据转换为2D表格（展平时间步维度）
4. 输出为Excel文件（包含训练集和验证集两个sheet）

更新内容:
- v0.2: 修复metadata兼容性问题，支持v0.3的'feature_columns'和v0.4的'feature_columns_example'
- v0.2: 当metadata中找不到特征列名时，自动生成默认列名

使用方法:
    python pt_to_excel_v0.2_20251225170308.py \\
        --data_dir /path/to/preprocess_data_v0.4_20251225165509 \\
        --train_samples 10000-10010 \\
        --val_samples 5000-5010

输出:
    Excel文件保存在数据文件夹中，文件名格式：train_{start}-{end}_val_{start}-{end}.xlsx
"""

import torch
import pandas as pd
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ============================================================================
# 配置区域 - 请在此处修改参数
# ============================================================================

# 数据文件夹路径（必需）
# 示例: "/data/project_20251211/data/processed/preprocess_data_v0.5_20251224120208"
DATA_DIR = "/data/project_20251211/data/processed/preprocess_data_v0.4_20251225185320"

# 训练集样本范围（可选，格式: "start-end"，如 "10000-10010" 表示提取第10000到10010个样本）
# 设置为 None 表示不提取训练集
TRAIN_SAMPLES = "50000-50010"  # 示例：提取前11个样本（索引0到10）

# 验证集样本范围（可选，格式: "start-end"，如 "5000-5010" 表示提取第5000到5010个样本）
# 设置为 None 表示不提取验证集
VAL_SAMPLES = "10000-10010"  # 示例：提取前11个样本（索引0到10）

# 注意事项:
# 1. 训练集总样本数: 60093，有效索引范围: 0-60092
# 2. 验证集总样本数: 31675，有效索引范围: 0-31674
# 3. 至少需要设置 TRAIN_SAMPLES 或 VAL_SAMPLES 中的一个
# 4. 输出Excel文件将保存在 DATA_DIR 指定的文件夹中

# ============================================================================
# 脚本版本信息（请勿修改）
# ============================================================================
SCRIPT_VERSION = 'v0.2'
SCRIPT_TIMESTAMP = '20251225170308'


def parse_range(range_str: str) -> tuple:
    """
    解析范围字符串（如 '10000-10010'）
    
    Args:
        range_str: 范围字符串，格式为 'start-end'
    
    Returns:
        (start, end) 元组
    
    Raises:
        ValueError: 如果格式不正确
    """
    try:
        parts = range_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"范围格式错误: {range_str}，应为 'start-end' 格式")
        
        start = int(parts[0])
        end = int(parts[1])
        
        if start < 0 or end < 0:
            raise ValueError(f"范围索引不能为负数: {range_str}")
        
        if start > end:
            raise ValueError(f"起始索引不能大于结束索引: {range_str}")
        
        return start, end
    
    except ValueError as e:
        raise ValueError(f"解析范围字符串失败: {range_str}, 错误: {str(e)}")


def load_pt_file(pt_file: Path) -> dict:
    """
    加载.pt文件
    
    Args:
        pt_file: .pt文件路径
    
    Returns:
        包含X, y和metadata的字典
    """
    print(f"\n加载文件: {pt_file.name}")
    
    if not pt_file.exists():
        raise FileNotFoundError(f"文件不存在: {pt_file}")
    
    # 加载数据
    data = torch.load(pt_file, map_location='cpu', weights_only=False)
    
    # 检查数据结构
    if 'X' not in data or 'y' not in data or 'metadata' not in data:
        raise ValueError(f"文件格式错误: {pt_file}，缺少必要的键")
    
    X = data['X']
    metadata = data['metadata']
    
    print(f"  样本数: {X.shape[0]}")
    print(f"  序列长度: {X.shape[1]}")
    print(f"  特征数: {X.shape[2]}")
    print(f"  目标列: {metadata.get('target_column', 'unknown')}")
    
    return data


def extract_data(data: dict, start: int, end: int, data_type: str) -> tuple:
    """
    从数据中提取指定范围的样本
    
    Args:
        data: 包含X, y和metadata的字典
        start: 起始索引（包含）
        end: 结束索引（包含）
        data_type: 数据类型（'train'或'val'，用于错误提示）
    
    Returns:
        (X_extracted, y_extracted, feature_columns_list) 元组
        feature_columns_list: 每个样本对应的列名列表
    """
    X = data['X']
    y = data['y']
    metadata = data['metadata']
    
    total_samples = X.shape[0]
    
    # 检查边界
    if start >= total_samples:
        raise ValueError(
            f"{data_type}数据: 起始索引 {start} 超出范围，"
            f"总样本数为 {total_samples}（有效索引: 0-{total_samples-1}）"
        )
    
    if end >= total_samples:
        raise ValueError(
            f"{data_type}数据: 结束索引 {end} 超出范围，"
            f"总样本数为 {total_samples}（有效索引: 0-{total_samples-1}）"
        )
    
    # 提取数据（注意Python切片是左闭右开，所以需要end+1）
    X_extracted = X[start:end+1]
    y_extracted = y[start:end+1]
    actual_samples = X_extracted.shape[0]
    
    # 获取每个样本的特征列名（兼容不同版本的metadata结构）
    sample_feature_columns = metadata.get('sample_feature_columns', None)
    
    if sample_feature_columns is not None:
        # 新版本：使用每个样本的实际列名
        feature_columns_list = sample_feature_columns[start:end+1]
        print(f"\n{data_type}数据提取:")
        print(f"  范围: {start}-{end}")
        print(f"  提取样本数: {actual_samples}")
        print(f"  数据形状: X={X_extracted.shape}, y={y_extracted.shape}")
        print(f"  ✓ 使用每个样本的实际特征列名（支持不同市场的公司）")
    else:
        # 旧版本：使用统一的列名（兼容性）
        feature_columns = metadata.get('feature_columns', None)
        if feature_columns is None:
            feature_columns = metadata.get('feature_columns_example', None)
        
        # 如果都不存在，根据特征数量生成默认列名
        if feature_columns is None or len(feature_columns) == 0:
            num_features = X_extracted.shape[2]
            feature_columns = [f'feature_{i}' for i in range(num_features)]
            print(f"  警告: metadata中未找到特征列名，已自动生成 {num_features} 个默认列名")
        
        # 所有样本使用相同的列名
        feature_columns_list = [feature_columns] * actual_samples
        
        print(f"\n{data_type}数据提取:")
        print(f"  范围: {start}-{end}")
        print(f"  提取样本数: {actual_samples}")
        print(f"  数据形状: X={X_extracted.shape}, y={y_extracted.shape}")
        print(f"  ⚠ 使用统一的特征列名（旧版数据格式，可能存在列名不匹配）")
    
    return X_extracted, y_extracted, feature_columns_list


def tensor_to_dataframe(X: torch.Tensor, y: torch.Tensor, feature_columns_list: list) -> pd.DataFrame:
    """
    将3D tensor转换为2D DataFrame（垂直排列，每个样本展开为多行）
    
    Args:
        X: 输入tensor，形状为 [num_samples, seq_len, num_features]
        y: 目标tensor，形状为 [num_samples, 1]
        feature_columns_list: 每个样本的特征列名列表（列表的列表）
    
    Returns:
        DataFrame，每个样本占seq_len行，列名为特征名
        target列只在每个样本的第一行填充，其他行为空
    """
    num_samples = X.shape[0]
    seq_len = X.shape[1]
    num_features = X.shape[2]
    
    print("\n转换数据为DataFrame（垂直排列）:")
    print(f"  样本数: {num_samples}")
    print(f"  序列长度: {seq_len}")
    print(f"  特征数: {num_features}")
    print(f"  输出行数: {num_samples * seq_len} (每个样本{seq_len}行)")
    
    # 转换X为numpy数组
    print("\n重塑数据...")
    X_numpy = X.numpy()
    y_numpy = y.numpy().squeeze()
    
    # 为每个样本创建DataFrame，然后合并
    print("\n创建DataFrame（为每个样本使用对应的列名）...")
    all_sample_dfs = []
    
    for i in tqdm(range(num_samples), desc="处理样本", disable=num_samples < 100):
        # 获取当前样本的数据
        sample_data = X_numpy[i]  # 形状: [seq_len, num_features]
        
        # 获取当前样本的列名（保留完整列名）
        sample_columns = feature_columns_list[i]
        
        # 创建当前样本的DataFrame
        sample_df = pd.DataFrame(sample_data, columns=sample_columns)
        
        # 添加sample_id列
        sample_df.insert(0, 'sample_id', i)
        
        # 添加time_step列
        sample_df.insert(1, 'time_step', list(range(1, seq_len + 1)))
        
        # 添加target列：只在第一行填充
        target_col = [float(y_numpy[i])] + [np.nan] * (seq_len - 1)
        sample_df['target'] = target_col
        
        all_sample_dfs.append(sample_df)
    
    # 合并所有样本的DataFrame
    print("\n合并所有样本...")
    df = pd.concat(all_sample_dfs, ignore_index=True)
    
    print(f"\n  DataFrame形状: {df.shape}")
    print(f"  总行数: {df.shape[0]} (样本数{num_samples} × 序列长度{seq_len})")
    print(f"  总列数: {df.shape[1]} (sample_id + time_step + 特征列 + target)")
    print(f"  内存占用: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"  ✓ 每个样本使用其对应的实际列名")
    
    return df


def save_to_excel(
    train_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    output_file: Path = None
):
    """
    保存DataFrame到Excel文件
    
    Args:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        output_file: 输出文件路径
    """
    if train_df is None and val_df is None:
        raise ValueError("至少需要提供训练集或验证集数据")
    
    print(f"\n保存到Excel: {output_file}")
    
    # 使用ExcelWriter写入多个sheet
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if train_df is not None:
            print("  写入训练集sheet...")
            print(f"    形状: {train_df.shape}")
            train_df.to_excel(writer, sheet_name='train', index=False)
        
        if val_df is not None:
            print("  写入验证集sheet...")
            print(f"    形状: {val_df.shape}")
            val_df.to_excel(writer, sheet_name='val', index=False)
    
    # 获取文件大小
    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n文件已保存: {output_file}")
    print(f"文件大小: {file_size_mb:.2f} MB")


def extract_timestamp_from_path(data_dir: Path) -> str:
    """
    从数据文件夹路径中提取时间标志
    
    Args:
        data_dir: 数据文件夹路径
    
    Returns:
        时间标志字符串（如 '20251225183946'），提取失败返回 'unknown'
    """
    import re
    
    folder_name = data_dir.name  # 例如：preprocess_data_v0.4_20251225183946
    
    # 方法1：使用正则表达式提取最后的14位数字（时间戳格式）
    match = re.search(r'(\d{14})$', folder_name)
    if match:
        return match.group(1)
    
    # 方法2：按下划线分割，取最后一部分如果是数字
    parts = folder_name.split('_')
    last_part = parts[-1] if parts else ''
    if last_part.isdigit() and len(last_part) >= 12:  # 至少12位数字
        return last_part
    
    # 如果都提取不到，返回 'unknown'
    return "unknown"


def generate_output_filename(data_dir: Path, train_range: str = None, val_range: str = None) -> str:
    """
    生成输出文件名
    
    Args:
        data_dir: 数据文件夹路径
        train_range: 训练集范围（如 '10000-10010'）
        val_range: 验证集范围（如 '5000-5010'）
    
    Returns:
        文件名字符串（格式：时间标志_train_范围_val_范围.xlsx）
    """
    # 提取时间标志
    timestamp = extract_timestamp_from_path(data_dir)
    
    # 构建文件名部分
    parts = [timestamp]  # 时间标志放在最前面
    
    if train_range:
        parts.append(f"train_{train_range}")
    
    if val_range:
        parts.append(f"val_{val_range}")
    
    if len(parts) == 1:  # 只有时间标志，没有范围
        raise ValueError("至少需要提供训练集或验证集范围")
    
    filename = "_".join(parts) + ".xlsx"
    return filename


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将PyTorch .pt文件转换为Excel文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
    方法1: 直接运行（使用代码顶部配置区域的默认值）
        python %(prog)s
    
    方法2: 通过命令行参数覆盖默认值
        python %(prog)s \\
            --data_dir /path/to/data \\
            --train_samples 10000-10010 \\
            --val_samples 5000-5010

输出:
    Excel文件保存在数据文件夹中
    文件名格式: train_{start}-{end}_val_{start}-{end}.xlsx
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DATA_DIR,
        help=f'数据文件夹路径（默认: {DATA_DIR}）'
    )
    
    parser.add_argument(
        '--train_samples',
        type=str,
        default=TRAIN_SAMPLES,
        help=f'训练集样本范围，格式: start-end（默认: {TRAIN_SAMPLES}）'
    )
    
    parser.add_argument(
        '--val_samples',
        type=str,
        default=VAL_SAMPLES,
        help=f'验证集样本范围，格式: start-end（默认: {VAL_SAMPLES}）'
    )
    
    args = parser.parse_args()
    
    # 检查至少提供一个范围参数
    if args.train_samples is None and args.val_samples is None:
        print("错误: 至少需要在代码配置区域设置 TRAIN_SAMPLES 或 VAL_SAMPLES")
        print("      或通过命令行参数 --train_samples 或 --val_samples 指定")
        sys.exit(1)
    
    print("=" * 80)
    print(f"PyTorch数据转Excel工具 {SCRIPT_VERSION}")
    print(f"时间戳: {SCRIPT_TIMESTAMP}")
    print("=" * 80)
    
    # 显示配置信息
    print("\n配置信息:")
    print(f"  数据文件夹: {args.data_dir}")
    print(f"  训练集范围: {args.train_samples if args.train_samples else '不提取'}")
    print(f"  验证集范围: {args.val_samples if args.val_samples else '不提取'}")
    
    # 解析路径
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n错误: 数据文件夹不存在: {data_dir}")
        sys.exit(1)
    
    print(f"\n数据文件夹: {data_dir}")
    
    # 变量存储
    train_df = None
    val_df = None
    
    # 处理训练集
    if args.train_samples:
        print("\n" + "=" * 80)
        print("处理训练集")
        print("=" * 80)
        
        try:
            # 解析范围
            train_start, train_end = parse_range(args.train_samples)
            print(f"训练集范围: {train_start}-{train_end}")
            
            # 查找训练集文件
            train_files = list(data_dir.glob("train_*.pt"))
            if not train_files:
                raise FileNotFoundError(f"在 {data_dir} 中未找到训练集文件（train_*.pt）")
            
            train_file = train_files[0]
            print(f"训练集文件: {train_file.name}")
            
            # 加载数据
            train_data = load_pt_file(train_file)
            
            # 提取数据
            X_train, y_train, feature_columns_list = extract_data(
                train_data, train_start, train_end, 'train'
            )
            
            # 转换为DataFrame
            train_df = tensor_to_dataframe(X_train, y_train, feature_columns_list)
        
        except Exception as e:
            print(f"\n错误: 处理训练集时出错: {str(e)}")
            sys.exit(1)
    
    # 处理验证集
    if args.val_samples:
        print("\n" + "=" * 80)
        print("处理验证集")
        print("=" * 80)
        
        try:
            # 解析范围
            val_start, val_end = parse_range(args.val_samples)
            print(f"验证集范围: {val_start}-{val_end}")
            
            # 查找验证集文件
            val_files = list(data_dir.glob("val_*.pt"))
            if not val_files:
                raise FileNotFoundError(f"在 {data_dir} 中未找到验证集文件（val_*.pt）")
            
            val_file = val_files[0]
            print(f"验证集文件: {val_file.name}")
            
            # 加载数据
            val_data = load_pt_file(val_file)
            
            # 提取数据
            X_val, y_val, feature_columns_list = extract_data(
                val_data, val_start, val_end, 'val'
            )
            
            # 转换为DataFrame
            val_df = tensor_to_dataframe(X_val, y_val, feature_columns_list)
        
        except Exception as e:
            print(f"\n错误: 处理验证集时出错: {str(e)}")
            sys.exit(1)
    
    # 生成输出文件名
    print("\n" + "=" * 80)
    print("保存Excel文件")
    print("=" * 80)
    
    try:
        output_filename = generate_output_filename(
            data_dir=data_dir,
            train_range=args.train_samples,
            val_range=args.val_samples
        )
        output_file = data_dir / output_filename
        
        # 保存到Excel
        save_to_excel(train_df, val_df, output_file)
    
    except Exception as e:
        print(f"\n错误: 保存Excel文件时出错: {str(e)}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"\n输出文件: {output_file}")
    
    if train_df is not None:
        print(f"训练集: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
    
    if val_df is not None:
        print(f"验证集: {val_df.shape[0]} 行 × {val_df.shape[1]} 列")


if __name__ == '__main__':
    main()

