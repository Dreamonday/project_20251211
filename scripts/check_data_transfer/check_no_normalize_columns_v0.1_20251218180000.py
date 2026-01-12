#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查不归一化列的数据完整性
版本: v0.1
日期: 20251218180000

功能:
专门检查不参与归一化的列（开盘、收盘、最高、最低等）在数据转换过程中是否保持原值
"""

import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置路径
TRAIN_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/train_v0.3.pt"
VAL_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/val_v0.3.pt"
TRAIN_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/train_samples_index.parquet"
VAL_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/val_samples_index.parquet"

# 目标公司和数据
TARGET_COMPANY_FILE = "0296_赛力斯_601127_20251115_235412.parquet"
TARGET_TIMESTAMP = 1468368000000  # 对应的日期时间戳


def find_target_row_in_parquet(parquet_file: Path, timestamp: int):
    """在parquet文件中找到指定时间戳的行"""
    df = pd.read_parquet(parquet_file)
    
    # 尝试找到匹配的行
    # 假设日期列是datetime类型或者是timestamp
    if '日期' in df.columns:
        # 转换为时间戳（毫秒）
        if pd.api.types.is_datetime64_any_dtype(df['日期']):
            df['timestamp'] = (df['日期'].astype('int64') // 10**6)  # 转换为毫秒
        else:
            df['timestamp'] = df['日期']
        
        matching_rows = df[df['timestamp'] == timestamp]
        
        if len(matching_rows) > 0:
            return matching_rows.iloc[0], matching_rows.index[0]
        else:
            # 如果没找到精确匹配，显示可能的时间范围
            print(f"未找到精确匹配的时间戳 {timestamp}")
            print(f"文件中的时间戳范围:")
            print(f"  最小: {df['timestamp'].min()}")
            print(f"  最大: {df['timestamp'].max()}")
            print(f"\n前5行的时间戳:")
            for i in range(min(5, len(df))):
                print(f"  行{i}: {df['timestamp'].iloc[i]}")
            return None, None
    else:
        print("未找到'日期'列")
        return None, None


def check_specific_sample():
    """检查特定样本的不归一化列"""
    print("="*80)
    print("检查赛力斯公司的特定数据行")
    print("="*80)
    
    # 找到目标文件
    target_file = None
    for data_dir in [project_root / "data/raw/processed_data_20251212"]:
        possible_file = data_dir / TARGET_COMPANY_FILE
        if possible_file.exists():
            target_file = possible_file
            break
    
    if target_file is None:
        print(f"❌ 未找到文件: {TARGET_COMPANY_FILE}")
        return
    
    print(f"\n找到文件: {target_file}")
    
    # 读取完整数据
    df_full = pd.read_parquet(target_file)
    print(f"数据形状: {df_full.shape}")
    print(f"列: {df_full.columns.tolist()[:10]}...")
    
    # 查找目标行
    print(f"\n查找时间戳 {TARGET_TIMESTAMP} 对应的行...")
    target_row_data, target_row_idx = find_target_row_in_parquet(target_file, TARGET_TIMESTAMP)
    
    if target_row_data is None:
        print("未找到目标行，尝试手动查看数据...")
        
        # 手动查看前几行
        print("\n文件前5行的关键信息:")
        for i in range(min(5, len(df_full))):
            row = df_full.iloc[i]
            print(f"\n行 {i}:")
            print(f"  日期: {row['日期']}")
            if 'company_name' in row:
                print(f"  公司: {row['company_name']}")
            if 'stock_code' in row:
                print(f"  代码: {row['stock_code']}")
            if '开盘' in row:
                print(f"  开盘: {row['开盘']}")
                print(f"  收盘: {row['收盘']}")
                print(f"  最高: {row['最高']}")
                print(f"  最低: {row['最低']}")
        return
    
    print(f"\n✅ 找到目标行（索引: {target_row_idx}）")
    print(f"\n原始数据:")
    print(f"  日期: {target_row_data['日期']}")
    print(f"  公司: {target_row_data['company_name']}")
    print(f"  代码: {target_row_data['stock_code']}")
    print(f"  开盘: {target_row_data['开盘']}")
    print(f"  收盘: {target_row_data['收盘']}")
    print(f"  最高: {target_row_data['最高']}")
    print(f"  最低: {target_row_data['最低']}")
    if '振幅' in target_row_data:
        print(f"  振幅: {target_row_data['振幅']}")
    if '涨跌幅' in target_row_data:
        print(f"  涨跌幅: {target_row_data['涨跌幅']}")
    if '涨跌额' in target_row_data:
        print(f"  涨跌额: {target_row_data['涨跌额']}")
    if '换手率' in target_row_data:
        print(f"  换手率: {target_row_data['换手率']}")
    
    # 查找包含该行的样本
    print(f"\n" + "="*80)
    print("在索引文件中查找包含该行的样本")
    print("="*80)
    
    # 加载索引文件
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    val_index = pd.read_parquet(VAL_INDEX_FILE)
    
    # 查找包含目标文件的样本
    target_file_str = str(target_file)
    
    matching_train = train_index[train_index['source_file'] == target_file_str]
    matching_val = val_index[val_index['source_file'] == target_file_str]
    
    print(f"\n该公司在训练集的样本数: {len(matching_train)}")
    print(f"该公司在验证集的样本数: {len(matching_val)}")
    
    # 查找包含目标行的样本（作为输入窗口或目标）
    samples_containing_row = []
    
    for idx, sample in matching_train.iterrows():
        # 检查是否在输入窗口内
        if sample['input_row_start'] <= target_row_idx <= sample['input_row_end']:
            samples_containing_row.append(('train', idx, sample, 'input', 
                                          target_row_idx - sample['input_row_start']))
        # 检查是否是目标行
        if sample['target_row'] == target_row_idx:
            samples_containing_row.append(('train', idx, sample, 'target', None))
    
    for idx, sample in matching_val.iterrows():
        if sample['input_row_start'] <= target_row_idx <= sample['input_row_end']:
            samples_containing_row.append(('val', idx, sample, 'input',
                                          target_row_idx - sample['input_row_start']))
        if sample['target_row'] == target_row_idx:
            samples_containing_row.append(('val', idx, sample, 'target', None))
    
    print(f"\n找到 {len(samples_containing_row)} 个包含该行的样本")
    
    if len(samples_containing_row) == 0:
        print("❌ 该行不在任何样本的窗口内")
        return
    
    # 检查每个样本
    train_data = torch.load(TRAIN_PT_FILE)
    val_data = torch.load(VAL_PT_FILE)
    
    for dataset_name, sample_idx_in_df, sample_info, role, position_in_window in samples_containing_row[:5]:  # 最多检查5个
        print(f"\n" + "-"*80)
        print(f"样本: {sample_info['sample_id']}")
        print(f"数据集: {dataset_name}")
        print(f"角色: {role}")
        print(f"输入窗口: 行 {sample_info['input_row_start']} ~ {sample_info['input_row_end']}")
        print(f"目标行: {sample_info['target_row']}")
        
        if dataset_name == 'train':
            pt_data = train_data
            # 找到在.pt中的索引
            sample_idx = sample_idx_in_df
        else:
            pt_data = val_data
            sample_idx = sample_idx_in_df
        
        feature_columns = pt_data['metadata']['feature_columns']
        no_normalize_columns = pt_data['metadata']['no_normalize_columns']
        
        print(f"\n不归一化的列: {no_normalize_columns}")
        
        if role == 'input':
            print(f"\n该行在输入窗口的位置: 第 {position_in_window} 行")
            
            # 获取tensor数据
            X_tensor = pt_data['X'][sample_idx]  # shape: (seq_len, num_features)
            
            # 检查不归一化的列
            print(f"\n对比不归一化列的数值:")
            errors = []
            
            for col in no_normalize_columns:
                if col in feature_columns and col in target_row_data.index:
                    col_idx = feature_columns.index(col)
                    original_value = target_row_data[col]
                    pt_value = X_tensor[position_in_window, col_idx].item()
                    
                    diff = abs(original_value - pt_value)
                    
                    print(f"\n  {col}:")
                    print(f"    原始值: {original_value}")
                    print(f"    .pt值:  {pt_value}")
                    print(f"    差异:   {diff}")
                    
                    if diff > 1e-5:
                        rel_diff = diff / (abs(original_value) + 1e-10)
                        if rel_diff > 1e-4:
                            errors.append(f"{col}: 差异 {diff:.6e} (相对 {rel_diff:.6e})")
                            print(f"    ❌ 数值不匹配！")
                        else:
                            print(f"    ✅ 匹配（在浮点误差范围内）")
                    else:
                        print(f"    ✅ 完全匹配")
            
            if errors:
                print(f"\n❌ 发现 {len(errors)} 个不归一化列的数值错误:")
                for err in errors:
                    print(f"  - {err}")
            else:
                print(f"\n✅ 所有不归一化列的数值都正确")
        
        elif role == 'target':
            print(f"\n该行作为目标值")
            y_value = pt_data['y'][sample_idx].item()
            target_col = pt_data['metadata']['target_column']
            original_target = target_row_data[target_col]
            
            print(f"\n目标列: {target_col}")
            print(f"  原始值: {original_target}")
            print(f"  .pt值:  {y_value}")
            print(f"  差异:   {abs(original_target - y_value)}")
            
            if abs(original_target - y_value) < 1e-5:
                print(f"  ✅ 目标值匹配")
            else:
                print(f"  ❌ 目标值不匹配！")


def random_check_no_normalize_columns():
    """随机检查多个样本的不归一化列"""
    print("\n" + "="*80)
    print("随机检查不归一化列的数据完整性")
    print("="*80)
    
    import random
    random.seed(42)
    
    # 加载数据
    train_data = torch.load(TRAIN_PT_FILE)
    val_data = torch.load(VAL_PT_FILE)
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    val_index = pd.read_parquet(VAL_INDEX_FILE)
    
    feature_columns = train_data['metadata']['feature_columns']
    no_normalize_columns = train_data['metadata']['no_normalize_columns']
    
    print(f"\n不归一化的列: {no_normalize_columns}")
    print(f"总特征数: {len(feature_columns)}")
    
    # 随机选择10个样本
    num_samples = 10
    train_samples = random.sample(range(len(train_index)), min(5, len(train_index)))
    val_samples = random.sample(range(len(val_index)), min(5, len(val_index)))
    
    all_errors = []
    all_checks = 0
    
    for dataset_name, sample_indices, pt_data, index_df in [
        ('训练集', train_samples, train_data, train_index),
        ('验证集', val_samples, val_data, val_index)
    ]:
        print(f"\n检查 {dataset_name}...")
        
        for sample_idx in sample_indices:
            sample_info = index_df.iloc[sample_idx]
            source_file = sample_info['source_file']
            input_row_start = sample_info['input_row_start']
            input_row_end = sample_info['input_row_end']
            
            # 读取原始数据
            df_original = pd.read_parquet(source_file)
            df_input = df_original.iloc[input_row_start:input_row_end+1]
            
            # 获取tensor
            X_tensor = pt_data['X'][sample_idx]
            
            # 检查每个不归一化的列
            for col in no_normalize_columns:
                if col not in feature_columns or col not in df_input.columns:
                    continue
                
                col_idx = feature_columns.index(col)
                
                # 对比每一行
                for row_idx in range(len(df_input)):
                    original_value = df_input.iloc[row_idx][col]
                    pt_value = X_tensor[row_idx, col_idx].item()
                    
                    # 处理NaN
                    if pd.isna(original_value):
                        original_value = 0.0
                    
                    diff = abs(original_value - pt_value)
                    all_checks += 1
                    
                    if diff > 1e-5:
                        rel_diff = diff / (abs(original_value) + 1e-10)
                        if rel_diff > 1e-4:
                            all_errors.append({
                                'dataset': dataset_name,
                                'sample_id': sample_info['sample_id'],
                                'company': sample_info['company_name'],
                                'column': col,
                                'row_in_window': row_idx,
                                'original': original_value,
                                'pt_value': pt_value,
                                'diff': diff,
                                'rel_diff': rel_diff
                            })
    
    # 汇总结果
    print(f"\n" + "="*80)
    print("检查结果汇总")
    print("="*80)
    print(f"\n总检查数: {all_checks}")
    print(f"错误数: {len(all_errors)}")
    print(f"准确率: {(all_checks - len(all_errors)) / all_checks * 100:.2f}%")
    
    if len(all_errors) > 0:
        print(f"\n❌ 发现 {len(all_errors)} 个不归一化列的数值错误:")
        for i, err in enumerate(all_errors[:20]):  # 最多显示20个
            print(f"\n错误 #{i+1}:")
            print(f"  样本: {err['sample_id']} ({err['company']})")
            print(f"  列: {err['column']}")
            print(f"  窗口内位置: 第 {err['row_in_window']} 行")
            print(f"  原始值: {err['original']}")
            print(f"  .pt值: {err['pt_value']}")
            print(f"  差异: {err['diff']:.6e} (相对: {err['rel_diff']:.6e})")
        
        if len(all_errors) > 20:
            print(f"\n... 还有 {len(all_errors) - 20} 个错误未显示")
    else:
        print(f"\n✅ 所有不归一化列的数值都正确！")


def main():
    """主函数"""
    # 先检查特定样本
    check_specific_sample()
    
    # 再随机检查
    random_check_no_normalize_columns()


if __name__ == '__main__':
    main()
