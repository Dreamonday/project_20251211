#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度调试单个样本
检查原始数据、归一化过程、最终tensor的每一步
"""

import torch
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置路径
TRAIN_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/train_v0.3.pt"
COMPANY_STATS_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/company_stats_v0.3.json"
TRAIN_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/train_samples_index.parquet"

# 检查第一个失败的样本: Bio-Techne, 样本索引 83810
SAMPLE_IDX = 83810


def main():
    """主函数"""
    print("="*80)
    print(f"深度调试样本 #{SAMPLE_IDX}")
    print("="*80)
    
    # 加载数据
    train_data = torch.load(TRAIN_PT_FILE)
    with open(COMPANY_STATS_FILE, 'r', encoding='utf-8') as f:
        company_stats = json.load(f)
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    
    # 获取样本信息
    sample_info = train_index.iloc[SAMPLE_IDX]
    source_file = sample_info['source_file']
    input_row_start = sample_info['input_row_start']
    input_row_end = sample_info['input_row_end']
    
    print(f"\n样本信息:")
    print(f"  公司: {sample_info['company_name']}")
    print(f"  文件: {Path(source_file).name}")
    print(f"  行范围: {input_row_start} ~ {input_row_end}")
    
    # 读取原始数据
    df_full = pd.read_parquet(source_file)
    df_input = df_full.iloc[input_row_start:input_row_end+1].copy()
    
    print(f"\n原始数据形状: {df_input.shape}")
    print(f"日期范围: {df_input['日期'].iloc[0]} ~ {df_input['日期'].iloc[-1]}")
    
    # 检查一个有问题的列
    problem_col = '固定资产：物业、厂房及设备（PP&E）；Property, Plant and Equipment (PP&E)；固定资产净值'
    
    if problem_col in df_input.columns:
        print(f"\n检查问题列: {problem_col}")
        col_data = df_input[problem_col]
        
        print(f"\n原始数据前10行:")
        for i in range(min(10, len(col_data))):
            val = col_data.iloc[i]
            print(f"  行 {i}: {val} (类型: {type(val).__name__}, NaN: {pd.isna(val)})")
        
        # 统计NaN
        nan_count = col_data.isna().sum()
        print(f"\nNaN统计: {nan_count}/{len(col_data)} ({nan_count/len(col_data)*100:.1f}%)")
        
        # 查看非NaN的值
        non_nan_values = col_data[~col_data.isna()]
        if len(non_nan_values) > 0:
            print(f"\n非NaN值统计:")
            print(f"  数量: {len(non_nan_values)}")
            print(f"  最小值: {non_nan_values.min()}")
            print(f"  最大值: {non_nan_values.max()}")
            print(f"  均值: {non_nan_values.mean()}")
            print(f"\n前5个非NaN值:")
            for i, (idx, val) in enumerate(non_nan_values.head().items()):
                print(f"  行 {idx}: {val}")
        
        # 填充NaN后的数据
        col_filled = col_data.fillna(0.0)
        print(f"\n填充NaN为0后的前10行:")
        for i in range(min(10, len(col_filled))):
            print(f"  行 {i}: {col_filled.iloc[i]}")
        
        # 获取公司统计量
        company_stats_key = str(source_file)
        if company_stats_key in company_stats:
            stats = company_stats[company_stats_key]
            if problem_col in stats:
                mean = stats[problem_col]['mean']
                std = stats[problem_col]['std']
                print(f"\n公司统计量:")
                print(f"  均值: {mean}")
                print(f"  标准差: {std}")
                
                # 手动归一化
                print(f"\n手动归一化前10行:")
                for i in range(min(10, len(col_filled))):
                    original = col_filled.iloc[i]
                    if std > 0:
                        normalized = (original - mean) / std
                    else:
                        normalized = 0.0
                    print(f"  行 {i}: 原始={original:.2f}, 归一化={normalized:.6f}")
            else:
                print(f"\n❌ 统计量中没有该列")
        else:
            print(f"\n❌ 统计量中没有该公司")
        
        # 从.pt文件获取对应的tensor值
        feature_columns = train_data['metadata']['feature_columns']
        if problem_col in feature_columns:
            col_idx = feature_columns.index(problem_col)
            X_tensor = train_data['X'][SAMPLE_IDX]
            pt_values = X_tensor[:, col_idx].numpy()
            
            print(f"\n.pt文件中的值前10行:")
            for i in range(min(10, len(pt_values))):
                print(f"  行 {i}: {pt_values[i]:.6f}")
        else:
            print(f"\n❌ 特征列中没有该列")
    
    # 检查整个窗口的数据情况
    print(f"\n" + "="*80)
    print("检查整个输入窗口的数据分布")
    print("="*80)
    
    feature_columns = train_data['metadata']['feature_columns']
    normalize_columns = train_data['metadata']['normalize_columns']
    
    # 统计每列的NaN情况
    print(f"\n各列NaN统计（输入窗口）:")
    nan_summary = []
    for col in feature_columns:
        if col in df_input.columns:
            nan_count = df_input[col].isna().sum()
            nan_pct = nan_count / len(df_input) * 100
            if nan_count > 0:
                nan_summary.append((col, nan_count, nan_pct))
    
    nan_summary.sort(key=lambda x: x[1], reverse=True)
    for col, count, pct in nan_summary[:20]:
        print(f"  {col[:60]:60s}: {count:3d}/{len(df_input)} ({pct:5.1f}%)")
    
    if len(nan_summary) > 20:
        print(f"  ... 还有 {len(nan_summary) - 20} 列有NaN值")
    
    # 检查整个公司的数据NaN情况
    print(f"\n" + "="*80)
    print("检查整个公司数据的NaN情况")
    print("="*80)
    print(f"公司全部数据行数: {len(df_full)}")
    
    for col in [problem_col]:
        if col in df_full.columns:
            print(f"\n列: {col}")
            col_full = df_full[col]
            nan_count = col_full.isna().sum()
            print(f"  总NaN数: {nan_count}/{len(col_full)} ({nan_count/len(col_full)*100:.1f}%)")
            
            # 查看哪些行有数据
            non_nan_indices = col_full[~col_full.isna()].index.tolist()
            if len(non_nan_indices) > 0:
                print(f"  有数据的行数: {len(non_nan_indices)}")
                print(f"  第一个有数据的行: {non_nan_indices[0]}")
                print(f"  最后一个有数据的行: {non_nan_indices[-1]}")
                print(f"  当前样本窗口: {input_row_start} ~ {input_row_end}")
                
                # 检查样本窗口是否在有数据的范围内
                window_has_data = any(input_row_start <= idx <= input_row_end for idx in non_nan_indices)
                print(f"  样本窗口是否有数据: {window_has_data}")
                
                if window_has_data:
                    window_non_nan = [idx for idx in non_nan_indices if input_row_start <= idx <= input_row_end]
                    print(f"  样本窗口内有数据的行: {len(window_non_nan)} 个")
                    print(f"  这些行的索引: {window_non_nan[:10]}")


if __name__ == '__main__':
    main()
