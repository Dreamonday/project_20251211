#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查索引文件指向的数据源
版本: v0.1
日期: 20260115
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# 索引文件路径
INDEX_FILE = '/data/project_20251211/data/processed/roll_generate_index_v0.7_20260111_154320/train_samples_index.parquet'

def main():
    print("=" * 80)
    print("检查索引文件指向的数据源")
    print("=" * 80)
    
    # 读取索引文件
    df = pd.read_parquet(INDEX_FILE)
    print(f"\n索引文件总样本数: {len(df)}")
    
    if 'source_file' not in df.columns:
        print("错误: 索引文件中没有source_file列")
        return
    
    # 统计source_file路径
    source_files = df['source_file'].tolist()
    
    # 统计指向processed_data_20251212和processed_data_20251220的数量
    count_20251212 = sum(1 for f in source_files if 'processed_data_20251212' in str(f))
    count_20251220 = sum(1 for f in source_files if 'processed_data_20251220' in str(f))
    
    print(f"\n指向 processed_data_20251212: {count_20251212} 个样本")
    print(f"指向 processed_data_20251220: {count_20251220} 个样本")
    
    # 显示前5个样本的source_file
    print("\n前5个样本的source_file路径:")
    for i in range(min(5, len(df))):
        print(f"  {i+1}. {df.iloc[i]['source_file']}")
    
    # 检查是否有其他路径
    unique_paths = set()
    for f in source_files:
        path_str = str(f)
        if 'processed_data_20251212' in path_str:
            unique_paths.add('processed_data_20251212')
        elif 'processed_data_20251220' in path_str:
            unique_paths.add('processed_data_20251220')
        else:
            unique_paths.add('其他路径')
    
    print(f"\n数据源类型: {unique_paths}")
    
    # 检查第一个样本的实际文件列数
    if len(df) > 0:
        first_source = df.iloc[0]['source_file']
        print(f"\n检查第一个样本的数据文件: {first_source}")
        try:
            first_df = pd.read_parquet(first_source)
            print(f"  文件列数: {len(first_df.columns)}")
            print(f"  是否包含指数列（沪深300_开盘等）: {'沪深300_开盘' in first_df.columns}")
            
            # 统计指数列数量
            index_cols = [col for col in first_df.columns if any(idx in col for idx in ['沪深300', '中证500', '恒生指数', '恒生科技', '标普500', '纳斯达克'])]
            print(f"  指数列数量: {len(index_cols)}")
            if index_cols:
                print(f"  前5个指数列: {index_cols[:5]}")
        except Exception as e:
            print(f"  错误: 无法读取文件 - {e}")

if __name__ == '__main__':
    main()
