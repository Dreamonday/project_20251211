#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查processed_data_20251220文件夹中所有公司数据的特征数量
版本: v0.1
日期: 20260115

功能:
检查指定文件夹中所有parquet文件是否都有64个特征
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

def check_feature_count(data_dir: str, expected_features: int = 68):
    """
    检查所有parquet文件的特征数量
    
    Args:
        data_dir: 数据目录路径
        expected_features: 期望的特征数量（默认64）
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"错误：目录不存在 - {data_path}")
        return
    
    # 获取所有parquet文件
    parquet_files = list(data_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"错误：目录中没有找到parquet文件 - {data_path}")
        return
    
    print("=" * 100)
    print(f"开始检查特征数量")
    print(f"数据目录: {data_path}")
    print(f"期望特征数: {expected_features}")
    print(f"文件总数: {len(parquet_files)}")
    print("=" * 100)
    print()
    
    # 统计数据
    correct_count = 0
    incorrect_count = 0
    incorrect_companies = []
    feature_columns = None
    
    # 遍历所有文件
    for file_path in tqdm(parquet_files, desc="检查文件"):
        try:
            # 读取parquet文件
            df = pd.read_parquet(file_path)
            
            # 排除日期时间列，获取特征列
            # 假设特征列是数值类型的列，排除日期、时间等列
            feature_cols = [col for col in df.columns if col not in ['date', 'Date', 'datetime', 'DateTime', 'time', 'Time', 'index', 'Index']]
            
            # 如果是第一个文件，保存特征列作为参考
            if feature_columns is None:
                feature_columns = set(feature_cols)
            
            # 检查特征数量
            if len(feature_cols) == expected_features:
                correct_count += 1
            else:
                incorrect_count += 1
                company_name = file_path.stem  # 文件名（不含扩展名）
                incorrect_companies.append({
                    'file': file_path.name,
                    'company': company_name,
                    'feature_count': len(feature_cols),
                    'missing_or_extra': expected_features - len(feature_cols),
                    'feature_cols': feature_cols
                })
        
        except Exception as e:
            print(f"\n错误：读取文件失败 - {file_path.name}")
            print(f"  错误信息: {str(e)}")
            incorrect_count += 1
    
    # 打印统计结果
    print("\n" + "=" * 100)
    print("检查完成！统计结果：")
    print("=" * 100)
    print(f"总文件数: {len(parquet_files)}")
    print(f"特征数正确的公司数: {correct_count} ({correct_count/len(parquet_files)*100:.2f}%)")
    print(f"特征数不正确的公司数: {incorrect_count} ({incorrect_count/len(parquet_files)*100:.2f}%)")
    print("=" * 100)
    
    # 如果有异常公司，打印详细信息
    if incorrect_companies:
        print("\n" + "=" * 100)
        print("异常公司详细信息：")
        print("=" * 100)
        for idx, company in enumerate(incorrect_companies, 1):
            print(f"\n[{idx}] 公司: {company['company']}")
            print(f"    文件: {company['file']}")
            print(f"    实际特征数: {company['feature_count']}")
            print(f"    差异: {company['missing_or_extra']} (负数表示多余，正数表示缺少)")
            
            # 如果特征列数量不多，可以打印所有特征列名
            if company['feature_count'] <= 100:
                print(f"    特征列: {', '.join(company['feature_cols'][:10])}...")
            
            # 如果有参考特征列，找出缺失或多余的列
            if feature_columns is not None:
                current_cols = set(company['feature_cols'])
                missing_cols = feature_columns - current_cols
                extra_cols = current_cols - feature_columns
                
                if missing_cols:
                    print(f"    缺失的列: {', '.join(list(missing_cols)[:10])}...")
                if extra_cols:
                    print(f"    多余的列: {', '.join(list(extra_cols)[:10])}...")
        print("=" * 100)
    else:
        print("\n✓ 所有公司的特征数量都正确！")
    
    print()


if __name__ == '__main__':
    # 默认数据目录
    default_data_dir = '/data/project_20251211/data/raw/processed_data_20251220'
    
    # 如果命令行提供了参数，使用命令行参数
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_data_dir
    
    # 如果命令行提供了期望特征数，使用命令行参数
    if len(sys.argv) > 2:
        expected_features = int(sys.argv[2])
    else:
        expected_features = 64
    
    # 运行检查
    check_feature_count(data_dir, expected_features)
