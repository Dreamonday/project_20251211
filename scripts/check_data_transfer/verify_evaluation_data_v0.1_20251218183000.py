#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证评估数据的准确性
版本: v0.1
日期: 20251218183000

功能:
检查评估脚本生成的Excel中的真实值是否和原始parquet文件中的收盘价一致
"""

import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置路径
TRAIN_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/train_v0.3.pt"
VAL_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/val_v0.3.pt"
TRAIN_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/train_samples_index.parquet"
VAL_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/val_samples_index.parquet"
EVAL_EXCEL_FILE = project_root / "tests/val_data/itransformer_v0.1_20251212_20251218151534_20251218150457.xlsx"


def find_last_samples_per_company_WRONG(index_df: pd.DataFrame):
    """
    评估脚本中的错误实现（用于对比）
    """
    last_samples = {}
    
    for company_file, group in index_df.groupby('source_file'):
        max_idx = group['target_row'].idxmax()
        max_row = group.loc[max_idx]
        
        # 错误：返回在group中的相对位置
        sample_idx = group.index.get_loc(max_idx)
        
        last_samples[company_file] = {
            'sample_id': max_row['sample_id'],
            'target_row': max_row['target_row'],
            'sample_idx_wrong': sample_idx,  # 这是错的
            'sample_idx_correct': max_idx  # 这才是对的
        }
    
    return last_samples


def find_last_samples_per_company_CORRECT(index_df: pd.DataFrame):
    """
    正确的实现
    """
    last_samples = {}
    
    for company_file, group in index_df.groupby('source_file'):
        max_idx = group['target_row'].idxmax()
        max_row = group.loc[max_idx]
        
        # 正确：直接使用DataFrame的索引
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
    path = Path(source_file)
    return path.stem


def verify_evaluation_logic():
    """验证评估逻辑的问题"""
    print("="*80)
    print("验证评估脚本的问题")
    print("="*80)
    
    # 加载数据
    train_data = torch.load(TRAIN_PT_FILE)
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    
    print(f"\n训练集样本数: {len(train_index)}")
    print(f"训练集.pt样本数: {len(train_data['X'])}")
    
    # 使用错误和正确的方法找最后样本
    print("\n使用错误方法（评估脚本的实现）:")
    wrong_samples = find_last_samples_per_company_WRONG(train_index)
    
    print("\n使用正确方法:")
    correct_samples = find_last_samples_per_company_CORRECT(train_index)
    
    # 随机选择5个公司对比
    import random
    random.seed(42)
    sample_companies = random.sample(list(wrong_samples.keys()), min(5, len(wrong_samples)))
    
    print(f"\n对比 {len(sample_companies)} 个公司的索引:")
    print("-"*80)
    
    errors_found = 0
    
    for company_file in sample_companies:
        company_code = extract_company_code(company_file)
        wrong_info = wrong_samples[company_file]
        correct_info = correct_samples[company_file]
        
        wrong_idx = wrong_info['sample_idx_wrong']
        correct_idx = correct_info['sample_idx']
        
        print(f"\n公司: {company_code}")
        print(f"  sample_id: {wrong_info['sample_id']}")
        print(f"  target_row: {wrong_info['target_row']}")
        print(f"  错误的sample_idx（group内位置）: {wrong_idx}")
        print(f"  正确的sample_idx（DataFrame索引）: {correct_idx}")
        
        if wrong_idx != correct_idx:
            errors_found += 1
            print(f"  ❌ 索引不匹配！")
            
            # 获取对应的真实值
            wrong_y = train_data['y'][wrong_idx].item()
            correct_y = train_data['y'][correct_idx].item()
            
            print(f"\n  使用错误索引得到的y值: {wrong_y}")
            print(f"  使用正确索引得到的y值: {correct_y}")
            
            # 从原始文件验证
            df_original = pd.read_parquet(company_file)
            true_y = df_original.iloc[wrong_info['target_row']]['收盘']
            
            print(f"  原始parquet中的真实值: {true_y}")
            
            if abs(correct_y - true_y) < 1e-5:
                print(f"  ✅ 正确索引的y值与原始值匹配")
            else:
                print(f"  ❌ 正确索引的y值与原始值不匹配！")
            
            if abs(wrong_y - true_y) > 1e-5:
                print(f"  ❌ 错误索引的y值与原始值不匹配（相差 {abs(wrong_y - true_y):.2f}）")
        else:
            print(f"  ✅ 索引一致（该公司只有一个样本）")
    
    print("\n" + "="*80)
    print(f"发现 {errors_found} 个公司的索引存在问题")
    print("="*80)
    
    return errors_found > 0


def verify_excel_data():
    """验证Excel中的数据"""
    print("\n\n" + "="*80)
    print("验证Excel文件中的数据")
    print("="*80)
    
    # 读取Excel
    print(f"\n读取Excel: {EVAL_EXCEL_FILE.name}")
    df_excel = pd.read_excel(EVAL_EXCEL_FILE, sheet_name='评估结果')
    
    print(f"Excel中的公司数: {len(df_excel)}")
    print(f"\nExcel列: {df_excel.columns.tolist()}")
    
    # 加载训练数据
    train_data = torch.load(TRAIN_PT_FILE)
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    val_data = torch.load(VAL_PT_FILE)
    val_index = pd.read_parquet(VAL_INDEX_FILE)
    
    # 使用正确方法找最后样本
    train_correct_samples = find_last_samples_per_company_CORRECT(train_index)
    val_correct_samples = find_last_samples_per_company_CORRECT(val_index)
    
    # 随机检查10个公司
    import random
    random.seed(42)
    check_companies = random.sample(range(len(df_excel)), min(10, len(df_excel)))
    
    print(f"\n随机检查 {len(check_companies)} 个公司...")
    
    train_errors = 0
    val_errors = 0
    
    for idx in check_companies:
        row = df_excel.iloc[idx]
        company_code = row['公司代码']
        
        print(f"\n{'='*80}")
        print(f"公司代码: {company_code}")
        
        # 找到对应的source_file
        source_files = [f for f in train_correct_samples.keys() if company_code in f]
        if not source_files:
            print(f"  ⚠️  未找到匹配的source_file")
            continue
        
        source_file = source_files[0]
        print(f"  文件: {Path(source_file).name}")
        
        # 检查训练集
        if not pd.isna(row['训练_真实值']):
            excel_train_y = row['训练_真实值']
            print(f"\n  训练集:")
            print(f"    Excel中的真实值: {excel_train_y}")
            
            if source_file in train_correct_samples:
                correct_info = train_correct_samples[source_file]
                correct_idx = correct_info['sample_idx']
                target_row = correct_info['target_row']
                
                # 从.pt获取
                pt_y = train_data['y'][correct_idx].item()
                
                # 从原始文件获取
                df_original = pd.read_parquet(source_file)
                original_y = df_original.iloc[target_row]['收盘']
                
                print(f"    .pt中的真实值: {pt_y}")
                print(f"    原始parquet中的值: {original_y}")
                
                diff_excel_original = abs(excel_train_y - original_y)
                diff_excel_pt = abs(excel_train_y - pt_y)
                
                print(f"\n    Excel vs 原始: 差异 {diff_excel_original:.6f}")
                print(f"    Excel vs .pt: 差异 {diff_excel_pt:.6f}")
                
                if diff_excel_original > 1e-3:
                    print(f"    ❌ Excel中的值与原始值不匹配！")
                    train_errors += 1
                else:
                    print(f"    ✅ Excel中的值正确")
        
        # 检查验证集
        if not pd.isna(row['验证_真实值']):
            excel_val_y = row['验证_真实值']
            print(f"\n  验证集:")
            print(f"    Excel中的真实值: {excel_val_y}")
            
            if source_file in val_correct_samples:
                correct_info = val_correct_samples[source_file]
                correct_idx = correct_info['sample_idx']
                target_row = correct_info['target_row']
                
                # 从.pt获取
                pt_y = val_data['y'][correct_idx].item()
                
                # 从原始文件获取
                df_original = pd.read_parquet(source_file)
                original_y = df_original.iloc[target_row]['收盘']
                
                print(f"    .pt中的真实值: {pt_y}")
                print(f"    原始parquet中的值: {original_y}")
                
                diff_excel_original = abs(excel_val_y - original_y)
                diff_excel_pt = abs(excel_val_y - pt_y)
                
                print(f"\n    Excel vs 原始: 差异 {diff_excel_original:.6f}")
                print(f"    Excel vs .pt: 差异 {diff_excel_pt:.6f}")
                
                if diff_excel_original > 1e-3:
                    print(f"    ❌ Excel中的值与原始值不匹配！")
                    val_errors += 1
                else:
                    print(f"    ✅ Excel中的值正确")
    
    print("\n" + "="*80)
    print("Excel验证结果汇总")
    print("="*80)
    print(f"训练集错误数: {train_errors}")
    print(f"验证集错误数: {val_errors}")
    
    if train_errors > 0 or val_errors > 0:
        print(f"\n❌ 发现数据不匹配问题！")
    else:
        print(f"\n✅ 所有检查的数据都正确")


def main():
    """主函数"""
    # 先验证评估逻辑
    has_logic_error = verify_evaluation_logic()
    
    # 再验证Excel数据
    verify_excel_data()
    
    if has_logic_error:
        print("\n" + "="*80)
        print("结论")
        print("="*80)
        print("评估脚本存在索引错误：")
        print("  问题代码：sample_idx = group.index.get_loc(max_idx)")
        print("  这返回的是在group中的相对位置，而不是在整个DataFrame中的绝对位置")
        print("\n  修复方法：直接使用 sample_idx = max_idx")
        print("="*80)


if __name__ == '__main__':
    main()
