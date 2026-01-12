#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证归一化逻辑脚本
版本: v0.1
日期: 20251218181000

功能:
1. 验证原始数据为0的是否完全不参与归一化统计量计算
2. 检查训练过程中目标值"收盘"是否发生变化
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
COMPANY_STATS_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/company_stats_v0.3.json"
TRAIN_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/train_samples_index.parquet"


def verify_zero_not_in_stats():
    """验证原始数据中的0值是否被排除在统计量计算之外"""
    print("="*80)
    print("验证1: 原始数据为0的是否不参与归一化统计量计算")
    print("="*80)
    
    import json
    with open(COMPANY_STATS_FILE, 'r') as f:
        company_stats = json.load(f)
    
    # 随机选择几个公司检查
    import random
    random.seed(42)
    company_files = list(company_stats.keys())
    sample_companies = random.sample(company_files, min(5, len(company_files)))
    
    print(f"\n检查 {len(sample_companies)} 个公司的统计量计算逻辑\n")
    
    for company_file in sample_companies:
        print(f"\n公司文件: {Path(company_file).name}")
        
        # 读取公司数据
        df = pd.read_parquet(company_file)
        
        # 获取该公司的统计量
        stats = company_stats[company_file]
        
        # 选择一个需要归一化的列进行验证
        test_columns = [
            '净利润：持续经营业务税后净利润、持续经营净利润、Net Profit from Continuing Operations、税后净利润',
            '固定资产：物业、厂房及设备（PP&E）；Property, Plant and Equipment (PP&E)；固定资产净值'
        ]
        
        for col in test_columns:
            if col not in df.columns or col not in stats:
                continue
            
            print(f"\n  检查列: {col[:60]}...")
            
            # 获取原始列数据
            col_data = df[col]
            
            # 计算不同条件下的统计量
            all_values = col_data[col_data.notna()]
            non_zero_values = col_data[(col_data.notna()) & (col_data != 0)]
            zero_values = col_data[col_data == 0.0]
            
            print(f"    总行数: {len(df)}")
            print(f"    非NaN行数: {len(all_values)}")
            print(f"    非NaN且非0行数: {len(non_zero_values)}")
            print(f"    为0的行数: {len(zero_values)}")
            
            if len(non_zero_values) > 0:
                # 手动计算统计量（排除0）
                manual_mean = float(np.mean(non_zero_values))
                manual_std = float(np.std(non_zero_values))
                
                # 获取保存的统计量
                saved_mean = stats[col]['mean']
                saved_std = stats[col]['std']
                
                print(f"\n    手动计算（排除0值）:")
                print(f"      均值: {manual_mean:.6f}")
                print(f"      标准差: {manual_std:.6f}")
                
                print(f"\n    保存的统计量:")
                print(f"      均值: {saved_mean:.6f}")
                print(f"      标准差: {saved_std:.6f}")
                
                # 对比
                mean_diff = abs(manual_mean - saved_mean)
                std_diff = abs(manual_std - saved_std)
                
                print(f"\n    差异:")
                print(f"      均值差异: {mean_diff:.6e}")
                print(f"      标准差差异: {std_diff:.6e}")
                
                if mean_diff < 1e-3 and std_diff < 1e-3:
                    print(f"    ✅ 统计量匹配，确认0值未参与计算")
                else:
                    print(f"    ❌ 统计量不匹配！")
                    
                    # 额外测试：如果包含0值会是什么结果
                    if len(zero_values) > 0:
                        mean_with_zero = float(np.mean(all_values))
                        std_with_zero = float(np.std(all_values))
                        print(f"\n    如果包含0值的统计量:")
                        print(f"      均值: {mean_with_zero:.6f}")
                        print(f"      标准差: {std_with_zero:.6f}")
                        
                        if abs(saved_mean - mean_with_zero) < 1e-3:
                            print(f"    ⚠️  保存的统计量似乎包含了0值！")
            
            break  # 只检查第一个可用列
        
        break  # 只详细检查第一个公司


def verify_target_value_unchanged():
    """验证训练过程中目标值"收盘"是否保持不变"""
    print("\n\n" + "="*80)
    print("验证2: 训练数据中目标值'收盘'是否与原始数据一致")
    print("="*80)
    
    # 加载训练数据
    train_data = torch.load(TRAIN_PT_FILE)
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    
    target_column = train_data['metadata']['target_column']
    print(f"\n目标列: {target_column}")
    
    # 随机选择10个样本检查
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(train_index)), min(10, len(train_index)))
    
    print(f"检查 {len(sample_indices)} 个样本的目标值...\n")
    
    all_match = True
    mismatches = []
    
    for idx in sample_indices:
        sample_info = train_index.iloc[idx]
        source_file = sample_info['source_file']
        target_row = sample_info['target_row']
        
        # 读取原始数据
        df_original = pd.read_parquet(source_file)
        original_target = df_original.iloc[target_row][target_column]
        
        # 获取.pt中的目标值
        pt_target = train_data['y'][idx].item()
        
        # 对比
        diff = abs(original_target - pt_target)
        
        if diff > 1e-5:
            all_match = False
            mismatches.append({
                'sample_id': sample_info['sample_id'],
                'company': sample_info['company_name'],
                'original': original_target,
                'pt': pt_target,
                'diff': diff
            })
    
    if all_match:
        print("✅ 所有样本的目标值都与原始数据完全一致")
    else:
        print(f"❌ 发现 {len(mismatches)} 个样本的目标值不匹配:")
        for m in mismatches:
            print(f"\n  样本 {m['sample_id']} ({m['company']})")
            print(f"    原始: {m['original']}")
            print(f"    .pt:  {m['pt']}")
            print(f"    差异: {m['diff']:.6e}")


def verify_input_dimensions():
    """验证模型输入维度"""
    print("\n\n" + "="*80)
    print("验证3: 确认模型输入输出维度")
    print("="*80)
    
    # 加载训练数据
    train_data = torch.load(TRAIN_PT_FILE)
    
    X = train_data['X']
    y = train_data['y']
    metadata = train_data['metadata']
    
    print(f"\n训练数据形状:")
    print(f"  输入 X: {X.shape}")
    print(f"    - 样本数: {X.shape[0]}")
    print(f"    - 序列长度: {X.shape[1]}")
    print(f"    - 特征维度: {X.shape[2]}")
    print(f"\n  输出 y: {y.shape}")
    print(f"    - 样本数: {y.shape[0]}")
    print(f"    - 输出维度: {y.shape[1]}")
    
    print(f"\n元数据:")
    print(f"  特征数量: {metadata['num_features']}")
    print(f"  序列长度: {metadata['seq_len']}")
    print(f"  目标列: {metadata['target_column']}")
    print(f"  是否归一化: {metadata['normalize']}")
    print(f"  归一化方法: {metadata['normalize_method']}")
    
    print(f"\n特征列（前10个）:")
    for i, col in enumerate(metadata['feature_columns'][:10]):
        print(f"  {i+1}. {col}")
    print(f"  ... 还有 {len(metadata['feature_columns']) - 10} 个特征")
    
    print(f"\n不归一化的列:")
    for col in metadata['no_normalize_columns']:
        print(f"  - {col}")
    
    # 确认
    if X.shape[2] == 44 and y.shape[1] == 1:
        print(f"\n✅ 确认: 模型使用 44 个特征维度输入，预测 1 个目标值（收盘）")
    else:
        print(f"\n❌ 维度异常: 输入 {X.shape[2]} 维，输出 {y.shape[1]} 维")


def check_收盘_in_features():
    """检查"收盘"是否在输入特征中"""
    print("\n\n" + "="*80)
    print("验证4: 检查'收盘'是否在输入特征中")
    print("="*80)
    
    train_data = torch.load(TRAIN_PT_FILE)
    metadata = train_data['metadata']
    
    feature_columns = metadata['feature_columns']
    target_column = metadata['target_column']
    
    print(f"\n目标列: {target_column}")
    print(f"输入特征数量: {len(feature_columns)}")
    
    if target_column in feature_columns:
        col_idx = feature_columns.index(target_column)
        print(f"\n⚠️  '收盘' 在输入特征中（位置: {col_idx}）")
        print(f"这意味着模型可以看到当前时刻的收盘价")
        
        # 检查是否被归一化
        if target_column in metadata['no_normalize_columns']:
            print(f"✅ '收盘' 未被归一化（保持原始值）")
        else:
            print(f"❌ '收盘' 被归一化了")
    else:
        print(f"\n✅ '收盘' 不在输入特征中")
        print(f"模型使用其他 {len(feature_columns)} 个特征来预测收盘价")


def main():
    """主函数"""
    verify_zero_not_in_stats()
    verify_target_value_unchanged()
    verify_input_dimensions()
    check_收盘_in_features()
    
    print("\n" + "="*80)
    print("所有验证完成")
    print("="*80)


if __name__ == '__main__':
    main()
