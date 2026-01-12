#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证修复后的评估脚本逻辑
版本: v0.1
日期: 20251218184500

功能:
快速验证修复后的find_last_samples_per_company函数是否正确
"""

import torch
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置路径
TRAIN_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/train_v0.3.pt"
TRAIN_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/train_samples_index.parquet"


def find_last_samples_per_company(index_df: pd.DataFrame):
    """
    修复后的实现（从评估脚本复制）
    """
    last_samples = {}
    
    for company_file, group in index_df.groupby('source_file'):
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


def main():
    """主函数"""
    print("="*80)
    print("验证修复后的评估脚本")
    print("="*80)
    
    # 加载数据
    train_data = torch.load(TRAIN_PT_FILE)
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    
    print(f"\n训练集样本数: {len(train_index)}")
    
    # 使用修复后的方法
    samples = find_last_samples_per_company(train_index)
    
    # 随机检查5个公司
    import random
    random.seed(42)
    sample_companies = random.sample(list(samples.keys()), min(5, len(samples)))
    
    print(f"\n检查 {len(sample_companies)} 个公司...")
    
    all_correct = True
    
    for company_file in sample_companies:
        company_code = Path(company_file).stem
        info = samples[company_file]
        
        sample_idx = info['sample_idx']
        target_row = info['target_row']
        
        # 从.pt获取y值
        pt_y = train_data['y'][sample_idx].item()
        
        # 从原始文件获取真实值
        df_original = pd.read_parquet(company_file)
        original_y = df_original.iloc[target_row]['收盘']
        
        diff = abs(pt_y - original_y)
        
        print(f"\n公司: {company_code}")
        print(f"  sample_idx: {sample_idx}")
        print(f"  target_row: {target_row}")
        print(f"  .pt中的y值: {pt_y}")
        print(f"  原始文件中的y值: {original_y}")
        print(f"  差异: {diff:.6f}")
        
        if diff < 1e-5:
            print(f"  ✅ 数据匹配！")
        else:
            print(f"  ❌ 数据不匹配！")
            all_correct = False
    
    print("\n" + "="*80)
    if all_correct:
        print("✅ 修复成功！所有检查的公司数据都正确匹配")
    else:
        print("❌ 修复失败！仍然存在数据不匹配")
    print("="*80)


if __name__ == '__main__':
    main()
