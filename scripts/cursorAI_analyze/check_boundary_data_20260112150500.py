#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查时间分界线前后的数据差异
日期: 20260112150500
"""

import pandas as pd
from pathlib import Path

# 读取原始数据文件
files_to_check = [
    {
        "name": "TRIP.COM携程",
        "file": "/data/project_20251211/data/raw/processed_data_20251220/930_TRIP.COM ‑S 攜程集團 ‑ S_09961_20251219_222135.parquet",
        "boundary_date": "2025-09-17",
        "check_before": "2025-09-15",  # 预测值为0的时间
        "check_after": "2025-09-20"    # 预测值正常的时间
    },
    {
        "name": "CHINA TELECOM中国电信(港股)",
        "file": "/data/project_20251211/data/raw/processed_data_20251220/853_CHINA TELECOM中國電信_00728_20251219_202124.parquet",
        "boundary_date": "2023-05-09",
        "check_before": "2023-05-05",  # 预测值正常的时间
        "check_after": "2023-05-15"    # 预测值为0的时间
    }
]

for company_info in files_to_check:
    print("=" * 80)
    print(f"公司: {company_info['name']}")
    print(f"分界日期: {company_info['boundary_date']}")
    print("=" * 80)
    
    file_path = Path(company_info['file'])
    if not file_path.exists():
        print(f"✗ 文件不存在: {file_path}")
        print()
        continue
    
    # 读取parquet文件
    df = pd.read_parquet(file_path)
    
    print(f"\n数据形状: {df.shape}")
    print(f"日期范围: {df['日期'].min()} 到 {df['日期'].max()}")
    
    # 将日期转换为datetime
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 获取分界日期前后的数据
    date_before = pd.to_datetime(company_info['check_before'])
    date_after = pd.to_datetime(company_info['check_after'])
    
    df_before = df[df['日期'] == date_before]
    df_after = df[df['日期'] == date_after]
    
    if len(df_before) == 0:
        print(f"\n✗ 未找到日期 {company_info['check_before']} 的数据")
    if len(df_after) == 0:
        print(f"✗ 未找到日期 {company_info['check_after']} 的数据")
    
    if len(df_before) > 0 and len(df_after) > 0:
        print(f"\n分界前数据 ({company_info['check_before']}):")
        print(df_before[['日期', '收盘', '货币单位']].to_string())
        
        print(f"\n分界后数据 ({company_info['check_after']}):")
        print(df_after[['日期', '收盘', '货币单位']].to_string())
        
        # 检查哪些列发生了变化
        print(f"\n检查列值差异:")
        
        # 获取所有列
        cols = df.columns.tolist()
        
        # 比较每一列
        diff_cols = []
        for col in cols:
            if col == '日期':
                continue
            
            val_before = df_before[col].iloc[0]
            val_after = df_after[col].iloc[0]
            
            # 检查是否有显著差异
            if pd.isna(val_before) != pd.isna(val_after):
                diff_cols.append((col, val_before, val_after, "NaN变化"))
            elif pd.isna(val_before) and pd.isna(val_after):
                continue
            elif isinstance(val_before, (int, float)) and isinstance(val_after, (int, float)):
                if val_before == 0 and val_after != 0:
                    diff_cols.append((col, val_before, val_after, "从0变为非0"))
                elif val_before != 0 and val_after == 0:
                    diff_cols.append((col, val_before, val_after, "从非0变为0"))
            elif val_before != val_after:
                diff_cols.append((col, val_before, val_after, "值变化"))
        
        if diff_cols:
            print(f"\n发现 {len(diff_cols)} 个列有显著差异:")
            for col, val_before, val_after, reason in diff_cols[:20]:  # 只显示前20个
                print(f"  - {col}")
                print(f"      分界前: {val_before}")
                print(f"      分界后: {val_after}")
                print(f"      原因: {reason}")
        else:
            print("  未发现显著差异")
        
        # 检查财务数据列
        financial_cols = [col for col in cols if any(keyword in col for keyword in 
                         ['净利润', '资产', '负债', '收入', '营业', '现金', '利润'])]
        
        print(f"\n财务数据列的NaN情况:")
        print(f"分界前 ({company_info['check_before']}) - NaN列数: {df_before[financial_cols].isna().sum().sum()} / {len(financial_cols)}")
        print(f"分界后 ({company_info['check_after']}) - NaN列数: {df_after[financial_cols].isna().sum().sum()} / {len(financial_cols)}")
        
        # 列出为NaN的财务列
        nan_before = df_before[financial_cols].isna().sum()
        nan_after = df_after[financial_cols].isna().sum()
        
        new_nans = []
        for col in financial_cols:
            if nan_before[col] == 0 and nan_after[col] > 0:
                new_nans.append(col)
        
        if new_nans:
            print(f"\n分界后新增NaN的财务列 ({len(new_nans)}个):")
            for col in new_nans[:10]:  # 只显示前10个
                print(f"  - {col}")
    
    print("\n")
