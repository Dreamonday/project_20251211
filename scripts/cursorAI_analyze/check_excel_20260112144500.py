#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查已生成的推理结果Excel文件
日期: 20260112144500
"""

import pandas as pd
from pathlib import Path

# Excel文件路径
excel_paths = [
    "/data/project_20251211/tests/inference_results/timexer_mlp_v0.5_20260112122205_20260112120624_500120_more_continueTV_v0.2/1_中国铝业_601600.xlsx",
]

for excel_path in excel_paths:
    print("=" * 80)
    print(f"文件: {Path(excel_path).name}")
    print("=" * 80)
    
    # 读取第一个sheet（推理结果）
    df = pd.read_excel(excel_path, sheet_name='推理结果')
    
    print(f"\n数据形状: {df.shape}")
    print("\n列名:")
    print(df.columns.tolist())
    
    print("\n前10行数据:")
    print(df.head(10).to_string())
    
    # 检查预测值统计
    print("\n训练集预测收盘价统计:")
    if '训练_预测收盘价' in df.columns:
        train_pred = df['训练_预测收盘价'].dropna()
        if len(train_pred) > 0:
            print(f"  数量: {len(train_pred)}")
            print(f"  最小值: {train_pred.min():.6f}")
            print(f"  最大值: {train_pred.max():.6f}")
            print(f"  平均值: {train_pred.mean():.6f}")
            print(f"  标准差: {train_pred.std():.6f}")
            print(f"  前10个值: {train_pred.head(10).tolist()}")
        else:
            print("  无数据")
    
    print("\n验证集预测收盘价统计:")
    if '验证_预测收盘价' in df.columns:
        val_pred = df['验证_预测收盘价'].dropna()
        if len(val_pred) > 0:
            print(f"  数量: {len(val_pred)}")
            print(f"  最小值: {val_pred.min():.6f}")
            print(f"  最大值: {val_pred.max():.6f}")
            print(f"  平均值: {val_pred.mean():.6f}")
            print(f"  标准差: {val_pred.std():.6f}")
            print(f"  前10个值: {val_pred.head(10).tolist()}")
        else:
            print("  无数据")
    
    # 读取第二个sheet（统计汇总）
    print("\n" + "-" * 80)
    print("统计汇总:")
    print("-" * 80)
    stats_df = pd.read_excel(excel_path, sheet_name='统计汇总')
    print(stats_df.to_string())
    
    print("\n")
