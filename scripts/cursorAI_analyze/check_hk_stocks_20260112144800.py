#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查港股公司的推理结果
日期: 20260112144800
"""

import pandas as pd
from pathlib import Path

# 港股公司Excel文件路径
base_dir = Path("/data/project_20251211/tests/inference_results/timexer_mlp_v0.5_20260112122205_20260112120624_500120_more_continueTV_v0.2")

excel_files = [
    base_dir / "922_JD ‑ SW京東集團 ‑ SW_09618.xlsx",
    base_dir / "928_BIDU ‑ SW百度集團 ‑ SW_09888.xlsx",
    base_dir / "930_TRIP.COM ‑S 攜程集團 ‑ S_09961.xlsx",
    base_dir / "886_TME ‑ SW 騰訊音樂 ‑ SW_01698.xlsx",
]

for excel_path in excel_files:
    if not excel_path.exists():
        print(f"文件不存在: {excel_path}")
        continue
        
    print("=" * 80)
    print(f"文件: {excel_path.name}")
    print("=" * 80)
    
    # 读取第一个sheet（推理结果）
    df = pd.read_excel(excel_path, sheet_name='推理结果')
    
    print(f"\n数据形状: {df.shape}")
    
    print("\n前5行数据:")
    print(df.head(5).to_string())
    
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
            print(f"  所有值是否为0: {(train_pred == 0).all()}")
            print(f"  为0的数量: {(train_pred == 0).sum()}")
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
            print(f"  所有值是否为0: {(val_pred == 0).all()}")
            print(f"  为0的数量: {(val_pred == 0).sum()}")
        else:
            print("  无数据")
    
    # 读取第二个sheet（统计汇总）
    print("\n统计汇总:")
    stats_df = pd.read_excel(excel_path, sheet_name='统计汇总')
    print(stats_df[['company_name', '训练_绝对相对误差_平均值', '验证_绝对相对误差_平均值']].to_string())
    
    print("\n")
