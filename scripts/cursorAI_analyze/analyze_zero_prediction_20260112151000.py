#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析预测值为0的公司的输入特征
日期: 20260112151000
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 数据路径
PREPROCESSED_DIR = "/data/project_20251211/data/processed/preprocess_data_v0.5_20260112120624_500120_more_continueTV"

print("=" * 80)
print("加载索引文件")
print("=" * 80)

# 加载训练集索引
train_index_path = Path(PREPROCESSED_DIR) / "train_index_v0.5.parquet"
train_index = pd.read_parquet(train_index_path)

print(f"\n训练集索引数量: {len(train_index)}")
print(f"索引列: {train_index.columns.tolist()}")

# 查找特定公司的样本
target_companies = {
    "JD ‑ SW京東集團 ‑ SW": "922_JD",
    "BIDU ‑ SW百度集團 ‑ SW": "928_BIDU",
    "TRIP.COM ‑S 攜程集團 ‑ S": "930_TRIP",
    "TME ‑ SW 騰訊音樂 ‑ SW": "886_TME",
    "CHINA TELECOM中國電信": "853_CHINA"
}

print("\n" + "=" * 80)
print("查找各公司的样本")
print("=" * 80)

for company_name, company_prefix in target_companies.items():
    # 按公司名称筛选
    company_samples = train_index[train_index['company_name'] == company_name]
    
    if len(company_samples) == 0:
        # 尝试模糊匹配
        company_samples = train_index[train_index['company_name'].str.contains(company_prefix.split('_')[1], na=False)]
    
    print(f"\n公司: {company_name}")
    print(f"  训练样本数: {len(company_samples)}")
    
    if len(company_samples) > 0:
        # 显示日期范围
        if 'target_date' in company_samples.columns:
            print(f"  日期范围: {company_samples['target_date'].min()} 到 {company_samples['target_date'].max()}")
        
        # 显示前几个样本的索引
        print(f"  前3个样本的sample_id: {company_samples['sample_id'].head(3).tolist()}")
        print(f"  后3个样本的sample_id: {company_samples['sample_id'].tail(3).tolist()}")

# 现在加载训练数据张量，检查这些样本的输入特征
print("\n" + "=" * 80)
print("加载训练数据张量（这可能需要几分钟...）")
print("=" * 80)

train_pt_path = Path(PREPROCESSED_DIR) / "train_v0.5.pt"
print(f"加载: {train_pt_path}")
print("（由于文件很大，这可能会超时，但我们至少能看到部分信息）")
