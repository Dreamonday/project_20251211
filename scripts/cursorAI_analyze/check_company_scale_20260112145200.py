#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查特定公司的规模信息
日期: 20260112145200
"""

import json
from pathlib import Path

# 读取company_scale_info
scale_info_path = Path("/data/project_20251211/data/processed/preprocess_data_v0.5_20260112120624_500120_more_continueTV/company_scale_info_v0.5.json")

print("=" * 80)
print("读取公司规模信息")
print("=" * 80)

with open(scale_info_path, 'r', encoding='utf-8') as f:
    scale_info = json.load(f)

print(f"\n公司数量: {len(scale_info)}")

# 查找几个关键公司
target_companies = {
    "1": "中国铝业",
    "922": "JD京东",
    "928": "BIDU百度",
    "930": "TRIP.COM携程",
    "886": "TME腾讯音乐"
}

for company_id, expected_name in target_companies.items():
    print(f"\n{'=' * 80}")
    print(f"查找公司ID: {company_id} ({expected_name})")
    print("=" * 80)
    
    if company_id in scale_info:
        info = scale_info[company_id]
        print(f"✓ 找到公司")
        print(f"  公司名称: {info.get('company_name', 'N/A')}")
        print(f"  股票代码: {info.get('stock_code', 'N/A')}")
        print(f"  货币单位: {info.get('currency', 'N/A')}")
        print(f"  Scale: {info.get('scale', 'N/A')}")
        
        # 打印所有键
        print(f"  所有字段: {list(info.keys())}")
    else:
        print("✗ 未找到该公司的信息")

print(f"\n{'=' * 80}")
print("检查完成")
print("=" * 80)
