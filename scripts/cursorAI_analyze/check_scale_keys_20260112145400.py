#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查scale_info的键格式
日期: 20260112145400
"""

import json
from pathlib import Path

# 读取company_scale_info
scale_info_path = Path("/data/project_20251211/data/processed/preprocess_data_v0.5_20260112120624_500120_more_continueTV/company_scale_info_v0.5.json")

with open(scale_info_path, 'r', encoding='utf-8') as f:
    scale_info = json.load(f)

print(f"公司数量: {len(scale_info)}")

# 打印前20个键
print("\n前20个公司的键:")
for i, key in enumerate(list(scale_info.keys())[:20]):
    info = scale_info[key]
    print(f"{i+1:3d}. 键='{key}' | 公司名={info.get('company_name', 'N/A')[:30]:<30} | 股票代码={info.get('stock_code', 'N/A'):<10}")

# 搜索包含特定关键字的公司
print("\n" + "=" * 80)
print("搜索包含关键字的公司:")
print("=" * 80)

keywords = ["中国铝业", "京东", "百度", "携程", "腾讯音乐", "JD", "BIDU", "TRIP", "TME"]

for keyword in keywords:
    print(f"\n关键字: '{keyword}'")
    found = []
    for key, info in scale_info.items():
        company_name = info.get('company_name', '')
        stock_code = info.get('stock_code', '')
        if keyword.lower() in company_name.lower() or keyword.lower() in stock_code.lower():
            found.append((key, company_name, stock_code))
    
    if found:
        for key, name, code in found[:5]:  # 只显示前5个
            print(f"  键='{key}' | {name} | {code}")
    else:
        print(f"  未找到")
