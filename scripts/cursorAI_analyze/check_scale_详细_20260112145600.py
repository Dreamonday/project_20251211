#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查scale_info的详细内容
日期: 20260112145600
"""

import json
from pathlib import Path

# 读取company_scale_info
scale_info_path = Path("/data/project_20251211/data/processed/preprocess_data_v0.5_20260112120624_500120_more_continueTV/company_scale_info_v0.5.json")

with open(scale_info_path, 'r', encoding='utf-8') as f:
    scale_info = json.load(f)

# 打印前3个公司的完整信息
print("=" * 80)
print("前3个公司的完整scale信息:")
print("=" * 80)

for i, (key, info) in enumerate(list(scale_info.items())[:3]):
    print(f"\n{i+1}. 文件: {Path(key).name}")
    print(f"   完整信息: {json.dumps(info, indent=4, ensure_ascii=False)}")

# 搜索特定公司
print("\n" + "=" * 80)
print("搜索特定公司的scale信息:")
print("=" * 80)

target_files = [
    "1_中国铝业_601600",
    "922_JD",
    "928_BIDU",
    "930_TRIP",
    "886_TME"
]

for target in target_files:
    print(f"\n查找: {target}")
    found = False
    for key, info in scale_info.items():
        if target in key:
            print(f"  ✓ 找到: {Path(key).name}")
            print(f"     Scale: {info.get('scale', 'N/A')}")
            print(f"     完整信息: {json.dumps(info, indent=6, ensure_ascii=False)}")
            found = True
            break
    
    if not found:
        print(f"  ✗ 未找到")
