#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查交易日数量和验证集样本为0的原因
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# 配置
SOURCE_DATA_DIR = '/data/project_20251211/data/raw/processed_data_20251220'
SPLIT_DATE = '2024-01-01'  # 从终端输出看，实际使用的是2024-01-01
END_DATE = '2025-12-31'
INPUT_WINDOW = 500
PREDICTION_DAY = 120
MIN_REQUIRED_DAYS = INPUT_WINDOW + PREDICTION_DAY  # 620天

print("=" * 80)
print("交易日数量分析")
print("=" * 80)
print(f"划分日期: {SPLIT_DATE}")
print(f"结束日期: {END_DATE}")
print(f"最小要求天数: {MIN_REQUIRED_DAYS} 天 (输入窗口{INPUT_WINDOW} + 预测第{PREDICTION_DAY}天)")
print("=" * 80)

# 读取一个示例文件来查看实际的交易日
source_dir = Path(SOURCE_DATA_DIR)
parquet_files = list(source_dir.glob('*.parquet'))

if not parquet_files:
    print(f"❌ 未找到数据文件在: {source_dir}")
    exit(1)

# 读取第一个文件作为示例
sample_file = parquet_files[0]
print(f"\n读取示例文件: {sample_file.name}")
df = pd.read_parquet(sample_file)

if '日期' not in df.columns:
    print("❌ 数据文件中没有'日期'列")
    exit(1)

df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values('日期').reset_index(drop=True)

# 筛选2024-01-01之后的数据
split_date = pd.to_datetime(SPLIT_DATE)
end_date = pd.to_datetime(END_DATE)

val_df = df[(df['日期'] >= split_date) & (df['日期'] <= end_date)].copy()
val_days = len(val_df)

print(f"\n示例文件 ({sample_file.name}):")
print(f"  总数据天数: {len(df)} 天")
print(f"  日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
print(f"  划分日期({SPLIT_DATE})后的数据天数: {val_days} 天")
print(f"  需要的最小天数: {MIN_REQUIRED_DAYS} 天")

if val_days < MIN_REQUIRED_DAYS:
    print(f"\n❌ 验证集数据不足！")
    print(f"   实际: {val_days} 天")
    print(f"   需要: {MIN_REQUIRED_DAYS} 天")
    print(f"   缺少: {MIN_REQUIRED_DAYS - val_days} 天")
else:
    print(f"\n✓ 验证集数据充足")

# 计算从2024-01-01到2025-12-31的理论交易日数
# 使用pandas的bdate_range（工作日范围）
start = pd.to_datetime(SPLIT_DATE)
end = pd.to_datetime(END_DATE)

# 计算工作日（不包括周末，但包括所有日期）
business_days = pd.bdate_range(start=start, end=end)
print(f"\n理论工作日数（2024-01-01 至 2025-12-31，不包括周末）: {len(business_days)} 天")

# 但实际交易日还要排除节假日，所以实际交易日会更少
print(f"\n注意: 实际交易日数 = 工作日数 - 节假日数")
print(f"      通常一年约有250个交易日，两年约500个交易日")
print(f"      从2024-01-01到2025-12-31，实际交易日数约为: 约500天左右")

# 检查多个文件
print("\n" + "=" * 80)
print("检查多个文件的验证集数据量")
print("=" * 80)

val_days_list = []
for file_path in parquet_files[:10]:  # 检查前10个文件
    try:
        df_temp = pd.read_parquet(file_path)
        if '日期' not in df_temp.columns:
            continue
        df_temp['日期'] = pd.to_datetime(df_temp['日期'])
        val_df_temp = df_temp[(df_temp['日期'] >= split_date) & (df_temp['日期'] <= end_date)]
        val_days_list.append(len(val_df_temp))
    except Exception as e:
        continue

if val_days_list:
    print(f"\n检查了 {len(val_days_list)} 个文件:")
    print(f"  验证集数据天数范围: {min(val_days_list)} - {max(val_days_list)} 天")
    print(f"  平均验证集数据天数: {sum(val_days_list) / len(val_days_list):.1f} 天")
    insufficient_count = sum(1 for d in val_days_list if d < MIN_REQUIRED_DAYS)
    print(f"  数据不足的文件数: {insufficient_count}/{len(val_days_list)}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print(f"1. 从2024-01-01到2025-12-31，实际交易日数约为500天左右（少于620天）")
print(f"2. 代码要求验证集至少需要{MIN_REQUIRED_DAYS}天数据（输入窗口{INPUT_WINDOW}天 + 预测第{PREDICTION_DAY}天）")
print(f"3. 由于实际数据不足，验证集样本数为0")
print(f"4. 建议:")
print(f"   - 将SPLIT_DATE改为更早的日期（如2023-01-01），这样验证集有更多数据")
print(f"   - 或者减小INPUT_WINDOW或PREDICTION_DAY的值")
print(f"   - 或者等待更多数据积累")
print("=" * 80)
