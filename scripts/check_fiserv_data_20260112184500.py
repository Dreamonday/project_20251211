"""
检查Fiserv公司的原始数据
"""
import pandas as pd
from pathlib import Path

result_dir = Path("/data/project_20251211/tests/inference_results/timexer_mlp_v0.5_20260112122205_20260112120624_500120_more_continueTV_20260112181215")

# 查找Fiserv的个人文件
import glob
fiserv_files = list(result_dir.glob("*Fiserv*.parquet"))

print("=" * 80)
print("查找Fiserv的数据文件")
print("=" * 80)

if fiserv_files:
    print(f"\n找到文件: {fiserv_files[0].name}")
    
    # 读取文件
    df = pd.read_parquet(fiserv_files[0])
    
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查是否有验证集数据
    if 'split' in df.columns or 'data_type' in df.columns or 'dataset_type' in df.columns:
        split_col = 'split' if 'split' in df.columns else ('data_type' if 'data_type' in df.columns else 'dataset_type')
        print(f"\n数据集划分 ({split_col}):")
        print(df[split_col].value_counts())
    
    # 查看前几行和后几行
    print("\n前5行:")
    print(df.head())
    
    print("\n后5行:")
    print(df.tail())
    
    # 检查是否有验证集
    print("\n" + "=" * 80)
    print("检查验证集数据")
    print("=" * 80)
    
    # 尝试多种可能的列名
    possible_split_cols = ['split', 'data_type', 'dataset_type', 'type', 'set_type']
    split_col_found = None
    
    for col in possible_split_cols:
        if col in df.columns:
            split_col_found = col
            break
    
    if split_col_found:
        print(f"使用列: {split_col_found}")
        val_data = df[df[split_col_found].str.contains('val', case=False, na=False) | 
                      df[split_col_found].str.contains('验证', case=False, na=False)]
        print(f"验证集数据量: {len(val_data)}")
        
        if len(val_data) == 0:
            print("\n⚠️  该公司没有验证集数据！")
            print("这就是为什么在验证集排名中显示为NaN的原因。")
    else:
        print("未找到数据集划分列，尝试其他方法...")
        print(f"所有列: {list(df.columns)}")
        
        # 检查最后一行是否是统计汇总
        if '统计标记' in df.columns:
            stats_row = df[df['统计标记'] == '统计汇总']
            if len(stats_row) > 0:
                print("\n找到统计汇总行:")
                print(stats_row.T)
                
                # 检查验证集相关的统计
                val_cols = [col for col in df.columns if '验证' in col or 'val' in col.lower()]
                if val_cols:
                    print(f"\n验证集相关列:")
                    for col in val_cols:
                        val = stats_row[col].iloc[0]
                        print(f"  {col}: {val}")
                        if pd.isna(val):
                            print(f"    ⚠️  {col} 为 NaN")
else:
    print("\n未找到Fiserv的数据文件")
    print("尝试查找所有parquet文件:")
    all_files = list(result_dir.glob("*.parquet"))
    for f in all_files[:10]:
        print(f"  {f.name}")
