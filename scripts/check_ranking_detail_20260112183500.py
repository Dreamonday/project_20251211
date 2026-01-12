"""
详细检查排名文件的数据完整性
"""
import pandas as pd
from pathlib import Path

result_dir = Path("/data/project_20251211/tests/inference_results/timexer_mlp_v0.5_20260112122205_20260112120624_500120_more_continueTV_20260112181215")

# 检查Excel文件
print("=" * 80)
print("检查Excel文件")
print("=" * 80)

excel_path = result_dir / "不同误差下的公司排名.xlsx"
xl_file = pd.ExcelFile(excel_path)

# 检查每个sheet
for sheet_name in xl_file.sheet_names:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"\nSheet: {sheet_name}")
    print(f"  总行数: {len(df)}")
    print(f"  缺失值: {df.isnull().sum().sum()}")
    
    # 检查是否有空字符串
    for col in ['company_name', 'stock_code']:
        if col in df.columns:
            empty_count = (df[col].astype(str).str.strip() == '').sum()
            if empty_count > 0:
                print(f"  列'{col}'有{empty_count}个空字符串")

# 检查Parquet文件
print("\n" + "=" * 80)
print("检查Parquet文件")
print("=" * 80)

parquet_path = result_dir / "不同误差下的公司排名.parquet"
if parquet_path.exists():
    df_parquet = pd.read_parquet(parquet_path)
    print(f"\n总行数: {len(df_parquet)}")
    print(f"总列数: {len(df_parquet.columns)}")
    print(f"列名: {list(df_parquet.columns)}")
    print(f"\n缺失值统计:")
    print(df_parquet.isnull().sum())
    
    # 检查是否有重复的排名方式
    print(f"\n排名方式分布:")
    print(df_parquet['排名方式'].value_counts())
    
    # 检查每个排名方式的数据
    print("\n" + "=" * 80)
    print("每个排名方式的数据统计")
    print("=" * 80)
    
    for ranking_method in df_parquet['排名方式'].unique():
        subset = df_parquet[df_parquet['排名方式'] == ranking_method]
        print(f"\n{ranking_method}:")
        print(f"  行数: {len(subset)}")
        print(f"  缺失值: {subset.isnull().sum().sum()}")
        print(f"  排名范围: {subset['排名'].min()} - {subset['排名'].max()}")
        print(f"  排名缺失数: {subset['排名'].isnull().sum()}")
        
        # 检查空字符串
        if 'company_name' in subset.columns:
            empty_name = (subset['company_name'].astype(str).str.strip() == '').sum()
            if empty_name > 0:
                print(f"  公司名称为空: {empty_name}")
                print(subset[subset['company_name'].astype(str).str.strip() == ''])

# 检查特殊字符问题
print("\n" + "=" * 80)
print("检查特殊字符")
print("=" * 80)

df_first = pd.read_excel(excel_path, sheet_name=xl_file.sheet_names[0])

# 查找包含特殊字符的公司名称
print("\n检查公司名称中的特殊字符:")
for idx, row in df_first.iterrows():
    name = str(row['company_name'])
    # 检查是否包含不可见字符或特殊Unicode
    if any(ord(c) > 127 and ord(c) < 160 for c in name):
        print(f"  行{idx+1}: {repr(name)} - 包含特殊字符")
    # 检查是否包含换行符
    if '\n' in name or '\r' in name or '\t' in name:
        print(f"  行{idx+1}: {repr(name)} - 包含换行符或制表符")

# 检查特定的公司
print("\n" + "=" * 80)
print("检查用户提到的公司")
print("=" * 80)

companies_to_check = [
    (853, "CHINA TELECOM"),
    (1129, "Fiserv")
]

for company_id, name_part in companies_to_check:
    matching = df_first[
        (df_first['company_id'] == company_id) | 
        (df_first['company_name'].str.contains(name_part, case=False, na=False))
    ]
    if len(matching) > 0:
        print(f"\n查找 company_id={company_id} 或 包含'{name_part}':")
        print(matching)
    else:
        print(f"\n未找到 company_id={company_id} 或 包含'{name_part}'的记录")

# 检查company_id=1129的公司
print("\n查找 company_id=1129:")
matching_1129 = df_first[df_first['company_id'] == 1129]
print(matching_1129)
