"""
检查排名Excel文件中的数据问题
"""
import pandas as pd
from pathlib import Path

# 读取Excel文件
excel_path = Path("/data/project_20251211/tests/inference_results/timexer_mlp_v0.5_20260112122205_20260112120624_500120_more_continueTV_20260112181215/不同误差下的公司排名.xlsx")

print(f"读取文件: {excel_path}")
print("=" * 80)

# 获取所有sheet名称
xl_file = pd.ExcelFile(excel_path)
print(f"\nSheet数量: {len(xl_file.sheet_names)}")
print(f"Sheet名称: {xl_file.sheet_names}")

# 读取第一个sheet查看问题
sheet_name = xl_file.sheet_names[0]
df = pd.read_excel(excel_path, sheet_name=sheet_name)

print(f"\n检查 Sheet: {sheet_name}")
print(f"总行数: {len(df)}")
print(f"列名: {list(df.columns)}")
print(f"\n数据类型:")
print(df.dtypes)

# 查找有问题的行（缺少数据的行）
print("\n" + "=" * 80)
print("查找缺少数据的行")
print("=" * 80)

# 检查每列的缺失情况
print("\n每列的缺失值数量:")
print(df.isnull().sum())

# 查找行中有缺失值的记录
rows_with_na = df[df.isnull().any(axis=1)]
print(f"\n有缺失值的行数: {len(rows_with_na)}")

if len(rows_with_na) > 0:
    print("\n前20行有缺失值的记录:")
    print(rows_with_na.head(20))
    
    print("\n后20行有缺失值的记录:")
    print(rows_with_na.tail(20))

# 特别查看第1398行附近
print("\n" + "=" * 80)
print("检查第1398行附近的数据")
print("=" * 80)
if len(df) >= 1398:
    print(f"\n第1395-1405行:")
    print(df.iloc[1394:1404])
    
    # 详细查看1398行
    print(f"\n第1398行详细信息:")
    row = df.iloc[1397]
    for col in df.columns:
        print(f"  {col}: {row[col]} (类型: {type(row[col])})")

# 检查是否有空字符串
print("\n" + "=" * 80)
print("检查是否有空字符串或特殊值")
print("=" * 80)

for col in ['company_name', 'stock_code']:
    if col in df.columns:
        empty_or_na = df[df[col].isna() | (df[col] == '') | (df[col].astype(str).str.strip() == '')]
        if len(empty_or_na) > 0:
            print(f"\n列 '{col}' 有 {len(empty_or_na)} 行为空或NA:")
            print(empty_or_na[['排名', 'company_id', 'company_name', 'stock_code', '指标值']].head(20))
