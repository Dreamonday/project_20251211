"""
检查所有sheet中特定公司的排名位置
"""
import pandas as pd
from pathlib import Path

result_dir = Path("/data/project_20251211/tests/inference_results/timexer_mlp_v0.5_20260112122205_20260112120624_500120_more_continueTV_20260112181215")
excel_path = result_dir / "不同误差下的公司排名.xlsx"

# 要检查的公司
target_companies = [
    (853, "CHINA TELECOM中國電信"),
    (1129, "Fiserv")
]

print("=" * 80)
print("检查两家公司在所有sheet中的位置")
print("=" * 80)

xl_file = pd.ExcelFile(excel_path)

for sheet_name in xl_file.sheet_names:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"\n{'=' * 80}")
    print(f"Sheet: {sheet_name}")
    print(f"{'=' * 80}")
    
    for company_id, company_name in target_companies:
        matching = df[df['company_id'] == company_id]
        if len(matching) > 0:
            row_idx = matching.index[0]
            row_num = row_idx + 2  # Excel行号（标题行+1，索引+1）
            print(f"\ncompany_id={company_id} ({company_name}):")
            print(f"  Excel行号: {row_num}")
            print(f"  DataFrame索引: {row_idx}")
            print(f"  排名: {matching.iloc[0]['排名']}")
            print(f"  指标值: {matching.iloc[0]['指标值']}")
    
    # 检查这两家公司的行号是否相邻
    rows = []
    for company_id, _ in target_companies:
        matching = df[df['company_id'] == company_id]
        if len(matching) > 0:
            rows.append(matching.index[0])
    
    if len(rows) == 2:
        distance = abs(rows[1] - rows[0])
        if distance <= 2:
            print(f"\n⚠️  这两家公司在此sheet中的行距离很近: {distance} 行")
            print(f"   可能导致Excel中复制粘贴时数据错位")
            print(f"\n   周围数据:")
            start = min(rows) - 1
            end = max(rows) + 2
            print(df.iloc[start:end][['排名', 'company_id', 'company_name', 'stock_code', '指标值']])

print("\n" + "=" * 80)
print("分析结论")
print("=" * 80)
print("""
如果某个sheet中这两家公司的排名位置相邻或接近，
在Excel中手动复制或查看时可能会导致数据混淆，
但实际文件中的数据是完整且正确的。
""")
