#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务数据预处理脚本 v0.2_20251209

功能：
1. 从1400+个Excel文件中提取"财报_日K对齐"sheet的数据
2. 提取公司标识信息（从文件名和第一行数据）
3. 删除不需要的列（股票代码、货币单位、财报数据截止日期、匹配财报公告日期）
4. 合并所有数据并按日期排序
5. 填充数值列的缺失值为0（排除标识列）
6. 保存为Parquet格式，便于AI训练时快速加载
7. 生成元数据文件（列信息、统计信息等）

输出：
- merged_data_v0.2.parquet: 合并后的完整数据
- data_metadata_v0.2.json: 元数据信息
- company_mapping_v0.2.json: 公司映射表
- processing_log_v0.2.txt: 处理日志
"""

import os
import re
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================== 配置区域 ===========================
# Excel文件所在目录
EXCEL_DIR = "01上市公司财务数据与日K融合-陈俊同-20251120"

# 输出目录
OUTPUT_DIR = "preprocessing_data_20251209/processed_data_20251210"

# 需要删除的列
COLS_TO_DROP = [
    "股票代码",
    "货币单位", 
    "财报数据截止日期",
    "匹配财报公告日期"
]

# 需要从第一行提取作为公司标识的列（提取后也会删除）
COLS_TO_EXTRACT = [
    "股票代码",
    "货币单位"
]

# 批次处理大小（每次处理N个文件后合并，避免内存溢出）
BATCH_SIZE = 50

# =================================================================


def parse_filename(filename):
    """
    解析文件名，提取公司信息
    格式: {序号}_{公司名}_{股票代码}_{时间戳}.xlsx
    例如: 0001_中国铝业_601600_20251115_235412.xlsx
    """
    match = re.match(r'^(\d+)_(.+?)_(.+?)_(\d{8}_\d{6})\.xlsx$', filename)
    if match:
        sequence_id = int(match.group(1))
        company_name = match.group(2)
        stock_code = match.group(3)
        timestamp = match.group(4)
        return {
            'sequence_id': sequence_id,
            'company_name': company_name,
            'stock_code': stock_code,
            'timestamp': timestamp
        }
    return None


def extract_company_info_from_data(df, filename_info):
    """
    从数据第一行提取公司信息
    """
    company_info = filename_info.copy()
    
    # 从第一行提取股票代码和货币单位
    if len(df) > 0:
        for col in COLS_TO_EXTRACT:
            if col in df.columns:
                value = df[col].iloc[0]
                if pd.notna(value):
                    # 将列名转换为key（去掉空格，转为小写）
                    key = col.replace(' ', '_').lower()
                    company_info[key] = str(value)
    
    return company_info


def process_single_file(file_path, filename_info):
    """
    处理单个Excel文件
    
    Returns:
        tuple: (DataFrame, company_info, error_message)
    """
    try:
        # 读取"财报_日K对齐"sheet
        df = pd.read_excel(file_path, sheet_name='财报_日K对齐')
        
        if df.empty:
            return None, None, "数据为空"
        
        # 提取公司信息（从第一行）
        company_info = extract_company_info_from_data(df, filename_info)
        
        # 删除不需要的列
        cols_to_drop_actual = [col for col in COLS_TO_DROP if col in df.columns]
        if cols_to_drop_actual:
            df = df.drop(columns=cols_to_drop_actual)
        
        # 添加公司标识列
        df['sequence_id'] = company_info['sequence_id']
        df['company_name'] = company_info['company_name']
        df['stock_code'] = company_info['stock_code']
        
        # 确保日期列存在且为datetime类型
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df = df.dropna(subset=['日期'])  # 删除日期为空的记录
        
        if df.empty:
            return None, None, "删除无效日期后数据为空"
        
        return df, company_info, None
        
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        return None, None, error_msg


def get_all_excel_files(excel_dir):
    """
    获取所有数字开头的Excel文件
    """
    excel_dir = Path(excel_dir)
    excel_files = []
    
    for file_path in excel_dir.glob("*.xlsx"):
        filename = file_path.name
        # 检查是否以数字开头
        if re.match(r'^\d+_', filename):
            excel_files.append(file_path)
    
    # 按文件名排序（确保顺序一致）
    excel_files.sort(key=lambda x: x.name)
    
    return excel_files


def collect_all_columns(excel_files, sample_size=10):
    """
    扫描所有文件，收集所有可能的列名
    """
    print(f"\n扫描前{sample_size}个文件，收集所有列名...")
    all_columns = set()
    
    for file_path in excel_files[:sample_size]:
        try:
            df = pd.read_excel(file_path, sheet_name='财报_日K对齐', nrows=1)
            all_columns.update(df.columns.tolist())
        except Exception as e:
            print(f"  警告: 无法读取 {file_path.name}: {e}")
    
    # 移除要删除的列
    all_columns = all_columns - set(COLS_TO_DROP)
    
    # 添加公司标识列
    all_columns.add('sequence_id')
    all_columns.add('company_name')
    all_columns.add('stock_code')
    
    # 确保日期列在最前面
    column_list = ['日期'] if '日期' in all_columns else []
    column_list.extend(sorted([col for col in all_columns if col != '日期']))
    
    print(f"  找到 {len(column_list)} 个列")
    return column_list


def process_all_files(excel_dir, output_dir):
    """
    处理所有Excel文件并合并数据
    """
    print("="*80)
    print("开始处理Excel文件")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有Excel文件
    excel_files = get_all_excel_files(excel_dir)
    total_files = len(excel_files)
    print(f"\n找到 {total_files} 个Excel文件")
    
    if total_files == 0:
        print("错误: 未找到任何Excel文件")
        return
    
    # 收集所有列名
    all_columns = collect_all_columns(excel_files)
    
    # 存储所有数据
    all_dataframes = []
    company_mapping = {}
    processing_log = []
    
    # 统计信息
    stats = {
        'total_files': total_files,
        'processed': 0,
        'success': 0,
        'failed': 0,
        'total_rows': 0,
        'errors': []
    }
    
    print(f"\n开始处理文件...")
    print(f"批次大小: {BATCH_SIZE}")
    
    # 分批处理
    for batch_start in range(0, total_files, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = excel_files[batch_start:batch_end]
        
        print(f"\n处理批次 {batch_start//BATCH_SIZE + 1}: 文件 {batch_start+1}-{batch_end}/{total_files}")
        
        batch_dataframes = []
        
        for file_path in batch_files:
            filename = file_path.name
            filename_info = parse_filename(filename)
            
            if filename_info is None:
                error_msg = f"文件名格式无法解析: {filename}"
                stats['failed'] += 1
                stats['errors'].append(error_msg)
                processing_log.append({
                    'file': filename,
                    'status': 'failed',
                    'error': error_msg
                })
                continue
            
            # 处理文件
            df, company_info, error = process_single_file(file_path, filename_info)
            
            if df is None or error:
                stats['failed'] += 1
                error_msg = error or "未知错误"
                stats['errors'].append(f"{filename}: {error_msg}")
                processing_log.append({
                    'file': filename,
                    'status': 'failed',
                    'error': error_msg
                })
                print(f"  ✗ {filename}: {error_msg}")
            else:
                stats['success'] += 1
                stats['total_rows'] += len(df)
                batch_dataframes.append(df)
                
                # 保存公司信息
                company_id = filename_info['sequence_id']
                company_mapping[company_id] = company_info
                
                processing_log.append({
                    'file': filename,
                    'status': 'success',
                    'rows': len(df),
                    'date_range': {
                        'start': str(df['日期'].min()),
                        'end': str(df['日期'].max())
                    }
                })
                
                if stats['success'] % 10 == 0:
                    print(f"  ✓ 已处理 {stats['success']} 个文件，当前: {filename} ({len(df)} 行)")
            
            stats['processed'] += 1
        
        # 合并当前批次的数据
        if batch_dataframes:
            batch_merged = pd.concat(batch_dataframes, ignore_index=True)
            all_dataframes.append(batch_merged)
            print(f"  批次合并完成: {len(batch_merged)} 行")
    
    # 合并所有数据
    print("\n合并所有数据...")
    if not all_dataframes:
        print("错误: 没有成功处理任何文件")
        return
    
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"合并完成: 总计 {len(merged_df)} 行")
    
    # 统一列结构（确保所有文件列一致）
    print("\n统一列结构...")
    for col in all_columns:
        if col not in merged_df.columns:
            merged_df[col] = None
    
    # 重新排列列顺序
    available_columns = [col for col in all_columns if col in merged_df.columns]
    merged_df = merged_df[available_columns]
    
    # 按日期排序（升序，时间序列顺序）
    print("\n按日期排序...")
    merged_df = merged_df.sort_values('日期', ascending=True).reset_index(drop=True)
    
    # 优化数据类型
    print("\n优化数据类型...")
    # 日期列保持datetime
    if '日期' in merged_df.columns:
        merged_df['日期'] = pd.to_datetime(merged_df['日期'])
    
    # 数值列转换为float32（节省内存）
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ['sequence_id']:  # sequence_id保持int
            merged_df[col] = merged_df[col].astype('float32')
    
    # sequence_id转为int32
    if 'sequence_id' in merged_df.columns:
        merged_df['sequence_id'] = merged_df['sequence_id'].astype('int32')
    
    # 填充数值列的缺失值为0（排除标识列）
    print("\n填充缺失值...")
    numeric_cols = merged_df.select_dtypes(include=['float32', 'float64']).columns
    exclude_cols = ['sequence_id']  # 标识列不填充
    fill_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # 统计填充前的缺失值
    missing_before = merged_df[fill_cols].isna().sum().sum()
    print(f"  填充前缺失值总数: {missing_before:,}")
    
    # 填充缺失值为0
    merged_df[fill_cols] = merged_df[fill_cols].fillna(0)
    
    # 验证填充结果
    missing_after = merged_df[fill_cols].isna().sum().sum()
    print(f"  填充后缺失值总数: {missing_after:,}")
    print(f"  ✓ 已填充 {len(fill_cols)} 个数值列的缺失值")
    
    # 保存为Parquet格式
    print("\n保存数据...")
    version_suffix = "v0.2"
    parquet_path = os.path.join(output_dir, f"merged_data_{version_suffix}.parquet")
    merged_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
    print(f"  ✓ 已保存: {parquet_path}")
    print(f"  文件大小: {os.path.getsize(parquet_path) / 1024 / 1024:.2f} MB")
    
    # 生成元数据
    print("\n生成元数据...")
    metadata = {
        'version': 'v0.2_20251209',
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_rows': len(merged_df),
        'total_companies': len(company_mapping),
        'date_range': {
            'start': str(merged_df['日期'].min()),
            'end': str(merged_df['日期'].max())
        },
        'columns': {
            'total': len(merged_df.columns),
            'list': merged_df.columns.tolist(),
            'date_columns': ['日期'],
            'company_columns': ['sequence_id', 'company_name', 'stock_code']
        },
        'statistics': {
            'total_files': stats['total_files'],
            'processed': stats['processed'],
            'success': stats['success'],
            'failed': stats['failed'],
            'success_rate': f"{stats['success']/stats['total_files']*100:.2f}%"
        },
        'missing_value_filling': {
            'enabled': True,
            'method': 'fillna(0)',
            'columns_filled': len(fill_cols),
            'missing_values_before': int(missing_before),
            'missing_values_after': int(missing_after)
        }
    }
    
    metadata_path = os.path.join(output_dir, f"data_metadata_{version_suffix}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已保存: {metadata_path}")
    
    # 保存公司映射表
    company_mapping_path = os.path.join(output_dir, f"company_mapping_{version_suffix}.json")
    with open(company_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(company_mapping, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已保存: {company_mapping_path}")
    
    # 保存处理日志
    log_path = os.path.join(output_dir, f"processing_log_{version_suffix}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"数据处理日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"版本: v0.2_20251209\n")
        f.write(f"总计文件数: {stats['total_files']}\n")
        f.write(f"成功处理: {stats['success']}\n")
        f.write(f"处理失败: {stats['failed']}\n")
        f.write(f"总数据行数: {stats['total_rows']}\n")
        f.write(f"\n缺失值填充:\n")
        f.write(f"  填充前缺失值总数: {missing_before:,}\n")
        f.write(f"  填充后缺失值总数: {missing_after:,}\n")
        f.write(f"  填充列数: {len(fill_cols)}\n\n")
        
        if stats['errors']:
            f.write("错误列表:\n")
            f.write("-"*80 + "\n")
            for error in stats['errors'][:50]:  # 只显示前50个错误
                f.write(f"{error}\n")
            if len(stats['errors']) > 50:
                f.write(f"... 还有 {len(stats['errors']) - 50} 个错误未显示\n")
    
    print(f"  ✓ 已保存: {log_path}")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("处理完成！")
    print("="*80)
    print(f"总计文件数: {stats['total_files']}")
    print(f"成功处理: {stats['success']}")
    print(f"处理失败: {stats['failed']}")
    print(f"总数据行数: {len(merged_df):,}")
    print(f"公司数量: {len(company_mapping)}")
    print(f"日期范围: {merged_df['日期'].min()} 到 {merged_df['日期'].max()}")
    print(f"列数: {len(merged_df.columns)}")
    print(f"缺失值填充: {missing_before:,} -> {missing_after:,}")
    print(f"\n输出目录: {output_dir}")


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.parent
    excel_dir = script_dir / EXCEL_DIR
    output_dir = script_dir / OUTPUT_DIR
    
    if not excel_dir.exists():
        print(f"错误: Excel目录不存在: {excel_dir}")
        return
    
    print(f"Excel目录: {excel_dir}")
    print(f"输出目录: {output_dir}")
    
    process_all_files(excel_dir, output_dir)


if __name__ == "__main__":
    main()
