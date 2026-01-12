#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据完整性检查脚本
版本: v0.1
日期: 20251218173000

功能:
验证从原始parquet文件到最终.pt文件的数据转换链路是否正确

检查内容:
1. 随机抽样验证（训练集和验证集各5个样本）
2. 追溯原始数据源
3. 对比转换后的数据
4. 检查归一化计算是否正确
"""

import torch
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置路径
TRAIN_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/train_v0.3.pt"
VAL_PT_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/val_v0.3.pt"
COMPANY_STATS_FILE = project_root / "data/processed/preprocess_data_v0.3_20251218150457/company_stats_v0.3.json"
TRAIN_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/train_samples_index.parquet"
VAL_INDEX_FILE = project_root / "data/processed/roll_generate_index_v0.2_20251216_092024/val_samples_index.parquet"

# 检查样本数量
NUM_SAMPLES_TO_CHECK = 5


def load_pt_data(pt_file: Path) -> Dict:
    """加载.pt文件"""
    print(f"加载 {pt_file.name}...")
    data = torch.load(pt_file)
    print(f"  - 样本数: {data['metadata']['num_samples']}")
    print(f"  - 特征数: {data['metadata']['num_features']}")
    print(f"  - 序列长度: {data['metadata']['seq_len']}")
    return data


def load_company_stats(stats_file: Path) -> Dict:
    """加载公司统计量"""
    print(f"\n加载公司统计量: {stats_file.name}...")
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    print(f"  - 公司数量: {len(stats)}")
    return stats


def manual_normalize(value: float, mean: float, std: float) -> float:
    """手动标准化归一化"""
    if std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std


def check_sample(
    sample_idx: int,
    pt_data: Dict,
    index_df: pd.DataFrame,
    company_stats: Dict,
    dataset_name: str
) -> Dict:
    """
    检查单个样本的数据完整性
    
    Returns:
        检查结果字典
    """
    print(f"\n{'='*80}")
    print(f"检查 {dataset_name} 样本 #{sample_idx}")
    print(f"{'='*80}")
    
    # 从.pt中获取数据
    X_tensor = pt_data['X'][sample_idx]  # shape: (seq_len, num_features)
    y_tensor = pt_data['y'][sample_idx]  # shape: (1,)
    
    # 从索引文件获取元数据
    sample_info = index_df.iloc[sample_idx]
    sample_id = sample_info['sample_id']
    source_file = sample_info['source_file']
    input_row_start = sample_info['input_row_start']
    input_row_end = sample_info['input_row_end']
    target_row = sample_info['target_row']
    
    print(f"\n样本信息:")
    print(f"  - Sample ID: {sample_id}")
    print(f"  - 公司: {sample_info['company_name']}")
    print(f"  - 股票代码: {sample_info['stock_code']}")
    print(f"  - 输入窗口: {sample_info['start_date']} ~ {sample_info['input_end_date']}")
    print(f"  - 目标日期: {sample_info['target_date']}")
    print(f"  - 原始文件: {Path(source_file).name}")
    print(f"  - 行范围: {input_row_start} ~ {input_row_end} (目标: {target_row})")
    
    # 从原始parquet文件读取数据
    print(f"\n从原始文件读取数据...")
    df_original = pd.read_parquet(source_file)
    
    # 获取输入窗口的原始数据
    df_input_original = df_original.iloc[input_row_start:input_row_end+1].copy()
    
    # 获取目标值的原始数据
    target_original = df_original.iloc[target_row]
    
    # 获取特征列和归一化配置
    feature_columns = pt_data['metadata']['feature_columns']
    normalize_columns = pt_data['metadata']['normalize_columns']
    no_normalize_columns = pt_data['metadata']['no_normalize_columns']
    target_column = pt_data['metadata']['target_column']
    
    print(f"\n特征配置:")
    print(f"  - 总特征数: {len(feature_columns)}")
    print(f"  - 需要归一化: {len(normalize_columns)}")
    print(f"  - 不归一化: {len(no_normalize_columns)}")
    print(f"  - 目标列: {target_column}")
    
    # 获取公司统计量
    company_stats_key = str(source_file)
    if company_stats_key not in company_stats:
        print(f"\n❌ 错误: 未找到公司统计量: {company_stats_key}")
        return {'status': 'error', 'message': '未找到公司统计量'}
    
    stats = company_stats[company_stats_key]
    
    # 检查结果
    results = {
        'sample_id': sample_id,
        'company': sample_info['company_name'],
        'status': 'ok',
        'errors': [],
        'warnings': [],
        'details': {}
    }
    
    # 检查序列长度
    expected_seq_len = input_row_end - input_row_start + 1
    actual_seq_len = X_tensor.shape[0]
    if expected_seq_len != actual_seq_len:
        results['errors'].append(f"序列长度不匹配: 期望 {expected_seq_len}, 实际 {actual_seq_len}")
    
    # 检查特征数量
    if X_tensor.shape[1] != len(feature_columns):
        results['errors'].append(f"特征数量不匹配: 期望 {len(feature_columns)}, 实际 {X_tensor.shape[1]}")
    
    # 检查每个特征列
    print(f"\n开始逐列检查...")
    column_errors = []
    column_matches = []
    
    for col_idx, col_name in enumerate(feature_columns):
        if col_name not in df_input_original.columns:
            column_errors.append(f"列 '{col_name}' 不存在于原始数据中")
            continue
        
        # 获取原始列数据
        original_values = df_input_original[col_name].values
        
        # 处理NaN（与预处理脚本一致）
        original_values = pd.Series(original_values).fillna(0.0).values
        
        # 获取.pt中的数据
        pt_values = X_tensor[:, col_idx].numpy()
        
        # 如果该列需要归一化，进行手动归一化
        if col_name in normalize_columns:
            if col_name in stats:
                mean = stats[col_name]['mean']
                std = stats[col_name]['std']
                
                # 手动归一化原始数据
                expected_values = np.array([manual_normalize(v, mean, std) for v in original_values])
            else:
                results['warnings'].append(f"列 '{col_name}' 在统计量中不存在")
                expected_values = original_values
        else:
            # 不归一化的列，直接使用原始值
            expected_values = original_values
        
        # 对比数据
        # 由于浮点数精度问题，使用相对误差
        if len(expected_values) != len(pt_values):
            column_errors.append(f"列 '{col_name}' 长度不匹配: 期望 {len(expected_values)}, 实际 {len(pt_values)}")
        else:
            # 计算差异
            diff = np.abs(expected_values - pt_values)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            # 检查是否匹配（允许1e-5的误差）
            if max_diff < 1e-5:
                column_matches.append(col_name)
            else:
                # 检查是否是相对误差可接受
                non_zero_mask = np.abs(expected_values) > 1e-10
                if np.any(non_zero_mask):
                    rel_diff = diff[non_zero_mask] / (np.abs(expected_values[non_zero_mask]) + 1e-10)
                    max_rel_diff = np.max(rel_diff)
                    
                    if max_rel_diff < 1e-4:  # 0.01% 相对误差
                        column_matches.append(col_name)
                    else:
                        column_errors.append(
                            f"列 '{col_name}' 数据不匹配 - "
                            f"最大绝对误差: {max_diff:.6e}, "
                            f"平均绝对误差: {mean_diff:.6e}, "
                            f"最大相对误差: {max_rel_diff:.6e}"
                        )
                        # 显示前几个不匹配的值
                        mismatch_indices = np.where(rel_diff > 1e-4)[0][:3]
                        for idx in mismatch_indices:
                            actual_idx = np.where(non_zero_mask)[0][idx]
                            print(f"    行 {actual_idx}: 期望 {expected_values[actual_idx]:.6f}, 实际 {pt_values[actual_idx]:.6f}")
                else:
                    # 都是零或接近零
                    if max_diff < 1e-5:
                        column_matches.append(col_name)
                    else:
                        column_errors.append(
                            f"列 '{col_name}' 数据不匹配 - "
                            f"最大绝对误差: {max_diff:.6e}"
                        )
    
    print(f"\n列检查结果:")
    print(f"  - 匹配的列: {len(column_matches)}/{len(feature_columns)}")
    print(f"  - 错误的列: {len(column_errors)}")
    
    if column_errors:
        print(f"\n列错误详情:")
        for err in column_errors[:10]:  # 最多显示10个
            print(f"  - {err}")
        if len(column_errors) > 10:
            print(f"  ... 还有 {len(column_errors) - 10} 个错误")
    
    # 检查目标值
    print(f"\n检查目标值...")
    target_original_value = target_original[target_column]
    target_pt_value = y_tensor.item()
    
    print(f"  - 原始目标值: {target_original_value}")
    print(f"  - .pt目标值: {target_pt_value}")
    
    target_diff = abs(target_original_value - target_pt_value)
    if target_diff < 1e-5:
        print(f"  - ✅ 目标值匹配")
    else:
        rel_diff = target_diff / (abs(target_original_value) + 1e-10)
        if rel_diff < 1e-4:
            print(f"  - ✅ 目标值匹配 (相对误差: {rel_diff:.6e})")
        else:
            error_msg = f"目标值不匹配 - 绝对误差: {target_diff:.6e}, 相对误差: {rel_diff:.6e}"
            print(f"  - ❌ {error_msg}")
            results['errors'].append(error_msg)
    
    # 汇总结果
    results['errors'].extend(column_errors)
    results['details'] = {
        'matched_columns': len(column_matches),
        'total_columns': len(feature_columns),
        'error_columns': len(column_errors),
        'target_match': target_diff < 1e-5 or (target_diff / (abs(target_original_value) + 1e-10) < 1e-4)
    }
    
    if results['errors']:
        results['status'] = 'error'
        print(f"\n❌ 样本检查失败: {len(results['errors'])} 个错误")
    else:
        print(f"\n✅ 样本检查通过")
    
    return results


def main():
    """主函数"""
    print("="*80)
    print("数据完整性检查")
    print("="*80)
    
    # 加载数据
    train_data = load_pt_data(TRAIN_PT_FILE)
    val_data = load_pt_data(VAL_PT_FILE)
    company_stats = load_company_stats(COMPANY_STATS_FILE)
    
    print(f"\n加载索引文件...")
    train_index = pd.read_parquet(TRAIN_INDEX_FILE)
    val_index = pd.read_parquet(VAL_INDEX_FILE)
    print(f"  - 训练索引: {len(train_index)} 条")
    print(f"  - 验证索引: {len(val_index)} 条")
    
    # 随机选择样本
    train_sample_indices = random.sample(range(len(train_index)), min(NUM_SAMPLES_TO_CHECK, len(train_index)))
    val_sample_indices = random.sample(range(len(val_index)), min(NUM_SAMPLES_TO_CHECK, len(val_index)))
    
    print(f"\n将检查以下样本:")
    print(f"  - 训练集: {train_sample_indices}")
    print(f"  - 验证集: {val_sample_indices}")
    
    # 检查训练集样本
    train_results = []
    for idx in train_sample_indices:
        result = check_sample(idx, train_data, train_index, company_stats, "训练集")
        train_results.append(result)
    
    # 检查验证集样本
    val_results = []
    for idx in val_sample_indices:
        result = check_sample(idx, val_data, val_index, company_stats, "验证集")
        val_results.append(result)
    
    # 汇总结果
    print("\n" + "="*80)
    print("检查结果汇总")
    print("="*80)
    
    all_results = train_results + val_results
    total_samples = len(all_results)
    passed_samples = sum(1 for r in all_results if r['status'] == 'ok')
    failed_samples = total_samples - passed_samples
    
    print(f"\n总样本数: {total_samples}")
    print(f"通过: {passed_samples}")
    print(f"失败: {failed_samples}")
    
    if failed_samples > 0:
        print(f"\n失败样本详情:")
        for result in all_results:
            if result['status'] == 'error':
                print(f"\n  样本: {result['sample_id']} ({result['company']})")
                print(f"  错误数: {len(result['errors'])}")
                for err in result['errors'][:5]:
                    print(f"    - {err}")
                if len(result['errors']) > 5:
                    print(f"    ... 还有 {len(result['errors']) - 5} 个错误")
    
    # 列级别统计
    print(f"\n列级别统计:")
    if all_results:
        avg_matched = np.mean([r['details']['matched_columns'] for r in all_results if 'details' in r])
        avg_total = np.mean([r['details']['total_columns'] for r in all_results if 'details' in r])
        print(f"  - 平均匹配列数: {avg_matched:.1f} / {avg_total:.1f}")
        print(f"  - 匹配率: {avg_matched/avg_total*100:.2f}%")
    
    print("\n" + "="*80)
    if failed_samples == 0:
        print("✅ 所有样本检查通过，数据转换链路正确！")
    else:
        print("❌ 发现数据不匹配问题，请检查预处理流程！")
    print("="*80)


if __name__ == '__main__':
    main()
