#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打印特征名称和位置索引
版本: v1.0
日期: 20260107

从预处理数据文件中读取特征列名，打印每个特征名称和对应的位置索引
用于区分内生数据和宏观数据
"""

import torch
from pathlib import Path
import sys


def print_feature_names_and_indices(pt_file_path: str):
    """
    打印特征名称和位置索引
    
    Args:
        pt_file_path: 预处理数据文件路径（.pt文件）
    """
    pt_file_path = Path(pt_file_path)
    
    if not pt_file_path.exists():
        print(f"错误: 文件不存在: {pt_file_path}")
        return
    
    print("=" * 80)
    print(f"加载预处理数据文件: {pt_file_path.name}")
    print("=" * 80)
    
    # 加载数据
    try:
        data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"错误: 无法加载文件: {e}")
        return
    
    # 获取metadata
    metadata = data.get('metadata', {})
    
    if not metadata:
        print("错误: metadata为空")
        return
    
    # 获取特征列名（优先使用 feature_columns_example，否则用 feature_columns）
    feature_columns = metadata.get('feature_columns_example')
    if feature_columns is None:
        feature_columns = metadata.get('feature_columns')
    
    if feature_columns is None:
        print("错误: metadata中未找到特征列名（feature_columns_example 或 feature_columns）")
        print(f"metadata中的键: {list(metadata.keys())}")
        return
    
    # 获取数据信息
    num_features = metadata.get('num_features', len(feature_columns))
    num_samples = metadata.get('num_samples', 0)
    seq_len = metadata.get('seq_len', 0)
    
    print(f"\n数据信息:")
    print(f"  样本数: {num_samples}")
    print(f"  序列长度: {seq_len}")
    print(f"  特征数量: {num_features}")
    print(f"  特征列名数量: {len(feature_columns)}")
    
    # 验证特征数量
    if num_features != len(feature_columns):
        print(f"\n警告: 特征数量 ({num_features}) 与特征列名数量 ({len(feature_columns)}) 不一致")
    
    # 检查是否有 sample_feature_columns（每个样本的列名）
    sample_feature_columns = metadata.get('sample_feature_columns')
    if sample_feature_columns is not None:
        print(f"\n注意: 检测到 sample_feature_columns（每个样本的列名）")
        print(f"  样本列名数量: {len(sample_feature_columns)}")
        print(f"  特征列名说明: {metadata.get('feature_columns_note', '无')}")
        
        # 检查前几个样本的列名是否相同
        if len(sample_feature_columns) > 0:
            first_sample_cols = sample_feature_columns[0]
            print(f"\n第一个样本的特征列名（共 {len(first_sample_cols)} 个）:")
            for idx, col_name in enumerate(first_sample_cols):
                print(f"  位置 {idx:3d}: {col_name}")
            
            # 检查是否所有样本的列名顺序相同
            all_same = True
            for i, sample_cols in enumerate(sample_feature_columns[1:6]):  # 检查前6个样本
                if sample_cols != first_sample_cols:
                    all_same = False
                    print(f"\n警告: 样本 {i+1} 的列名顺序与第一个样本不同")
                    break
            
            if all_same:
                print(f"\n✓ 前6个样本的特征列名顺序相同（假设所有样本顺序相同）")
                print(f"  使用第一个样本的列名作为标准")
    
    print("\n" + "=" * 80)
    print("特征名称和位置索引:")
    print("=" * 80)
    print(f"{'位置':<8} {'特征名称'}")
    print("-" * 80)
    
    # 打印每个特征名称和位置索引
    for idx, col_name in enumerate(feature_columns):
        print(f"{idx:<8} {col_name}")
    
    print("-" * 80)
    print(f"总计: {len(feature_columns)} 个特征")
    print("=" * 80)
    
    # 提供配置建议
    print("\n配置建议:")
    print("在 timexer_mlp_config.yaml 中可以这样配置:")
    print("  model:")
    print("    # 方式1: 使用范围（推荐）")
    print("    endogenous_start: 0")
    print("    endogenous_end: XX    # 根据上面的列表，填入内生数据的最后一个位置")
    print("    exogenous_start: YY   # 根据上面的列表，填入宏观数据的第一个位置")
    print("    exogenous_end: ZZ     # 根据上面的列表，填入宏观数据的最后一个位置")
    print("")
    print("    # 方式2: 使用索引列表（如果需要跳过某些特征）")
    print("    endogenous_indices: [0, 1, 2, ..., XX]")
    print("    exogenous_indices: [YY, YY+1, ..., ZZ]")


def main():
    """主函数"""
    # 默认使用训练脚本中指定的预处理数据目录
    project_root = Path(__file__).parent.parent.parent
    default_preprocessed_dir = project_root / "data" / "processed" / "preprocess_data_v0.5_20251225210740"
    
    # 从命令行参数获取文件路径，或使用默认路径
    if len(sys.argv) > 1:
        pt_file_path = sys.argv[1]
    else:
        # 查找训练集文件
        train_files = list(default_preprocessed_dir.glob("train_*.pt"))
        if train_files:
            pt_file_path = train_files[0]
            print(f"使用默认训练集文件: {pt_file_path}")
        else:
            print("错误: 未找到预处理数据文件")
            print(f"请指定文件路径，或确保以下目录存在训练集文件:")
            print(f"  {default_preprocessed_dir}")
            return
    
    print_feature_names_and_indices(pt_file_path)


if __name__ == '__main__':
    main()
