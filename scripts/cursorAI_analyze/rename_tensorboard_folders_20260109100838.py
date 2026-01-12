#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重命名 experiments 文件夹中 tensorboard 子文件夹的脚本
将 tensorboard 子文件夹名称修改为与父模型文件夹相同的名称
"""

import os
from pathlib import Path
from datetime import datetime

def rename_tensorboard_folders(experiments_dir: Path):
    """
    重命名 experiments 文件夹中所有 tensorboard 子文件夹
    
    Args:
        experiments_dir: experiments 文件夹路径
    """
    experiments_dir = Path(experiments_dir)
    
    if not experiments_dir.exists():
        print(f"错误: 文件夹 {experiments_dir} 不存在")
        return
    
    renamed_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"开始处理 experiments 文件夹: {experiments_dir}")
    print("=" * 80)
    
    # 遍历所有模型文件夹
    for model_folder in sorted(experiments_dir.iterdir()):
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        tensorboard_dir = model_folder / "tensorboard"
        
        # 检查是否存在 tensorboard 文件夹
        if not tensorboard_dir.exists() or not tensorboard_dir.is_dir():
            continue
        
        print(f"\n处理模型文件夹: {model_name}")
        
        # 获取 tensorboard 文件夹下的所有子文件夹
        subfolders = [f for f in tensorboard_dir.iterdir() if f.is_dir()]
        
        if not subfolders:
            print(f"  - tensorboard 文件夹为空，跳过")
            skipped_count += 1
            continue
        
        # 处理每个子文件夹
        for subfolder in subfolders:
            subfolder_name = subfolder.name
            
            # 如果子文件夹名称已经与模型文件夹名称相同，跳过
            if subfolder_name == model_name:
                print(f"  - 子文件夹 '{subfolder_name}' 已正确命名，跳过")
                skipped_count += 1
                continue
            
            # 重命名子文件夹
            new_subfolder_path = tensorboard_dir / model_name
            
            # 如果目标文件夹已存在（可能是之前重命名过的），先检查
            if new_subfolder_path.exists() and new_subfolder_path != subfolder:
                print(f"  - 警告: 目标文件夹 '{model_name}' 已存在，跳过重命名 '{subfolder_name}'")
                skipped_count += 1
                continue
            
            try:
                print(f"  - 重命名: '{subfolder_name}' -> '{model_name}'")
                subfolder.rename(new_subfolder_path)
                renamed_count += 1
                print(f"    ✓ 成功")
            except Exception as e:
                print(f"    ✗ 失败: {str(e)}")
                error_count += 1
    
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"  成功重命名: {renamed_count} 个文件夹")
    print(f"  跳过: {skipped_count} 个文件夹")
    print(f"  错误: {error_count} 个文件夹")


def main():
    """主函数"""
    # experiments 文件夹路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    experiments_dir = project_root / "experiments"
    
    print("=" * 80)
    print("TensorBoard 文件夹重命名脚本")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    rename_tensorboard_folders(experiments_dir)


if __name__ == "__main__":
    main()
