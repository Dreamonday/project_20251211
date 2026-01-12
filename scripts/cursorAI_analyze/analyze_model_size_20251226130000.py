#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析模型文件大小差异和TSMixer参数量大的原因
版本: 20251226130000
"""

import torch
import sys
from pathlib import Path
import importlib.util

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_checkpoint_size(checkpoint_path: Path):
    """分析checkpoint文件各部分的大小"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n分析checkpoint文件: {checkpoint_path.name}")
    print("=" * 80)
    
    total_size = 0
    
    # 模型参数
    if 'model_state_dict' in checkpoint:
        model_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        model_size = model_params * 4  # float32 = 4 bytes
        total_size += model_size
        print(f"模型参数:")
        print(f"  参数量: {model_params:,}")
        print(f"  大小: {model_size / 1024 / 1024:.2f} MB")
    
    # 优化器状态
    if 'optimizer_state_dict' in checkpoint:
        opt_params = sum(p.numel() if isinstance(p, torch.Tensor) else 0 
                         for state in checkpoint['optimizer_state_dict']['state'].values() 
                         for p in state.values() if isinstance(p, torch.Tensor))
        opt_size = opt_params * 4
        total_size += opt_size
        print(f"\n优化器状态:")
        print(f"  参数量: {opt_params:,}")
        print(f"  大小: {opt_size / 1024 / 1024:.2f} MB")
        if 'model_state_dict' in checkpoint:
            ratio = opt_params / model_params
            print(f"  相对于模型参数: {ratio:.2f}x")
    
    # 调度器状态
    if 'scheduler_state_dict' in checkpoint:
        sched_size = len(str(checkpoint['scheduler_state_dict']))
        total_size += sched_size
        print(f"\n调度器状态:")
        print(f"  大小: {sched_size / 1024 / 1024:.2f} MB (估算)")
    
    # 训练历史
    if 'train_history' in checkpoint:
        import json
        hist_size = len(json.dumps(checkpoint['train_history']).encode('utf-8'))
        total_size += hist_size
        print(f"\n训练历史:")
        print(f"  大小: {hist_size / 1024 / 1024:.2f} MB (估算)")
    
    print(f"\n总计估算大小: {total_size / 1024 / 1024:.2f} MB")
    
    return model_params if 'model_state_dict' in checkpoint else None


def analyze_tsmixer_structure(seq_len=500, n_features=64, n_blocks=2):
    """分析TSMixer模型的结构和参数量"""
    print("\n" + "=" * 80)
    print("TSMixer模型结构分析")
    print("=" * 80)
    print(f"输入: seq_len={seq_len}, n_features={n_features}, n_blocks={n_blocks}")
    print()
    
    # 计算各部分参数量
    print("1. TimeMixingBlock (每个块):")
    print("-" * 80)
    
    # TimeMixingBlock参数量
    # up1: seq_len → 512
    up1_params = seq_len * 512 + 512
    print(f"   up1 ({seq_len}→512): {up1_params:,}")
    
    # res1: ResidualBlock(512) = 5层 × (512×512 + 512)
    res1_params = 5 * (512 * 512 + 512)
    print(f"   res1 (ResidualBlock 512, 5层): {res1_params:,}")
    
    # up2: 512 → 1024
    up2_params = 512 * 1024 + 1024
    print(f"   up2 (512→1024): {up2_params:,}")
    
    # res2: ResidualBlock(1024) = 5层 × (1024×1024 + 1024)
    res2_params = 5 * (1024 * 1024 + 1024)
    print(f"   res2 (ResidualBlock 1024, 5层): {res2_params:,}")
    
    # down1: 1024 → 512
    down1_params = 1024 * 512 + 512
    print(f"   down1 (1024→512): {down1_params:,}")
    
    # res3: ResidualBlock(512)
    res3_params = res1_params
    print(f"   res3 (ResidualBlock 512, 5层): {res3_params:,}")
    
    # down2: 512 → seq_len
    down2_params = 512 * seq_len + seq_len
    print(f"   down2 (512→{seq_len}): {down2_params:,}")
    
    time_mixing_params = up1_params + res1_params + up2_params + res2_params + down1_params + res3_params + down2_params
    print(f"   总计一个TimeMixingBlock: {time_mixing_params:,}")
    
    print("\n2. FeatureMixingBlock (每个块):")
    print("-" * 80)
    
    # FeatureMixingBlock参数量
    # up1: n_features → 512
    feat_up1_params = n_features * 512 + 512
    print(f"   up1 ({n_features}→512): {feat_up1_params:,}")
    
    # res1: ResidualBlock(512)
    feat_res1_params = res1_params
    print(f"   res1 (ResidualBlock 512, 5层): {feat_res1_params:,}")
    
    # up2: 512 → 1024
    feat_up2_params = up2_params
    print(f"   up2 (512→1024): {feat_up2_params:,}")
    
    # res2: ResidualBlock(1024)
    feat_res2_params = res2_params
    print(f"   res2 (ResidualBlock 1024, 5层): {feat_res2_params:,}")
    
    # down1: 1024 → 512
    feat_down1_params = down1_params
    print(f"   down1 (1024→512): {feat_down1_params:,}")
    
    # res3: ResidualBlock(512)
    feat_res3_params = res1_params
    print(f"   res3 (ResidualBlock 512, 5层): {feat_res3_params:,}")
    
    # down2: 512 → n_features
    feat_down2_params = 512 * n_features + n_features
    print(f"   down2 (512→{n_features}): {feat_down2_params:,}")
    
    feature_mixing_params = feat_up1_params + feat_res1_params + feat_up2_params + feat_res2_params + feat_down1_params + feat_res3_params + feat_down2_params
    print(f"   总计一个FeatureMixingBlock: {feature_mixing_params:,}")
    
    print("\n3. TSMixerBlock (每个块 = TimeMixing + FeatureMixing):")
    print("-" * 80)
    tsmixer_block_params = time_mixing_params + feature_mixing_params
    print(f"   一个TSMixerBlock: {tsmixer_block_params:,}")
    
    print("\n4. 所有TSMixer块:")
    print("-" * 80)
    all_blocks_params = tsmixer_block_params * n_blocks
    print(f"   {n_blocks}个TSMixerBlock: {all_blocks_params:,}")
    
    print("\n5. 输出投影层:")
    print("-" * 80)
    # output_res1: ResidualBlock(n_features)
    output_res1_params = 5 * (n_features * n_features + n_features)
    print(f"   output_res1 (ResidualBlock {n_features}, 5层): {output_res1_params:,}")
    
    # output_proj1: n_features → 64
    output_proj1_params = n_features * 64 + 64
    print(f"   output_proj1 ({n_features}→64): {output_proj1_params:,}")
    
    # output_res2: ResidualBlock(64)
    output_res2_params = 5 * (64 * 64 + 64)
    print(f"   output_res2 (ResidualBlock 64, 5层): {output_res2_params:,}")
    
    # output_proj2: 64 → 32
    output_proj2_params = 64 * 32 + 32
    print(f"   output_proj2 (64→32): {output_proj2_params:,}")
    
    # output_res3: ResidualBlock(32)
    output_res3_params = 5 * (32 * 32 + 32)
    print(f"   output_res3 (ResidualBlock 32, 5层): {output_res3_params:,}")
    
    # output_proj3: 32 → 16
    output_proj3_params = 32 * 16 + 16
    print(f"   output_proj3 (32→16): {output_proj3_params:,}")
    
    # output_proj4: 16 → 1
    output_proj4_params = 16 * 1 + 1
    print(f"   output_proj4 (16→1): {output_proj4_params:,}")
    
    output_params = output_res1_params + output_proj1_params + output_res2_params + output_proj2_params + output_res3_params + output_proj3_params + output_proj4_params
    print(f"   输出投影层总计: {output_params:,}")
    
    print("\n6. 总计:")
    print("-" * 80)
    total_params = all_blocks_params + output_params
    print(f"   总参数量: {total_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n7. 参数量大的主要原因:")
    print("-" * 80)
    print(f"   - TimeMixingBlock中的ResidualBlock(1024): {res2_params:,} 参数")
    print(f"   - FeatureMixingBlock中的ResidualBlock(1024): {feat_res2_params:,} 参数")
    print(f"   - 每个ResidualBlock有5层全连接，参数量 = 5 × (dim² + dim)")
    print(f"   - seq_len=500时，TimeMixingBlock的Linear层参数量很大")
    
    return total_params


def suggest_reductions():
    """提供减少TSMixer模型大小的建议"""
    print("\n" + "=" * 80)
    print("减少TSMixer模型大小的建议")
    print("=" * 80)
    
    suggestions = [
        "1. 减少ResidualBlock的层数:",
        "   - 当前: 5层全连接",
        "   - 建议: 改为3层，可减少约40%的参数量",
        "",
        "2. 降低中间维度:",
        "   - 当前: 512 → 1024 → 512",
        "   - 建议: 改为 256 → 512 → 256，可减少约75%的参数量",
        "",
        "3. 减少TSMixer块的数量:",
        "   - 当前: n_blocks=2",
        "   - 建议: 改为1，可减少约50%的参数量",
        "",
        "4. 简化输出投影层:",
        "   - 当前: 3个ResidualBlock + 4个Linear层",
        "   - 建议: 移除ResidualBlock，只保留Linear层，可减少约70%的参数量",
        "",
        "5. 减少TimeMixingBlock的维度:",
        "   - 当前: seq_len(500) → 512 → 1024 → 512 → seq_len(500)",
        "   - 建议: 改为 seq_len(500) → 256 → 512 → 256 → seq_len(500)",
        "",
        "6. 组合优化方案:",
        "   - ResidualBlock层数: 5 → 3",
        "   - 中间维度: 1024 → 512",
        "   - TSMixer块数: 2 → 1",
        "   - 简化输出投影层",
        "   - 预计可减少约60-70%的参数量"
    ]
    
    for suggestion in suggestions:
        print(suggestion)


def main():
    """主函数"""
    print("=" * 80)
    print("模型文件大小和结构分析")
    print("=" * 80)
    
    # 分析实际checkpoint文件
    checkpoint_files = [
        project_root / "experiments" / "itransformer_v0.1_20251226091316_20251225210740" / "checkpoints" / "best_model.pth",
        project_root / "experiments" / "tsmixer_v0.2_20251230113226_20251225210740" / "checkpoints" / "best_model.pth"
    ]
    
    for checkpoint_path in checkpoint_files:
        if checkpoint_path.exists():
            analyze_checkpoint_size(checkpoint_path)
    
    # 分析TSMixer结构
    analyze_tsmixer_structure(seq_len=500, n_features=64, n_blocks=2)
    
    # 提供减少模型大小的建议
    suggest_reductions()
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
