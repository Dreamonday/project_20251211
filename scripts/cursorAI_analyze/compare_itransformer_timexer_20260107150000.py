#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比分析 iTransformer 和 TimeXer-MLP 模型的参数量、架构复杂度和训练速度
版本: v0.1
日期: 20260107

分析为什么 TimeXer-MLP 训练速度比 iTransformer 快很多
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import yaml
import importlib.util
from datetime import datetime

# 添加项目路径
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # scripts -> project_20251211
sys.path.insert(0, str(project_root))


def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_model_size(model: nn.Module) -> tuple[int, int]:
    """
    计算模型参数量和存储大小
    
    Returns:
        (参数量, 存储大小字节数)
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = num_params * 4  # 假设float32，每个参数4字节
    return num_params, size_bytes


def format_size(size_bytes: int) -> str:
    """格式化存储大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.2f} MB"


def analyze_model_architecture(model: nn.Module, model_name: str):
    """分析模型架构复杂度"""
    print(f"\n{'='*80}")
    print(f"{model_name} 架构分析")
    print(f"{'='*80}")
    
    # 统计不同类型的层
    layer_stats = {
        'Linear': 0,
        'LayerNorm': 0,
        'MultiheadAttention': 0,
        'Dropout': 0,
        'GELU': 0,
        'ReLU': 0,
        '其他': 0
    }
    
    total_params = 0
    layer_details = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            module_type = type(module).__name__
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            
            if param_count > 0:
                total_params += param_count
                layer_details.append((name, module_type, param_count))
            
            if 'Linear' in module_type:
                layer_stats['Linear'] += 1
            elif 'LayerNorm' in module_type:
                layer_stats['LayerNorm'] += 1
            elif 'MultiheadAttention' in module_type or 'Attention' in module_type:
                layer_stats['MultiheadAttention'] += 1
            elif 'Dropout' in module_type:
                layer_stats['Dropout'] += 1
            elif 'GELU' in module_type:
                layer_stats['GELU'] += 1
            elif 'ReLU' in module_type:
                layer_stats['ReLU'] += 1
            else:
                layer_stats['其他'] += 1
    
    print(f"\n层类型统计:")
    for layer_type, count in layer_stats.items():
        if count > 0:
            print(f"  {layer_type}: {count}")
    
    print(f"\n参数量统计:")
    print(f"  总参数量: {total_params:,}")
    
    # 显示参数量最大的10个模块
    print(f"\n参数量最大的10个模块:")
    sorted_details = sorted(layer_details, key=lambda x: x[2], reverse=True)[:10]
    for name, module_type, param_count in sorted_details:
        print(f"  {name[:60]:<60} {module_type:>20} {param_count:>12,}")
    
    return layer_stats, total_params


def compare_computational_complexity():
    """比较计算复杂度"""
    print(f"\n{'='*80}")
    print("计算复杂度对比")
    print(f"{'='*80}")
    
    # iTransformer配置
    seq_len_itransformer = 100
    d_model = 512
    n_heads = 4
    n_layers = 6
    
    # TimeXer-MLP配置
    seq_len_timexer = 500
    n_features = 64
    endogenous_features = 44
    exogenous_features = 20
    
    print(f"\n1. Attention机制复杂度 (iTransformer):")
    # Multi-Head Attention: O(seq_len² * d_model)
    attention_ops_per_layer = seq_len_itransformer * seq_len_itransformer * d_model
    total_attention_ops = attention_ops_per_layer * n_layers
    print(f"   单层Attention操作数: {attention_ops_per_layer:,}")
    print(f"   {n_layers}层总操作数: {total_attention_ops:,}")
    print(f"   复杂度: O(seq_len² × d_model × n_layers) = O({seq_len_itransformer}² × {d_model} × {n_layers})")
    
    print(f"\n2. MLP操作复杂度 (TimeXer-MLP):")
    # MLP时间混合: O(seq_len × n_features)
    time_mixing_ops = seq_len_timexer * n_features
    # MLP特征混合: O(seq_len × n_features)
    feature_mixing_ops = seq_len_timexer * n_features
    total_mlp_ops = (time_mixing_ops + feature_mixing_ops) * (3 + 2)  # 内生3块 + 宏观2块
    print(f"   单块MLP操作数: {time_mixing_ops + feature_mixing_ops:,}")
    print(f"   5块总操作数: {total_mlp_ops:,}")
    print(f"   复杂度: O(seq_len × n_features × n_blocks) = O({seq_len_timexer} × {n_features} × 5)")
    
    print(f"\n3. 复杂度对比:")
    ratio = total_attention_ops / total_mlp_ops if total_mlp_ops > 0 else float('inf')
    print(f"   iTransformer操作数 / TimeXer-MLP操作数 ≈ {ratio:.2f}x")
    print(f"   注意: 这是粗略估计，实际还涉及FFN、ResNet等操作")


def main():
    """主函数"""
    print("=" * 80)
    print("iTransformer vs TimeXer-MLP 模型对比分析")
    print("=" * 80)
    
    # 导入模型
    print("\n加载模型...")
    
    # iTransformer
    itransformer_models_path = project_root / "src" / "models" / "v0.1_20251212"
    itransformer_decoder_path = itransformer_models_path / "itransformer_decoder.py"
    if not itransformer_decoder_path.exists():
        raise FileNotFoundError(f"找不到iTransformer模型文件: {itransformer_decoder_path}")
    itransformer_module = _load_module(itransformer_decoder_path, "itransformer_decoder")
    iTransformerDecoder = itransformer_module.iTransformerDecoder
    
    # TimeXer-MLP
    timexer_models_path = project_root / "src" / "models" / "v0.5_20260107"
    timexer_mlp_path = timexer_models_path / "timexer_mlp.py"
    if not timexer_mlp_path.exists():
        raise FileNotFoundError(f"找不到TimeXer-MLP模型文件: {timexer_mlp_path}")
    timexer_module = _load_module(timexer_mlp_path, "timexer_mlp")
    TimeXerMLP = timexer_module.TimeXerMLP
    
    # 加载配置
    print("\n加载配置文件...")
    itransformer_config_path = project_root / "configs" / "v0.1TB_20251225" / "itransformer_config.yaml"
    timexer_config_path = project_root / "configs" / "v0.5_20260107" / "timexer_mlp_config.yaml"
    
    with open(itransformer_config_path, 'r', encoding='utf-8') as f:
        itransformer_config = yaml.safe_load(f)
    
    with open(timexer_config_path, 'r', encoding='utf-8') as f:
        timexer_config = yaml.safe_load(f)
    
    itransformer_cfg = itransformer_config['model']
    timexer_cfg = timexer_config['model']
    
    # 创建模型
    print("\n创建模型...")
    
    # iTransformer模型
    itransformer_model = iTransformerDecoder(
        input_features=itransformer_cfg['input_features'],
        seq_len=itransformer_cfg['seq_len'],
        d_model=itransformer_cfg['d_model'],
        n_layers=itransformer_cfg['n_layers'],
        n_heads=itransformer_cfg['n_heads'],
        d_ff=itransformer_cfg['d_ff'],
        dropout=itransformer_cfg['dropout'],
        activation=itransformer_cfg['activation'],
        decoder_config=itransformer_cfg.get('decoder', {}),
        input_resnet_config=itransformer_cfg.get('input_resnet', {}),
        output_resnet_config=itransformer_cfg.get('output_resnet', {}),
        final_output_config=itransformer_cfg.get('final_output', {})
    )
    
    # TimeXer-MLP模型
    timexer_model = TimeXerMLP(
        seq_len=timexer_cfg['seq_len'],
        n_features=timexer_cfg['n_features'],
        endogenous_features=timexer_cfg.get('endogenous_features', 44),
        exogenous_features=timexer_cfg.get('exogenous_features', 20),
        prediction_len=timexer_cfg.get('prediction_len', 1),
        endogenous_indices=timexer_cfg.get('endogenous_indices', None),
        exogenous_indices=timexer_cfg.get('exogenous_indices', None),
        endogenous_blocks=timexer_cfg.get('endogenous_blocks', 3),
        endogenous_hidden_dim=timexer_cfg.get('endogenous_hidden_dim', 256),
        exogenous_blocks=timexer_cfg.get('exogenous_blocks', 2),
        exogenous_hidden_dim=timexer_cfg.get('exogenous_hidden_dim', 256),
        shared_time_mixing=timexer_cfg.get('shared_time_mixing', True),
        mlp_fusion_ff_dim=timexer_cfg.get('mlp_fusion_ff_dim', 512),
        dropout=timexer_cfg.get('dropout', 0.1),
        activation=timexer_cfg.get('activation', 'gelu'),
        use_layernorm=timexer_cfg.get('use_layernorm', False),
        use_residual=timexer_cfg.get('use_residual', True),
        norm_type=timexer_cfg.get('norm_type', 'layer')
    )
    
    # 计算参数量
    print("\n" + "=" * 80)
    print("参数量对比")
    print("=" * 80)
    
    itransformer_params, itransformer_size = calculate_model_size(itransformer_model)
    timexer_params, timexer_size = calculate_model_size(timexer_model)
    
    print(f"\niTransformer:")
    print(f"  参数量: {itransformer_params:,}")
    print(f"  存储大小: {format_size(itransformer_size)}")
    print(f"  模型方法返回: {itransformer_model.get_num_parameters():,}")
    
    print(f"\nTimeXer-MLP:")
    print(f"  参数量: {timexer_params:,}")
    print(f"  存储大小: {format_size(timexer_size)}")
    print(f"  模型方法返回: {timexer_model.get_num_parameters():,}")
    
    print(f"\n对比:")
    param_ratio = itransformer_params / timexer_params if timexer_params > 0 else float('inf')
    print(f"  iTransformer参数量 / TimeXer-MLP参数量 = {param_ratio:.2f}x")
    
    # 架构分析
    print("\n" + "=" * 80)
    print("架构复杂度分析")
    print("=" * 80)
    
    itransformer_stats, _ = analyze_model_architecture(itransformer_model, "iTransformer")
    timexer_stats, _ = analyze_model_architecture(timexer_model, "TimeXer-MLP")
    
    # 计算复杂度对比
    compare_computational_complexity()
    
    # 总结
    print("\n" + "=" * 80)
    print("训练速度差异原因总结")
    print("=" * 80)
    
    print(f"\n1. 参数量差异:")
    print(f"   - iTransformer: {itransformer_params:,} 参数")
    print(f"   - TimeXer-MLP: {timexer_params:,} 参数")
    if param_ratio > 1.5:
        print(f"   - iTransformer参数量是TimeXer-MLP的 {param_ratio:.2f} 倍")
        print(f"   - 更多参数意味着更多的梯度计算和优化器更新开销")
    
    print(f"\n2. 架构复杂度差异:")
    print(f"   - iTransformer使用Multi-Head Attention机制")
    print(f"     * 计算复杂度: O(seq_len² × d_model)")
    print(f"     * 需要计算注意力矩阵，内存和计算开销大")
    print(f"     * 有 {itransformer_stats.get('MultiheadAttention', 0)} 个Attention层")
    
    print(f"\n   - TimeXer-MLP使用MLP架构")
    print(f"     * 计算复杂度: O(seq_len × n_features)")
    print(f"     * 线性操作，计算效率高")
    print(f"     * 没有Attention层，避免了二次复杂度")
    
    print(f"\n3. 序列长度差异:")
    print(f"   - iTransformer: seq_len = {itransformer_cfg['seq_len']}")
    print(f"   - TimeXer-MLP: seq_len = {timexer_cfg['seq_len']}")
    seq_ratio = timexer_cfg['seq_len'] / itransformer_cfg['seq_len']
    print(f"   - TimeXer-MLP序列长度是iTransformer的 {seq_ratio:.1f} 倍")
    print(f"   - 但MLP的线性复杂度使得即使序列更长，计算仍然高效")
    
    print(f"\n4. 其他优化因素:")
    print(f"   - TimeXer-MLP使用共享时间混合层，减少参数量")
    print(f"   - 双分支架构可以并行处理，提高GPU利用率")
    print(f"   - MLP操作对GPU友好，容易优化")
    
    print(f"\n5. 建议:")
    print(f"   - 如果iTransformer训练慢，可以考虑:")
    print(f"     * 启用混合精度训练 (mixed_precision: true)")
    print(f"     * 使用torch.compile()优化模型")
    print(f"     * 减少Attention头数或层数")
    print(f"     * 减少序列长度（如果可能）")
    
    print("\n" + "=" * 80)
    
    # 测试前向传播速度（可选）
    print("\n测试前向传播速度...")
    batch_size = 32
    
    # iTransformer
    x_itransformer = torch.randn(batch_size, itransformer_cfg['seq_len'], itransformer_cfg['input_features'])
    itransformer_model.eval()
    
    with torch.no_grad():
        import time
        # 预热
        _ = itransformer_model(x_itransformer)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 测试
        start = time.time()
        for _ in range(10):
            _ = itransformer_model(x_itransformer)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        itransformer_time = (time.time() - start) / 10
    
    # TimeXer-MLP
    x_timexer = torch.randn(batch_size, timexer_cfg['seq_len'], timexer_cfg['n_features'])
    timexer_model.eval()
    
    with torch.no_grad():
        # 预热
        _ = timexer_model(x_timexer)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 测试
        start = time.time()
        for _ in range(10):
            _ = timexer_model(x_timexer)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        timexer_time = (time.time() - start) / 10
    
    print(f"\n前向传播时间 (batch_size={batch_size}, 10次平均):")
    print(f"  iTransformer: {itransformer_time*1000:.2f} ms")
    print(f"  TimeXer-MLP: {timexer_time*1000:.2f} ms")
    if timexer_time > 0:
        speedup = itransformer_time / timexer_time
        print(f"  TimeXer-MLP速度提升: {speedup:.2f}x")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
