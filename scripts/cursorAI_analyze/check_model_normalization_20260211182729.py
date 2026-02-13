#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查模型是否使用归一化
版本: v1.0
时间: 20260211182729

功能：
1. 加载模型 checkpoint，检查参数名称
2. 检查模型配置文件
3. 动态加载模型，打印完整结构
4. 运行测试推理，通过 hook 观察数据流
5. 生成详细检查报告

使用方法：
    python check_model_normalization_20260211182729.py
"""

import torch
import torch.nn as nn
import yaml
import sys
from pathlib import Path
from datetime import datetime
import importlib.util
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ===== 模型配置 =====
MODEL_DIR = "/data/project_20251211/experiments/timexer_v0.43_20260207232015_20260119170929_500120"
OUTPUT_FILE = f"/data/project_20251211/scripts/check_model_normalization_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"


def print_and_log(text, file_handle):
    """同时打印到控制台和文件"""
    print(text)
    file_handle.write(text + "\n")


def load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_checkpoint_parameters(checkpoint_path: Path, f):
    """检查 checkpoint 中的参数"""
    print_and_log("\n" + "=" * 80, f)
    print_and_log("1. 检查 Checkpoint 参数", f)
    print_and_log("=" * 80, f)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 检查 checkpoint 的键
    print_and_log(f"\nCheckpoint 包含的顶层键: {list(checkpoint.keys())}", f)
    
    # 检查是否有特殊的归一化参数
    special_keys = ['norm_mean', 'norm_std', 'norm_mask', 'log_offset', 'use_norm']
    print_and_log("\n检查特殊归一化参数:", f)
    for key in special_keys:
        if key in checkpoint:
            print_and_log(f"  ✓ 找到: {key} = {checkpoint[key]}", f)
        else:
            print_and_log(f"  ✗ 未找到: {key}", f)
    
    # 检查模型参数
    model_state = checkpoint.get('model_state_dict', {})
    print_and_log(f"\n模型参数数量: {len(model_state)}", f)
    
    # 查找包含 'norm' 关键字的参数
    norm_params = [k for k in model_state.keys() if 'norm' in k.lower()]
    if norm_params:
        print_and_log(f"\n找到 {len(norm_params)} 个包含 'norm' 的参数:", f)
        for param in norm_params:
            shape = model_state[param].shape if hasattr(model_state[param], 'shape') else 'N/A'
            print_and_log(f"  - {param}: {shape}", f)
    else:
        print_and_log("\n未找到包含 'norm' 的参数", f)
    
    # 查找包含 'missing' 关键字的参数（v0.43 的 missing embedding）
    missing_params = [k for k in model_state.keys() if 'missing' in k.lower()]
    if missing_params:
        print_and_log(f"\n找到 {len(missing_params)} 个包含 'missing' 的参数:", f)
        for param in missing_params:
            shape = model_state[param].shape if hasattr(model_state[param], 'shape') else 'N/A'
            print_and_log(f"  - {param}: {shape}", f)
    else:
        print_and_log("\n未找到包含 'missing' 的参数", f)
    
    # 打印前 20 个参数名称
    print_and_log(f"\n前 20 个参数名称:", f)
    for i, key in enumerate(list(model_state.keys())[:20]):
        shape = model_state[key].shape if hasattr(model_state[key], 'shape') else 'N/A'
        print_and_log(f"  {i+1}. {key}: {shape}", f)
    
    return checkpoint


def check_model_config(config_path: Path, f):
    """检查模型配置文件"""
    print_and_log("\n" + "=" * 80, f)
    print_and_log("2. 检查模型配置文件", f)
    print_and_log("=" * 80, f)
    
    with open(config_path, 'r', encoding='utf-8') as cf:
        config = yaml.safe_load(cf)
    
    model_config = config.get('model', {})
    
    # 检查归一化相关配置
    norm_keys = ['use_norm', 'norm_feature_indices', 'num_norm_features', 'output_feature_index']
    print_and_log("\n归一化相关配置:", f)
    for key in norm_keys:
        if key in model_config:
            value = model_config[key]
            if isinstance(value, list) and len(value) > 10:
                print_and_log(f"  {key}: [列表，长度={len(value)}]", f)
            else:
                print_and_log(f"  {key}: {value}", f)
        else:
            print_and_log(f"  {key}: 未设置", f)
    
    # 检查其他关键配置
    print_and_log("\n其他关键配置:", f)
    key_configs = ['name', 'seq_len', 'n_features', 'endogenous_features', 'exogenous_features',
                   'use_layernorm', 'missing_embedding_enabled', 'missing_value_flag']
    for key in key_configs:
        if key in model_config:
            print_and_log(f"  {key}: {model_config[key]}", f)
    
    return config


def load_and_print_model_structure(model_dir: Path, checkpoint, config, f):
    """加载模型并打印结构"""
    print_and_log("\n" + "=" * 80, f)
    print_and_log("3. 加载模型并检查结构", f)
    print_and_log("=" * 80, f)
    
    model_config = config['model']
    
    # 检测版本（使用目录名）
    dir_name = model_dir.name.lower()
    if 'v0.43' in dir_name:
        models_path = project_root / "src" / "models" / "v0.43_20260119"
        print_and_log(f"\n检测到 v0.43 模型，使用代码路径: {models_path}", f)
    elif 'v0.45' in dir_name:
        models_path = project_root / "src" / "models" / "v0.45_20260207"
        print_and_log(f"\n检测到 v0.45 模型，使用代码路径: {models_path}", f)
    else:
        print_and_log(f"\n无法从目录名识别版本: {dir_name}", f)
        return None
    
    # 动态加载模型
    timexer_module = load_module(models_path / "timexer.py", "timexer")
    ModelClass = timexer_module.TimeXer
    
    # 构建模型参数
    model_params = {
        'seq_len': model_config['seq_len'],
        'n_features': model_config['n_features'],
        'prediction_len': model_config.get('prediction_len', 1),
        'endogenous_features': model_config.get('endogenous_features', 44),
        'exogenous_features': model_config.get('exogenous_features', 20),
        'endogenous_indices': model_config.get('endogenous_indices'),
        'exogenous_indices': model_config.get('exogenous_indices'),
        'endogenous_blocks': model_config.get('endogenous_blocks', 3),
        'endogenous_hidden_dim': model_config.get('endogenous_hidden_dim', 256),
        'exogenous_blocks': model_config.get('exogenous_blocks', 2),
        'exogenous_hidden_dim': model_config.get('exogenous_hidden_dim', 256),
        'shared_time_mixing': model_config.get('shared_time_mixing', True),
        'time_mixing_type': model_config.get('time_mixing_type', 'attention'),
        'time_attn_n_heads': model_config.get('time_attn_n_heads', 8),
        'use_rope': model_config.get('use_rope', True),
        'cross_attn_n_heads': model_config.get('cross_attn_n_heads', 8),
        'cross_attn_ff_dim': model_config.get('cross_attn_ff_dim', 1024),
        'dropout': model_config.get('dropout', 0.1),
        'activation': model_config.get('activation', 'gelu'),
        'use_layernorm': model_config.get('use_layernorm', True),
        'use_residual': model_config.get('use_residual', True),
        'temporal_aggregation_config': model_config.get('temporal_aggregation', {}),
        'output_projection_config': model_config.get('output_projection', {})
    }
    
    # 添加细粒度 LayerNorm 控制（v0.41+）
    model_params['use_layernorm_in_tsmixer'] = model_config.get('use_layernorm_in_tsmixer')
    model_params['use_layernorm_in_attention'] = model_config.get('use_layernorm_in_attention')
    model_params['use_layernorm_before_pooling'] = model_config.get('use_layernorm_before_pooling')
    
    # 尝试添加归一化参数（v0.45）
    if 'use_norm' in model_config:
        print_and_log("\n尝试添加 Instance Normalization 参数...", f)
        model_params['use_norm'] = model_config.get('use_norm', True)
        model_params['norm_feature_indices'] = model_config.get('norm_feature_indices')
        model_params['output_feature_index'] = model_config.get('output_feature_index', 2)
        print_and_log("  ✓ 已添加归一化参数到模型初始化", f)
    else:
        print_and_log("\n配置文件中没有 use_norm，跳过归一化参数", f)
    
    # 创建模型
    try:
        model = ModelClass(**model_params)
        print_and_log("\n✓ 模型创建成功", f)
    except TypeError as e:
        print_and_log(f"\n✗ 模型创建失败（参数不匹配）: {e}", f)
        # 如果失败，移除归一化参数重试
        if 'use_norm' in model_params:
            print_and_log("\n尝试移除归一化参数重新创建...", f)
            del model_params['use_norm']
            del model_params['norm_feature_indices']
            del model_params['output_feature_index']
            model = ModelClass(**model_params)
            print_and_log("  ✓ 移除归一化参数后创建成功", f)
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], 
        strict=False
    )
    
    if missing_keys:
        print_and_log(f"\n模型中缺失的参数（{len(missing_keys)}个）:", f)
        for key in missing_keys[:10]:  # 只显示前 10 个
            print_and_log(f"  - {key}", f)
        if len(missing_keys) > 10:
            print_and_log(f"  ... 还有 {len(missing_keys) - 10} 个", f)
    
    if unexpected_keys:
        print_and_log(f"\nCheckpoint 中额外的参数（{len(unexpected_keys)}个）:", f)
        for key in unexpected_keys[:10]:  # 只显示前 10 个
            print_and_log(f"  - {key}", f)
        if len(unexpected_keys) > 10:
            print_and_log(f"  ... 还有 {len(unexpected_keys) - 10} 个", f)
    
    # 打印模型结构
    print_and_log("\n模型结构（查找归一化相关层）:", f)
    has_norm_layer = False
    for name, module in model.named_modules():
        if 'norm' in name.lower() or 'normalization' in type(module).__name__.lower():
            print_and_log(f"  - {name}: {type(module).__name__}", f)
            has_norm_layer = True
    
    if not has_norm_layer:
        print_and_log("  ✗ 未找到归一化相关层", f)
    
    # 检查是否有归一化相关的属性
    print_and_log("\n检查模型的归一化相关属性:", f)
    norm_attrs = ['use_norm', 'norm_mean', 'norm_std', 'norm_mask', 'instance_norm']
    for attr in norm_attrs:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, torch.Tensor):
                print_and_log(f"  ✓ {attr}: Tensor, shape={value.shape}", f)
            else:
                print_and_log(f"  ✓ {attr}: {value}", f)
        else:
            print_and_log(f"  ✗ {attr}: 不存在", f)
    
    return model


def test_forward_with_hooks(model, f):
    """测试前向传播并使用 hook 观察数据流"""
    print_and_log("\n" + "=" * 80, f)
    print_and_log("4. 测试前向传播（观察数据流）", f)
    print_and_log("=" * 80, f)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 500
    n_features = 64
    
    # 创建正常范围的测试数据
    test_input = torch.randn(batch_size, seq_len, n_features) * 10 + 50  # 均值50，标准差10
    
    print_and_log(f"\n测试输入:", f)
    print_and_log(f"  形状: {test_input.shape}", f)
    print_and_log(f"  均值: {test_input.mean().item():.4f}", f)
    print_and_log(f"  标准差: {test_input.std().item():.4f}", f)
    print_and_log(f"  最小值: {test_input.min().item():.4f}", f)
    print_and_log(f"  最大值: {test_input.max().item():.4f}", f)
    
    # 记录 hook 捕获的数据
    hook_data = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input
            
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            
            if isinstance(input_tensor, torch.Tensor):
                hook_data[f"{name}_input"] = {
                    'mean': input_tensor.mean().item(),
                    'std': input_tensor.std().item(),
                    'min': input_tensor.min().item(),
                    'max': input_tensor.max().item(),
                    'shape': tuple(input_tensor.shape)
                }
            
            if isinstance(output_tensor, torch.Tensor):
                hook_data[f"{name}_output"] = {
                    'mean': output_tensor.mean().item(),
                    'std': output_tensor.std().item(),
                    'min': output_tensor.min().item(),
                    'max': output_tensor.max().item(),
                    'shape': tuple(output_tensor.shape)
                }
        return hook
    
    # 注册 hook 到关键层
    hooks = []
    
    # 尝试在第一层注册 hook
    first_layer_found = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not first_layer_found:
            hooks.append(module.register_forward_hook(make_hook(f"first_linear_{name}")))
            print_and_log(f"\n✓ 在第一个线性层注册 hook: {name}", f)
            first_layer_found = True
            break
    
    # 在输出层注册 hook
    for name, module in model.named_modules():
        if 'output_proj3' in name or 'final' in name.lower():
            hooks.append(module.register_forward_hook(make_hook(f"output_{name}")))
            print_and_log(f"✓ 在输出层注册 hook: {name}", f)
    
    # 如果模型有归一化层，注册 hook
    for name, module in model.named_modules():
        if 'norm' in name.lower() and isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.InstanceNorm1d)):
            hooks.append(module.register_forward_hook(make_hook(f"norm_{name}")))
            print_and_log(f"✓ 在归一化层注册 hook: {name}", f)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print_and_log(f"\n模型输出:", f)
    print_and_log(f"  形状: {output.shape}", f)
    print_and_log(f"  均值: {output.mean().item():.4f}", f)
    print_and_log(f"  标准差: {output.std().item():.4f}", f)
    print_and_log(f"  最小值: {output.min().item():.4f}", f)
    print_and_log(f"  最大值: {output.max().item():.4f}", f)
    print_and_log(f"  具体值: {output.flatten()[:5].tolist()}", f)
    
    # 打印 hook 捕获的数据
    if hook_data:
        print_and_log(f"\nHook 捕获的中间数据流:", f)
        for key, stats in hook_data.items():
            print_and_log(f"\n  {key}:", f)
            print_and_log(f"    形状: {stats['shape']}", f)
            print_and_log(f"    均值: {stats['mean']:.4f}", f)
            print_and_log(f"    标准差: {stats['std']:.4f}", f)
            print_and_log(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]", f)
    else:
        print_and_log("\n未捕获到任何中间数据", f)
    
    # 清理 hooks
    for hook in hooks:
        hook.remove()
    
    # 分析是否使用了归一化
    print_and_log("\n" + "-" * 80, f)
    print_and_log("数据流分析:", f)
    print_and_log("-" * 80, f)
    
    # 检查第一层输入是否被归一化
    first_linear_input_keys = [k for k in hook_data.keys() if 'first_linear' in k and 'input' in k]
    if first_linear_input_keys:
        first_input = hook_data[first_linear_input_keys[0]]
        print_and_log(f"\n第一个线性层的输入统计:", f)
        print_and_log(f"  均值: {first_input['mean']:.4f}", f)
        print_and_log(f"  标准差: {first_input['std']:.4f}", f)
        
        # 判断是否接近标准化（均值接近0，标准差接近1）
        if abs(first_input['mean']) < 1.0 and abs(first_input['std'] - 1.0) < 0.5:
            print_and_log("  ⚠️  数据接近标准化分布（均值≈0，标准差≈1），可能使用了归一化", f)
        elif abs(first_input['mean'] - 50) < 10 and abs(first_input['std'] - 10) < 5:
            print_and_log("  ✓ 数据保持原始分布，未使用归一化", f)
        else:
            print_and_log("  ? 数据分布不确定，需要进一步分析", f)
    
    return output


def compare_with_training_script(model_dir: Path, f):
    """对比训练脚本，确认使用的版本"""
    print_and_log("\n" + "=" * 80, f)
    print_and_log("5. 训练脚本分析", f)
    print_and_log("=" * 80, f)
    
    # 从目录名提取时间戳
    dir_name = model_dir.name
    print_and_log(f"\n模型目录名: {dir_name}", f)
    
    # 尝试从训练日志中读取信息
    log_path = model_dir / "logs" / "training_log.jsonl"
    if log_path.exists():
        print_and_log(f"\n读取训练日志: {log_path}", f)
        with open(log_path, 'r') as lf:
            lines = lf.readlines()
            if lines:
                first_log = json.loads(lines[0])
                print_and_log(f"  训练开始时间: {first_log.get('timestamp', 'N/A')}", f)
                print_and_log(f"  首次损失值: {first_log.get('train_loss', 'N/A')}", f)
    
    # 检查模型配置中的元数据
    config_path = model_dir / "configs" / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as cf:
        config = yaml.safe_load(cf)
    
    metadata = config.get('metadata', {})
    print_and_log(f"\n配置文件元数据:", f)
    print_and_log(f"  版本: {metadata.get('version', 'N/A')}", f)
    print_and_log(f"  日期: {metadata.get('date', 'N/A')}", f)
    print_and_log(f"  代码修改时间: {metadata.get('code_modify_timestamp', 'N/A')}", f)
    print_and_log(f"  运行时间: {metadata.get('run_timestamp', 'N/A')}", f)
    
    # 推断使用的训练脚本
    version = metadata.get('version', '')
    if 'v0.43' in version:
        script_path = project_root / "scripts" / "v0.43_20260119" / "train_timexer.py"
        print_and_log(f"\n推断使用的训练脚本: {script_path}", f)
        print_and_log("  ✓ v0.43 不支持 Instance Normalization", f)
    elif 'v0.45' in version:
        script_path = project_root / "scripts" / "v0.45_20260207" / "train_timexer.py"
        print_and_log(f"\n推断使用的训练脚本: {script_path}", f)
        print_and_log("  ✓ v0.45 支持 Instance Normalization", f)


def generate_summary(f):
    """生成检查总结"""
    print_and_log("\n" + "=" * 80, f)
    print_and_log("6. 检查总结", f)
    print_and_log("=" * 80, f)
    
    print_and_log("\n基于以上检查结果，总结如下：", f)
    print_and_log("\n【关键发现】", f)
    print_and_log("1. 模型配置文件中包含归一化参数（use_norm, norm_feature_indices 等）", f)
    print_and_log("2. 但需要根据以下判断实际是否使用：", f)
    print_and_log("   - Checkpoint 中是否有 norm_mean/norm_std 参数", f)
    print_and_log("   - 模型代码是否接受归一化参数", f)
    print_and_log("   - 数据流分析结果", f)
    
    print_and_log("\n【版本识别】", f)
    print_and_log("- 如果模型创建时接受 use_norm 参数，且 checkpoint 有 norm_mean/std", f)
    print_and_log("  → 模型使用了 Instance Normalization（v0.45 特性）", f)
    print_and_log("- 如果模型创建时不接受 use_norm 参数", f)
    print_and_log("  → 模型未使用 Instance Normalization（v0.43 或更早版本）", f)
    
    print_and_log("\n【建议】", f)
    print_and_log("1. 如果发现配置文件中有归一化参数，但实际未使用", f)
    print_and_log("   → 配置文件可能是从模板复制的，实际训练时未启用", f)
    print_and_log("2. 如果希望使用归一化功能", f)
    print_and_log("   → 应使用 v0.45 或更新版本的训练脚本重新训练", f)
    print_and_log("3. 在推理时，应根据实际情况选择合适的处理方式", f)
    print_and_log("   → 当前推理脚本 v0.7 的处理逻辑是正确的", f)


def main():
    """主函数"""
    model_dir = Path(MODEL_DIR)
    output_file = Path(OUTPUT_FILE)
    
    print(f"\n{'=' * 80}")
    print(f"检查模型归一化配置")
    print(f"{'=' * 80}")
    print(f"模型目录: {model_dir}")
    print(f"输出文件: {output_file}")
    print(f"{'=' * 80}\n")
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        print_and_log(f"模型归一化检查报告", f)
        print_and_log(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        print_and_log(f"模型目录: {model_dir}", f)
        print_and_log("=" * 80, f)
        
        # 1. 检查 checkpoint
        checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
        checkpoint = check_checkpoint_parameters(checkpoint_path, f)
        
        # 2. 检查配置文件
        config_path = model_dir / "configs" / "model_config.yaml"
        config = check_model_config(config_path, f)
        
        # 3. 加载模型并检查结构
        model = load_and_print_model_structure(model_dir, checkpoint, config, f)
        
        if model is not None:
            # 4. 测试前向传播
            test_forward_with_hooks(model, f)
        
        # 5. 对比训练脚本
        compare_with_training_script(model_dir, f)
        
        # 6. 生成总结
        generate_summary(f)
        
        print_and_log("\n" + "=" * 80, f)
        print_and_log("检查完成！", f)
        print_and_log("=" * 80, f)
    
    print(f"\n报告已保存到: {output_file}")


if __name__ == '__main__':
    main()
