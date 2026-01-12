#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU验证脚本
用于验证PyTorch和CUDA是否正常工作，以及GPU是否可用于AI计算
"""

import sys
import time

def print_section(title):
    """打印分隔线"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_pytorch():
    """检查PyTorch安装"""
    print_section("PyTorch 检查")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        return torch
    except ImportError:
        print("✗ PyTorch未安装")
        sys.exit(1)

def check_cuda(torch):
    """检查CUDA可用性"""
    print_section("CUDA 检查")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {'✓ 是' if cuda_available else '✗ 否'}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN启用: {torch.backends.cudnn.enabled}")
    else:
        print("警告: CUDA不可用，将使用CPU进行计算")
    
    return cuda_available

def get_gpu_info(torch):
    """获取GPU信息"""
    print_section("GPU 信息")
    if not torch.cuda.is_available():
        print("无GPU可用")
        return None
    
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    return gpu_count

def test_gpu_computation(torch, use_gpu=True):
    """测试GPU计算性能"""
    print_section("GPU 计算测试")
    
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")
    
    # 创建测试数据
    size = 5000
    print(f"\n测试矩阵大小: {size} x {size}")
    
    # 创建随机矩阵
    print("创建随机矩阵...")
    start_time = time.time()
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    creation_time = time.time() - start_time
    print(f"矩阵创建时间: {creation_time:.4f} 秒")
    
    # 矩阵乘法测试
    print("执行矩阵乘法...")
    start_time = time.time()
    c = torch.matmul(a, b)
    if use_gpu:
        torch.cuda.synchronize()  # 等待GPU计算完成
    computation_time = time.time() - start_time
    print(f"矩阵乘法时间: {computation_time:.4f} 秒")
    
    # 验证结果
    result_sum = c.sum().item()
    print(f"结果矩阵元素和: {result_sum:.2f}")
    
    # 内存使用情况
    if use_gpu and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU内存使用:")
        print(f"  已分配: {memory_allocated:.2f} GB")
        print(f"  已保留: {memory_reserved:.2f} GB")
    
    return computation_time

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  GPU验证脚本 - 验证AI计算环境")
    print("="*60)
    
    # 检查PyTorch
    torch = check_pytorch()
    
    # 检查CUDA
    cuda_available = check_cuda(torch)
    
    # 获取GPU信息
    gpu_count = get_gpu_info(torch)
    
    # 测试GPU计算
    if cuda_available:
        print("\n开始GPU计算测试...")
        gpu_time = test_gpu_computation(torch, use_gpu=True)
        
        print("\n开始CPU计算测试（对比）...")
        cpu_time = test_gpu_computation(torch, use_gpu=False)
        
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\n加速比: {speedup:.2f}x (GPU比CPU快{speedup:.2f}倍)")
    else:
        print("\n由于CUDA不可用，仅进行CPU计算测试...")
        test_gpu_computation(torch, use_gpu=False)
    
    # 总结
    print_section("验证结果")
    if cuda_available:
        print("✓ PyTorch安装正常")
        print("✓ CUDA可用")
        print(f"✓ 检测到 {gpu_count} 个GPU")
        print("✓ GPU计算测试通过")
        print("\n结论: GPU环境配置正确，可用于AI模型训练！")
    else:
        print("✓ PyTorch安装正常")
        print("✗ CUDA不可用")
        print("⚠ 将使用CPU进行计算（速度较慢）")
        print("\n建议: 检查CUDA驱动和PyTorch CUDA版本是否匹配")

if __name__ == "__main__":
    main()

