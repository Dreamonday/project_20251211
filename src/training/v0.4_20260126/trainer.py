#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器类（集成TensorBoard，支持mask + 对数变换）
版本: v0.4
日期: 20260126

封装训练循环、验证、模型保存、TensorBoard可视化等功能
适用于TSMixer、TimeXer等各种模型

v0.4新增：
- 支持对数变换：训练在对数空间，评估在原始空间
- 自动检测dataset的对数变换配置
- 分别计算对数空间损失和原始空间指标
- 对数空间损失用于训练和early stopping
- 原始空间指标用于展示和TensorBoard记录
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict
import time
import json
import math
from datetime import datetime
import importlib.util

# 动态导入metrics_utils
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入metrics工具
project_root = Path(__file__).parent.parent.parent.parent
utils_path = project_root / "src" / "utils" / "v0.1_20251212"
try:
    metrics_module = _load_module(utils_path / "metrics_utils.py", "metrics_utils")
    compute_metrics = metrics_module.compute_metrics
    format_metrics = metrics_module.format_metrics
except Exception as e:
    raise ImportError(f"Failed to import metrics_utils: {e}") from e


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        初始化早停
        
        Args:
            patience: 耐心值（多少个epoch没有改进就停止）
            min_delta: 最小改进阈值
            mode: "min"表示越小越好，"max"表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前得分
        
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """判断当前得分是否更好"""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class Trainer:
    """
    训练器类（集成TensorBoard，支持mask + 对数变换）
    封装训练循环、验证、模型保存、TensorBoard可视化等功能
    
    v0.3新增：支持mask机制，兼容v0.1和v0.2数据集
    v0.4新增：支持对数变换，训练在对数空间，评估在原始空间
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        save_dir: Optional[str] = None,
        save_best: bool = True,
        best_metric: str = "loss",
        best_metric_mode: str = "min",
        early_stopping: Optional[EarlyStopping] = None,
        mixed_precision: bool = False,
        val_interval: int = 1,
        save_interval: int = 10,
        log_file: Optional[str] = None,
        history_file: Optional[str] = None,
        # TensorBoard相关参数
        tensorboard_enabled: bool = True,
        tensorboard_dir: Optional[str] = None,
        tb_log_interval: int = 50,
        tb_histogram_interval: int = 500,
        tb_log_histograms: bool = True,
        # 评估指标相关参数
        max_relative_error: Optional[float] = None,
        epsilon: float = 1e-8
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            scheduler: 学习率调度器（可选）
            device: 设备（"cuda"或"cpu"）
            save_dir: 模型保存目录
            save_best: 是否保存最佳模型
            best_metric: 选择最佳模型的评估指标（如"loss", "mape", "mae"等）
            best_metric_mode: 指标优化方向（"min"或"max"）
            early_stopping: 早停机制（可选）
            mixed_precision: 是否使用混合精度训练
            val_interval: 每N个epoch验证一次
            save_interval: 每N个epoch保存一次模型
            log_file: 日志文件路径（可选）
            history_file: 训练历史文件路径（可选）
            tensorboard_enabled: 是否启用TensorBoard
            tensorboard_dir: TensorBoard日志目录
            tb_log_interval: 每N个batch记录一次标量到TensorBoard
            tb_histogram_interval: 每N个batch记录一次直方图到TensorBoard
            tb_log_histograms: 是否记录参数和梯度直方图
            max_relative_error: MAPE计算时的相对误差上限（可选，避免接近0的样本使MAPE爆炸）
            epsilon: 防止除零的小值（用于MAPE计算）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.log_file = log_file
        self.history_file = history_file
        
        # 评估指标配置
        self.max_relative_error = max_relative_error
        self.epsilon = epsilon
        
        # TensorBoard配置
        self.tensorboard_enabled = tensorboard_enabled
        self.tb_log_interval = tb_log_interval
        self.tb_histogram_interval = tb_histogram_interval
        self.tb_log_histograms = tb_log_histograms
        self.writer = None
        
        # 初始化TensorBoard
        if self.tensorboard_enabled and tensorboard_dir:
            tb_dir = Path(tensorboard_dir)
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard日志目录: {tb_dir}")
            print(f"启动TensorBoard: tensorboard --logdir={tb_dir}")
        
        # 混合精度训练的scaler
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # v0.4新增：检测数据集是否启用了对数变换
        self.log_transform_enabled = False
        self.log_offset = None
        
        # 尝试从train_loader的dataset获取对数变换信息
        if hasattr(train_loader.dataset, 'log_transform'):
            self.log_transform_enabled = train_loader.dataset.log_transform
            if self.log_transform_enabled:
                self.log_offset = train_loader.dataset.log_offset
                print(f"\n检测到对数变换已启用:")
                print(f"  offset = {self.log_offset:.4f}")
                print(f"  训练损失将在对数空间计算")
                print(f"  评估指标将在原始空间计算（还原后）")
        
        # 训练历史
        self.train_history = {
            "loss": [],  # 对数空间的损失（用于训练）
            "val_loss": [],  # 对数空间的损失（用于early stopping）
            "metrics": [],  # 原始空间的评估指标（用于展示）
            "val_metrics": [],  # 原始空间的评估指标（用于展示）
            "learning_rate": []
        }
        
        # 最佳模型指标（基于对数空间的loss）
        if self.best_metric_mode == "min":
            self.best_val_metric = float('inf')
        else:
            self.best_val_metric = float('-inf')
        self.best_epoch = 0
        
        # 全局步数（用于TensorBoard）
        self.global_step = 0
    
    def _inverse_transform(self, y_log: torch.Tensor) -> torch.Tensor:
        """
        对数变换的反变换
        
        Args:
            y_log: 对数空间的值
        
        Returns:
            原始空间的值
        """
        if not self.log_transform_enabled:
            return y_log
        
        return torch.exp(y_log) - self.log_offset
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch（支持mask + 对数变换）
        
        Args:
            epoch: 当前epoch编号
        
        Returns:
            指标字典，包含：
            - loss: 对数空间的损失（用于训练）
            - metrics（原始空间）: mae, mse, rmse, mape, r2
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        total_batches = len(self.train_loader)
        print_interval = max(1, total_batches // 100)  # 每1%打印一次进度
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # 检测数据格式
            if len(batch_data) == 3:
                # 支持mask的数据集
                X, y, mask = batch_data
                X = X.to(self.device)
                y = y.to(self.device)  # 这里的y已经是对数空间（如果启用了变换）
                mask = mask.to(self.device)
            else:
                # 不支持mask的数据集
                X, y = batch_data
                X = X.to(self.device)
                y = y.to(self.device)
                mask = None
            
            # 检查输入数据是否有NaN或Inf
            if torch.isnan(X).any() or torch.isinf(X).any():
                print(f"\n警告: Batch {batch_idx + 1} 输入数据包含NaN或Inf!")
                print(f"  X NaN数量: {torch.isnan(X).sum().item()}")
                print(f"  X Inf数量: {torch.isinf(X).sum().item()}")
                print(f"  X 统计: min={X.min().item():.6f}, max={X.max().item():.6f}, mean={X.mean().item():.6f}")
                continue
            
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"\n警告: Batch {batch_idx + 1} 目标数据包含NaN或Inf!")
                print(f"  y NaN数量: {torch.isnan(y).sum().item()}")
                print(f"  y Inf数量: {torch.isinf(y).sum().item()}")
                print(f"  y 统计: min={y.min().item():.6f}, max={y.max().item():.6f}, mean={y.mean().item():.6f}")
                continue
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播（传递mask）
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    pred = self.model(X, mask=mask) if mask is not None else self.model(X)
                    # 损失在对数空间计算
                    loss = self.criterion(pred, y)
                
                # 检查模型输出和loss
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"\n错误: Batch {batch_idx + 1} 模型输出包含NaN或Inf!")
                    print(f"  pred NaN数量: {torch.isnan(pred).sum().item()}")
                    print(f"  pred Inf数量: {torch.isinf(pred).sum().item()}")
                    print(f"  pred 统计: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
                    raise ValueError("模型输出包含NaN或Inf，训练停止")
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n错误: Batch {batch_idx + 1} Loss为NaN或Inf!")
                    print(f"  pred 统计: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
                    print(f"  y 统计: min={y.min().item():.6f}, max={y.max().item():.6f}, mean={y.mean().item():.6f}")
                    raise ValueError("Loss为NaN或Inf，训练停止")
                
                # 反向传播（混合精度）
                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(X, mask=mask) if mask is not None else self.model(X)
                # 损失在对数空间计算
                loss = self.criterion(pred, y)
                
                # 检查模型输出和loss
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"\n错误: Batch {batch_idx + 1} 模型输出包含NaN或Inf!")
                    print(f"  pred NaN数量: {torch.isnan(pred).sum().item()}")
                    print(f"  pred Inf数量: {torch.isinf(pred).sum().item()}")
                    print(f"  pred 统计: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
                    raise ValueError("模型输出包含NaN或Inf，训练停止")
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n错误: Batch {batch_idx + 1} Loss为NaN或Inf!")
                    print(f"  pred 统计: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
                    print(f"  y 统计: min={y.min().item():.6f}, max={y.max().item():.6f}, mean={y.mean().item():.6f}")
                    raise ValueError("Loss为NaN或Inf，训练停止")
                
                # 反向传播
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 记录损失（用于实时进度显示）
            loss_value = loss.item()
            if not (math.isnan(loss_value) or math.isinf(loss_value)):
                total_loss += loss_value
            else:
                print(f"\n警告: Batch {batch_idx + 1} Loss为NaN或Inf，跳过记录")
            
            # 收集预测和目标（对数空间）
            all_preds.append(pred.detach().cpu())
            all_targets.append(y.detach().cpu())
            
            # ===== TensorBoard: 记录训练loss（每N个batch） =====
            if self.writer and (self.global_step % self.tb_log_interval == 0):
                self.writer.add_scalar('Loss_Log_Space/train_batch', loss_value, self.global_step)
            
            # ===== TensorBoard: 记录参数和梯度直方图（每M个batch） =====
            if self.writer and self.tb_log_histograms and (self.global_step % self.tb_histogram_interval == 0):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        # 记录参数分布
                        self.writer.add_histogram(f'Parameters/{name}', param.data, self.global_step)
                        # 记录梯度分布
                        if param.grad is not None:
                            self.writer.add_histogram(f'Gradients/{name}', param.grad, self.global_step)
            
            self.global_step += 1
            
            # 打印进度（显示对数空间的loss）
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                loss_display = loss_value if not (math.isnan(loss_value) or math.isinf(loss_value)) else float('nan')
                space_name = "对数空间" if self.log_transform_enabled else "原始空间"
                print(f"\r训练进度: [{batch_idx + 1}/{total_batches}] ({progress:.1f}%) | "
                      f"Loss ({space_name}): {loss_display:.6f}", end="", flush=True)
        
        # 训练完成后换行
        print()
        
        # 拼接所有预测值和真实值（对数空间）
        all_preds_log = torch.cat(all_preds, dim=0)
        all_targets_log = torch.cat(all_targets, dim=0)
        
        # 计算对数空间的损失（用于训练和early stopping）
        criterion_loss_log = self.criterion(all_preds_log, all_targets_log).item()
        
        # v0.4新增：反变换到原始空间并计算评估指标
        if self.log_transform_enabled:
            # 反变换
            all_preds_original = self._inverse_transform(all_preds_log)
            all_targets_original = self._inverse_transform(all_targets_log)
            
            # 计算原始空间的评估指标
            metrics_original = compute_metrics(
                all_targets_original.numpy(), 
                all_preds_original.numpy(),
                max_relative_error=self.max_relative_error,
                epsilon=self.epsilon
            )
            
            print(f"  训练集 - 对数空间损失: {criterion_loss_log:.6f}")
            print(f"  训练集 - 原始空间指标: MAE={metrics_original['mae']:.4f}, "
                  f"MSE={metrics_original['mse']:.2f}, "
                  f"RMSE={metrics_original['rmse']:.4f}, "
                  f"MAPE={metrics_original['mape']:.2f}%")
        else:
            # 如果没有对数变换，直接计算
            # 优先使用传入的参数，其次从criterion读取
            max_relative_error = self.max_relative_error
            epsilon = self.epsilon
            if max_relative_error is None and hasattr(self.criterion, 'max_relative_error'):
                max_relative_error = self.criterion.max_relative_error
            if hasattr(self.criterion, 'epsilon'):
                epsilon = self.criterion.epsilon
            metrics_original = compute_metrics(
                all_targets_log.numpy(), 
                all_preds_log.numpy(), 
                max_relative_error=max_relative_error, 
                epsilon=epsilon
            )
        
        # ===== TensorBoard: 记录epoch级别的训练指标 =====
        if self.writer:
            # 对数空间损失
            self.writer.add_scalar('Loss_Log_Space/train_epoch', criterion_loss_log, epoch)
            # 原始空间指标
            for metric_name, metric_value in metrics_original.items():
                self.writer.add_scalar(f'Metrics_Original/train_{metric_name}', metric_value, epoch)
        
        # 返回指标
        return {
            "loss": criterion_loss_log,  # 对数空间的损失（用于训练）
            **metrics_original  # 原始空间的指标（用于展示）
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        验证一个epoch（支持mask + 对数变换）
        
        Args:
            epoch: 当前epoch编号
        
        Returns:
            指标字典，包含：
            - loss: 对数空间的损失（用于early stopping）
            - metrics（原始空间）: mae, mse, rmse, mape, r2
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        total_batches = len(self.val_loader)
        print_interval = max(1, total_batches // 100)  # 每1%打印一次进度
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # 检测数据格式
                if len(batch_data) == 3:
                    # 支持mask的数据集
                    X, y, mask = batch_data
                    X = X.to(self.device)
                    y = y.to(self.device)  # 这里的y已经是对数空间（如果启用了变换）
                    mask = mask.to(self.device)
                else:
                    # 不支持mask的数据集
                    X, y = batch_data
                    X = X.to(self.device)
                    y = y.to(self.device)
                    mask = None
                
                # 前向传播（传递mask）
                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        pred = self.model(X, mask=mask) if mask is not None else self.model(X)
                        # 损失在对数空间计算
                        loss = self.criterion(pred, y)
                else:
                    pred = self.model(X, mask=mask) if mask is not None else self.model(X)
                    # 损失在对数空间计算
                    loss = self.criterion(pred, y)
                
                # 记录损失（用于实时进度显示）
                loss_value = loss.item()
                total_loss += loss_value
                
                # 收集预测和目标（对数空间）
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())
                
                # 打印验证进度（显示对数空间的loss）
                if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_batches:
                    progress = (batch_idx + 1) / total_batches * 100
                    space_name = "对数空间" if self.log_transform_enabled else "原始空间"
                    print(f"\r验证进度: [{batch_idx + 1}/{total_batches}] ({progress:.1f}%) | "
                          f"Loss ({space_name}): {loss_value:.6f}", end="", flush=True)
        
        # 验证完成后换行
        print()
        
        # 拼接所有预测值和真实值（对数空间）
        all_preds_log = torch.cat(all_preds, dim=0)
        all_targets_log = torch.cat(all_targets, dim=0)
        
        # 计算对数空间的损失（用于early stopping）
        criterion_loss_log = self.criterion(all_preds_log, all_targets_log).item()
        
        # v0.4新增：反变换到原始空间并计算评估指标
        if self.log_transform_enabled:
            # 反变换
            all_preds_original = self._inverse_transform(all_preds_log)
            all_targets_original = self._inverse_transform(all_targets_log)
            
            # 计算原始空间的评估指标
            metrics_original = compute_metrics(
                all_targets_original.numpy(), 
                all_preds_original.numpy(),
                max_relative_error=self.max_relative_error,
                epsilon=self.epsilon
            )
            
            print(f"  验证集 - 对数空间损失: {criterion_loss_log:.6f}")
            print(f"  验证集 - 原始空间指标: MAE={metrics_original['mae']:.4f}, "
                  f"MSE={metrics_original['mse']:.2f}, "
                  f"RMSE={metrics_original['rmse']:.4f}, "
                  f"MAPE={metrics_original['mape']:.2f}%")
        else:
            # 如果没有对数变换，直接计算
            # 优先使用传入的参数，其次从criterion读取
            max_relative_error = self.max_relative_error
            epsilon = self.epsilon
            if max_relative_error is None and hasattr(self.criterion, 'max_relative_error'):
                max_relative_error = self.criterion.max_relative_error
            if hasattr(self.criterion, 'epsilon'):
                epsilon = self.criterion.epsilon
            metrics_original = compute_metrics(
                all_targets_log.numpy(), 
                all_preds_log.numpy(), 
                max_relative_error=max_relative_error, 
                epsilon=epsilon
            )
        
        # ===== TensorBoard: 记录验证指标 =====
        if self.writer:
            # 对数空间损失
            self.writer.add_scalar('Loss_Log_Space/val_epoch', criterion_loss_log, epoch)
            # 原始空间指标
            for metric_name, metric_value in metrics_original.items():
                self.writer.add_scalar(f'Metrics_Original/val_{metric_name}', metric_value, epoch)
            
            # ===== TensorBoard: 可视化预测 vs 真实值（原始空间） =====
            # 每隔几个epoch记录一次散点图，避免过多I/O
            if epoch % 5 == 0:
                # 采样前1000个样本进行可视化
                sample_size = min(1000, len(all_preds_log))
                
                if self.log_transform_enabled:
                    # 使用原始空间的值绘图
                    sample_preds = all_preds_original[:sample_size].numpy().flatten()
                    sample_targets = all_targets_original[:sample_size].numpy().flatten()
                else:
                    sample_preds = all_preds_log[:sample_size].numpy().flatten()
                    sample_targets = all_targets_log[:sample_size].numpy().flatten()
                
                # 使用matplotlib创建散点图
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(sample_targets, sample_preds, alpha=0.5, s=10)
                    
                    # 绘制y=x参考线
                    min_val = min(sample_targets.min(), sample_preds.min())
                    max_val = max(sample_targets.max(), sample_preds.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                    
                    ax.set_xlabel('True Values (Original Space)')
                    ax.set_ylabel('Predictions (Original Space)')
                    ax.set_title(f'Prediction vs True (Epoch {epoch})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    self.writer.add_figure('Predictions/scatter_plot_original', fig, epoch)
                    plt.close(fig)
                except ImportError:
                    pass  # matplotlib未安装，跳过可视化
        
        # 返回指标
        return {
            "loss": criterion_loss_log,  # 对数空间的损失（用于early stopping）
            **metrics_original  # 原始空间的指标（用于展示）
        }
    
    def train(self, num_epochs: int) -> Dict[str, list]:
        """
        完整训练循环
        
        Args:
            num_epochs: 训练轮数
        
        Returns:
            训练历史字典
        """
        print(f"开始训练，设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        if self.tensorboard_enabled:
            print(f"TensorBoard已启用 (每 {self.tb_log_interval} batch记录标量)")
        if self.log_transform_enabled:
            print(f"对数变换已启用 (offset={self.log_offset:.4f})")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # ===== TensorBoard: 记录学习率 =====
            if self.writer:
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 更新学习率（基于对数空间的损失）
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # 记录训练指标
            self.train_history["loss"].append(train_metrics["loss"])  # 对数空间的损失
            self.train_history["metrics"].append(train_metrics)  # 原始空间的指标
            self.train_history["learning_rate"].append(current_lr)
            
            # 验证
            if epoch % self.val_interval == 0:
                val_metrics = self.validate_epoch(epoch)
                self.train_history["val_loss"].append(val_metrics["loss"])  # 对数空间的损失
                self.train_history["val_metrics"].append(val_metrics)  # 原始空间的指标
                
                # 检查是否是最佳模型（使用对数空间的loss）
                current_metric_value = val_metrics["loss"]  # 始终使用loss进行模型选择
                is_better = False
                
                if self.best_metric_mode == "min":
                    if current_metric_value < self.best_val_metric:
                        is_better = True
                else:  # max
                    if current_metric_value > self.best_val_metric:
                        is_better = True
                
                if is_better:
                    self.best_val_metric = current_metric_value
                    self.best_epoch = epoch
                    
                    # 保存最佳模型
                    if self.save_best and self.save_dir:
                        self.save_checkpoint(epoch, is_best=True)
                
                # 早停检查（使用对数空间的loss）
                if self.early_stopping:
                    early_stop_metric = val_metrics["loss"]
                    if self.early_stopping(early_stop_metric):
                        print(f"\n早停触发于epoch {epoch}")
                        break
            else:
                val_metrics = None
            
            # 打印进度
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_info(epoch, train_metrics, val_metrics, epoch_time, current_lr)
            
            # 定期保存模型
            if epoch % self.save_interval == 0 and self.save_dir:
                self.save_checkpoint(epoch, is_best=False)
            
            # 记录日志
            if self.log_file:
                self._log_epoch(epoch, train_metrics, val_metrics)
            
            # 保存训练历史
            if self.history_file:
                self._save_history()
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")
        print(f"最佳验证损失（对数空间）: {self.best_val_metric:.6f} (epoch {self.best_epoch})")
        
        # 关闭TensorBoard writer
        if self.writer:
            self.writer.close()
            print("TensorBoard日志已保存")
        
        return self.train_history
    
    def _print_epoch_info(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        epoch_time: float,
        lr: float
    ):
        """打印epoch信息，显示所有评估指标（原始空间）"""
        # 第一行：基本信息
        print(f"{'='*100}")
        print(f"Epoch {epoch:4d} | LR: {lr:.2e} | Time: {epoch_time:.2f}s")
        if self.log_transform_enabled:
            print(f"  训练Loss (对数空间): {train_metrics['loss']:.6f}")
            if val_metrics:
                print(f"  验证Loss (对数空间): {val_metrics['loss']:.6f}")
        print(f"{'-'*100}")
        
        # 第二行：训练集指标（原始空间）
        print(f"Train  | "
              f"MAE: {train_metrics.get('mae', 0):.6f} | "
              f"MSE: {train_metrics.get('mse', 0):.2f} | "
              f"RMSE: {train_metrics.get('rmse', 0):.6f} | "
              f"MAPE: {train_metrics.get('mape', 0):.2f}% | "
              f"R²: {train_metrics.get('r2', 0):.6f}")
        
        # 第三行：验证集指标（原始空间，如果有）
        if val_metrics:
            print(f"Val    | "
                  f"MAE: {val_metrics.get('mae', 0):.6f} | "
                  f"MSE: {val_metrics.get('mse', 0):.2f} | "
                  f"RMSE: {val_metrics.get('rmse', 0):.6f} | "
                  f"MAPE: {val_metrics.get('mape', 0):.2f}% | "
                  f"R²: {val_metrics.get('r2', 0):.6f}")
        
        print(f"{'='*100}")
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ):
        """记录epoch日志到文件"""
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train": train_metrics,
            "val": val_metrics
        }
        
        log_file = Path(self.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def _save_history(self):
        """保存训练历史到文件"""
        history_file = Path(self.history_file)
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, ensure_ascii=False, indent=2)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存模型检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否是最佳模型
        """
        if self.save_dir is None:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'best_metric': self.best_metric,
            'best_metric_mode': self.best_metric_mode,
            'train_history': self.train_history,
            # v0.4新增：保存对数变换信息
            'log_transform_enabled': self.log_transform_enabled,
            'log_offset': self.log_offset
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最佳模型（只保存best_model.pth，不保存checkpoint_epoch文件）
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
        else:
            # 保存常规检查点（定期保存时）
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"保存检查点到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            load_optimizer: 是否加载优化器状态
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        # 加载对数变换信息
        if 'log_transform_enabled' in checkpoint:
            self.log_transform_enabled = checkpoint['log_transform_enabled']
        if 'log_offset' in checkpoint:
            self.log_offset = checkpoint['log_offset']
        
        # 向后兼容：优先加载新的 best_val_metric，否则尝试 best_val_loss
        if 'best_val_metric' in checkpoint:
            self.best_val_metric = checkpoint['best_val_metric']
        elif 'best_val_loss' in checkpoint:
            self.best_val_metric = checkpoint['best_val_loss']
        
        print(f"加载检查点: {checkpoint_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        best_metric_name = checkpoint.get('best_metric', self.best_metric)
        best_metric_value = checkpoint.get('best_val_metric', checkpoint.get('best_val_loss', 'unknown'))
        print(f"Best Val Metric ({best_metric_name}): {best_metric_value}")
        if self.log_transform_enabled:
            print(f"对数变换: enabled (offset={self.log_offset:.4f})")
