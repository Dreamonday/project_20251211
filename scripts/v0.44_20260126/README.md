# TimeXer v0.44 训练脚本

版本：v0.44
日期：2026-01-26

## 核心改进

### 问题
不同公司的股票价格差异巨大（-10到+1000），导致：
1. 模型难以处理大跨度预测
2. MAPE/SMAPE损失函数对接近0的值过度敏感
3. 不同价格尺度的股票损失不可比

### 解决方案

#### 1. 对数变换
- 目标值进行对数变换：`y_log = log(y + offset)`
- offset自动计算：`abs(min(y)) + 1.0`
- 效果：将[-10, 1000]压缩到[0, 7]左右的对数空间
- 好处：不同价格尺度的股票误差变得可比

#### 2. Huber Loss
- 替代MAPE/SMAPE，在对数空间计算
- delta=1.0：小误差用MSE（平滑），大误差用MAE（鲁棒）
- 不需要max_relative_error：对数空间 + Huber已经足够稳定
- 自带截断机制，不会损失爆炸

#### 3. 双空间评估
- **训练损失**：对数空间Huber Loss（用于梯度更新和early stopping）
- **评估指标**：原始价格空间（MAE, MSE, RMSE, MAPE, R²）（给用户看）
- TensorBoard同时记录两个空间的指标

## 文件结构

```
v0.44_20260126/
├── train_timexer.py          # 训练脚本
└── README.md                  # 本文件

/data/project_20251211/
├── src/
│   ├── data/
│   │   └── v0.4_20260126/
│   │       └── preprocessed_dataset.py  # 支持对数变换的数据集
│   └── training/
│       └── v0.4_20260126/
│           └── trainer.py     # 支持对数变换的训练器
└── configs/
    └── v0.44_20260126/
        ├── training_config.yaml   # Huber Loss + 对数变换配置
        └── timexer_config.yaml    # 模型配置（继承v0.43）
```

## 使用方法

### 训练命令

```bash
cd /data/project_20251211/scripts/v0.44_20260126

# 使用默认配置（对数变换 + Huber Loss）
python train_timexer.py --config_version v0.44_20260126 --use_preprocessed

# 使用自定义预处理数据目录
python train_timexer.py \
    --config_version v0.44_20260126 \
    --use_preprocessed \
    --preprocessed_dir data/processed/preprocess_data_v1.0_20260119170929_500120
```

### 配置说明

#### training_config.yaml
```yaml
training:
  loss:
    type: "huber"  # 使用Huber Loss
    delta: 1.0     # 分界点参数
  
  best_metric: "loss"      # 使用对数空间loss选择最佳模型
  best_metric_mode: "min"
  
  early_stopping:
    metric: "loss"         # 使用对数空间loss进行early stopping
    mode: "min"
```

#### 数据集参数（在train_timexer.py中）
```python
train_dataset = PreprocessedStockDataset(
    pt_file_path=str(train_pt_file),
    log_transform=True,      # 启用对数变换
    log_offset=None,         # 自动计算offset
    log_margin=1.0,          # offset边距
    # ... 其他参数
)
```

## 训练输出

### 控制台输出示例
```
Epoch 1/15
训练进度: [500/500] (100.0%) | Loss (对数空间): 0.156234
  训练集 - 对数空间损失: 0.156234
  训练集 - 原始空间指标: MAE=8.92, MSE=123.45, RMSE=11.11, MAPE=12.34%

验证进度: [200/200] (100.0%) | Loss (对数空间): 0.178456
  验证集 - 对数空间损失: 0.178456
  验证集 - 原始空间指标: MAE=10.23, MSE=178.45, RMSE=13.36, MAPE=18.34%

================================================================================
Epoch    1 | LR: 1.00e-04 | Time: 45.23s
  训练Loss (对数空间): 0.156234
  验证Loss (对数空间): 0.178456
--------------------------------------------------------------------------------
Train  | MAE: 8.920000 | MSE: 123.45 | RMSE: 11.110000 | MAPE: 12.34% | R²: 0.856000
Val    | MAE: 10.230000 | MSE: 178.45 | RMSE: 13.360000 | MAPE: 18.34% | R²: 0.823000
================================================================================
```

### TensorBoard可视化

启动TensorBoard：
```bash
tensorboard --logdir=/data/project_20251211/experiments/timexer_v0.44_*/tensorboard
```

可视化内容：
- **Loss_Log_Space/**: 对数空间的训练和验证损失
- **Metrics_Original/**: 原始空间的评估指标
  - train_mae, train_mse, train_rmse, train_mape, train_r2
  - val_mae, val_mse, val_rmse, val_mape, val_r2
- **Learning_Rate**: 学习率变化
- **Predictions/scatter_plot_original**: 预测vs真实值散点图（原始空间）

## 关键优势

### 1. 数值稳定性
- 对数空间误差范围小且均匀
- Huber Loss自带截断，不会爆炸
- 无需手动设置max_relative_error

### 2. 模型性能
- 不同价格尺度的股票损失可比
- 梯度更平滑，训练更稳定
- 模型关注相对变化而非绝对值

### 3. 评估直观性
- 评估指标在原始价格空间
- 用户可以直接理解MAE、MAPE等指标
- TensorBoard同时展示两个空间

### 4. 使用便捷
- 自动检测和应用对数变换
- 验证时自动反变换
- 无需手动处理，一切自动完成

## 技术细节

### 对数变换流程

```
数据加载（Dataset.__getitem__）:
  原始价格 → [+offset] → [log] → 对数空间

训练（Trainer.train_epoch）:
  对数空间预测 → Huber Loss → 反向传播

验证（Trainer.validate_epoch）:
  对数空间预测 → Huber Loss → 早停检查
  对数空间预测 → [exp] → [-offset] → 原始空间
  原始空间 → 计算指标 → 展示给用户
```

### Early Stopping策略
- 使用对数空间的Huber Loss
- 与训练目标完全一致
- 最佳模型也基于对数空间loss选择

### 模型保存
- 检查点包含对数变换信息
- 推理时自动使用正确的offset
- 向后兼容：可以加载v0.43模型

## 与v0.43的对比

| 特性 | v0.43 | v0.44 |
|------|-------|-------|
| 损失函数 | SMAPE + clip | Huber |
| 数据空间 | 原始价格 | 对数空间 |
| 损失范围 | [0, 2] | 连续实数 |
| 大跨度处理 | 依赖clip | 对数压缩 |
| 不同尺度 | 不可比 | 可比 |
| 数值稳定性 | 需要epsilon | 天然稳定 |
| 评估空间 | 原始 | 双空间 |

## 注意事项

1. **数据预处理**：无需修改已有的预处理数据，对数变换在数据加载时动态完成
2. **offset一致性**：验证集必须使用与训练集相同的offset
3. **负值处理**：offset自动处理负值，无需担心log(负数)
4. **推理时**：需要从模型或数据集metadata中读取offset进行反变换

## 未来改进

1. 可选配置：对数变换可以通过配置文件开关
2. 自适应offset：根据数据分布动态调整offset
3. 多目标支持：支持多个目标列的独立变换
4. 变换选择：支持其他变换方法（如Box-Cox）

## 相关文件

- 数据集：`/data/project_20251211/src/data/v0.4_20260126/preprocessed_dataset.py`
- 训练器：`/data/project_20251211/src/training/v0.4_20260126/trainer.py`
- 配置：`/data/project_20251211/configs/v0.44_20260126/`
- 模型：继续使用`/data/project_20251211/src/models/v0.43_20260119/`（无需修改）
