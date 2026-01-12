# 如何运行训练 - 完整指南

## 🚀 运行训练脚本

### 运行文件位置

**文件路径**：`scripts/v0.1_20251212/train.py`

### 运行方法

#### 方法1：使用默认配置（最简单）

```bash
cd /data/project_20251211
python scripts/v0.1_20251212/train.py
```

#### 方法2：指定配置版本

```bash
python scripts/v0.1_20251212/train.py --config_version v0.1_20251212
```

#### 方法3：使用自定义配置文件

```bash
python scripts/v0.1_20251212/train.py \
  --model_config /path/to/model_config.yaml \
  --training_config /path/to/training_config.yaml \
  --dataset_config /path/to/dataset_config.yaml
```

#### 方法4：使用虚拟环境（推荐）

```bash
cd /data/project_20251211
conda activate stock_prediction_20251211  # 或你的虚拟环境名称
python scripts/v0.1_20251212/train.py
```

---

## 📺 训练时会打印的信息

### 1. 初始化阶段

```
================================================================================
iTransformer 训练脚本
================================================================================
模型配置: /data/project_20251211/configs/v0.1_20251212/itransformer_config.yaml
训练配置: /data/project_20251211/configs/v0.1_20251212/training_config.yaml
数据集配置: /data/project_20251211/configs/v0.1_20251212/dataset_config.yaml
================================================================================

加载配置文件...

输出目录: /data/project_20251211/experiments/runs/itransformer_v0.1_20251212_20251212_143022
```

### 2. 数据集初始化阶段

```
初始化数据集...
计算训练集特征统计量...
特征统计量计算完成，共 43 个特征
特征统计量已保存到: /data/project_20251211/experiments/runs/.../feature_stats.json

训练集样本数: 21684
验证集样本数: 11478
特征数量: 43
序列长度: 1000
```

### 3. 自动调整信息（如果有）

```
警告: 配置中的特征数量 (40) 与实际特征数量 (43) 不同
更新模型配置中的特征数量为: 43

自动调整输入ResNet第一个模块的维度:
  配置中的维度: 40
  实际特征数量: 43
  更新第一个模块: [40, 40] -> [43, 43]
```

### 4. 模型创建阶段

```
创建数据加载器...

初始化模型...
模型参数数量: 2,345,678

创建优化器...
创建学习率调度器...
创建损失函数...

创建训练器...

保存配置...
```

### 5. 训练阶段

```
================================================================================
开始训练
================================================================================
开始训练，设备: cuda
训练样本数: 21684
验证样本数: 11478
模型参数数量: 2,345,678
--------------------------------------------------------------------------------

Epoch    1 | Train Loss: 0.123456 | Train MAE: 0.345678 | LR: 1.00e-04 | Time: 45.23s | Val Loss: 0.112345 | Val MAE: 0.323456
Epoch    2 | Train Loss: 0.098765 | Train MAE: 0.298765 | LR: 1.00e-04 | Time: 44.56s | Val Loss: 0.095432 | Val MAE: 0.287654
Epoch    3 | Train Loss: 0.087654 | Train MAE: 0.276543 | LR: 1.00e-04 | Time: 44.89s | Val Loss: 0.084321 | Val MAE: 0.265432
...
保存最佳模型到: /data/project_20251211/experiments/runs/.../checkpoints/best_model.pth
...
```

### 6. 训练完成

```
训练完成！总耗时: 2.35小时
最佳验证损失: 0.045678 (epoch 45)

训练完成！所有结果已保存到: /data/project_20251211/experiments/runs/itransformer_v0.1_20251212_20251212_143022
```

---

## 📊 训练过程中的关键信息解读

### Epoch信息格式

```
Epoch    1 | Train Loss: 0.123456 | Train MAE: 0.345678 | LR: 1.00e-04 | Time: 45.23s | Val Loss: 0.112345 | Val MAE: 0.323456
```

**各部分含义**：
- `Epoch 1`：当前是第几个epoch
- `Train Loss`：训练集损失（越小越好）
- `Train MAE`：训练集平均绝对误差（越小越好）
- `LR`：当前学习率
- `Time`：这个epoch耗时（秒）
- `Val Loss`：验证集损失（越小越好）
- `Val MAE`：验证集平均绝对误差（越小越好）

### 正常训练的表现

**好的训练**：
- ✅ 训练损失和验证损失都在下降
- ✅ 验证损失略高于训练损失（正常）
- ✅ 损失值逐渐稳定

**异常情况**：
- ⚠️ 验证损失上升：可能过拟合
- ⚠️ 损失不下降：学习率可能太小，或模型太简单
- ⚠️ 损失震荡：学习率可能太大

---

## 🔍 如何查看训练进度

### 实时查看

训练脚本会在终端实时打印每个epoch的信息。

### 查看日志文件

训练日志保存在：
```
experiments/runs/itransformer_v0.1_20251212_YYYYMMDD_HHMMSS/training_log.jsonl
```

每行是一个epoch的记录，格式为JSON。

### 查看训练历史

训练完成后，可以查看：
```
experiments/runs/itransformer_v0.1_20251212_YYYYMMDD_HHMMSS/training_history.json
```

包含所有训练指标的历史记录。

---

## ⚠️ 常见问题

### Q1: 运行后没有输出？

**检查**：
1. 是否在正确的目录：`cd /data/project_20251211`
2. Python路径是否正确
3. 是否有权限执行

### Q2: 出现 "CUDA out of memory" 错误？

**解决方法**：
1. 减小 `batch_size`（在 `training_config.yaml` 中）
2. 减小模型大小（`d_model`, `n_layers` 等）
3. 使用CPU训练（`device: "cpu"`）

### Q3: 训练很慢？

**可能原因**：
1. 使用CPU而不是GPU
2. `batch_size` 太小
3. `num_workers` 设置不合理

**优化建议**：
- 使用GPU：`device: "cuda"`
- 增大 `batch_size`（如果内存允许）
- 设置 `num_workers: 4` 或更多

---

## 📝 完整运行示例

```bash
# 1. 进入项目目录
cd /data/project_20251211

# 2. 激活虚拟环境（如果有）
# conda activate stock_prediction_20251211

# 3. 运行训练
python scripts/v0.1_20251212/train.py

# 4. 等待训练完成（可能需要几小时）
# 训练过程中会实时显示进度

# 5. 训练完成后，查看结果
# 结果保存在：experiments/runs/itransformer_v0.1_20251212_YYYYMMDD_HHMMSS/
```

---

## 🎯 快速检查清单

训练前检查：
- [ ] 配置文件路径正确
- [ ] 数据文件存在
- [ ] GPU可用（如果使用CUDA）
- [ ] 虚拟环境已激活（如果需要）

训练中观察：
- [ ] 损失是否下降
- [ ] 验证损失是否正常
- [ ] 是否有错误信息

训练后检查：
- [ ] 最佳模型已保存
- [ ] 训练历史已记录
- [ ] 配置文件已备份

