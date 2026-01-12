# iTransformer 股票预测模型训练指南

## 📚 目录
1. [配置文件位置](#配置文件位置)
2. [模型配置详解](#模型配置详解)
3. [训练配置详解](#训练配置详解)
4. [数据集配置详解](#数据集配置详解)
5. [如何运行训练](#如何运行训练)
6. [输出文件说明](#输出文件说明)
7. [常见问题](#常见问题)

---

## 📁 配置文件位置

所有配置文件都在 `configs/v0.1_20251212/` 目录下：

```
configs/v0.1_20251212/
├── itransformer_config.yaml    # 模型架构配置
├── training_config.yaml         # 训练过程配置
└── dataset_config.yaml          # 数据集配置
```

**重要提示**：这三个文件是YAML格式，使用空格缩进（不要用Tab），冒号后面要有空格。

---

## 🤖 模型配置详解 (`itransformer_config.yaml`)

这个文件控制模型的架构，就像设计房子的蓝图。

### 1. 基础维度设置

```yaml
input_features: 45    # 输入特征数量（会自动检测，一般不用改）
seq_len: 1000         # 输入序列长度（历史数据的天数）
output_dim: 1         # 输出维度（预测1个值：收盘价）
```

**说明**：
- `input_features`：你的数据有多少个特征列（股价、成交量、财务数据等）
- `seq_len`：用多少天的历史数据来预测（1000天≈4年）
- `output_dim`：预测几个值（1表示只预测收盘价）

### 2. 模型核心参数

```yaml
d_model: 512          # 模型隐藏维度（越大模型越复杂，学习能力越强）
n_layers: 6           # Decoder层数（层数越多，模型越深）
n_heads: 8            # 注意力头数（多头注意力，一般8或16）
d_ff: 2048            # Feed-Forward网络隐藏维度（通常是d_model的4倍）
dropout: 0.1          # Dropout比率（防止过拟合，0.1表示10%的神经元随机关闭）
activation: "gelu"    # 激活函数："gelu"（推荐）或"relu"
```

**参数选择建议**：
- **小模型**（快速实验）：`d_model: 256, n_layers: 3, n_heads: 4`
- **中等模型**（推荐）：`d_model: 512, n_layers: 6, n_heads: 8`（当前配置）
- **大模型**（高性能）：`d_model: 1024, n_layers: 12, n_heads: 16`

**Dropout说明**：
- 0.0 = 不使用Dropout（可能过拟合）
- 0.1 = 轻微正则化（推荐）
- 0.3 = 强正则化（如果模型过拟合严重）

### 3. Decoder层配置

```yaml
decoder:
  use_causal_mask: true    # 是否使用因果掩码（时间序列必须用true）
  norm_type: "pre"         # 归一化类型："pre"（推荐）或"post"
  attention_dropout: 0.1   # 注意力层Dropout
  ff_dropout: 0.1          # Feed-Forward层Dropout
  use_bias: true           # 是否使用bias（一般用true）
```

**说明**：
- `use_causal_mask: true`：确保模型只能看到过去的数据，不能看到未来（必须保持true）
- `norm_type: "pre"`：Pre-Norm通常训练更稳定（推荐）
- `norm_type: "post"`：Post-Norm是传统Transformer的方式

### 4. 输入ResNet模块（可选）

```yaml
input_resnet:
  enabled: true            # 是否启用（true=使用，false=不使用）
  modules:
    - name: "input_resnet_1"
      layers: 2             # 这个模块有几层
      hidden_dims: [256, 128]  # 每层的维度（列表长度要等于layers）
      dropout: 0.1
      activation: "gelu"
      use_bias: true
```

**作用**：在数据进入Transformer之前，先用全连接层处理一下。

**如何设置**：
- `enabled: false`：不使用（模型更简单，训练更快）
- `enabled: true`：使用（可以增强模型表达能力）

**示例配置**：
```yaml
# 简单配置（1层）
modules:
  - name: "input_resnet_1"
    layers: 1
    hidden_dims: [256]

# 复杂配置（多层）
modules:
  - name: "input_resnet_1"
    layers: 3
    hidden_dims: [512, 256, 128]
```

### 5. 输出ResNet模块（可选）

```yaml
output_resnet:
  enabled: true
  modules:
    - name: "output_resnet_1"
      layers: 2
      hidden_dims: [256, 128]
      dropout: 0.1
      activation: "gelu"
      use_bias: true
```

**作用**：在Transformer输出之后，再用全连接层处理一下，再输出最终结果。

**设置方式**：和输入ResNet一样

### 6. 最终输出层

```yaml
final_output:
  hidden_dim: 256      # 最终全连接层的隐藏维度
  dropout: 0.1         # Dropout比率
  output_dim: 1        # 输出维度（预测收盘价=1）
```

**说明**：这是模型的最后一层，把特征转换成预测值。

---

## 🏋️ 训练配置详解 (`training_config.yaml`)

这个文件控制训练过程，就像训练计划。

### 1. 基础训练参数

```yaml
batch_size: 32         # 每次训练多少个样本（越大训练越快，但需要更多内存）
num_epochs: 100        # 训练多少轮（把所有数据过一遍算1轮）
num_workers: 4         # 数据加载的线程数（CPU核心数的一半）
pin_memory: true       # 加速GPU数据传输（有GPU就用true）
```

**参数选择建议**：
- **batch_size**：
  - GPU内存8GB：16或32
  - GPU内存16GB：32或64
  - GPU内存24GB+：64或128
- **num_epochs**：从50开始，如果还没收敛就增加到100或200

### 2. 优化器配置

```yaml
optimizer:
  type: "adamw"        # 优化器类型："adam"、"adamw"（推荐）、"sgd"
  lr: 0.0001          # 学习率（最重要！控制学习速度）
  weight_decay: 0.01   # 权重衰减（防止过拟合）
  betas: [0.9, 0.999]  # Adam优化器的参数（一般不改）
```

**学习率选择**：
- **太大**（如0.01）：训练不稳定，损失震荡
- **太小**（如0.00001）：训练太慢，可能不收敛
- **推荐范围**：0.0001 ~ 0.001

**优化器选择**：
- `adamw`：推荐，训练稳定，效果好
- `adam`：也可以，但不如AdamW
- `sgd`：传统方法，需要更多调参

### 3. 学习率调度器

```yaml
scheduler:
  type: "cosine"       # 类型："cosine"（推荐）、"step"、"plateau"、"none"
  T_max: 100          # Cosine退火的最大周期数（通常等于num_epochs）
  eta_min: 1e-6       # 最小学习率
  step_size: 30       # StepLR：每30个epoch降低一次学习率
  gamma: 0.1          # StepLR：每次降低到原来的10%
  factor: 0.5         # Plateau：每次降低到原来的50%
  patience: 10        # Plateau：10个epoch没改进就降低学习率
```

**调度器选择**：
- `cosine`：学习率像余弦波一样平滑下降（推荐）
- `step`：每隔固定epoch降低学习率
- `plateau`：验证损失不下降时降低学习率
- `none`：学习率不变

### 4. 损失函数

```yaml
loss:
  type: "mse"         # 类型："mse"（均方误差，推荐）、"mae"（平均绝对误差）、"huber"
  reduction: "mean"   # 一般用"mean"
```

**损失函数选择**：
- `mse`：对异常值敏感，适合大多数情况（推荐）
- `mae`：对异常值不敏感，更稳健
- `huber`：介于MSE和MAE之间

### 5. 验证和保存

```yaml
val_interval: 1        # 每1个epoch验证一次（1表示每个epoch都验证）
save_interval: 10      # 每10个epoch保存一次模型检查点
save_best: true        # 是否保存最佳模型（推荐true）
early_stopping:
  enabled: true        # 是否启用早停（防止过拟合）
  patience: 20        # 20个epoch没改进就停止训练
  min_delta: 0.0001   # 最小改进阈值（改进小于这个值不算改进）
```

**说明**：
- `val_interval: 1`：每个epoch都验证，可以看到训练进度
- `save_best: true`：自动保存验证损失最低的模型
- `early_stopping`：如果模型不再改进，自动停止训练，节省时间

### 6. 设备配置

```yaml
device: "cuda"         # "cuda"（有GPU）或"cpu"（只有CPU）
mixed_precision: false # 是否使用混合精度训练（可以加速，但可能不稳定）
```

**说明**：
- 有NVIDIA GPU：用`"cuda"`
- 只有CPU：用`"cpu"`（训练会很慢）
- `mixed_precision: true`：可以加速训练，但需要GPU支持

---

## 📊 数据集配置详解 (`dataset_config.yaml`)

这个文件告诉程序在哪里找数据，用什么数据训练。

### 1. 索引文件路径

```yaml
index_dir: "/data/project_20251211/data/processed/roll_generate_index_v0.2_20251212_171113"
train_index_file: "train_samples_index.parquet"
val_index_file: "val_samples_index.parquet"
```

**说明**：
- `index_dir`：索引文件所在的文件夹路径
- `train_index_file`：训练集索引文件名
- `val_index_file`：验证集索引文件名

**如何修改**：如果用了新的索引文件，修改`index_dir`路径即可。

### 2. 原始数据路径

```yaml
data_dir: "/data/project_20251211/data/raw/processed_data_20251212"
```

**说明**：原始股票数据文件所在的文件夹。

### 3. 特征配置（重要！）

这是**设置哪些特征用于训练，哪个特征作为输出**的地方。

```yaml
features:
  exclude_columns:    # 需要排除的列（这些列不用于训练）
    - "日期"
    - "company_name"
    - "sequence_id"
    - "stock_code"
    - "货币单位"
  
  target_column: "收盘"    # 要预测的列（输出目标）
  
  normalize: true          # 是否标准化特征（推荐true）
  normalize_method: "standard"  # 标准化方法："standard"（Z-score）或"minmax"
```

#### 如何选择输入特征（用于训练的特征）

**原理**：所有列 - 排除列 = 输入特征

例如，如果你的数据有48列：
- 排除了5列（日期、公司名等）
- 那么输入特征 = 48 - 5 = **43个特征**

**如何修改**：
1. 查看你的数据文件有哪些列
2. 在`exclude_columns`列表中添加不想用于训练的列名
3. 剩下的列会自动成为输入特征

**示例**：如果你想排除更多列（比如"货币单位"和"sequence_id"），可以这样写：
```yaml
exclude_columns:
  - "日期"
  - "company_name"
  - "sequence_id"
  - "stock_code"
  - "货币单位"
  - "其他不想用的列名"
```

#### 如何选择输出特征（要预测的目标）

**设置位置**：`target_column` 参数

**当前设置**：`"收盘"` - 预测收盘价

**如何修改**：改成你想预测的列名，例如：
```yaml
target_column: "收盘"      # 预测收盘价
# 或者
target_column: "开盘"      # 预测开盘价
# 或者
target_column: "成交量"    # 预测成交量
```

**注意**：`target_column`不能出现在`exclude_columns`列表中！

#### 自动维度调整

**好消息**：选择好特征后，模型会**自动调整**输入和输出维度！

**工作原理**：
1. 训练脚本会自动计算实际的特征数量（总列数 - 排除列数）
2. 如果配置文件中的`input_features`与实际不符，会自动更新
3. 模型会使用正确的特征数量创建

**你不需要手动修改** `itransformer_config.yaml` 中的 `input_features`，程序会自动处理！

**示例输出**：
```
特征数量: 43
警告: 配置中的特征数量 (45) 与实际特征数量 (43) 不同
更新模型配置中的特征数量为: 43
```

**说明**：
- `exclude_columns`：这些是元数据列，不是特征，所以要排除
- `target_column`：要预测什么（"收盘"表示预测收盘价）
- `normalize: true`：标准化可以让训练更稳定（强烈推荐）
- `normalize_method`：
  - `"standard"`：Z-score标准化（推荐）
  - `"minmax"`：缩放到0-1之间

### 4. 数据加载配置

```yaml
cache_enabled: true    # 是否启用文件缓存（推荐true，加速数据加载）
cache_size: 100        # 缓存文件数量上限（100表示最多缓存100个公司的数据文件）
```

**说明**：
- `cache_enabled: true`：启用缓存，重复读取同一文件时直接从内存读取，更快
- `cache_size: 100`：如果内存够大，可以增加到200或更多

---

## 🚀 如何运行训练

### 方法1：使用默认配置（最简单）

```bash
cd /data/project_20251211
python scripts/v0.1_20251212/train.py
```

### 方法2：指定配置版本

```bash
python scripts/v0.1_20251212/train.py --config_version v0.1_20251212
```

### 方法3：使用自定义配置文件

```bash
python scripts/v0.1_20251212/train.py \
  --model_config /path/to/your/model_config.yaml \
  --training_config /path/to/your/training_config.yaml \
  --dataset_config /path/to/your/dataset_config.yaml
```

### 完整示例

```bash
# 1. 进入项目目录
cd /data/project_20251211

# 2. 激活虚拟环境（如果有）
# conda activate stock_prediction_20251211

# 3. 运行训练
python scripts/v0.1_20251212/train.py
```

---

## 📂 输出文件说明

训练完成后，所有结果会保存在：

```
experiments/runs/itransformer_v0.1_20251212_YYYYMMDD_HHMMSS/
├── checkpoints/                    # 模型检查点
│   ├── best_model.pth             # 最佳模型（验证损失最低）
│   ├── checkpoint_epoch_10.pth    # 第10个epoch的检查点
│   ├── checkpoint_epoch_20.pth    # 第20个epoch的检查点
│   └── ...
├── model_config.yaml              # 模型配置（备份）
├── training_config.yaml           # 训练配置（备份）
├── dataset_config.yaml            # 数据集配置（备份）
├── feature_stats.json             # 特征统计量（用于标准化）
├── training_log.jsonl             # 训练日志（每行一个epoch的记录）
└── training_history.json          # 训练历史（所有指标）
```

**重要文件**：
- `best_model.pth`：这是最好的模型，用于后续预测
- `training_history.json`：包含所有训练指标，可以画图分析

---

## ❓ 常见问题

### Q1: 如何知道模型参数数量？

运行训练脚本时，会打印：
```
模型参数数量: 2,345,678
```

参数越多，模型越复杂，需要更多内存和计算时间。

### Q2: 训练需要多长时间？

取决于：
- 数据量（样本数）
- 模型大小（参数数量）
- GPU性能
- batch_size大小

**估算**：
- 小模型（100万参数）+ 3万样本：约1-2小时
- 中等模型（500万参数）+ 3万样本：约3-5小时
- 大模型（1000万参数）+ 3万样本：约6-10小时

### Q3: 如何判断训练是否正常？

**正常情况**：
- 训练损失逐渐下降
- 验证损失也逐渐下降（但可能比训练损失高）
- 学习率按调度器设置变化

**异常情况**：
- 损失不下降：学习率可能太小，或模型太简单
- 损失震荡：学习率可能太大
- 验证损失上升：可能过拟合，需要增加dropout或减少模型复杂度

### Q4: 如何选择batch_size？

**原则**：
1. 先用小的（如16或32）确保能运行
2. 如果GPU内存还有剩余，逐渐增大
3. 如果出现"out of memory"错误，减小batch_size

**参考**：
- GPU内存8GB：batch_size = 16
- GPU内存16GB：batch_size = 32-64
- GPU内存24GB+：batch_size = 64-128

### Q5: 如何调整学习率？

**策略**：
1. 从0.0001开始
2. 如果损失下降很慢：增大到0.0005或0.001
3. 如果损失震荡：减小到0.00005或0.00001

**经验值**：
- 小模型：0.001
- 中等模型：0.0001（推荐）
- 大模型：0.00005

### Q6: 训练中断了怎么办？

检查点文件会定期保存，可以：
1. 找到最新的检查点文件（如`checkpoint_epoch_50.pth`）
2. 修改代码加载检查点继续训练（需要修改trainer代码）

### Q7: 如何修改输出目录？

训练脚本会自动创建带版本号和时间戳的目录，格式：
```
experiments/runs/itransformer_v0.1_20251212_20251212_143022/
```

其中：
- `v0.1_20251212` 是配置版本号（从 `--config_version` 参数获取，默认为 `v0.1_20251212`）
- `20251212_143022` 是训练开始的时间戳（年月日_时分秒）

如果需要自定义，可以修改`scripts/v0.1_20251212/train.py`中的`output_dir`设置。

---

## 🎯 快速开始示例

### 示例1：快速实验（小模型）

**目标**：快速测试代码是否能运行

**修改 `itransformer_config.yaml`**：
```yaml
d_model: 256
n_layers: 3
n_heads: 4
```

**修改 `training_config.yaml`**：
```yaml
batch_size: 16
num_epochs: 10
```

### 示例2：标准训练（推荐）

使用默认配置即可，已经是最佳实践。

### 示例3：高性能训练（大模型）

**修改 `itransformer_config.yaml`**：
```yaml
d_model: 1024
n_layers: 12
n_heads: 16
```

**修改 `training_config.yaml`**：
```yaml
batch_size: 64
num_epochs: 200
```

---

## 📝 配置检查清单

训练前，请检查：

- [ ] 三个配置文件路径正确
- [ ] `index_dir`指向正确的索引文件夹
- [ ] `data_dir`指向正确的数据文件夹
- [ ] `batch_size`适合你的GPU内存
- [ ] `device: "cuda"`（如果有GPU）
- [ ] `num_epochs`设置合理（50-200）
- [ ] `lr`设置合理（0.0001-0.001）

---

## 🎓 总结

1. **模型配置**：控制模型架构（大小、层数等）
2. **训练配置**：控制训练过程（学习率、优化器等）
3. **数据集配置**：控制数据来源和特征选择

**建议**：
- 初学者：先用默认配置，确保能运行
- 有经验后：根据实际情况调整参数
- 遇到问题：查看训练日志，分析损失曲线

祝训练顺利！🚀

