# ResNet模块配置详细指南

## 📍 Batch Size 调整位置

**位置**：`configs/v0.1_20251212/training_config.yaml`

```yaml
training:
  batch_size: 32    # 在这里修改batch_size
```

**如何修改**：
- 直接修改这个数字即可
- 建议值：16, 32, 64, 128（根据GPU内存选择）

---

## 🏗️ 复杂ResNet结构配置示例

### 你的需求

**输入路径**：
```
40特征 -> 40 (2层) -> 128 (1层) -> 128 (2层) -> 256 (1层) -> 256 (2层) -> 512 (1层) -> 512 (2层) -> Decoder
```

**输出路径**：
```
Decoder -> 512 -> 512 (2层) -> 256 (1层) -> 256 (2层) -> 128 (1层) -> 128 (2层) -> 40 (1层) -> 40 (2层) -> 1
```

### 配置方法

在 `configs/v0.1_20251212/itransformer_config.yaml` 中配置：

#### 1. 输入ResNet配置（Decoder之前）

```yaml
input_resnet:
  enabled: true
  modules:
    # 模块1: 40 -> 40 (2层)
    - name: "input_block_1"
      layers: 2
      hidden_dims: [40, 40]  # 2层，都是40维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块2: 40 -> 128 (1层)
    - name: "input_block_2"
      layers: 1
      hidden_dims: [128]  # 1层，输出128维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块3: 128 -> 128 (2层)
    - name: "input_block_3"
      layers: 2
      hidden_dims: [128, 128]  # 2层，都是128维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块4: 128 -> 256 (1层)
    - name: "input_block_4"
      layers: 1
      hidden_dims: [256]  # 1层，输出256维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块5: 256 -> 256 (2层)
    - name: "input_block_5"
      layers: 2
      hidden_dims: [256, 256]  # 2层，都是256维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块6: 256 -> 512 (1层)
    - name: "input_block_6"
      layers: 1
      hidden_dims: [512]  # 1层，输出512维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块7: 512 -> 512 (2层)
    - name: "input_block_7"
      layers: 2
      hidden_dims: [512, 512]  # 2层，都是512维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
```

#### 2. 输出ResNet配置（Decoder之后）

```yaml
output_resnet:
  enabled: true
  modules:
    # 模块1: 512 -> 512 (2层)
    - name: "output_block_1"
      layers: 2
      hidden_dims: [512, 512]  # 2层，都是512维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块2: 512 -> 256 (1层)
    - name: "output_block_2"
      layers: 1
      hidden_dims: [256]  # 1层，输出256维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块3: 256 -> 256 (2层)
    - name: "output_block_3"
      layers: 2
      hidden_dims: [256, 256]  # 2层，都是256维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块4: 256 -> 128 (1层)
    - name: "output_block_4"
      layers: 1
      hidden_dims: [128]  # 1层，输出128维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块5: 128 -> 128 (2层)
    - name: "output_block_5"
      layers: 2
      hidden_dims: [128, 128]  # 2层，都是128维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块6: 128 -> 40 (1层)
    - name: "output_block_6"
      layers: 1
      hidden_dims: [40]  # 1层，输出40维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
    
    # 模块7: 40 -> 40 (2层)
    - name: "output_block_7"
      layers: 2
      hidden_dims: [40, 40]  # 2层，都是40维
      dropout: 0.1
      activation: "gelu"
      use_bias: true
```

#### 3. 最终输出层配置

```yaml
final_output:
  hidden_dim: 40    # 从40维降到1维
  dropout: 0.1
  output_dim: 1     # 最终输出1个值
```

---

## 📝 配置规则说明

### 模块顺序很重要！

**规则**：模块按顺序执行，前一个模块的输出维度 = 后一个模块的输入维度

**示例**：
```yaml
modules:
  - name: "block_1"
    layers: 2
    hidden_dims: [40, 40]    # 输入40，输出40
  
  - name: "block_2"
    layers: 1
    hidden_dims: [128]       # 输入40（来自block_1），输出128
  
  - name: "block_3"
    layers: 2
    hidden_dims: [128, 128]  # 输入128（来自block_2），输出128
```

### hidden_dims 列表长度必须等于 layers

**正确**：
```yaml
layers: 2
hidden_dims: [40, 40]  # 列表长度=2，等于layers
```

**错误**：
```yaml
layers: 2
hidden_dims: [40]      # 列表长度=1，不等于layers（会报错）
```

### 维度变化规则

**维度变化发生在模块之间**：
- 模块1的输出维度 = 模块2的输入维度
- 每个模块内部的最后一层维度 = 该模块的输出维度

**示例**：
```yaml
# 模块1: 40 -> 40 (2层)
hidden_dims: [40, 40]  # 最后一层是40，所以输出40

# 模块2: 40 -> 128 (1层)
hidden_dims: [128]     # 输入40（来自模块1），输出128
```

---

## 🎯 简化配置示例

如果你想要更简单的结构，可以这样配置：

### 简单输入ResNet（3个模块）

```yaml
input_resnet:
  enabled: true
  modules:
    - name: "input_1"
      layers: 2
      hidden_dims: [128, 128]
    
    - name: "input_2"
      layers: 1
      hidden_dims: [256]
    
    - name: "input_3"
      layers: 2
      hidden_dims: [512, 512]
```

**效果**：40 -> 128 -> 128 -> 256 -> 512 -> 512 -> Decoder

### 简单输出ResNet（3个模块）

```yaml
output_resnet:
  enabled: true
  modules:
    - name: "output_1"
      layers: 2
      hidden_dims: [256, 256]
    
    - name: "output_2"
      layers: 1
      hidden_dims: [128]
    
    - name: "output_3"
      layers: 1
      hidden_dims: [1]
```

**效果**：Decoder -> 512 -> 256 -> 256 -> 128 -> 1

---

## ⚠️ 注意事项

1. **输入维度**：第一个输入ResNet模块的输入维度 = 特征数量（会自动检测）
2. **输出维度**：最后一个输出ResNet模块的输出维度应该接近最终输出维度
3. **Decoder输入**：输入ResNet的最后一个模块输出维度应该等于或接近 `d_model`
4. **Decoder输出**：Decoder输出维度 = `d_model`（通常是512）

---

## 🔍 检查配置是否正确

训练时会打印模型结构，检查：
1. 输入ResNet的输入维度是否等于特征数量
2. 输入ResNet的输出维度是否等于 `d_model`
3. 输出ResNet的输入维度是否等于 `d_model`
4. 最终输出维度是否为1

如果维度不匹配，程序会报错并提示。

