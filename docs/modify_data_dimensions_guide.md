# 修改数据维度指南

## 📋 支持的修改类型

### 1. 修改输入序列长度（例如：100 → 200）

**步骤：**
1. 修改生成索引文件的脚本参数（`input_window_size`）
2. 重新生成索引文件
3. 重新运行预处理脚本
4. 训练脚本会自动检测新的序列长度

**示例：**
```bash
# 1. 生成新的索引文件（假设有生成脚本）
python scripts/generate_index.py --input_window 200

# 2. 重新预处理
python scripts/v0.1_20251212/preprocess_data_20251216.py --config_version v0.1_20251212

# 3. 直接训练（自动检测新维度）
python scripts/v0.1_20251212/train.py
```

---

### 2. 修改特征数量（例如：44 → 60）

**原因：** 添加或删除特征列

**步骤：**
1. 修改 `dataset_config.yaml` 中的 `exclude_columns`
2. 重新运行预处理脚本
3. 训练脚本会自动检测并更新模型配置

**示例：**
```yaml
# configs/v0.1_20251212/dataset_config.yaml
features:
  exclude_columns:
    - "日期"
    - "company_name"
    # 添加或删除要排除的列
```

```bash
# 重新预处理
python scripts/v0.1_20251212/preprocess_data_20251216.py --config_version v0.1_20251212

# 训练时会自动调整
python scripts/v0.1_20251212/train.py
```

---

### 3. 修改训练集/验证集数量

**原因：** 重新划分数据集

**步骤：**
1. 重新生成索引文件（修改划分比例）
2. 重新运行预处理脚本

**示例：**
```bash
# 1. 生成新索引（假设有生成脚本，修改train_ratio参数）
python scripts/generate_index.py --train_ratio 0.8

# 2. 重新预处理
python scripts/v0.1_20251212/preprocess_data_20251216.py --config_version v0.1_20251212

# 3. 训练
python scripts/v0.1_20251212/train.py
```

---

### 4. 修改目标列（例如：预测"最高价"而不是"收盘价"）

**步骤：**
1. 修改 `dataset_config.yaml` 中的 `target_column`
2. 重新运行预处理脚本

**示例：**
```yaml
# configs/v0.1_20251212/dataset_config.yaml
features:
  target_column: "最高"  # 从"收盘"改为"最高"
```

---

## ⚠️ 重要提示

### 1. **必须重新预处理**
修改任何数据维度后，**必须重新运行预处理脚本**：
```bash
python scripts/v0.1_20251212/preprocess_data_20251216.py --config_version v0.1_20251212
```

### 2. **自动检测机制**
训练脚本会自动检测：
- ✅ 特征数量变化 → 自动更新模型配置
- ✅ 序列长度变化 → 自动更新模型配置
- ⚠️ 如果预处理数据与当前配置不匹配 → 报错提示

### 3. **版本管理**
每次预处理会创建新的时间戳文件夹，不会覆盖旧数据：
```
data/processed/
├── preprocess_data_20251216122731/  # 旧版本（44特征，100序列）
└── preprocess_data_20251216153000/  # 新版本（60特征，200序列）
```

---

## 🎯 快速检查清单

修改数据后，检查以下内容：

- [ ] 索引文件是否重新生成？
- [ ] 预处理脚本是否重新运行？
- [ ] 新的预处理目录是否创建？
- [ ] 训练脚本是否自动检测到新维度？
- [ ] 模型配置是否自动更新？

---

## 💡 示例：完整的修改流程

假设你要：
- 输入序列长度：100 → 150
- 添加5个新特征：44 → 49
- 调整数据集划分：8:2 → 7:3

```bash
# 1. 修改配置（如果需要）
# 编辑 dataset_config.yaml

# 2. 重新生成索引
python scripts/generate_index.py \
  --input_window 150 \
  --train_ratio 0.7

# 3. 重新预处理
python scripts/v0.1_20251212/preprocess_data_20251216.py \
  --config_version v0.1_20251212

# 4. 训练（自动适配）
python scripts/v0.1_20251212/train.py

# 输出示例：
# 训练集样本数: XXXXX (新数量)
# 特征数量: 49 (自动检测)
# 序列长度: 150 (自动检测)
# 警告: 配置中的特征数量 (40) 与实际特征数量 (49) 不同
# 更新模型配置中的特征数量为: 49
```

---

## 📝 总结

预处理脚本 `preprocess_data_20251216.py` 的特点：

✅ **灵活性高** - 自动适配任何维度的数据
✅ **自动检测** - 训练时自动读取实际维度
✅ **版本管理** - 保留所有历史预处理结果
✅ **配置驱动** - 通过YAML文件控制特征选择

**核心原则：数据变化 → 重新预处理 → 自动训练**

