# 中文古诗词训练指南

## 📚 数据集信息

已成功预处理 **chinese-poetry-collection** 数据集：

| 指标 | 数值 |
|------|------|
| 训练样本 | **367,896** 条 |
| 验证样本 | **19,363** 条 |
| 测试样本 | **1,710** 条 |
| 总字符数 | **~1365万** 字符 |
| 平均长度 | **37** 字符/样本 |
| 词表大小 | **10,000** |

---

## ⚙️ 超参数配置

### 诗词优化配置 (推荐)

已针对诗词数据集创建优化配置 `config_poetry.py`：

```python
# 模型架构
vocab_size = 10,000      # 词表大小
max_seq_len = 128        # 序列长度（诗词平均37字符）
d_model = 384            # 模型维度（增大50%）
n_layers = 8             # Transformer层数（增加2层）
d_ff = 1536              # 前馈网络维度

# 训练参数
batch_size = 64          # 批次大小（翻倍）
learning_rate = 5e-4     # 学习率
max_epochs = 20          # 训练轮数（翻倍）
max_steps = 100,000      # 最大步数（翻倍）
warmup_steps = 2,000     # 预热步数（翻倍）

# 推理参数
temperature = 0.8        # 采样温度（降低以提高质量）
top_k = 40               # Top-K采样
top_p = 0.85             # Top-P采样
```

### 配置对比

| 参数 | 原始配置 | 诗词配置 | 变化 |
|------|---------|---------|------|
| 模型参数量 | 9.9M | **21.9M** | ⬆️ 2.2x |
| 序列长度 | 256 | **128** | ⬇️ 50% |
| 批次大小 | 32 | **64** | ⬆️ 100% |
| 训练步数 | 50K | **100K** | ⬆️ 100% |

**查看详细对比**: 运行 `python scripts/compare_configs.py`

---

## 🚀 快速开始

### 1️⃣ 数据已预处理 ✅

数据已经预处理完成，生成的文件：

```
data/processed/
├── train.txt          # 训练数据 (367,896条)
├── val.txt            # 验证数据 (19,363条)
├── test.txt           # 测试数据 (1,710条)
├── vocab.json         # 词表 (10,000词)
└── dataset_info.json  # 数据集信息
```

### 2️⃣ 查看配置

```bash
# 查看诗词优化配置
python config_poetry.py

# 对比两个配置
python scripts/compare_configs.py
```

### 3️⃣ 开始训练

```bash
# 使用诗词优化配置训练（推荐）
python train_poetry.py

# 或使用原始配置（不推荐，效果较差）
python train.py
```

### 4️⃣ 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir logs_poetry

# 在浏览器打开
http://localhost:6006
```

---

## 📊 训练预期

### 训练统计

```
每轮步数:     5,748 步
训练轮数:     20 轮
总训练步数:   100,000 步
保存检查点:   50 次
评估次数:     100 次
```

### 时间估算

| GPU型号 | 预计时间 |
|---------|---------|
| RTX 3090 | **~14 小时** |
| RTX 3080 | **~16 小时** |
| RTX 3060 | **~22 小时** |

### 显存需求

- **最低**: 4GB（需调整batch_size=32）
- **推荐**: 6-8GB
- **理想**: 10GB+

---

## 📈 训练监控指标

### 健康的训练曲线

```
Epoch 1:  train_loss=4.5, val_loss=4.3, perplexity=90.0
Epoch 5:  train_loss=3.2, val_loss=3.1, perplexity=24.5
Epoch 10: train_loss=2.5, val_loss=2.6, perplexity=13.5
Epoch 15: train_loss=2.0, val_loss=2.2, perplexity=9.0
Epoch 20: train_loss=1.8, val_loss=2.1, perplexity=8.2
```

### 关键指标

1. **训练损失** - 应持续下降，目标 < 2.0
2. **验证损失** - 应跟随训练损失，如上升则过拟合
3. **困惑度** - 越低越好，目标 < 10

---

## 🔧 显存不足？

如果遇到 `CUDA Out of Memory` 错误：

### 方案1: 减小批次大小

```python
# 在 config_poetry.py 中修改
batch_size = 32  # 从64减到32
```

### 方案2: 减小模型维度

```python
d_model = 256    # 从384减到256
d_ff = 1024      # 从1536减到1024
n_layers = 6     # 从8减到6
```

### 方案3: 减小序列长度

```python
max_seq_len = 96  # 从128减到96
```

---

## 🎨 生成参数调优

训练完成后，可以调整生成参数：

### 保守策略（高质量）

```python
temperature = 0.6
top_k = 20
top_p = 0.75
```

### 平衡策略（推荐）

```python
temperature = 0.8  # 默认
top_k = 40
top_p = 0.85
```

### 创新策略（多样性）

```python
temperature = 1.0
top_k = 60
top_p = 0.95
```

---

## 📁 文件说明

### 配置文件

- `config.py` - 原始配置（通用）
- `config_poetry.py` - **诗词优化配置（推荐）**

### 训练脚本

- `train.py` - 使用原始配置训练
- `train_poetry.py` - **使用诗词配置训练（推荐）**

### 数据处理

- `scripts/prepare_poetry_data.py` - 诗词数据预处理脚本
- `scripts/compare_configs.py` - 配置对比脚本

### 文档

- `docs/POETRY_HYPERPARAMETERS.md` - **超参数详细说明**
- `docs/TRAINING_GUIDE.md` - 训练指南

---

## 💡 最佳实践

### ✅ 推荐

1. ✅ 使用 `config_poetry.py` 配置
2. ✅ 运行 `train_poetry.py` 训练
3. ✅ 使用GPU训练（至少RTX 3060）
4. ✅ 定期检查验证损失
5. ✅ 使用TensorBoard监控

### ❌ 避免

1. ❌ 在CPU上训练（太慢）
2. ❌ 批次大小太小（<16）
3. ❌ 学习率太高（>1e-3）
4. ❌ 不做验证（无法发现过拟合）
5. ❌ 只保存最后一个检查点

---

## 🎯 训练建议

### 基于数据量的建议

数据集有 **36.7万** 训练样本，属于**中等规模**数据集：

1. **模型容量**: 21.9M参数合适，不会过拟合
2. **训练轮数**: 20轮足够，可以充分学习
3. **批次大小**: 64合适，训练稳定
4. **学习率**: 5e-4合适，收敛速度快

### 训练阶段

**阶段1 (0-20%步数)**: 快速下降期
- 损失快速下降
- 模型学习基本语法

**阶段2 (20-60%步数)**: 稳定学习期
- 损失平稳下降
- 模型学习诗词结构

**阶段3 (60-100%步数)**: 精细调优期
- 损失缓慢下降
- 模型学习韵律和意境

---

## 📞 遇到问题？

### 常见问题

1. **显存不足** → 减小batch_size或d_model
2. **训练太慢** → 检查是否使用GPU
3. **损失不下降** → 降低学习率
4. **过拟合** → 增加dropout或减少训练轮数

### 查看日志

```bash
# 查看训练日志
cat logs_poetry/train.log

# 查看GPU状态
nvidia-smi

# 查看TensorBoard
tensorboard --logdir logs_poetry
```

---

## 📚 参考文档

- **超参数详细说明**: `docs/POETRY_HYPERPARAMETERS.md`
- **训练指南**: `docs/TRAINING_GUIDE.md`
- **项目状态**: `docs/PROJECT_STATUS.md`
- **快速开始**: `QUICK_START.md`

---

## 🎉 开始训练

一切准备就绪！运行以下命令开始训练：

```bash
python train_poetry.py
```

祝训练顺利！🚀

---

**最后更新**: 2026-02-09  
**作者**: @xiaomin.zhang
