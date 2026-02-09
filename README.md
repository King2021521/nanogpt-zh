# NanoGPT-ZH

<div align="center">

**从零手搓的中文GPT模型 | A Tiny GPT Implementation from Scratch for Chinese**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 项目简介

**NanoGPT-ZH** 是一个从0到1手工实现的中文GPT文本生成模型，基于PyTorch和Transformer架构。这是一个教育性质的项目，目标是通过从零实现一个小型GPT模型（~7M参数），深入理解Transformer架构和深度学习的核心概念。

**NanoGPT-ZH 特点：**
- 🔧 **完全手工实现** - 不依赖transformers库，从零实现所有组件
- 🎯 **小规模参数** - 默认7M参数，可在个人电脑训练
- 🇨🇳 **中文优化** - 专为中文文本生成优化
- 📚 **教育友好** - 代码清晰，注释详细，适合学习
- ⚡ **完整流程** - 包含数据处理、训练、推理全流程

## 项目结构

```
nanogpt-zh/
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── embedding.py           # 嵌入层和位置编码
│   ├── attention.py           # 多头注意力机制
│   ├── transformer.py         # Transformer Block
│   └── gpt.py                 # 完整GPT模型
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── tokenizer.py           # 中文分词器
│   ├── data_loader.py         # 数据加载
│   └── trainer.py             # 训练器
├── scripts/                   # 脚本工具
│   ├── extract_wiki.py        # 维基百科数据提取
│   ├── prepare_data.py        # 数据准备脚本
│   └── process_wiki_data.py   # 维基数据处理
├── tests/                     # 测试脚本
│   ├── test_model.py          # 模型测试
│   ├── test_gpu.py            # GPU测试
│   └── install_pytorch_gpu.py # GPU安装指南
├── examples/                  # 示例代码
│   └── inference.py           # 推理示例
├── docs/                      # 文档
│   ├── STORY.md               # 项目故事
│   ├── DESIGN.md              # 设计文档
│   ├── MODEL_PARAMETERS_GUIDE.md      # 模型参数说明
│   ├── TRANSFORMER_ARCHITECTURE.md    # Transformer架构文档
│   └── *.md                   # 其他文档
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后的数据
├── checkpoints/               # 模型检查点
├── logs/                      # 训练日志
├── config.py                  # 配置文件
├── train.py                   # 训练脚本
├── requirements.txt           # 依赖列表
├── .gitignore                 # Git忽略文件
└── README.md                  # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

**推荐环境：**
- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM
- GPU（可选，但强烈推荐）

### 2. 数据准备

```bash
# 准备示例数据并构建词表
python scripts/prepare_data.py

# 或者使用维基百科数据
python scripts/extract_wiki.py
```

这会创建示例数据集。实际使用时，建议使用更大的数据集：
- 中文维基百科
- 新闻语料库
- 小说、文章等

### 3. 训练模型

```bash
# 开始训练
python train.py
```

训练过程中会：
- 自动保存检查点
- 记录训练日志到TensorBoard
- 定期在验证集上评估

**查看训练日志：**
```bash
tensorboard --logdir=logs
```

### 4. 文本生成

```bash
# 交互式文本生成
python examples/inference.py
```

## 模型配置

主要超参数在 `config.py` 中配置：

```python
# 模型架构
vocab_size = 10000      # 词表大小
d_model = 256           # 模型维度
n_layers = 6            # Transformer层数
n_heads = 8             # 注意力头数
max_seq_len = 256       # 最大序列长度

# 训练参数
batch_size = 32
learning_rate = 3e-4
max_epochs = 10
```

## 模型架构

```
输入文本
    ↓
Token Embedding
    ↓
Positional Encoding
    ↓
[Transformer Block] × N
    ├── Multi-Head Attention
    ├── Layer Normalization
    ├── Feed Forward Network
    └── Residual Connection
    ↓
Layer Normalization
    ↓
Language Model Head
    ↓
输出概率分布
```

## 核心组件

### 1. 多头注意力机制
```python
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 2. 位置编码
使用正弦和余弦函数为序列位置编码：
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3. 前馈网络
```python
FFN(x) = GELU(xW1 + b1)W2 + b2
```

## 训练技巧

1. **学习率预热**：前1000步线性增长
2. **余弦退火**：学习率逐渐衰减
3. **梯度裁剪**：防止梯度爆炸
4. **权重共享**：Token嵌入和输出层共享权重
5. **Pre-LN**：在子层之前进行Layer Normalization

## 性能优化

- **混合精度训练**：使用torch.cuda.amp（如果有GPU）
- **梯度累积**：模拟更大的batch size
- **模型量化**：减少推理时的内存占用

## 测试模型组件

```bash
# 测试模型组件
python tests/test_model.py

# 测试GPU可用性
python tests/test_gpu.py
```

## 常见问题

### Q: 显存不足怎么办？
A: 减小batch_size、d_model或n_layers，或使用梯度累积。

### Q: 训练太慢怎么办？
A: 使用GPU，或减小模型规模和数据量。

### Q: 生成的文本质量不好？
A: 增加训练数据量，延长训练时间，调整采样参数。

### Q: 如何使用自己的数据？
A: 修改`scripts/prepare_data.py`，读取您的数据文件，然后运行数据准备流程。

## 扩展方向

- [ ] 支持更大的模型规模
- [ ] 实现分布式训练
- [ ] 添加更多采样策略
- [ ] 支持微调（Fine-tuning）
- [ ] 实现RLHF（人类反馈强化学习）
- [ ] 添加Web界面（Gradio/Streamlit）

## 学习资源

- **论文**：Attention Is All You Need
- **代码**：nanoGPT by Andrej Karpathy
- **教程**：The Annotated Transformer

## 项目灵感

本项目受到以下优秀项目的启发：
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 作者

@Author xiaomin.zhang

## 许可证

MIT License

## Star History

如果这个项目对您有帮助，欢迎给个 ⭐️ Star！

---

<div align="center">
Made with ❤️ by xiaomin.zhang | NanoGPT-ZH © 2026
</div>
