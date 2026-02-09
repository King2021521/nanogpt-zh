# NanoGPT-ZH 项目结构说明

@Author xiaomin.zhang

## 目录结构

**NanoGPT-ZH** 采用模块化设计，按照功能和类型组织代码和文档。

```
nanogpt-zh/
├── models/                    # 核心模型实现
├── utils/                     # 工具函数
├── scripts/                   # 数据处理和辅助脚本
├── tests/                     # 测试代码
├── examples/                  # 示例代码
├── docs/                      # 项目文档
├── data/                      # 数据文件
├── checkpoints/               # 模型检查点
├── logs/                      # 训练日志
├── config.py                  # 全局配置
├── train.py                   # 训练入口
├── requirements.txt           # 依赖管理
├── .gitignore                 # Git忽略规则
└── README.md                  # 项目说明
```

## 详细说明

### 1. models/ - 模型实现

包含所有神经网络模型的定义。

```
models/
├── __init__.py                # 模块初始化
├── embedding.py               # Token嵌入和位置编码
├── attention.py               # 多头注意力机制
├── transformer.py             # Transformer Block
└── gpt.py                     # 完整GPT模型
```

**核心组件：**
- `TokenEmbedding`: 将token ID转换为向量
- `PositionalEncoding`: 为序列添加位置信息
- `MultiHeadAttention`: 多头自注意力机制
- `CausalSelfAttention`: 因果自注意力（用于GPT）
- `FeedForward`: 前馈神经网络
- `TransformerBlock`: 完整的Transformer层
- `GPTModel`: 完整的GPT模型

### 2. utils/ - 工具函数

包含训练、数据处理等辅助功能。

```
utils/
├── __init__.py                # 模块初始化
├── tokenizer.py               # 中文分词器
├── data_loader.py             # 数据加载器
└── trainer.py                 # 训练器
```

**核心功能：**
- `ChineseTokenizer`: 基于jieba的中文分词
- `TextDataset`: 文本数据集类
- `create_dataloader`: 数据加载器工厂函数
- `Trainer`: 完整的训练流程管理

### 3. scripts/ - 脚本工具

包含数据处理和辅助脚本。

```
scripts/
├── extract_wiki.py            # 维基百科数据提取
├── prepare_data.py            # 数据准备脚本
└── process_wiki_data.py       # 维基数据处理
```

**功能说明：**
- `extract_wiki.py`: 从维基百科XML文件提取纯文本
- `prepare_data.py`: 准备训练数据和构建词表
- `process_wiki_data.py`: 处理WikiExtractor输出

### 4. tests/ - 测试代码

包含单元测试和功能测试。

```
tests/
├── test_model.py              # 模型组件测试
├── test_gpu.py                # GPU可用性测试
└── install_pytorch_gpu.py     # GPU安装指南
```

**测试内容：**
- 各模型组件的功能测试
- GPU和CUDA环境检测
- 安装和配置指南

### 5. examples/ - 示例代码

包含使用示例和演示代码。

```
examples/
└── inference.py               # 文本生成示例
```

**示例内容：**
- 模型加载和推理
- 交互式文本生成
- 不同采样策略演示

### 6. docs/ - 项目文档

包含所有项目文档。

```
docs/
├── STORY.md                   # 项目故事和背景
├── DESIGN.md                  # 设计方案
├── MODEL_PARAMETERS_GUIDE.md  # 模型参数说明
├── TRANSFORMER_ARCHITECTURE.md # Transformer架构文档
├── PROJECT_STRUCTURE.md       # 项目结构说明（本文件）
└── *.md                       # 其他文档
```

**文档内容：**
- 项目背景和目标
- 技术方案和设计
- 模型架构详解
- 参数计算和优化
- 开发指南

### 7. data/ - 数据文件

存储训练数据。

```
data/
├── raw/                       # 原始数据
│   ├── .gitkeep
│   └── zhwiki-*.xml.bz2      # 维基百科原始数据（不提交）
└── processed/                 # 处理后的数据
    ├── .gitkeep
    ├── train.txt              # 训练集（不提交）
    ├── val.txt                # 验证集（不提交）
    └── vocab.json             # 词表（不提交）
```

**说明：**
- `raw/`: 存放原始数据文件
- `processed/`: 存放清洗和分词后的数据
- 大文件不提交到Git，使用`.gitkeep`保留目录结构

### 8. checkpoints/ - 模型检查点

存储训练过程中的模型检查点。

```
checkpoints/
├── .gitkeep
├── checkpoint_epoch_*.pt      # 训练检查点（不提交）
├── best_model.pt              # 最佳模型（可选提交）
└── final_model.pt             # 最终模型（可选提交）
```

**说明：**
- 自动保存训练检查点
- 保留验证集上最佳模型
- 训练结束保存最终模型

### 9. logs/ - 训练日志

存储TensorBoard日志和训练记录。

```
logs/
└── events.out.tfevents.*      # TensorBoard日志（不提交）
```

**说明：**
- TensorBoard可视化日志
- 训练过程指标记录
- 不提交到Git

### 10. 根目录文件

```
├── config.py                  # 全局配置文件
├── train.py                   # 训练入口脚本
├── requirements.txt           # Python依赖列表
├── .gitignore                 # Git忽略规则
└── README.md                  # 项目说明文档
```

**文件说明：**
- `config.py`: 模型和训练的所有超参数配置
- `train.py`: 训练流程的主入口
- `requirements.txt`: 项目依赖包列表
- `.gitignore`: Git版本控制忽略规则
- `README.md`: 项目快速入门指南

## 文件命名规范

### Python文件
- 模块文件：小写+下划线，如 `data_loader.py`
- 类名：大驼峰，如 `GPTModel`
- 函数名：小写+下划线，如 `create_dataloader`
- 常量：大写+下划线，如 `MAX_SEQ_LEN`

### 文档文件
- 全大写+下划线，如 `README.md`、`DESIGN.md`
- 描述性命名，清晰表达内容

### 数据文件
- 小写+下划线，如 `train.txt`、`vocab.json`
- 使用标准扩展名

## Git管理策略

### 提交到Git
- 所有源代码（`*.py`）
- 配置文件（`config.py`, `requirements.txt`）
- 文档（`docs/*.md`, `README.md`）
- 目录结构标记（`.gitkeep`）

### 不提交到Git
- 虚拟环境（`.venv/`）
- Python缓存（`__pycache__/`）
- 训练数据（`data/raw/*`, `data/processed/*.txt`）
- 模型文件（`checkpoints/*.pt`，除非特别标记）
- 训练日志（`logs/`）
- IDE配置（`.vscode/`, `.idea/`）

详见 `.gitignore` 文件。

## 开发工作流

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
```bash
# 使用示例数据
python scripts/prepare_data.py

# 或使用维基百科数据
python scripts/extract_wiki.py
```

### 3. 模型训练
```bash
# 开始训练
python train.py

# 监控训练（另开终端）
tensorboard --logdir=logs
```

### 4. 模型推理
```bash
# 交互式生成
python examples/inference.py
```

### 5. 测试验证
```bash
# 测试模型组件
python tests/test_model.py

# 测试GPU
python tests/test_gpu.py
```

## 扩展开发

### 添加新模型组件
1. 在 `models/` 下创建新文件
2. 实现模型类，继承 `nn.Module`
3. 在 `models/__init__.py` 中导出
4. 在 `tests/test_model.py` 中添加测试

### 添加新工具函数
1. 在 `utils/` 下创建新文件或扩展现有文件
2. 实现工具函数或类
3. 在 `utils/__init__.py` 中导出
4. 添加相应的测试

### 添加新脚本
1. 在 `scripts/` 下创建新文件
2. 实现脚本功能
3. 在文档中说明用途和用法

### 添加新文档
1. 在 `docs/` 下创建Markdown文件
2. 使用清晰的标题和结构
3. 在 `README.md` 中添加链接

## 最佳实践

1. **代码注释**：所有代码注释携带 `@Author xiaomin.zhang`
2. **模块化**：保持模块职责单一，便于维护
3. **文档化**：重要功能添加文档说明
4. **测试**：新功能添加相应测试
5. **版本控制**：合理使用Git，编写清晰的commit信息
6. **配置管理**：超参数统一在 `config.py` 管理
7. **日志记录**：使用TensorBoard记录训练过程

## 总结

**NanoGPT-ZH** 采用清晰的模块化结构，将代码、文档、数据、测试分离管理，便于开发、维护和协作。遵循Python和深度学习项目的最佳实践，为学习和扩展提供良好的基础。

---

<div align="center">
NanoGPT-ZH © 2026 | Made with ❤️ by xiaomin.zhang
</div>
