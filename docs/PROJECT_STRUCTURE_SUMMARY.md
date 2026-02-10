# NanoGPT-ZH 项目结构总结

@Author xiaomin.zhang

## 优化后的目录结构

```
nanogpt-zh/
├── 📁 models/              # 核心模型实现
│   ├── embedding.py        # Token嵌入和位置编码
│   ├── attention.py        # 多头注意力机制
│   ├── transformer.py      # Transformer Block
│   └── gpt.py              # 完整GPT模型
│
├── 📁 utils/               # 工具函数
│   ├── tokenizer.py        # 中文分词器
│   ├── data_loader.py      # 数据加载器
│   └── trainer.py          # 训练器
│
├── 📁 scripts/             # 数据处理脚本
│   ├── extract_wiki.py     # 维基百科数据提取
│   ├── prepare_data.py     # 数据准备
│   └── process_wiki_data.py # 维基数据处理
│
├── 📁 tests/               # 测试代码
│   ├── test_model.py       # 模型测试
│   ├── test_gpu.py         # GPU测试
│   └── install_pytorch_gpu.py # GPU安装指南
│
├── 📁 examples/            # 示例代码
│   └── inference.py        # 推理示例
│
├── 📁 docs/                # 项目文档
│   ├── STORY.md            # 项目故事
│   ├── DESIGN.md           # 设计文档
│   ├── MODEL_PARAMETERS_GUIDE.md # 模型参数说明
│   ├── TRANSFORMER_ARCHITECTURE.md # Transformer架构
│   ├── PROJECT_STRUCTURE.md # 详细结构说明
│   └── *.md                # 其他文档
│
├── 📁 data/                # 数据文件
│   ├── raw/                # 原始数据
│   │   ├── .gitkeep
│   │   └── zhwiki-*.xml.bz2 (不提交)
│   └── processed/          # 处理后的数据
│       ├── .gitkeep
│       ├── train.txt (不提交)
│       ├── val.txt (不提交)
│       └── vocab.json (不提交)
│
├── 📁 checkpoints/         # 模型检查点
│   ├── .gitkeep
│   └── *.pt (不提交)
│
├── 📁 logs/                # 训练日志
│   └── events.* (不提交)
│
├── 📄 config.py            # 全局配置
├── 📄 train.py             # 训练入口
├── 📄 requirements.txt     # 依赖管理
├── 📄 .gitignore           # Git忽略规则
└── 📄 README.md            # 项目说明
```

## 优化要点

### 1. 按功能分类
- **models/**: 核心模型实现
- **utils/**: 通用工具函数
- **scripts/**: 数据处理和辅助脚本
- **tests/**: 测试代码
- **examples/**: 使用示例

### 2. 按类型分类
- **docs/**: 所有文档集中管理
- **data/**: 数据文件独立存储
- **checkpoints/**: 模型检查点
- **logs/**: 训练日志

### 3. Git管理
- 创建 `.gitignore` 忽略大文件和临时文件
- 使用 `.gitkeep` 保留空目录结构
- 只提交源代码和文档

## 快速使用

### 数据准备
```bash
python scripts/prepare_data.py      # 示例数据
python scripts/extract_wiki.py      # 维基百科数据
```

### 模型训练
```bash
python train.py
```

### 模型推理
```bash
python examples/inference.py
```

### 测试
```bash
python tests/test_model.py          # 测试模型
python tests/test_gpu.py            # 测试GPU
```

## 文档说明

- **README.md**: 快速入门指南
- **docs/PROJECT_STRUCTURE.md**: 详细的项目结构说明
- **docs/TRANSFORMER_ARCHITECTURE.md**: Transformer架构详解
- **docs/MODEL_PARAMETERS_GUIDE.md**: 模型参数计算说明

## 注意事项

1. 所有代码注释携带 `@Author xiaomin.zhang`
2. 大文件（数据、模型）不提交到Git
3. 使用虚拟环境管理依赖
4. 遵循Python命名规范

---

<div align="center">

**NanoGPT-ZH** - 从零手搓的中文GPT模型

详细说明请查看 `docs/PROJECT_STRUCTURE.md`

Made with ❤️ by xiaomin.zhang | © 2026

</div>
