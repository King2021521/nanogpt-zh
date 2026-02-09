# NanoGPT-ZH 快速开始

<div align="center">

**从零手搓的中文GPT模型 | 7M参数 | 教育友好**

</div>

## ⚡ 5分钟快速开始

### 1️⃣ 克隆项目
```bash
git clone https://github.com/King2021521/nanogpt-zh.git
cd nanogpt-zh
```

### 2️⃣ 安装依赖
```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3️⃣ 准备数据
```bash
# 使用示例数据
python scripts/prepare_data.py

# 或使用维基百科数据（需要先下载）
python scripts/extract_wiki.py
```

### 4️⃣ 开始训练
```bash
python train.py
```

### 5️⃣ 生成文本
```bash
python examples/inference.py
```

## 📊 项目信息

| 项目 | 信息 |
|------|------|
| **名称** | NanoGPT-ZH |
| **参数量** | 7.3M |
| **架构** | Decoder-Only Transformer |
| **语言** | 中文 |
| **用途** | 教育学习 |

## 🎯 核心特点

- 🔧 **100%手工实现** - 不依赖transformers库
- 🇨🇳 **中文优化** - 专为中文文本生成设计
- 📚 **教育友好** - 代码清晰，注释详细
- ⚡ **轻量级** - 7M参数，个人电脑可训练
- 🚀 **完整流程** - 数据处理→训练→推理

## 📁 项目结构

```
nanogpt-zh/
├── models/          # 模型实现
├── utils/           # 工具函数
├── scripts/         # 数据处理脚本
├── tests/           # 测试代码
├── examples/        # 使用示例
├── docs/            # 文档
├── config.py        # 配置文件
└── train.py         # 训练脚本
```

## 🔧 配置说明

主要配置在 `config.py`:

```python
vocab_size = 10000      # 词表大小
d_model = 256           # 模型维度
n_layers = 6            # Transformer层数
n_heads = 8             # 注意力头数
batch_size = 32         # 批次大小
learning_rate = 3e-4    # 学习率
```

## 💡 常用命令

### 训练
```bash
# 基础训练
python train.py

# 查看训练日志
tensorboard --logdir=logs
```

### 推理
```bash
# 交互式生成
python examples/inference.py

# 指定模型路径
python examples/inference.py --model checkpoints/best_model.pt
```

### 测试
```bash
# 测试模型组件
python tests/test_model.py

# 测试GPU
python tests/test_gpu.py
```

## 📚 学习路径

### 初学者
1. 阅读 `README.md` 了解项目
2. 查看 `docs/TRANSFORMER_ARCHITECTURE.md` 学习架构
3. 运行 `tests/test_model.py` 理解组件
4. 使用示例数据训练小模型

### 进阶用户
1. 阅读 `docs/MODEL_PARAMETERS_GUIDE.md` 了解参数
2. 修改 `config.py` 调整模型规模
3. 使用维基百科数据训练
4. 优化训练策略

### 高级用户
1. 研究源码实现细节
2. 扩展模型功能
3. 优化训练性能
4. 贡献代码

## 🐛 常见问题

### Q: 训练太慢？
A: 
- 使用GPU（如果有）
- 减小batch_size
- 减少模型层数

### Q: 显存不足？
A:
- 减小batch_size
- 减小d_model
- 使用梯度累积

### Q: 生成质量不好？
A:
- 增加训练数据
- 延长训练时间
- 调整采样参数（temperature, top_k, top_p）

## 📖 文档导航

- **快速开始**: `QUICK_START.md`（本文件）
- **详细说明**: `README.md`
- **架构文档**: `docs/TRANSFORMER_ARCHITECTURE.md`
- **参数指南**: `docs/MODEL_PARAMETERS_GUIDE.md`
- **项目结构**: `docs/PROJECT_STRUCTURE.md`
- **更新日志**: `docs/CHANGELOG.md`

## 🤝 贡献

欢迎贡献！请查看 `CONTRIBUTING.md`

## 📄 许可证

MIT License - 详见 `LICENSE`

## 🙏 致谢

感谢 [nanoGPT](https://github.com/karpathy/nanoGPT) 的启发！

---

<div align="center">

**NanoGPT-ZH** © 2026 | Made with ❤️ by xiaomin.zhang

[GitHub](https://github.com/your-username/nanogpt-zh) | [文档](docs/) | [问题反馈](https://github.com/your-username/nanogpt-zh/issues)

</div>
