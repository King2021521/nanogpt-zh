# 诗词创作指令微调完整指南

> @Author xiaomin.zhang  
> 项目: nanogpt_zh  
> 创建日期: 2026-02-13

---

## 🎯 项目简介

本项目提供了一套**完整的诗词创作指令微调解决方案**，帮助你训练一个能够根据提示词创作中国古典诗词的AI模型。

### ✨ 特点

- 📚 **详细的教程文档** - 从数据准备到模型训练的完整指南
- 🎨 **高质量示例数据** - 30条精选的诗词创作训练样本
- 🔧 **实用工具脚本** - 数据验证、格式转换、可视化分析
- 🚀 **快速开始指南** - 5分钟即可开始训练
- 📊 **数据分析报告** - 自动生成的统计和可视化

---

## 📖 文档导航

### 🌟 推荐阅读顺序

#### 1️⃣ 新手入门
👉 **[快速开始指南](QUICKSTART_INSTRUCTION_FINETUNING.md)**
- 5分钟快速上手
- 环境配置
- 训练和测试示例
- 常见问题解答

#### 2️⃣ 深入学习
👉 **[完整教程](instruction_finetuning_tutorial.md)**
- 数据规模建议
- 数据结构设计
- 详细示例说明
- 质量保证方法
- 评估指标体系

#### 3️⃣ 数据使用
👉 **[数据集说明](../data/poetry_instruction/README.md)**
- 数据集概览
- 格式说明
- 使用方法
- 扩展指南

#### 4️⃣ 项目总览
👉 **[项目总结](POETRY_FINETUNING_SUMMARY.md)**
- 完整项目结构
- 文件说明
- 统计信息
- 学习路径

---

## 🚀 快速开始

### 第一步：查看数据

```bash
# 运行数据处理脚本
python scripts/prepare_poetry_data.py

# 生成可视化分析
python scripts/visualize_poetry_data.py
```

### 第二步：查看分析报告

```bash
# 查看训练集报告
cat data/poetry_instruction/analysis/train_report.md

# 查看数据对比
cat data/poetry_instruction/analysis/comparison_report.md
```

### 第三步：开始训练

```bash
# 使用LLaMA-Factory（推荐）
cd LLaMA-Factory
llamafactory-cli webui

# 或使用自定义脚本
python train_poetry.py
```

---

## 📊 数据集概览

### 基本信息

| 项目 | 数值 |
|------|------|
| 总样本数 | 30条 |
| 训练集 | 20条 (66.7%) |
| 验证集 | 5条 (16.7%) |
| 测试集 | 5条 (16.7%) |
| 验证通过率 | 100% |

### 诗词类型

```
五言绝句 ████████████████████ 46.7%
七言绝句 ███████████████ 33.3%
五言律诗 ████ 10.0%
七言律诗 █ 3.3%
古体诗   ██ 6.7%
```

### 风格分布

```
清新 ████████████ 33.3%
豪放 ██████████ 26.7%
婉约 ████████ 23.3%
古典 █████ 13.3%
现代 █ 3.3%
```

### 难度分布

```
初级 ██████████ 30.0%
中级 ████████████████ 50.0%
高级 ████████ 20.0%
```

---

## 📁 项目结构

```
nanogpt_zh/
│
├── 📄 README_POETRY_FINETUNING.md          # 本文档（项目入口）
│
├── 📂 docs/                                 # 文档目录
│   ├── instruction_finetuning_tutorial.md  # 完整教程（主文档）
│   ├── QUICKSTART_INSTRUCTION_FINETUNING.md # 快速开始
│   └── POETRY_FINETUNING_SUMMARY.md        # 项目总结
│
├── 📂 data/poetry_instruction/              # 数据集目录
│   ├── train.jsonl                         # 训练集（20条）
│   ├── val.jsonl                           # 验证集（5条）
│   ├── test.jsonl                          # 测试集（5条）
│   ├── train_alpaca.jsonl                  # Alpaca格式
│   ├── train_chatgpt.jsonl                 # ChatGPT格式
│   ├── metadata.json                       # 元信息
│   ├── README.md                           # 数据集说明
│   └── analysis/                           # 分析报告
│       ├── train_report.md                 # 训练集报告
│       ├── val_report.md                   # 验证集报告
│       ├── test_report.md                  # 测试集报告
│       ├── comparison_report.md            # 对比报告
│       └── *.png                           # 可视化图表
│
└── 📂 scripts/                              # 工具脚本
    ├── prepare_poetry_data.py              # 数据处理
    └── visualize_poetry_data.py            # 数据可视化
```

---

## 💡 数据示例

### 基础示例

```json
{
  "instruction": "写一首描写春天的五言绝句",
  "input": "",
  "output": "春风拂柳绿，\n燕子绕梁飞。\n桃花三月雨，\n诗意满园归。",
  "metadata": {
    "poem_type": "五言绝句",
    "style": "清新",
    "season": "春",
    "difficulty": "初级"
  }
}
```

### 带情感要求

```json
{
  "instruction": "创作一首表达思乡之情的七言绝句",
  "input": "",
  "output": "明月千里照故乡，\n秋风落叶满庭霜。\n游子天涯心未老，\n梦回故里泪两行。",
  "metadata": {
    "poem_type": "七言绝句",
    "style": "婉约",
    "emotion": "思乡",
    "difficulty": "中级"
  }
}
```

### 带约束条件

```json
{
  "instruction": "写一首五言绝句，要求包含'风'、'雨'两个字，表达离别之情",
  "input": "关键词：风、雨",
  "output": "风起雨纷纷，\n离人泪满襟。\n天涯各一方，\n何日再相寻。",
  "metadata": {
    "poem_type": "五言绝句",
    "style": "婉约",
    "emotion": "离别",
    "keywords": ["风", "雨"],
    "difficulty": "中级"
  }
}
```

---

## 🔧 工具脚本

### 1. 数据处理脚本

**功能**:
- ✅ 加载和验证数据
- ✅ 格律检查
- ✅ 统计分析
- ✅ 格式转换

**使用**:
```bash
python scripts/prepare_poetry_data.py
```

**输出**:
```
正在加载数据集...
训练集: 20 条
验证集: 5 条
测试集: 5 条
总计: 30 条

正在验证数据集...
有效样本: 30
无效样本: 0

数据集统计信息
==================================================
总样本数: 30
平均指令长度: 14.73 字符
平均输出长度: 36.77 字符
...
```

### 2. 数据可视化脚本

**功能**:
- ✅ 生成分析报告
- ✅ 绘制分布图表
- ✅ 数据对比分析

**使用**:
```bash
python scripts/visualize_poetry_data.py
```

**输出**:
- Markdown格式报告
- PNG格式图表
- 数据对比分析

---

## 📈 训练建议

### 数据规模

| 场景 | 建议规模 | 说明 |
|------|----------|------|
| 学习测试 | 30-100条 | 快速验证流程 |
| 最小可用 | 1,000-2,000条 | 基础功能 |
| 推荐规模 | 5,000-10,000条 | 良好效果 |
| 理想规模 | 20,000+条 | 最佳效果 |

### 训练参数

```python
# 推荐配置
learning_rate = 5e-5
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
warmup_steps = 100
```

### 模型选择

| 模型 | 参数量 | GPU内存 | 推荐场景 |
|------|--------|---------|----------|
| Qwen2-0.5B | 0.5B | 4GB | 快速测试 |
| Qwen2-1.5B | 1.5B | 8GB | 平衡性能 |
| Qwen2-7B | 7B | 16GB+ | 最佳效果 |

---

## 🎯 评估指标

### 自动评估

- **格律准确率**: 是否符合诗词格律
- **BLEU分数**: 与参考诗词的相似度
- **困惑度**: 模型生成的流畅度

### 人工评估

| 维度 | 评分标准 | 权重 |
|------|----------|------|
| 相关性 | 是否符合指令要求 | 30% |
| 创意性 | 是否有新意 | 20% |
| 艺术性 | 意境和美感 | 30% |
| 流畅度 | 语言是否通顺 | 10% |
| 格律性 | 是否符合格律 | 10% |

---

## 🐛 常见问题

### Q1: 数据规模太小怎么办？

**A**: 当前30条数据适合学习和测试。实际使用建议：
- 从古诗词数据库扩充
- 使用AI辅助生成（需人工审核）
- 数据增强技术
- 众包标注

### Q2: GPU内存不足？

**A**: 尝试以下方法：
- 减小batch_size
- 使用gradient_accumulation
- 使用LoRA微调
- 使用量化（8bit/4bit）

### Q3: 生成的诗词不符合格律？

**A**: 改进方法：
- 增加训练数据
- 强调格律要求
- 后处理验证
- 使用约束解码

### Q4: 如何扩充数据集？

**A**: 三种方法：
1. **人工创作** - 质量最高
2. **古诗词改编** - 效率较高
3. **AI辅助生成** - 需要审核

---

## 📚 学习路径

### 🌱 初学者（1-2天）

1. ✅ 阅读快速开始指南
2. ✅ 运行数据处理脚本
3. ✅ 查看示例数据
4. ✅ 使用LLaMA-Factory训练
5. ✅ 测试生成效果

### 🌿 进阶者（1-2周）

1. ✅ 阅读完整教程
2. ✅ 扩充数据到1000+条
3. ✅ 调整训练参数
4. ✅ 实现自定义评估
5. ✅ 优化生成质量

### 🌳 高级者（持续）

1. ✅ 研究数据增强
2. ✅ 实现格律验证
3. ✅ 开发标注工具
4. ✅ 部署生产服务
5. ✅ 持续优化迭代

---

## 🤝 贡献指南

欢迎贡献！你可以：

- 📝 添加高质量诗词数据
- 🔧 改进工具脚本
- 📖 完善文档
- 🐛 报告问题
- 💡 提出建议

### 贡献流程

1. Fork本项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

---

## 📊 项目统计

### 文件统计

| 类型 | 数量 |
|------|------|
| 文档 | 5个 |
| 数据文件 | 7个 |
| 脚本 | 2个 |
| 报告 | 4个 |
| 图表 | 15个 |

### 代码统计

| 项目 | 数量 |
|------|------|
| Python代码 | ~800行 |
| 文档内容 | ~3500行 |
| 数据样本 | 30条 |
| 示例代码 | 10+个 |

---

## 🔗 相关资源

### 数据资源

- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) - 中文诗歌数据库
- [古诗文网](https://www.gushiwen.cn/) - 古诗词资源

### 训练框架

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 大模型微调框架
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face库
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调

### 学习资料

- 《唐诗三百首》
- 《宋词三百首》
- 《诗词格律》- 王力
- Stanford Alpaca论文

---

## 📄 许可证

本项目采用 **MIT** 许可证。

---

## 📮 联系方式

- **作者**: xiaomin.zhang
- **项目**: nanogpt_zh
- **创建日期**: 2026-02-13

---

## 🎉 致谢

感谢所有为中文诗词文化传承做出贡献的人们！

---

## 🌟 开始你的诗词创作之旅

现在就开始吧！从[快速开始指南](QUICKSTART_INSTRUCTION_FINETUNING.md)开始你的第一次训练。

**祝你训练顺利，创作出优美的诗词！** 🎨📝✨

---

<div align="center">

**[📖 完整教程](instruction_finetuning_tutorial.md)** | 
**[🚀 快速开始](QUICKSTART_INSTRUCTION_FINETUNING.md)** | 
**[📊 数据集](../data/poetry_instruction/README.md)** | 
**[📝 项目总结](POETRY_FINETUNING_SUMMARY.md)**

Made with ❤️ by xiaomin.zhang

</div>
