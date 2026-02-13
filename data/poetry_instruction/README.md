# 诗词创作指令微调数据集

> @Author xiaomin.zhang

## 📖 简介

本数据集专门用于训练大语言模型进行中国古典诗词创作。通过指令微调，模型可以根据用户的提示词（如主题、风格、情感等）创作符合要求的诗词作品。

## 📊 数据集概览

- **总样本数**: 30条（示例数据）
- **训练集**: 20条
- **验证集**: 5条
- **测试集**: 5条

### 诗词类型覆盖

- 五言绝句
- 七言绝句
- 五言律诗
- 七言律诗
- 古体诗

### 风格类型

- 清新自然
- 婉约派
- 豪放派
- 古典风格
- 现代风格

## 📁 文件结构

```
poetry_instruction/
├── train.jsonl              # 训练集（20条）
├── val.jsonl                # 验证集（5条）
├── test.jsonl               # 测试集（5条）
├── metadata.json            # 数据集元信息
├── train_alpaca.jsonl       # Alpaca格式训练集
├── train_chatgpt.jsonl      # ChatGPT格式训练集
├── README.md                # 本文档
└── analysis/                # 数据分析报告（运行脚本后生成）
    ├── train_report.md
    ├── val_report.md
    ├── test_report.md
    └── comparison_report.md
```

## 🔧 数据格式

### 标准格式（JSONL）

每行一个JSON对象，包含以下字段：

```json
{
  "instruction": "用户的指令/提示词",
  "input": "可选的额外输入信息（通常为空）",
  "output": "模型应该生成的诗词",
  "metadata": {
    "poem_type": "诗词类型",
    "style": "风格",
    "theme": "主题（可选）",
    "emotion": "情感（可选）",
    "season": "季节（可选）",
    "keywords": ["关键词列表（可选）"],
    "imitate": "模仿对象（可选）",
    "difficulty": "难度等级"
  }
}
```

### 示例数据

#### 示例1：基础创作

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

#### 示例2：带情感要求

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

#### 示例3：带约束条件

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

## 🚀 使用方法

### 1. 加载数据

```python
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 加载训练数据
train_data = load_jsonl('train.jsonl')
val_data = load_jsonl('val.jsonl')
test_data = load_jsonl('test.jsonl')
```

### 2. 数据处理与分析

使用提供的脚本进行数据处理：

```bash
# 数据验证和格式化
python scripts/prepare_poetry_data.py

# 数据可视化分析
python scripts/visualize_poetry_data.py
```

### 3. 模型训练

#### 使用LLaMA-Factory

```bash
# 安装LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# 配置数据集
# 在 data/dataset_info.json 中添加：
{
  "poetry_instruction": {
    "file_name": "path/to/train.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}

# 开始训练
llamafactory-cli train \
    --model_name_or_path your_base_model \
    --dataset poetry_instruction \
    --output_dir output/poetry_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4
```

#### 使用自定义训练脚本

```python
# @Author xiaomin.zhang

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# 加载模型和分词器
model_name = "your_base_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 准备数据
def format_instruction(example):
    prompt = f"### 指令:\n{example['instruction']}\n\n"
    if example.get('input'):
        prompt += f"### 输入:\n{example['input']}\n\n"
    prompt += f"### 输出:\n{example['output']}"
    return prompt

# 训练配置
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=10,
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

## 📈 数据质量保证

### 格律检查

所有诗词都经过格律验证：

- **五言绝句**: 4句，每句5字
- **七言绝句**: 4句，每句7字
- **五言律诗**: 8句，每句5字
- **七言律诗**: 8句，每句7字

### 内容质量

- ✅ 意境优美，语言流畅
- ✅ 内容与指令高度相关
- ✅ 避免生硬、牵强的表达
- ✅ 符合中国古典诗词的审美标准

## 🔍 数据统计

详细的数据统计信息请查看 `metadata.json` 文件，或运行可视化脚本生成分析报告。

### 快速统计

| 维度 | 统计 |
|------|------|
| 诗词类型 | 5种 |
| 风格类型 | 5种 |
| 主题数量 | 18个 |
| 情感类型 | 7种 |
| 难度级别 | 3级 |

## 📝 扩展数据集

### 方法1：人工创作

1. 参考现有格式创建新样本
2. 确保符合格律要求
3. 添加完整的metadata信息
4. 运行验证脚本检查

### 方法2：从古诗词库改编

```python
# @Author xiaomin.zhang

# 示例：从古诗词生成训练数据
def create_from_poem(poem_dict):
    return {
        "instruction": f"写一首{poem_dict['type']}，主题是{poem_dict['theme']}",
        "input": "",
        "output": poem_dict['content'],
        "metadata": {
            "poem_type": poem_dict['type'],
            "style": poem_dict['style'],
            "difficulty": "中级"
        }
    }
```

### 方法3：AI辅助生成

1. 使用大模型生成初稿
2. 人工审核和修改
3. 确保质量达标
4. 添加到数据集

## 🎯 评估指标

### 自动评估

- **格律准确率**: 是否符合诗词格律
- **BLEU分数**: 与参考诗词的相似度
- **困惑度**: 模型生成的流畅度

### 人工评估

| 维度 | 评分标准 |
|------|----------|
| 相关性 | 是否符合指令要求 (1-5分) |
| 创意性 | 是否有新意 (1-5分) |
| 艺术性 | 意境和美感 (1-5分) |
| 流畅度 | 语言是否通顺 (1-5分) |

## 🤝 贡献指南

欢迎贡献高质量的诗词数据！

### 贡献流程

1. Fork本项目
2. 创建新的数据样本
3. 运行验证脚本确保格式正确
4. 提交Pull Request
5. 等待审核

### 数据要求

- 符合格律规范
- 意境优美
- 指令清晰明确
- metadata信息完整

## 📚 参考资源

### 数据来源

- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) - 中文诗歌数据库
- 人工创作
- AI辅助生成（经人工审核）

### 相关工具

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 大模型微调框架
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - 指令微调方法
- [pypinyin](https://github.com/mozillazg/python-pinyin) - 拼音处理工具

### 学习资料

- 《唐诗三百首》
- 《宋词三百首》
- 《诗词格律》- 王力

## 📄 许可证

本数据集采用 MIT 许可证。

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 作者: xiaomin.zhang
- 项目地址: [nanogpt_zh](https://github.com/your-repo/nanogpt_zh)

## 🔄 更新日志

### v1.0.0 (2026-02-13)

- ✨ 初始版本发布
- 📊 包含30条高质量示例数据
- 🔧 提供数据处理和可视化脚本
- 📖 完整的文档和使用指南

---

**最后更新**: 2026-02-13  
**作者**: xiaomin.zhang
