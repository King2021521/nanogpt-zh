# 诗词创作指令微调快速开始指南

> @Author xiaomin.zhang

## 🚀 5分钟快速开始

本指南将帮助你快速开始诗词创作模型的指令微调。

## 📋 前置要求

- Python 3.8+
- PyTorch 1.12+
- transformers 4.30+
- 至少8GB GPU内存（推荐）

## 📦 安装依赖

```bash
# 基础依赖
pip install torch transformers datasets accelerate

# 数据处理依赖
pip install matplotlib pandas

# 可选：使用LLaMA-Factory进行训练
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 📊 查看数据集

### 1. 数据统计

```bash
# 运行数据处理脚本
python scripts/prepare_poetry_data.py

# 生成可视化分析
python scripts/visualize_poetry_data.py
```

### 2. 查看生成的报告

```bash
# 查看训练集报告
cat data/poetry_instruction/analysis/train_report.md

# 查看数据对比报告
cat data/poetry_instruction/analysis/comparison_report.md
```

## 🎯 数据集概览

当前示例数据集包含：

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| 训练集 | 20条 | 用于模型训练 |
| 验证集 | 5条 | 用于训练过程中的验证 |
| 测试集 | 5条 | 用于最终评估 |
| **总计** | **30条** | 高质量示例数据 |

### 诗词类型分布

- 五言绝句: 46.7%
- 七言绝句: 33.3%
- 五言律诗: 10.0%
- 七言律诗: 3.3%
- 古体诗: 6.7%

## 🔧 训练模型

### 方法1：使用LLaMA-Factory（推荐）

#### 步骤1：配置数据集

在 `LLaMA-Factory/data/dataset_info.json` 中添加：

```json
{
  "poetry_instruction": {
    "file_name": "../../data/poetry_instruction/train.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

#### 步骤2：启动训练

```bash
cd LLaMA-Factory

# 使用Web UI（推荐新手）
llamafactory-cli webui

# 或使用命令行
llamafactory-cli train \
    --stage sft \
    --model_name_or_path Qwen/Qwen2-1.5B \
    --dataset poetry_instruction \
    --template qwen \
    --finetuning_type lora \
    --output_dir ../output/poetry_lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16
```

### 方法2：使用自定义训练脚本

创建 `train_poetry.py`:

```python
# @Author xiaomin.zhang

import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# 1. 加载数据
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 2. 格式化数据
def format_instruction(example):
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']
    
    if input_text:
        prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n"
    else:
        prompt = f"### 指令:\n{instruction}\n\n### 输出:\n"
    
    return prompt + output

# 3. 加载模型
model_name = "Qwen/Qwen2-1.5B"  # 或其他中文模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    device_map="auto"
)

# 4. 准备数据集
train_data = load_data("data/poetry_instruction/train.jsonl")
val_data = load_data("data/poetry_instruction/val.jsonl")

train_texts = [format_instruction(x) for x in train_data]
val_texts = [format_instruction(x) for x in val_data]

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

train_dataset = Dataset.from_dict({'text': train_texts})
val_dataset = Dataset.from_dict({'text': val_texts})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./output/poetry_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    save_total_limit=3,
    fp16=True,
    report_to="tensorboard"
)

# 6. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()

# 7. 保存模型
trainer.save_model("./output/poetry_model_final")
tokenizer.save_pretrained("./output/poetry_model_final")

print("训练完成！")
```

运行训练：

```bash
python train_poetry.py
```

## 🧪 测试模型

创建 `test_poetry.py`:

```python
# @Author xiaomin.zhang

from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载微调后的模型
model_path = "./output/poetry_model_final"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    device_map="auto"
)

def generate_poem(instruction, input_text=""):
    """生成诗词"""
    if input_text:
        prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n"
    else:
        prompt = f"### 指令:\n{instruction}\n\n### 输出:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取输出部分
    if "### 输出:" in result:
        result = result.split("### 输出:")[1].strip()
    
    return result

# 测试示例
test_cases = [
    "写一首描写春天的五言绝句",
    "创作一首表达思乡之情的七言绝句",
    "以'月'为主题，写一首诗",
]

print("="*50)
print("诗词创作测试")
print("="*50)

for instruction in test_cases:
    print(f"\n指令: {instruction}")
    print("-"*50)
    poem = generate_poem(instruction)
    print(poem)
    print("-"*50)
```

运行测试：

```bash
python test_poetry.py
```

## 📈 评估模型

### 1. 自动评估

```python
# @Author xiaomin.zhang

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu

# 加载测试数据
test_data = []
with open("data/poetry_instruction/test.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))

# 加载模型
model_path = "./output/poetry_model_final"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 评估
total_bleu = 0
for item in test_data:
    instruction = item['instruction']
    reference = item['output']
    
    # 生成
    prompt = f"### 指令:\n{instruction}\n\n### 输出:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 计算BLEU
    bleu = sentence_bleu([list(reference)], list(generated))
    total_bleu += bleu
    
    print(f"指令: {instruction}")
    print(f"参考: {reference}")
    print(f"生成: {generated}")
    print(f"BLEU: {bleu:.4f}\n")

avg_bleu = total_bleu / len(test_data)
print(f"平均BLEU分数: {avg_bleu:.4f}")
```

### 2. 人工评估

评估维度（1-5分）：

- **相关性**: 是否符合指令要求
- **创意性**: 是否有新意
- **艺术性**: 意境和美感
- **流畅度**: 语言是否通顺
- **格律性**: 是否符合诗词格律

## 📊 监控训练

### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir ./output/poetry_model/runs

# 在浏览器中打开 http://localhost:6006
```

### 查看训练日志

```bash
# 查看最新的训练日志
tail -f ./output/poetry_model/trainer_log.txt
```

## 🎨 优化建议

### 1. 扩充数据集

当前只有30条示例数据，建议扩充到：

- **最小**: 1,000+ 条
- **推荐**: 5,000+ 条
- **理想**: 20,000+ 条

### 2. 调整超参数

```python
# 学习率
learning_rate = 5e-5  # 可尝试 1e-5 到 1e-4

# 训练轮数
num_train_epochs = 3  # 可尝试 3-10

# 批次大小
per_device_train_batch_size = 4  # 根据GPU内存调整

# 温预热步数
warmup_steps = 100  # 约为总步数的10%
```

### 3. 使用LoRA微调

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # LoRA秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

## 🐛 常见问题

### Q1: GPU内存不足

**解决方案**：
- 减小batch_size
- 使用gradient_accumulation_steps
- 使用LoRA而不是全参数微调
- 使用8bit或4bit量化

### Q2: 生成的诗词不符合格律

**解决方案**：
- 增加训练数据
- 在训练数据中强调格律要求
- 后处理验证格律

### Q3: 模型过拟合

**解决方案**：
- 增加训练数据
- 使用数据增强
- 降低学习率
- 增加dropout

## 📚 下一步

1. ✅ 扩充数据集到1000+条
2. ✅ 尝试不同的基础模型
3. ✅ 调整超参数优化性能
4. ✅ 添加格律验证后处理
5. ✅ 部署为API服务

## 🔗 相关资源

- [完整教程](./instruction_finetuning_tutorial.md)
- [数据集README](../data/poetry_instruction/README.md)
- [LLaMA-Factory文档](https://github.com/hiyouga/LLaMA-Factory)

---

**作者**: xiaomin.zhang  
**更新时间**: 2026-02-13
