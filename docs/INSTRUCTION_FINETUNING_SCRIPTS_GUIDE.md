# 指令微调脚本使用指南

> @Author xiaomin.zhang  
> 创建日期: 2026-02-13

## 📋 脚本概览

本项目提供了三个核心脚本用于诗词创作的指令微调：

| 脚本 | 功能 | 说明 |
|------|------|------|
| `train_instruction_finetuning.py` | 训练脚本 | 完整的指令微调训练流程 |
| `test_instruction_model.py` | 测试脚本 | 测试训练好的模型 |
| `evaluate_instruction_model.py` | 评估脚本 | 评估模型性能指标 |

---

## 🚀 快速开始

### 1. 训练模型

```bash
# 基础训练（使用默认参数）
python train_instruction_finetuning.py

# 使用LoRA微调（推荐，节省内存）
python train_instruction_finetuning.py --use_lora

# 自定义参数（在脚本中修改）
```

### 2. 测试模型

```bash
# 运行测试用例
python test_instruction_model.py --mode test

# 交互式测试
python test_instruction_model.py --mode interactive

# 参数对比测试
python test_instruction_model.py --mode compare
```

### 3. 评估模型

```bash
# 评估模型性能
python evaluate_instruction_model.py

# 指定模型路径
python evaluate_instruction_model.py --model_path ./output/poetry_instruction_model
```

---

## 📖 详细使用说明

### 一、训练脚本 (train_instruction_finetuning.py)

#### 功能特点

- ✅ 支持多种预训练模型（Qwen、ChatGLM等）
- ✅ 支持LoRA高效微调
- ✅ 支持多种提示词模板（Alpaca、ChatGPT）
- ✅ 自动数据处理和验证
- ✅ TensorBoard可视化
- ✅ 梯度检查点节省内存

#### 主要参数配置

在脚本的 `main()` 函数中修改参数：

```python
# 模型参数
model_args = ModelArguments(
    model_name_or_path="Qwen/Qwen2-1.5B",  # 模型路径
    use_lora=True,                          # 是否使用LoRA
    lora_r=8,                               # LoRA秩
    lora_alpha=32                           # LoRA alpha
)

# 数据参数
data_args = DataArguments(
    train_file="data/poetry_instruction/train.jsonl",
    val_file="data/poetry_instruction/val.jsonl",
    max_length=512,                         # 最大序列长度
    prompt_template="alpaca"                # 提示词模板
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./output/poetry_instruction_model",
    num_train_epochs=3,                     # 训练轮数
    per_device_train_batch_size=2,          # 批次大小
    gradient_accumulation_steps=8,          # 梯度累积
    learning_rate=5e-5,                     # 学习率
    fp16=True,                              # 混合精度训练
    # ... 更多参数
)
```

#### 训练流程

1. **加载模型和分词器**
   - 自动下载预训练模型
   - 配置LoRA（如果启用）
   - 启用梯度检查点

2. **准备数据**
   - 加载JSONL格式数据
   - 格式化为指令模板
   - 分词和padding

3. **开始训练**
   - 使用Trainer API
   - 自动保存检查点
   - TensorBoard日志

4. **保存模型**
   - 保存完整模型
   - 保存LoRA权重（如果使用）
   - 保存tokenizer

#### 输出文件

```
output/poetry_instruction_model/
├── config.json                 # 模型配置
├── pytorch_model.bin           # 模型权重
├── tokenizer_config.json       # 分词器配置
├── special_tokens_map.json     # 特殊token映射
├── lora_weights/               # LoRA权重（如果使用）
│   ├── adapter_config.json
│   └── adapter_model.bin
└── logs/                       # TensorBoard日志
    └── events.out.tfevents.*
```

#### 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir ./output/poetry_instruction_model/logs

# 在浏览器打开
# http://localhost:6006
```

#### 常见问题

**Q: GPU内存不足？**
```python
# 解决方案：
# 1. 启用LoRA
use_lora=True

# 2. 减小batch size
per_device_train_batch_size=1

# 3. 增加梯度累积
gradient_accumulation_steps=16

# 4. 启用梯度检查点
gradient_checkpointing=True
```

**Q: 训练速度慢？**
```python
# 解决方案：
# 1. 启用混合精度
fp16=True

# 2. 增加batch size
per_device_train_batch_size=4

# 3. 增加dataloader workers
dataloader_num_workers=4
```

---

### 二、测试脚本 (test_instruction_model.py)

#### 功能特点

- ✅ 三种测试模式
- ✅ 支持LoRA模型
- ✅ 可调节生成参数
- ✅ 交互式测试

#### 使用方法

##### 1. 测试用例模式

运行预定义的测试用例：

```bash
python test_instruction_model.py \
    --model_path ./output/poetry_instruction_model \
    --mode test
```

输出示例：
```
============================================================
测试用例 1/5
============================================================
指令: 写一首描写春天的五言绝句
------------------------------------------------------------
生成结果:
春风拂柳绿，
燕子绕梁飞。
桃花三月雨，
诗意满园归。
------------------------------------------------------------
```

##### 2. 交互式模式

与模型实时对话：

```bash
python test_instruction_model.py \
    --model_path ./output/poetry_instruction_model \
    --mode interactive
```

交互示例：
```
============================================================
诗词创作交互式测试
============================================================
输入指令来生成诗词，输入 'quit' 或 'exit' 退出
============================================================

请输入指令: 写一首关于秋天的七言绝句
输入额外信息（可选，直接回车跳过）: 

生成中...

------------------------------------------------------------
生成结果:
------------------------------------------------------------
秋风萧瑟叶飘零，
明月当空照古城。
游子思乡心切切，
何时归去见亲情。
------------------------------------------------------------
```

##### 3. 参数对比模式

对比不同temperature参数的效果：

```bash
python test_instruction_model.py \
    --model_path ./output/poetry_instruction_model \
    --mode compare
```

#### 生成参数说明

在脚本中可以调整以下参数：

```python
poem = generator.generate_poem(
    instruction="写一首诗",
    input_text="",
    max_new_tokens=128,        # 最大生成token数
    temperature=0.7,           # 温度（0.1-2.0）
    top_p=0.9,                 # nucleus sampling
    top_k=50,                  # top-k sampling
    repetition_penalty=1.1,    # 重复惩罚
    do_sample=True             # 是否采样
)
```

参数效果：

| 参数 | 低值效果 | 高值效果 | 推荐值 |
|------|----------|----------|--------|
| temperature | 更保守、重复 | 更随机、创新 | 0.7 |
| top_p | 更确定性 | 更多样性 | 0.9 |
| repetition_penalty | 允许重复 | 避免重复 | 1.1 |

---

### 三、评估脚本 (evaluate_instruction_model.py)

#### 功能特点

- ✅ 自动评估测试集
- ✅ 多维度评估指标
- ✅ 生成详细报告
- ✅ 保存JSON结果

#### 使用方法

```bash
# 基础评估
python evaluate_instruction_model.py

# 指定路径
python evaluate_instruction_model.py \
    --model_path ./output/poetry_instruction_model \
    --test_file data/poetry_instruction/test.jsonl \
    --output_dir ./output/evaluation
```

#### 评估指标

1. **格式准确率**
   - 检查诗词是否符合格律
   - 验证句数和字数

2. **BLEU分数**
   - 衡量与参考诗词的相似度
   - 范围：0-1，越高越好

3. **字符准确率**
   - 生成文本与参考文本的字符重叠率
   - 范围：0-100%

#### 输出文件

```
output/evaluation/
├── evaluation_results.json     # JSON格式结果
└── evaluation_report.md        # Markdown格式报告
```

#### 评估报告示例

```markdown
# 诗词创作模型评估报告

**模型路径**: ./output/poetry_instruction_model
**测试数据**: data/poetry_instruction/test.jsonl
**测试样本数**: 5

## 评估指标

- **格式准确率**: 80.00% (4/5)
- **平均BLEU分数**: 0.3245
- **平均字符准确率**: 65.23%

## 详细结果

### 样本 1

**指令**: 写一首关于秋月的五言绝句

**参考输出**:
```
秋月照窗明，
清辉洒满庭。
举头望明月，
低头思故乡。
```

**生成输出**:
```
秋月挂天边，
清光洒人间。
举杯邀明月，
对影成三人。
```

**诗词类型**: 五言绝句
**格式正确**: ✅
**BLEU分数**: 0.3521
**字符准确率**: 68.42%

---
```

---

## 🎯 完整工作流程

### 步骤1：准备数据

```bash
# 验证数据格式
python scripts/prepare_poetry_data.py

# 查看数据统计
python scripts/visualize_poetry_data.py
```

### 步骤2：训练模型

```bash
# 开始训练
python train_instruction_finetuning.py

# 监控训练（另开终端）
tensorboard --logdir ./output/poetry_instruction_model/logs
```

### 步骤3：测试模型

```bash
# 快速测试
python test_instruction_model.py --mode test

# 交互式测试
python test_instruction_model.py --mode interactive
```

### 步骤4：评估模型

```bash
# 完整评估
python evaluate_instruction_model.py

# 查看报告
cat output/evaluation/evaluation_report.md
```

---

## 🔧 高级配置

### 1. 使用不同的基础模型

```python
# 在 train_instruction_finetuning.py 中修改

# Qwen系列
model_name_or_path="Qwen/Qwen2-1.5B"
model_name_or_path="Qwen/Qwen2-7B"

# ChatGLM系列
model_name_or_path="THUDM/chatglm3-6b"

# 其他中文模型
model_name_or_path="baichuan-inc/Baichuan2-7B-Chat"
```

### 2. 自定义提示词模板

```python
# 在 InstructionDataProcessor 类中添加新模板

def format_custom_prompt(self, instruction, input_text, output):
    """自定义格式"""
    return f"问：{instruction}\n答：{output}"
```

### 3. 调整LoRA参数

```python
# 更大的秩 = 更强的表达能力，但需要更多内存
lora_r=16  # 默认8

# 更大的alpha = 更强的适应能力
lora_alpha=64  # 默认32

# 目标模块（根据模型架构调整）
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

### 4. 优化训练速度

```python
# 启用混合精度训练
fp16=True  # 或 bf16=True

# 增加批次大小
per_device_train_batch_size=4

# 使用更快的优化器
optim="adamw_torch_fused"  # 需要PyTorch 2.0+

# 增加workers
dataloader_num_workers=8
```

---

## 📊 性能基准

### 训练时间（参考）

| 配置 | GPU | 数据量 | 时间 |
|------|-----|--------|------|
| LoRA + FP16 | RTX 3090 (24GB) | 20条 | ~5分钟 |
| LoRA + FP16 | RTX 3090 (24GB) | 1000条 | ~2小时 |
| 全参数 | A100 (40GB) | 1000条 | ~6小时 |

### 内存占用（参考）

| 配置 | 模型大小 | 内存占用 |
|------|----------|----------|
| Qwen2-1.5B + LoRA | 1.5B | ~6GB |
| Qwen2-7B + LoRA | 7B | ~16GB |
| Qwen2-7B 全参数 | 7B | ~32GB |

---

## 🐛 故障排除

### 问题1：CUDA out of memory

```python
# 解决方案：
# 1. 减小batch size
per_device_train_batch_size=1

# 2. 使用LoRA
use_lora=True

# 3. 启用梯度检查点
gradient_checkpointing=True

# 4. 减小max_length
max_length=256
```

### 问题2：训练loss不下降

```python
# 解决方案：
# 1. 调整学习率
learning_rate=1e-4  # 尝试更大的学习率

# 2. 增加warmup
warmup_steps=200

# 3. 检查数据质量
# 运行 prepare_poetry_data.py 验证数据
```

### 问题3：生成结果不符合格律

```bash
# 解决方案：
# 1. 增加训练数据
# 2. 增加训练轮数
num_train_epochs=5

# 3. 在提示词中强调格律
instruction="写一首严格符合格律的五言绝句"

# 4. 后处理验证
# 使用 evaluate_instruction_model.py 检查格式
```

---

## 📚 相关文档

- [完整教程](./instruction_finetuning_tutorial.md)
- [快速开始](./QUICKSTART_INSTRUCTION_FINETUNING.md)
- [数据集说明](../data/poetry_instruction/README.md)
- [项目总结](./POETRY_FINETUNING_SUMMARY.md)

---

## 🤝 贡献

欢迎改进脚本！提交PR时请确保：

- 代码符合PEP 8规范
- 添加必要的注释
- 更新相关文档

---

**作者**: xiaomin.zhang  
**更新时间**: 2026-02-13
