# NanoGPT-ZH

从零实现的中文 GPT 诗词生成模型。当前主线已经从通用教学版 nanoGPT，推进到一个约 92.74M 参数的中文古诗词模型，并在本地完成了一轮面向“按要求创作”的指令微调。

本项目不依赖 Hugging Face `transformers` 训练框架，核心模型、注意力、Transformer Block、Tokenizer、训练循环和 SFT 数据管线都在仓库内实现，适合学习 Decoder-Only Transformer 的完整工程链路。

## 当前状态

| 项目 | 结果 |
| --- | --- |
| 基座任务 | 中文古诗词自回归生成 |
| 模型结构 | Decoder-Only Transformer / GPT |
| 参数规模 | 92.74M |
| 词表 | 字符级中文词表，约 9,837 个有效 token |
| 最大上下文 | 128 token |
| 预训练数据 | `data/processed/train.txt`，372,911 条 |
| 预训练最佳模型 | `checkpoints/best_model.pt` |
| 预训练最佳验证损失 | `3.8024`，global step `62000` |
| SFT 数据 | `data/poetry_sft_v2_10000`，10,000 条 |
| SFT 最佳模型 | `checkpoints_poetry_sft_v2/best_model.pt` |
| SFT 最佳验证损失 | `2.8906`，global step `1800` |
| 当前设备验证 | Apple M4 / MPS |

说明：预训练损失和 SFT 损失来自不同数据与不同 label mask 方式，不能当作同一任务上的严格横向比较。SFT 损失更适合用于判断指令微调阶段是否收敛。

## 背景

最初版本的 NanoGPT-ZH 是一个小规模中文 GPT 教学项目，用来验证从数据处理、词表构建、模型实现到训练推理的完整流程。后续项目转向中文古诗词生成任务，原因是：

- 古诗词文本结构短，适合小上下文模型训练和观察效果。
- 字符级建模可以覆盖冷门字、古汉语用字和标点，不强依赖分词质量。
- 诗词有较明确的体裁、主题、关键词约束，适合做从预训练到 SFT 的能力演进实验。

当前基座模型已经能生成古诗词风格文本，但原始预训练模型只学习“续写分布”，不天然理解“写一首七言绝句，包含某些字”这类显式要求。因此本轮新增了指令数据集生成与 SFT 训练流程，让模型学习“要求 -> 作品”的映射。

## 算法核心

模型是标准 GPT 风格的 Decoder-Only Transformer：

```text
token ids
  -> token embedding
  -> sinusoidal positional encoding
  -> [Pre-LN Transformer Block] x 12
  -> final layer norm
  -> tied language modeling head
  -> next-token logits
```

核心组件：

- `models/attention.py`：多头自注意力和因果 mask。每个位置只能看到自己及之前的 token。
- `models/transformer.py`：Pre-LN Transformer Block，结构为 `x + Attention(LN(x))` 和 `x + FFN(LN(x))`。
- `models/gpt.py`：GPT 主模型，包含 embedding、位置编码、12 层 Transformer、最终 LayerNorm 和 LM Head。
- `utils/tokenizer.py`：中文 Tokenizer。当前诗词词表会被检测为字符级词表，因此编码时按字符切分。

注意力公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

训练目标：

```text
给定 x_1, x_2, ..., x_t
最大化 P(x_{t+1} | x_1, ..., x_t)
```

也就是用交叉熵训练下一个 token 预测。`pad_id=0` 会在 loss 中被忽略。

## 模型配置

当前诗词模型配置在 `config_poetry.py`：

| 参数 | 值 |
| --- | --- |
| `vocab_size` | 10000 |
| `max_seq_len` | 128 |
| `d_model` | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 |
| `d_ff` | 3072 |
| `dropout` | 0.1 |
| `grad_clip` | 1.0 |

实际初始化参数量：

```text
92.74M
```

## 项目结构

```text
nanogpt-zh/
├── models/                         # GPT、注意力、Transformer Block
├── utils/                          # Tokenizer、DataLoader、Trainer
├── scripts/
│   ├── prepare_poetry_data.py       # 诗词数据处理
│   ├── rebuild_vocab.py             # 字符级词表重建
│   └── build_poetry_sft_dataset.py  # SFT 指令数据集生成
├── data/
│   ├── processed/                   # 预训练数据与词表
│   └── poetry_sft_v2_10000/         # 本轮 SFT 数据集
├── checkpoints/                     # 预训练模型
├── checkpoints_poetry_sft_v2/       # 本轮 SFT 模型
├── logs_poetry_sft_v2/              # TensorBoard 日志
├── train_poetry.py                  # 诗词预训练入口
├── train_poetry_sft.py              # 指令微调入口
└── tests/                           # 单元测试和管线测试
```

## 环境准备

推荐使用当前已验证的 Conda 环境：

```bash
conda create -n nanogpt python=3.10 -y
conda activate nanogpt
pip install -r requirements.txt
```

在 Apple Silicon 上，PyTorch 使用 MPS；在 NVIDIA GPU 上会自动使用 CUDA。

快速检查：

```bash
python tests/test_model.py
python tests/test_gpu.py
```

## 预训练过程

预训练目标是让模型先学习中文古诗词的字符分布、句式、韵律和常见意象。数据来自本地处理后的诗词语料：

```text
data/processed/train.txt  372,911 条
data/processed/val.txt      9,813 条
data/processed/test.txt     9,814 条
data/processed/vocab.json   字符级词表
```

数据准备流程：

```bash
conda activate nanogpt
python scripts/prepare_poetry_data.py
python scripts/rebuild_vocab.py
```

启动预训练：

```bash
conda activate nanogpt
python train_poetry.py
```

预训练训练器使用：

- AdamW 优化器
- warmup + cosine decay 学习率调度
- 梯度裁剪
- TensorBoard 日志
- 定期验证与 checkpoint 保存

当前可用的预训练最佳模型：

```text
checkpoints/best_model.pt
epoch: 7
global_step: 62000
best_val_loss: 3.8024
```

这个模型可以生成诗词风格文本，但它没有经过指令对齐，对“指定体裁、主题、关键词”的遵循不稳定。

## SFT 数据集

本轮重新生成了 `poetry-sft-v2-10000`，用于让模型学习按创作要求输出诗词。

生成命令：

```bash
conda activate nanogpt
PYTHONPATH=. python scripts/build_poetry_sft_dataset.py \
  --input data/processed/train.txt \
  --output_dir data/poetry_sft_v2_10000 \
  --total 10000 \
  --seed 43 \
  --max_samples_per_poem 6 \
  --reservoir_cap_per_group 5000
```

数据规模：

| split | 样本数 |
| --- | ---: |
| train | 9000 |
| val | 500 |
| test | 500 |

指令类型分布：

| 类型 | 数量 | 目的 |
| --- | ---: | --- |
| `keyword` | 3500 | 强化包含指定字词 |
| `combined` | 3000 | 同时约束体裁、主题、关键词 |
| `theme` | 2000 | 强化主题创作 |
| `basic` | 1000 | 基础体裁创作 |
| `season` | 500 | 季节类主题 |

体裁分布：

| 体裁 | 数量 |
| --- | ---: |
| 七言绝句 | 3250 |
| 五言绝句 | 3247 |
| 五言律诗 | 3249 |
| 七言律诗 | 254 |

数据质量统计：

```text
candidate_samples: 75358
unique_instruction_count: 2423
unique_output_count: 9627
duplicate_output_count: 373
constraint_errors: 0
```

SFT 文本格式很紧凑，以适配当前 `max_seq_len=128`：

```text
要求：写一首七言绝句，包含“山”、“水”。
作品：万树山河古战场，天生壮士一杯觞。...
```

训练时会 mask 掉 prompt 部分，只在 `作品：` 之后的输出 token 上计算 loss。

## SFT 训练过程

SFT 从预训练最佳模型继续训练：

```bash
conda activate nanogpt
PYTHONPATH=. PYTHONUNBUFFERED=1 python -u train_poetry_sft.py \
  --train_file data/poetry_sft_v2_10000/train.jsonl \
  --val_file data/poetry_sft_v2_10000/val.jsonl \
  --init_checkpoint checkpoints/best_model.pt \
  --output_dir checkpoints_poetry_sft_v2 \
  --log_dir logs_poetry_sft_v2 \
  --epochs 5 \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --eval_interval 100 \
  --save_interval 1000 \
  --log_interval 20 \
  --early_stopping_patience 2 \
  --early_stopping_min_delta 0.002
```

本轮训练设备：

```text
Apple M4 GPU via PyTorch MPS
16GB unified memory
```

训练结果：

```text
steps: 1800
best_val_loss: 2.8906
elapsed: 47.9 minutes
best checkpoint: checkpoints_poetry_sft_v2/best_model.pt
final checkpoint: checkpoints_poetry_sft_v2/final_model.pt
```

验证损失走势摘要：

| step | val loss |
| ---: | ---: |
| 100 | 3.6487 |
| 200 | 3.2166 |
| 500 | 3.0121 |
| epoch 1 | 2.9989 |
| 1000 | 2.9357 |
| epoch 2 | 2.9251 |
| 1400 | 2.9065 |
| epoch 3 | 2.8922 |
| 1800 | 2.8906 |

第 4 轮 step 1800 触发 early stopping。日志里 `early_stop_best` 使用 `min_delta=0.002` 判定有效改善，checkpoint 则保存严格最小的 `checkpoint_best=2.8906`。

## 推理测试结果

本轮 SFT 后，用 `checkpoints_poetry_sft_v2/best_model.pt` 做了指令生成测试。

测试参数：

```text
temperature: 0.82
top_k: 50
top_p: 0.9
max_new_tokens: 84
```

样例 1：

```text
要求：写一首七言绝句，包含“山”、“水”。
输出：万树山河古战场，天生壮士一杯觞。至今海上无风雨，不知何处是金堂。
检查：山=True，水=False
```

样例 2：

```text
要求：写一首五言绝句，包含“月”、“江”。
输出：南州古驿边，一望一悠然。夜月江边水，秋风树杪船。
检查：月=True，江=True
```

样例 3：

```text
要求：以“离别”为主题，写一首七言绝句。
输出：四海干戈满，一身天地老风埃。不须辛苦学张翰，自是英雄已是才。
```

样例 4：

```text
要求：写一首关于春天的五言绝句。
输出：江山千古意，花木一时春。我亦江湖客，江南问水滨。
```

样例 5：

```text
要求：写一首七言律诗，包含“风”、“花”。
输出：幽禽啄叶鸣桑柘，日暮村村花乱开。雨后园林芳草遍，风前林木亦花来。行人莫道春无计，不是风光不是才。不是天公留意久，却缘风雨为花催。
检查：风=True，花=True
```

结论：

- 指令跟随能力较预训练模型明显增强。
- 关键词约束已有改善，但还不是 100% 可靠。
- 体裁和主题大体能跟随，但个别输出仍有句式混乱或字数不严的问题。
- 当前模型适合作为“小参数诗词指令模型”的实验基线，不应视为稳定的对话模型。

如果要进一步提高可控生成质量，下一步更有效的是在推理侧增加多候选采样、关键词检查、体裁检查和重排；训练侧可以继续补强七言律诗、严格格律样本和失败样本反例。

## 验证命令

本轮修改后通过了以下检查：

```bash
conda run --no-capture-output -n nanogpt env PYTHONPATH=. python tests/test_poetry_sft_pipeline.py
conda run --no-capture-output -n nanogpt env PYTHONPATH=. python -m py_compile scripts/build_poetry_sft_dataset.py train_poetry_sft.py tests/test_poetry_sft_pipeline.py
conda run --no-capture-output -n nanogpt env PYTHONPATH=. python tests/test_model.py
```

结果：

```text
tests/test_poetry_sft_pipeline.py: 4 tests OK
py_compile: OK
tests/test_model.py: all tests passed
```

## 常用命令

查看 TensorBoard：

```bash
tensorboard --logdir logs_poetry_sft_v2
```

查看 checkpoint 元信息：

```bash
conda run --no-capture-output -n nanogpt env PYTHONPATH=. python - <<'PY'
import torch

for path in ["checkpoints/best_model.pt", "checkpoints_poetry_sft_v2/best_model.pt"]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(path)
    print("  epoch:", ckpt.get("epoch"))
    print("  global_step:", ckpt.get("global_step"))
    print("  best_val_loss:", ckpt.get("best_val_loss"))
PY
```

重新生成 SFT 数据：

```bash
conda run --no-capture-output -n nanogpt env PYTHONPATH=. python scripts/build_poetry_sft_dataset.py
```

重新运行 SFT：

```bash
conda run --no-capture-output -n nanogpt env PYTHONPATH=. python -u train_poetry_sft.py
```

## 已知限制

- 当前上下文只有 128 token，复杂长指令和长诗会受限制。
- SFT 数据来自已有诗词样本重组，模型更像“按约束仿写古诗”，不是开放域聊天助手。
- 对关键词、字数、格律的遵循仍需推理后处理或更强的约束训练。
- 七言律诗样本较少，相关能力弱于绝句和五言律诗。
- Apple MPS 可完成训练，但吞吐明显低于中高端 NVIDIA GPU。

## 许可证

MIT License

## 作者

@Author xiaomin.zhang
