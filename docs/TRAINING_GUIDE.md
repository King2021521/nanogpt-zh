# NanoGPT-ZH 训练及调参操作手册

<div align="center">

**完整的训练流程 | 参数调优指南 | 问题排查**

</div>

---

## 📋 目录

- [1. 训练前准备](#1-训练前准备)
- [2. 基础训练流程](#2-基础训练流程)
- [3. 参数调优指南](#3-参数调优指南)
- [4. 高级训练技巧](#4-高级训练技巧)
- [5. 监控与调试](#5-监控与调试)
- [6. 常见问题排查](#6-常见问题排查)
- [7. 最佳实践](#7-最佳实践)

---

## 1. 训练前准备

### 1.1 环境检查

```bash
# 检查Python版本（需要3.8+）
python --version

# 检查PyTorch安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"

# 检查GPU可用性
python tests/test_gpu.py
```

### 1.2 数据准备

#### 方式一：使用示例数据（快速测试）
```bash
# 生成示例数据
python scripts/prepare_data.py
```

#### 方式二：使用维基百科数据（推荐）
```bash
# 1. 下载中文维基百科数据
# 访问: https://dumps.wikimedia.org/zhwiki/latest/
# 下载: zhwiki-latest-pages-articles.xml.bz2

# 2. 提取文本
python scripts/extract_wiki.py --input zhwiki-latest-pages-articles.xml.bz2

# 3. 处理数据
python scripts/process_wiki_data.py
```

#### 方式三：使用自定义数据
```bash
# 将文本文件放在 data/raw/ 目录
# 运行数据处理脚本
python scripts/prepare_data.py --input data/raw/your_data.txt
```

### 1.3 配置检查

编辑 `config.py` 确认配置：

```python
# 检查数据路径
train_data_path = "data/processed/train.txt"
val_data_path = "data/processed/val.txt"
vocab_path = "data/processed/vocab.json"

# 检查输出路径
checkpoint_dir = "checkpoints"
log_dir = "logs"
```

---

## 2. 基础训练流程

### 2.1 开始训练

```bash
# 基础训练（使用默认配置）
python train.py
```

### 2.2 训练过程说明

训练脚本会自动执行以下步骤：

1. **加载分词器** - 从 `data/processed/vocab.json` 加载词表
2. **创建数据加载器** - 加载训练和验证数据
3. **初始化模型** - 创建GPT模型（约7.3M参数）
4. **创建训练器** - 配置优化器、学习率调度器
5. **开始训练循环** - 训练并定期验证、保存检查点

### 2.3 训练输出

```
==================================================
中文GPT模型训练
==================================================

1. 加载分词器...
词表大小: 10000

2. 创建数据加载器...
训练样本数: 50000
验证样本数: 5000

3. 创建模型...
GPT模型初始化完成
参数量: 7.30M

4. 创建训练器...
训练器初始化完成
设备: cuda

5. 开始训练...
==================================================
开始训练
==================================================

Epoch 1/10: 100%|██████████| 1563/1563 [05:23<00:00, 4.83it/s, loss=4.2156, lr=0.000300]

Epoch 1/10
  Train Loss: 4.2156 | Train PPL: 67.71
  Val Loss: 3.9823 | Val PPL: 53.62
  Time: 5.38 min
```

### 2.4 查看训练日志

```bash
# 启动TensorBoard
tensorboard --logdir=logs

# 在浏览器打开
# http://localhost:6006
```

---

## 3. 参数调优指南

### 3.1 模型架构参数

#### 3.1.1 词表大小 (`vocab_size`)

```python
# config.py
vocab_size = 10000  # 默认值
```

**调优建议：**
- **小数据集**：5000-10000
- **中等数据集**：10000-20000
- **大数据集**：20000-50000

**影响：**
- ↑ 词表大小 → ↑ 表达能力 | ↑ 参数量 | ↑ 训练时间
- ↓ 词表大小 → ↓ 表达能力 | ↓ 参数量 | ↓ 训练时间

#### 3.1.2 模型维度 (`d_model`)

```python
# config.py
d_model = 256  # 默认值
```

**调优建议：**
- **轻量级**：128-256（个人电脑）
- **标准**：512-768（单GPU）
- **大型**：1024-2048（多GPU）

**影响：**
- ↑ d_model → ↑ 模型容量 | ↑ 显存占用 | ↑ 训练时间

#### 3.1.3 层数 (`n_layers`)

```python
# config.py
n_layers = 6  # 默认值
```

**调优建议：**
- **小模型**：4-6层（快速实验）
- **中等模型**：8-12层（平衡性能）
- **大模型**：16-24层（最佳性能）

**影响：**
- ↑ n_layers → ↑ 模型深度 | ↑ 表达能力 | ↑ 训练难度

#### 3.1.4 注意力头数 (`n_heads`)

```python
# config.py
n_heads = 8  # 默认值
```

**调优建议：**
- **约束**：`d_model` 必须能被 `n_heads` 整除
- **常见配置**：
  - d_model=256 → n_heads=8 (每头32维)
  - d_model=512 → n_heads=8 (每头64维)
  - d_model=768 → n_heads=12 (每头64维)

**影响：**
- ↑ n_heads → ↑ 多样性 | ↓ 每头维度

#### 3.1.5 前馈网络维度 (`d_ff`)

```python
# config.py
d_ff = 1024  # 默认值（4倍d_model）
```

**调优建议：**
- **标准**：`d_ff = 4 * d_model`
- **轻量**：`d_ff = 2 * d_model`
- **重量**：`d_ff = 8 * d_model`

#### 3.1.6 最大序列长度 (`max_seq_len`)

```python
# config.py
max_seq_len = 256  # 默认值
```

**调优建议：**
- **短文本**：128-256（对话、评论）
- **中等文本**：512-1024（文章段落）
- **长文本**：2048-4096（长文档）

**影响：**
- ↑ max_seq_len → ↑ 上下文长度 | ↑ 显存占用（平方级）

### 3.2 训练超参数

#### 3.2.1 批次大小 (`batch_size`)

```python
# config.py
batch_size = 32  # 默认值
```

**调优建议：**
- **显存受限**：8-16
- **标准训练**：32-64
- **大显存**：128-256

**动态调整：**
```python
# 根据显存自动调整
if torch.cuda.get_device_properties(0).total_memory < 8e9:  # <8GB
    batch_size = 16
elif torch.cuda.get_device_properties(0).total_memory < 16e9:  # <16GB
    batch_size = 32
else:
    batch_size = 64
```

**影响：**
- ↑ batch_size → ↑ 训练稳定性 | ↑ 显存占用 | ↑ 训练速度
- ↓ batch_size → ↓ 训练稳定性 | ↓ 显存占用 | 可能需要调整学习率

#### 3.2.2 学习率 (`learning_rate`)

```python
# config.py
learning_rate = 3e-4  # 默认值
```

**调优建议：**
- **小模型**：3e-4 到 5e-4
- **中等模型**：1e-4 到 3e-4
- **大模型**：5e-5 到 1e-4

**学习率与批次大小的关系：**
```python
# 线性缩放规则
base_lr = 3e-4
base_batch_size = 32
actual_batch_size = 64

learning_rate = base_lr * (actual_batch_size / base_batch_size)
# learning_rate = 3e-4 * (64 / 32) = 6e-4
```

**学习率调度策略：**

当前使用：**预热 + 余弦退火**

```python
# utils/trainer.py
def lr_lambda(step):
    # 预热阶段：线性增长
    if step < warmup_steps:
        return step / warmup_steps
    
    # 余弦退火
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
```

#### 3.2.3 预热步数 (`warmup_steps`)

```python
# config.py
warmup_steps = 1000  # 默认值
```

**调优建议：**
- **小数据集**：500-1000步
- **中等数据集**：1000-2000步
- **大数据集**：2000-5000步

**经验公式：**
```python
# 预热步数 = 总步数的 5-10%
warmup_steps = int(max_steps * 0.05)
```

#### 3.2.4 权重衰减 (`weight_decay`)

```python
# config.py
weight_decay = 0.01  # 默认值
```

**调优建议：**
- **标准值**：0.01
- **轻微正则化**：0.001
- **强正则化**：0.1

**影响：**
- ↑ weight_decay → ↑ 正则化 | ↓ 过拟合风险 | 可能 ↓ 训练速度

#### 3.2.5 Dropout (`dropout`)

```python
# config.py
dropout = 0.1  # 默认值
```

**调优建议：**
- **小数据集**：0.2-0.3（防止过拟合）
- **中等数据集**：0.1-0.2
- **大数据集**：0.0-0.1

#### 3.2.6 梯度裁剪 (`grad_clip`)

```python
# config.py
grad_clip = 1.0  # 默认值
```

**调优建议：**
- **标准值**：1.0
- **训练不稳定**：0.5（更严格）
- **训练稳定**：5.0（更宽松）

### 3.3 采样参数（推理时）

#### 3.3.1 温度 (`temperature`)

```python
# config.py
temperature = 1.0  # 默认值
```

**调优建议：**
- **确定性输出**：0.1-0.5（更保守）
- **平衡**：0.7-1.0
- **创造性输出**：1.0-2.0（更随机）

**示例：**
```python
# 生成时调整温度
model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8  # 稍微随机
)
```

#### 3.3.2 Top-K采样 (`top_k`)

```python
# config.py
top_k = 50  # 默认值
```

**调优建议：**
- **保守**：10-20
- **平衡**：40-60
- **多样**：80-100
- **不限制**：None

#### 3.3.3 Top-P采样 (`top_p`)

```python
# config.py
top_p = 0.9  # 默认值
```

**调优建议：**
- **保守**：0.7-0.8
- **平衡**：0.9-0.95
- **多样**：0.95-1.0

---

## 4. 高级训练技巧

### 4.1 梯度累积

当显存不足时，使用梯度累积模拟大批次：

```python
# 修改 utils/trainer.py
accumulation_steps = 4  # 累积4步

for batch_idx, (input_ids, target_ids) in enumerate(pbar):
    # 前向传播
    logits, loss = self.model(input_ids, target_ids)
    loss = loss / accumulation_steps  # 缩放损失
    
    # 反向传播
    loss.backward()
    
    # 每N步更新一次
    if (batch_idx + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
```

**效果：**
- 实际批次大小 = `batch_size * accumulation_steps`
- 显存占用 = `batch_size`

### 4.2 混合精度训练

使用FP16加速训练：

```python
# 修改 utils/trainer.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_idx, (input_ids, target_ids) in enumerate(pbar):
    with autocast():
        logits, loss = self.model(input_ids, target_ids)
    
    self.optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
    scaler.step(self.optimizer)
    scaler.update()
```

**优势：**
- ↑ 训练速度（1.5-3倍）
- ↓ 显存占用（约50%）

### 4.3 学习率查找

找到最佳学习率：

```python
# 创建学习率查找脚本
import torch
import matplotlib.pyplot as plt

def find_lr(model, train_loader, init_lr=1e-8, final_lr=10, beta=0.98):
    num_batches = len(train_loader)
    mult = (final_lr / init_lr) ** (1 / num_batches)
    lr = init_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    avg_loss = 0.
    best_loss = 0.
    losses = []
    lrs = []
    
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        # 训练一步
        optimizer.zero_grad()
        logits, loss = model(input_ids, target_ids)
        
        # 平滑损失
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_idx + 1))
        
        # 记录
        losses.append(smoothed_loss)
        lrs.append(lr)
        
        # 如果损失爆炸，停止
        if batch_idx > 0 and smoothed_loss > 4 * best_loss:
            break
        
        if smoothed_loss < best_loss or batch_idx == 0:
            best_loss = smoothed_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # 绘图
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.savefig('lr_finder.png')
    
    return lrs, losses
```

### 4.4 数据增强

对中文文本进行增强：

```python
# 同义词替换
def synonym_replacement(text, n=3):
    # 使用同义词词典替换n个词
    pass

# 随机插入
def random_insertion(text, n=3):
    # 随机插入n个词
    pass

# 随机删除
def random_deletion(text, p=0.1):
    # 以概率p删除每个词
    pass

# 随机交换
def random_swap(text, n=3):
    # 随机交换n对词
    pass
```

### 4.5 继续训练

从检查点继续训练：

```python
# 修改 train.py
def main():
    # ... 创建模型和训练器 ...
    
    # 加载检查点
    checkpoint_path = "checkpoints/checkpoint_step_10000.pt"
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
        print(f"从检查点继续训练: {checkpoint_path}")
    
    # 继续训练
    trainer.train()
```

---

## 5. 监控与调试

### 5.1 TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=logs --port=6006
```

**监控指标：**

1. **训练损失 (train/loss)**
   - 应该持续下降
   - 如果波动剧烈 → 降低学习率

2. **验证损失 (val/loss)**
   - 应该下降
   - 如果上升 → 过拟合，增加正则化

3. **困惑度 (perplexity)**
   - PPL = exp(loss)
   - 越低越好
   - 好的模型：PPL < 50

4. **学习率 (train/lr)**
   - 查看学习率调度是否正常
   - 预热阶段应该线性上升
   - 训练阶段应该余弦下降

### 5.2 训练曲线分析

#### 正常训练
```
Loss
  ^
  |  \
  |   \___
  |       ----___
  |              ----___
  +----------------------> Steps
```

#### 学习率过大
```
Loss
  ^
  |  \/\/\/\
  |  /\/\/\/
  |  \/\/\/\
  +----------------------> Steps
```
**解决：** 降低学习率

#### 学习率过小
```
Loss
  ^
  |  ___
  |     ___
  |        ___
  |           ___
  +----------------------> Steps
```
**解决：** 提高学习率

#### 过拟合
```
Loss
  ^
  |  train ----___________
  |  val   ----___/\/\/\/\
  |                  ↑
  |              开始过拟合
  +----------------------> Steps
```
**解决：**
- 增加dropout
- 增加weight_decay
- 使用更多数据
- 早停

### 5.3 梯度监控

添加梯度监控代码：

```python
# 在 utils/trainer.py 中添加
def log_gradients(self):
    """记录梯度统计"""
    total_norm = 0
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # 记录每层梯度
            self.writer.add_scalar(f'gradients/{name}', param_norm, self.global_step)
    
    total_norm = total_norm ** 0.5
    self.writer.add_scalar('gradients/total_norm', total_norm, self.global_step)
```

**正常梯度范围：** 0.1 - 10

**异常情况：**
- 梯度 > 100 → 梯度爆炸，降低学习率或减小grad_clip
- 梯度 < 0.001 → 梯度消失，检查模型初始化

### 5.4 模型输出采样

定期查看模型生成的文本：

```python
# 在训练循环中添加
if self.global_step % 1000 == 0:
    self.sample_generation()

def sample_generation(self):
    """生成样本文本"""
    self.model.eval()
    
    prompts = ["今天天气", "人工智能", "中国"]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        generated = self.model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )
        
        text = tokenizer.decode(generated[0].tolist())
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text}")
```

---

## 6. 常见问题排查

### 6.1 显存不足 (CUDA Out of Memory)

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**解决方案（按优先级）：**

1. **降低批次大小**
```python
batch_size = 16  # 从32降到16
```

2. **减小序列长度**
```python
max_seq_len = 128  # 从256降到128
```

3. **减小模型规模**
```python
d_model = 128      # 从256降到128
n_layers = 4       # 从6降到4
```

4. **使用梯度累积**
```python
# 见 4.1 节
```

5. **使用混合精度训练**
```python
# 见 4.2 节
```

6. **清理显存**
```python
import torch
torch.cuda.empty_cache()
```

### 6.2 训练损失不下降

**可能原因：**

1. **学习率过大**
   - 症状：损失波动剧烈或NaN
   - 解决：降低学习率（除以10）

2. **学习率过小**
   - 症状：损失下降极慢
   - 解决：提高学习率（乘以2-5）

3. **数据问题**
   - 症状：损失在某个值停滞
   - 解决：检查数据质量和预处理

4. **模型初始化问题**
   - 症状：损失从一开始就很高
   - 解决：检查权重初始化

5. **梯度消失/爆炸**
   - 症状：梯度过小(<1e-6)或过大(>100)
   - 解决：调整学习率、使用梯度裁剪

### 6.3 验证损失上升（过拟合）

**解决方案：**

1. **增加Dropout**
```python
dropout = 0.2  # 从0.1增加到0.2
```

2. **增加权重衰减**
```python
weight_decay = 0.1  # 从0.01增加到0.1
```

3. **使用更多数据**
```python
# 增加训练数据量
```

4. **减小模型规模**
```python
n_layers = 4  # 从6减到4
```

5. **早停**
```python
# 在 utils/trainer.py 中添加
if val_loss > self.best_val_loss:
    self.patience_counter += 1
    if self.patience_counter >= self.patience:
        print("Early stopping!")
        break
else:
    self.patience_counter = 0
```

### 6.4 生成质量差

**可能原因和解决方案：**

1. **训练不足**
   - 解决：延长训练时间

2. **数据质量差**
   - 解决：清洗数据，使用高质量语料

3. **采样参数不当**
   - 解决：调整temperature, top_k, top_p

4. **模型规模太小**
   - 解决：增加模型参数量

**生成参数调优：**
```python
# 更确定性的输出
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,    # 降低温度
    top_k=40,           # 降低top_k
    top_p=0.85          # 降低top_p
)

# 更多样性的输出
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=1.2,    # 提高温度
    top_k=100,          # 提高top_k
    top_p=0.95          # 提高top_p
)
```

### 6.5 训练速度慢

**优化方案：**

1. **使用GPU**
```python
# 检查GPU是否被使用
print(f"使用设备: {config.device}")
```

2. **增加批次大小**
```python
batch_size = 64  # 如果显存允许
```

3. **使用混合精度训练**
```python
# 见 4.2 节
```

4. **减少数据加载时间**
```python
# 在 create_dataloader 中
num_workers = 4  # 增加工作进程
pin_memory = True  # 使用固定内存
```

5. **使用更快的优化器**
```python
# 使用AdamW的融合版本
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    fused=True  # 需要CUDA 11.3+
)
```

### 6.6 损失变成NaN

**可能原因：**

1. **学习率过大**
   - 解决：降低学习率

2. **梯度爆炸**
   - 解决：降低grad_clip阈值

3. **数值不稳定**
   - 解决：使用混合精度训练

4. **数据问题**
   - 解决：检查是否有异常值

**调试代码：**
```python
# 添加NaN检测
if torch.isnan(loss):
    print("检测到NaN损失！")
    print(f"输入: {input_ids}")
    print(f"目标: {target_ids}")
    print(f"Logits: {logits}")
    raise ValueError("Loss is NaN")
```

---

## 7. 最佳实践

### 7.1 推荐配置

#### 个人电脑（无GPU或低端GPU）
```python
# config.py
vocab_size = 5000
max_seq_len = 128
d_model = 128
n_layers = 4
n_heads = 4
d_ff = 512
batch_size = 16
learning_rate = 3e-4
max_epochs = 5
```

**预期：**
- 参数量：~2M
- 训练时间：2-4小时（CPU）
- 显存占用：~2GB

#### 单GPU（RTX 3060/4060）
```python
# config.py
vocab_size = 10000
max_seq_len = 256
d_model = 256
n_layers = 6
n_heads = 8
d_ff = 1024
batch_size = 32
learning_rate = 3e-4
max_epochs = 10
```

**预期：**
- 参数量：~7M
- 训练时间：3-6小时
- 显存占用：~6GB

#### 高端GPU（RTX 3090/4090）
```python
# config.py
vocab_size = 20000
max_seq_len = 512
d_model = 512
n_layers = 12
n_heads = 8
d_ff = 2048
batch_size = 64
learning_rate = 2e-4
max_epochs = 20
```

**预期：**
- 参数量：~50M
- 训练时间：8-12小时
- 显存占用：~16GB

### 7.2 训练流程建议

#### 阶段1：快速验证（1-2小时）
```python
# 使用小配置快速验证代码和数据
vocab_size = 5000
d_model = 128
n_layers = 4
batch_size = 16
max_epochs = 2
```

**目标：** 确保代码运行正常，数据加载正确

#### 阶段2：中等规模训练（4-8小时）
```python
# 使用推荐配置训练
vocab_size = 10000
d_model = 256
n_layers = 6
batch_size = 32
max_epochs = 10
```

**目标：** 获得基本可用的模型

#### 阶段3：参数调优（按需）
- 调整学习率
- 调整正则化参数
- 尝试不同的采样策略

#### 阶段4：大规模训练（可选）
```python
# 如果有资源，训练更大的模型
d_model = 512
n_layers = 12
max_epochs = 20
```

### 7.3 检查清单

**训练前：**
- [ ] 数据已准备并验证
- [ ] 配置参数已检查
- [ ] GPU可用性已确认
- [ ] 磁盘空间充足（至少10GB）
- [ ] 已创建checkpoints和logs目录

**训练中：**
- [ ] 定期查看TensorBoard
- [ ] 监控损失曲线
- [ ] 检查生成样本质量
- [ ] 验证检查点保存正常

**训练后：**
- [ ] 保存最佳模型
- [ ] 记录最终指标
- [ ] 测试推理性能
- [ ] 备份重要检查点

### 7.4 实验记录模板

创建 `experiments.md` 记录实验：

```markdown
# 实验记录

## 实验1: 基线模型
- **日期**: 2026-02-09
- **配置**: 
  - d_model=256, n_layers=6, batch_size=32
  - lr=3e-4, max_epochs=10
- **数据**: 维基百科中文（50MB）
- **结果**:
  - 训练损失: 3.45
  - 验证损失: 3.78
  - 困惑度: 43.7
  - 训练时间: 4.5小时
- **观察**: 生成质量一般，有时重复
- **下一步**: 增加dropout防止过拟合

## 实验2: 增加正则化
- **日期**: 2026-02-10
- **配置**: 
  - 基于实验1
  - dropout=0.2 (从0.1增加)
  - weight_decay=0.05 (从0.01增加)
- **结果**:
  - 训练损失: 3.62
  - 验证损失: 3.65
  - 困惑度: 38.5
  - 训练时间: 4.8小时
- **观察**: 过拟合减少，生成质量提升
- **下一步**: 尝试更大的模型
```

---

## 📚 参考资料

### 相关文档
- [快速开始](QUICK_START.md)
- [模型架构](TRANSFORMER_ARCHITECTURE.md)
- [参数指南](MODEL_PARAMETERS_GUIDE.md)
- [项目结构](PROJECT_STRUCTURE.md)

### 推荐阅读
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2论文
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy的GPT实现

### 工具
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 训练可视化
- [Weights & Biases](https://wandb.ai/) - 实验跟踪
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - 性能分析

---

## 🤝 获取帮助

如果遇到问题：

1. **查看文档** - 先查阅本手册和其他文档
2. **检查日志** - 查看TensorBoard和终端输出
3. **搜索问题** - 在GitHub Issues中搜索
4. **提问** - 在GitHub Issues中提问，提供详细信息：
   - 配置文件
   - 错误信息
   - 训练日志
   - 系统信息

---

<div align="center">

**NanoGPT-ZH 训练及调参操作手册**

© 2026 | Made with ❤️ by xiaomin.zhang

[返回首页](../README.md) | [快速开始](QUICK_START.md) | [GitHub](https://github.com/King2021521/nanogpt-zh)

</div>
