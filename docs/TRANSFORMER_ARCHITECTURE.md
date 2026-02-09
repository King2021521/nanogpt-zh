# NanoGPT-ZH Transformer架构与实现原理详解

## 📚 文档说明

本文档详细介绍了 **NanoGPT-ZH** 项目中**完全原生实现的Transformer架构**，包括核心原理、代码实现和技术细节。

**NanoGPT-ZH 实现特点**：
- ✅ 100%手工实现，零外部Transformer依赖
- ✅ 基于论文"Attention Is All You Need"
- ✅ GPT风格的Decoder-Only架构
- ✅ 包含详细注释和公式推导
- ✅ 7.3M参数，适合个人电脑训练

---

## 🏗️ 整体架构概览

### 模型类型

**Decoder-Only Transformer（GPT风格）**

```
输入文本 (Token IDs)
    ↓
Token Embedding (词嵌入)
    ↓
Positional Encoding (位置编码)
    ↓
┌─────────────────────────────┐
│  Transformer Block 1        │
│  ├─ Causal Self-Attention   │
│  ├─ Layer Normalization     │
│  ├─ Feed Forward Network    │
│  └─ Residual Connections    │
├─────────────────────────────┤
│  Transformer Block 2        │
│  └─ ...                     │
├─────────────────────────────┤
│  ...                        │
├─────────────────────────────┤
│  Transformer Block N        │
│  └─ ...                     │
└─────────────────────────────┘
    ↓
Final Layer Normalization
    ↓
Language Model Head (输出层)
    ↓
Logits (词表概率分布)
```

### 架构特点

| 特性 | 说明 |
|------|------|
| **架构类型** | Decoder-Only（纯解码器） |
| **注意力类型** | Causal Self-Attention（因果自注意力） |
| **归一化** | Pre-Layer Normalization |
| **激活函数** | GELU |
| **位置编码** | Sinusoidal（正弦余弦） |
| **权重共享** | 嵌入层与输出层共享 |

---

## 🧩 核心组件详解

### 1. Token嵌入层（Token Embedding）

#### 原理

将离散的token ID转换为连续的向量表示，这是模型理解文本的第一步。

#### 数学公式

```
E(x) = Embedding(x) × √d_model
```

其中：
- `x`: token ID
- `Embedding`: 嵌入矩阵 (vocab_size × d_model)
- `√d_model`: 缩放因子（论文中的做法）

#### 代码实现

```python
# models/embedding.py
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        # 返回: (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
```

#### 参数量

```python
参数量 = vocab_size × d_model
当前配置 = 10,000 × 256 = 2,560,000 参数
```

#### 作用

- 将每个词映射为高维向量
- 相似的词在向量空间中距离更近
- 可学习的参数，随训练优化

---

### 2. 位置编码（Positional Encoding）

#### 原理

Transformer没有循环结构，无法感知位置信息。位置编码为每个位置添加唯一的标识。

#### 数学公式

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中：
- `pos`: 位置索引 (0, 1, 2, ...)
- `i`: 维度索引 (0, 1, 2, ..., d_model/2)
- 偶数维度使用sin，奇数维度使用cos

#### 代码实现

```python
# models/embedding.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 应用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        # 添加batch维度并注册为buffer（不参与训练）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

#### 特点

- ✅ **固定编码**：不参与训练，预先计算
- ✅ **周期性**：不同频率的sin/cos组合
- ✅ **相对位置**：能表达相对位置关系
- ✅ **外推性**：理论上可处理任意长度

#### 可视化

```
位置0: [sin(0/1), cos(0/1), sin(0/100), cos(0/100), ...]
位置1: [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
位置2: [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
...
```

---

### 3. 多头自注意力机制（Multi-Head Self-Attention）

#### 原理

注意力机制允许模型在处理每个位置时，关注序列中的所有位置，学习词与词之间的关系。

#### 数学公式

**单头注意力**：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**多头注意力**：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

其中 head_i = Attention(Q×W_Q^i, K×W_K^i, V×W_V^i)
```

参数：
- `Q` (Query): 查询矩阵
- `K` (Key): 键矩阵
- `V` (Value): 值矩阵
- `d_k = d_model / n_heads`: 每个头的维度
- `W_Q, W_K, W_V, W_O`: 投影矩阵

#### 代码实现

```python
# models/attention.py
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q, K, V的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """将d_model分割成(n_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        # (batch_size, seq_len, d_model) 
        # -> (batch_size, seq_len, n_heads, d_k)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def merge_heads(self, x):
        """合并多个头"""
        batch_size, n_heads, seq_len, d_k = x.size()
        # (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, seq_len, n_heads, d_k)
        x = x.transpose(1, 2)
        # -> (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        
        Args:
            Q: (batch_size, n_heads, seq_len, d_k)
            K: (batch_size, n_heads, seq_len, d_k)
            V: (batch_size, n_heads, seq_len, d_k)
            mask: (batch_size, 1, seq_len, seq_len)
        
        Returns:
            output: (batch_size, n_heads, seq_len, d_k)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        # 1. 计算注意力分数: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2. 应用mask（用于因果注意力）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 3. Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 4. 应用注意力权重到Value
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 1. 线性变换得到Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. 分割成多个头
        Q = self.split_heads(Q)  # (batch_size, n_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. 计算注意力
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 合并多个头
        attn_output = self.merge_heads(attn_output)  # (batch_size, seq_len, d_model)
        
        # 5. 最后的线性变换
        output = self.W_o(attn_output)
        
        return output
```

#### 因果自注意力（Causal Self-Attention）

GPT使用因果注意力，确保每个位置只能看到它之前的位置。

```python
# models/attention.py
class CausalSelfAttention(MultiHeadAttention):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__(d_model, n_heads, dropout)
        
        # 创建因果mask（下三角矩阵）
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', 
                           causal_mask.view(1, 1, max_seq_len, max_seq_len))
    
    def forward(self, x):
        seq_len = x.size(1)
        # 自动应用因果mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        return super().forward(x, mask)
```

#### 因果Mask示例

```
序列: [A, B, C, D]

注意力矩阵（允许关注的位置，1=允许，0=禁止）:
     A  B  C  D
A [  1  0  0  0 ]  # A只能看自己
B [  1  1  0  0 ]  # B能看A和自己
C [  1  1  1  0 ]  # C能看A、B和自己
D [  1  1  1  1 ]  # D能看所有之前的

这确保了自回归生成：预测下一个词时不能"偷看"未来
```

#### 参数量

```python
# 每个注意力层
W_q: d_model × d_model = 256 × 256 = 65,536
W_k: d_model × d_model = 256 × 256 = 65,536
W_v: d_model × d_model = 256 × 256 = 65,536
W_o: d_model × d_model = 256 × 256 = 65,536

总计 = 262,144 参数
```

#### 计算复杂度

```
时间复杂度: O(seq_len² × d_model)
空间复杂度: O(seq_len²)  # 注意力矩阵

这是Transformer的主要瓶颈，限制了序列长度
```

---

### 4. 前馈神经网络（Feed-Forward Network）

#### 原理

在每个Transformer Block中，注意力层之后是一个位置独立的前馈网络，对每个位置进行相同的非线性变换。

#### 数学公式

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

其中：
- `W₁`: d_model → d_ff（扩展）
- `W₂`: d_ff → d_model（压缩）
- `GELU`: 高斯误差线性单元激活函数
- 通常 `d_ff = 4 × d_model`

#### 代码实现

```python
# models/transformer.py
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GPT使用GELU
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        
        # 扩展到d_ff维
        x = self.linear1(x)  # -> (batch_size, seq_len, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 压缩回d_model维
        x = self.linear2(x)  # -> (batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        return x
```

#### GELU激活函数

```python
# GELU比ReLU更平滑
GELU(x) = x × Φ(x)
其中 Φ(x) 是标准正态分布的累积分布函数

近似公式:
GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

#### 参数量

```python
W₁: d_model × d_ff = 256 × 1024 = 262,144
b₁: d_ff = 1,024
W₂: d_ff × d_model = 1024 × 256 = 262,144
b₂: d_model = 256

总计 = 525,568 参数（单层中最多）
```

#### 作用

- 增加模型的非线性表达能力
- 每个位置独立处理（位置无关）
- 提供额外的参数容量

---

### 5. Transformer Block（完整层）

#### 结构

每个Transformer Block包含两个子层：
1. **自注意力子层**
2. **前馈网络子层**

每个子层都有：
- Layer Normalization（Pre-LN）
- Residual Connection（残差连接）
- Dropout

#### 代码实现

```python
# models/transformer.py
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        # 因果自注意力
        self.attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        使用Pre-LN结构
        
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        # 子层1: 自注意力 + 残差连接
        # Pre-LN: x + Sublayer(LayerNorm(x))
        attn_output = self.attention(self.ln1(x))
        x = x + self.dropout(attn_output)
        
        # 子层2: 前馈网络 + 残差连接
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x
```

#### Pre-LN vs Post-LN

**Pre-LN（本项目使用）**：
```
x → LN → Attention → Dropout → (+) → LN → FFN → Dropout → (+)
↓                              ↑    ↓                      ↑
└──────────────────────────────┘    └──────────────────────┘
```

**Post-LN（原始论文）**：
```
x → Attention → Dropout → (+) → LN → FFN → Dropout → (+) → LN
↓                         ↑         ↓                ↑
└─────────────────────────┘         └────────────────┘
```

**Pre-LN优势**：
- ✅ 训练更稳定
- ✅ 不需要学习率预热
- ✅ 更容易训练深层网络

#### 残差连接（Residual Connection）

```python
# 残差连接允许梯度直接流过
output = x + Sublayer(x)

# 缓解梯度消失问题
# 即使Sublayer学习到0，至少有恒等映射
```

#### 单层参数量

```python
注意力层: 262,144
前馈网络: 525,568
LayerNorm: 1,024
━━━━━━━━━━━━━━━━
总计: 788,736 参数/层
```

---

### 6. 完整GPT模型

#### 架构

```python
# models/gpt.py
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Token嵌入层
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        
        # 2. 位置编码
        self.pos_encoding = PositionalEncoding(
            config.d_model, 
            config.max_seq_len, 
            config.dropout
        )
        
        # 3. Transformer解码器层堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # 4. 最后的Layer Normalization
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # 5. 输出层（语言模型头）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 6. 权重共享：嵌入层和输出层共享权重
        self.lm_head.weight = self.token_embedding.embedding.weight
        
        # 7. 初始化参数
        self.apply(self._init_weights)
```

#### 前向传播

```python
def forward(self, input_ids, targets=None):
    """
    Args:
        input_ids: (batch_size, seq_len) token IDs
        targets: (batch_size, seq_len) 目标token IDs
    
    Returns:
        logits: (batch_size, seq_len, vocab_size)
        loss: 交叉熵损失（如果提供targets）
    """
    # 1. Token嵌入
    x = self.token_embedding(input_ids)  
    # -> (batch_size, seq_len, d_model)
    
    # 2. 添加位置编码
    x = self.pos_encoding(x)
    
    # 3. 通过所有Transformer层
    for block in self.transformer_blocks:
        x = block(x)
    
    # 4. 最后的Layer Normalization
    x = self.ln_f(x)
    
    # 5. 输出层：映射到词表
    logits = self.lm_head(x)  
    # -> (batch_size, seq_len, vocab_size)
    
    # 6. 计算损失（如果提供targets）
    loss = None
    if targets is not None:
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
    
    return logits, loss
```

#### 文本生成

```python
@torch.no_grad()
def generate(self, input_ids, max_new_tokens, temperature=1.0, 
             top_k=None, top_p=None):
    """
    自回归生成文本
    
    Args:
        input_ids: (batch_size, seq_len) 初始token IDs
        max_new_tokens: 最大生成token数
        temperature: 采样温度（越高越随机）
        top_k: Top-K采样
        top_p: Top-P (nucleus) 采样
    
    Returns:
        generated: (batch_size, seq_len + max_new_tokens)
    """
    self.eval()
    
    for _ in range(max_new_tokens):
        # 1. 截断到max_seq_len
        input_ids_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                        else input_ids[:, -self.config.max_seq_len:]
        
        # 2. 前向传播
        logits, _ = self(input_ids_cond)
        
        # 3. 只取最后一个位置的logits
        logits = logits[:, -1, :]  # (batch_size, vocab_size)
        
        # 4. 应用温度
        logits = logits / temperature
        
        # 5. Top-K采样
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # 6. Top-P采样
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # 7. 采样下一个token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 8. 拼接到序列
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids
```

#### 权重初始化

```python
def _init_weights(self, module):
    """
    使用正态分布初始化权重
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)
```

---

## 🔬 关键技术细节

### 1. 权重共享（Weight Tying）

```python
# 嵌入层和输出层共享权重
self.lm_head.weight = self.token_embedding.embedding.weight
```

**优势**：
- ✅ 减少参数量（节省2.56M参数）
- ✅ 提升性能（输入和输出使用相同的词表示）
- ✅ 正则化效果（减少过拟合）

**原理**：
- 输入嵌入：token ID → 向量
- 输出投影：向量 → token概率
- 两者本质上都是词的表示，共享合理

### 2. 缩放因子（Scaling Factor）

#### Token嵌入缩放

```python
return self.embedding(x) * math.sqrt(self.d_model)
```

**原因**：
- 位置编码的值在[-1, 1]范围
- 嵌入向量的值较小
- 缩放使两者量级相当，便于相加

#### 注意力分数缩放

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
```

**原因**：
- QK^T的方差与d_k成正比
- 除以√d_k使方差归一化
- 防止softmax进入饱和区

### 3. Dropout策略

```python
# 多处使用Dropout
self.dropout = nn.Dropout(dropout)

# 1. 位置编码后
x = self.dropout(x)

# 2. 注意力权重
attention_weights = self.dropout(attention_weights)

# 3. 前馈网络中间
x = self.dropout(x)

# 4. 残差连接前
x = x + self.dropout(sublayer_output)
```

**作用**：
- 防止过拟合
- 提高泛化能力
- 训练时随机丢弃，测试时不丢弃

### 4. Layer Normalization

```python
self.ln = nn.LayerNorm(d_model)
```

**公式**：
```
LN(x) = γ × (x - μ) / √(σ² + ε) + β
```

其中：
- `μ`: 均值
- `σ²`: 方差
- `γ, β`: 可学习参数
- `ε`: 数值稳定性常数

**作用**：
- 稳定训练
- 加速收敛
- 减少内部协变量偏移

### 5. 因果Mask

```python
# 创建下三角矩阵
causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

# 应用mask
scores = scores.masked_fill(mask == 0, float('-inf'))
```

**效果**：
```
原始注意力分数:
[[0.5, 0.3, 0.2],
 [0.4, 0.4, 0.2],
 [0.3, 0.3, 0.4]]

应用mask后:
[[0.5, -inf, -inf],
 [0.4, 0.4, -inf],
 [0.3, 0.3, 0.4]]

Softmax后:
[[1.0, 0.0, 0.0],
 [0.5, 0.5, 0.0],
 [0.33, 0.33, 0.34]]
```

---

## 📐 数学原理深入

### 注意力机制的直觉理解

**Query-Key-Value类比**：

想象一个图书馆检索系统：
- **Query（查询）**：你想找什么书
- **Key（键）**：书的标签/索引
- **Value（值）**：书的实际内容

**过程**：
1. 用Query和所有Key计算相似度（QK^T）
2. 相似度归一化（softmax）得到注意力权重
3. 用权重加权求和所有Value

**在NLP中**：
- Query：当前词想要什么信息
- Key：其他词能提供什么信息
- Value：其他词的实际表示

### 多头注意力的作用

**单头注意力**：
- 只能学习一种关系模式

**多头注意力**：
- 每个头学习不同的关系
- 例如：
  - 头1：语法关系（主谓宾）
  - 头2：语义关系（同义词）
  - 头3：位置关系（相邻词）
  - ...

**数学上**：
```
head_i = Attention(xW_Q^i, xW_K^i, xW_V^i)

不同的W矩阵 → 不同的子空间 → 不同的关系模式
```

### 残差连接的重要性

**梯度流动**：
```
没有残差：
y = F(x)
∂L/∂x = ∂L/∂y × ∂F/∂x  # 梯度可能消失

有残差：
y = x + F(x)
∂L/∂x = ∂L/∂y × (1 + ∂F/∂x)  # 至少有1，梯度不会消失
```

**效果**：
- 允许训练更深的网络
- 加速收敛
- 提供"快捷通道"

---

## 🎯 与原始Transformer的区别

### 架构差异

| 特性 | 原始Transformer | 本项目（GPT风格） |
|------|----------------|------------------|
| **结构** | Encoder-Decoder | Decoder-Only |
| **注意力** | 双向 + 交叉注意力 | 因果（单向） |
| **归一化** | Post-LN | Pre-LN |
| **应用** | 翻译等seq2seq | 文本生成 |

### 为什么选择Decoder-Only？

**优势**：
1. ✅ **架构简单**：只需要解码器
2. ✅ **训练高效**：统一的自回归目标
3. ✅ **扩展性好**：GPT-3证明了有效性
4. ✅ **通用性强**：可以处理各种任务

**GPT系列的成功**：
- GPT-1: 117M参数
- GPT-2: 1.5B参数
- GPT-3: 175B参数
- GPT-4: 估计1.76T参数

---

## 💻 代码组织结构

```
models/
├── __init__.py           # 模块导出
├── embedding.py          # Token嵌入和位置编码
│   ├── TokenEmbedding
│   └── PositionalEncoding
├── attention.py          # 注意力机制
│   ├── MultiHeadAttention
│   └── CausalSelfAttention
├── transformer.py        # Transformer Block
│   ├── FeedForward
│   └── TransformerBlock
└── gpt.py               # 完整GPT模型
    └── GPTModel
```

### 模块依赖关系

```
GPTModel
  ├── TokenEmbedding (embedding.py)
  ├── PositionalEncoding (embedding.py)
  ├── TransformerBlock × N (transformer.py)
  │   ├── CausalSelfAttention (attention.py)
  │   │   └── MultiHeadAttention (attention.py)
  │   └── FeedForward (transformer.py)
  └── nn.Linear (lm_head)
```

---

## 🔍 调试和可视化

### 查看模型结构

```python
from models import GPTModel
from config import config

model = GPTModel(config)
print(model)
```

### 查看参数量

```python
# 总参数
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 各部分参数
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {params:,}")
```

### 查看注意力权重

```python
# 修改attention.py的forward方法，返回attention_weights
def forward(self, x, mask=None):
    # ... 计算过程 ...
    attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
    # ...
    return output, attention_weights  # 返回权重

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# attention_weights: (batch_size, n_heads, seq_len, seq_len)
weights = attention_weights[0, 0].cpu().numpy()  # 第一个样本，第一个头

plt.figure(figsize=(10, 8))
sns.heatmap(weights, cmap='viridis')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Attention Weights')
plt.show()
```

---

## 📚 参考资料

### 核心论文

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - 原始Transformer论文
   - https://arxiv.org/abs/1706.03762

2. **Improving Language Understanding by Generative Pre-Training** (Radford et al., 2018)
   - GPT-1论文
   - https://openai.com/research/language-unsupervised

3. **Language Models are Few-Shot Learners** (Brown et al., 2020)
   - GPT-3论文
   - https://arxiv.org/abs/2005.14165

### 教程资源

1. **The Annotated Transformer**
   - Harvard NLP详细注释版
   - http://nlp.seas.harvard.edu/annotated-transformer/

2. **nanoGPT**
   - Andrej Karpathy的最小GPT实现
   - https://github.com/karpathy/nanoGPT

3. **The Illustrated Transformer**
   - Jay Alammar的可视化教程
   - http://jalammar.github.io/illustrated-transformer/

---

## ✅ 总结

### 本项目的特点

1. **完全原生实现** ⭐⭐⭐⭐⭐
   - 零外部Transformer依赖
   - 每一行代码都清晰可见
   - 适合学习和研究

2. **符合论文原理** ⭐⭐⭐⭐⭐
   - 严格遵循"Attention Is All You Need"
   - GPT风格的Decoder-Only架构
   - 包含现代改进（Pre-LN）

3. **代码质量高** ⭐⭐⭐⭐⭐
   - 详细注释
   - 清晰的模块划分
   - 良好的工程实践

4. **教育价值高** ⭐⭐⭐⭐⭐
   - 适合深度学习入门
   - 理解Transformer原理
   - 学习PyTorch实践

### 核心要点回顾

1. **Transformer = 注意力机制 + 前馈网络 + 残差连接 + 归一化**

2. **多头注意力允许模型学习不同类型的关系**

3. **因果注意力确保自回归生成（不能看未来）**

4. **Pre-LN比Post-LN训练更稳定**

5. **权重共享减少参数量并提升性能**

6. **位置编码提供序列的位置信息**

### 学习建议

1. **从小到大**：先理解单个组件，再理解整体
2. **动手实践**：修改代码，观察效果
3. **可视化**：画出注意力权重，理解机制
4. **对比学习**：与其他实现对比，理解差异

---

**@Author xiaomin.zhang**
**文档版本**: 1.0
**创建日期**: 2026-02-06

**致谢**：感谢"Attention Is All You Need"论文作者和开源社区的贡献！
