"""
Transformer Block实现
@Author xiaomin.zhang
"""

import torch
import torch.nn as nn
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    前馈神经网络（Position-wise Feed-Forward Network）
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度（通常是d_model的4倍）
            dropout: Dropout概率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GPT使用GELU激活函数
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer解码器块（用于GPT）
    包含：
    1. 因果自注意力层
    2. Layer Normalization
    3. 前馈网络
    4. 残差连接
    """
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            max_seq_len: 最大序列长度
            dropout: Dropout概率
        """
        super().__init__()
        
        # 因果自注意力
        self.attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization（使用Pre-LN，即在子层之前进行归一化）
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        使用Pre-LN结构：LN -> Attention -> Residual
        
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # 自注意力子层（带残差连接）
        # Pre-LN: x + Attention(LN(x))
        attn_output = self.attention(self.ln1(x))
        x = x + self.dropout(attn_output)
        
        # 前馈网络子层（带残差连接）
        # Pre-LN: x + FFN(LN(x))
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x


if __name__ == "__main__":
    # 测试代码
    print("测试FeedForward...")
    batch_size = 2
    seq_len = 10
    d_model = 256
    d_ff = 1024
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入shape: {x.shape}")
    
    # 测试前馈网络
    ffn = FeedForward(d_model, d_ff)
    output = ffn(x)
    print(f"FeedForward输出shape: {output.shape}")
    
    # 测试Transformer Block
    print("\n测试TransformerBlock...")
    n_heads = 8
    max_seq_len = 512
    
    block = TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
    output = block(x)
    print(f"TransformerBlock输出shape: {output.shape}")
    
    # 验证残差连接
    print(f"\n输入和输出shape是否一致: {x.shape == output.shape}")
    
    print("\n✓ Transformer Block测试通过！")
