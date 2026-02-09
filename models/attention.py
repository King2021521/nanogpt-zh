"""
多头注意力机制实现
@Author xiaomin.zhang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    实现论文 "Attention is All You Need" 中的Multi-Head Attention
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q, K, V的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出的线性变换层
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """
        将最后一维分割成(n_heads, d_k)
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # 重塑为 (batch_size, seq_len, n_heads, d_k)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # 转置为 (batch_size, n_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def merge_heads(self, x):
        """
        合并多个头
        Args:
            x: (batch_size, n_heads, seq_len, d_k)
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        # 转置为 (batch_size, seq_len, n_heads, d_k)
        x = x.transpose(1, 2)
        # 重塑为 (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        
        Args:
            Q: (batch_size, n_heads, seq_len, d_k) Query
            K: (batch_size, n_heads, seq_len, d_k) Key
            V: (batch_size, n_heads, seq_len, d_k) Value
            mask: (batch_size, 1, seq_len, seq_len) 或 (batch_size, 1, 1, seq_len)
        Returns:
            output: (batch_size, n_heads, seq_len, d_k)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        # 计算注意力分数: QK^T / sqrt(d_k)
        # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（用于padding或causal attention）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到Value
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: (batch_size, seq_len, d_model) 输入
            mask: (batch_size, 1, seq_len, seq_len) 注意力mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)
        
        # 线性变换得到Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分割成多个头
        Q = self.split_heads(Q)  # (batch_size, n_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 计算注意力
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多个头
        attn_output = self.merge_heads(attn_output)  # (batch_size, seq_len, d_model)
        
        # 最后的线性变换
        output = self.W_o(attn_output)
        
        return output


class CausalSelfAttention(MultiHeadAttention):
    """
    因果自注意力（用于GPT）
    确保每个位置只能关注它之前的位置（包括自己）
    """
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            max_seq_len: 最大序列长度
            dropout: Dropout概率
        """
        super().__init__(d_model, n_heads, dropout)
        
        # 创建因果mask（下三角矩阵）
        # 注册为buffer，不参与梯度更新
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len))
    
    def forward(self, x):
        """
        前向传播（自动应用因果mask）
        Args:
            x: (batch_size, seq_len, d_model) 输入
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 使用预计算的因果mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        return super().forward(x, mask)


if __name__ == "__main__":
    # 测试代码
    print("测试MultiHeadAttention...")
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入shape: {x.shape}")
    
    # 测试多头注意力
    mha = MultiHeadAttention(d_model, n_heads)
    output = mha(x)
    print(f"MultiHeadAttention输出shape: {output.shape}")
    
    # 测试因果注意力
    print("\n测试CausalSelfAttention...")
    causal_attn = CausalSelfAttention(d_model, n_heads, max_seq_len=512)
    output = causal_attn(x)
    print(f"CausalSelfAttention输出shape: {output.shape}")
    
    print("\n✓ 注意力机制测试通过！")
