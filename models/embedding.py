"""
嵌入层实现 - Token Embedding 和 Positional Encoding
@Author xiaomin.zhang
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token嵌入层
    将token ID转换为稠密向量表示
    """
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: 词表大小
            d_model: 嵌入维度
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) token IDs
        Returns:
            (batch_size, seq_len, d_model) 嵌入向量
        """
        # 乘以sqrt(d_model)是Transformer论文中的做法，用于缩放
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码
    使用正弦和余弦函数为序列中的每个位置生成唯一的编码
    """
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_seq_len: 最大序列长度
            dropout: Dropout概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度 (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) 输入嵌入
        Returns:
            (batch_size, seq_len, d_model) 添加位置编码后的嵌入
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # 测试代码
    print("测试TokenEmbedding...")
    vocab_size = 10000
    d_model = 256
    batch_size = 2
    seq_len = 10
    
    # 创建随机token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入shape: {token_ids.shape}")
    
    # Token嵌入
    token_emb = TokenEmbedding(vocab_size, d_model)
    embedded = token_emb(token_ids)
    print(f"Token嵌入输出shape: {embedded.shape}")
    
    # 位置编码
    print("\n测试PositionalEncoding...")
    pos_enc = PositionalEncoding(d_model, max_seq_len=100)
    output = pos_enc(embedded)
    print(f"位置编码输出shape: {output.shape}")
    
    print("\n✓ 嵌入层测试通过！")
