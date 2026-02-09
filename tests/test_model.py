"""
模型测试脚本 - 快速验证所有组件
@Author xiaomin.zhang
"""

import torch
import sys
import io

# 设置标准输出编码为UTF-8（Windows兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from config import config


def test_embedding():
    """测试嵌入层"""
    print("\n" + "=" * 50)
    print("测试嵌入层")
    print("=" * 50)
    
    from models.embedding import TokenEmbedding, PositionalEncoding
    
    batch_size = 2
    seq_len = 10
    
    # 测试Token嵌入
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    token_emb = TokenEmbedding(config.vocab_size, config.d_model)
    embedded = token_emb(token_ids)
    
    print(f"✓ Token嵌入: {token_ids.shape} -> {embedded.shape}")
    
    # 测试位置编码
    pos_enc = PositionalEncoding(config.d_model, config.max_seq_len)
    output = pos_enc(embedded)
    
    print(f"✓ 位置编码: {embedded.shape} -> {output.shape}")


def test_attention():
    """测试注意力机制"""
    print("\n" + "=" * 50)
    print("测试注意力机制")
    print("=" * 50)
    
    from models.attention import MultiHeadAttention, CausalSelfAttention
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # 测试多头注意力
    mha = MultiHeadAttention(config.d_model, config.n_heads)
    output = mha(x)
    
    print(f"✓ 多头注意力: {x.shape} -> {output.shape}")
    
    # 测试因果注意力
    causal_attn = CausalSelfAttention(config.d_model, config.n_heads, config.max_seq_len)
    output = causal_attn(x)
    
    print(f"✓ 因果注意力: {x.shape} -> {output.shape}")


def test_transformer():
    """测试Transformer Block"""
    print("\n" + "=" * 50)
    print("测试Transformer Block")
    print("=" * 50)
    
    from models.transformer import TransformerBlock
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    block = TransformerBlock(
        config.d_model,
        config.n_heads,
        config.d_ff,
        config.max_seq_len
    )
    output = block(x)
    
    print(f"✓ Transformer Block: {x.shape} -> {output.shape}")


def test_gpt_model():
    """测试完整GPT模型"""
    print("\n" + "=" * 50)
    print("测试GPT模型")
    print("=" * 50)
    
    from models.gpt import GPTModel
    
    model = GPTModel(config)
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # 测试前向传播
    logits, loss = model(input_ids, targets)
    
    print(f"✓ 前向传播: {input_ids.shape} -> {logits.shape}")
    print(f"✓ 损失计算: {loss.item():.4f}")
    
    # 测试生成
    generated = model.generate(input_ids[:1], max_new_tokens=5)
    
    print(f"✓ 文本生成: {input_ids[:1].shape} -> {generated.shape}")


def test_tokenizer():
    """测试分词器"""
    print("\n" + "=" * 50)
    print("测试分词器")
    print("=" * 50)
    
    from utils.tokenizer import Tokenizer
    
    texts = [
        "今天天气很好",
        "我喜欢学习人工智能",
        "深度学习很有趣"
    ]
    
    tokenizer = Tokenizer(vocab_size=100)
    tokenizer.build_vocab(texts)
    
    test_text = texts[0]
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"✓ 原始文本: {test_text}")
    print(f"✓ 编码: {token_ids}")
    print(f"✓ 解码: {decoded}")


def test_all():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print(" " * 20 + "模型组件测试")
    print("=" * 70)
    
    try:
        test_embedding()
        test_attention()
        test_transformer()
        test_gpt_model()
        test_tokenizer()
        
        print("\n" + "=" * 70)
        print(" " * 20 + "✓ 所有测试通过！")
        print("=" * 70)
        
        print("\n模型信息:")
        print(f"  参数规模: ~{config.d_model * config.d_model * config.n_layers * 12 / 1e6:.1f}M")
        print(f"  层数: {config.n_layers}")
        print(f"  模型维度: {config.d_model}")
        print(f"  注意力头数: {config.n_heads}")
        print(f"  词表大小: {config.vocab_size}")
        print(f"  最大序列长度: {config.max_seq_len}")
        print(f"  设备: {config.device}")
        
        print("\n下一步:")
        print("  1. 运行 'python prepare_data.py' 准备数据")
        print("  2. 运行 'python train.py' 开始训练")
        print("  3. 运行 'python inference.py' 生成文本")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_all()
