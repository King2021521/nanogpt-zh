"""
重建词表脚本
基于已处理的训练数据重新构建词表
@Author xiaomin.zhang
"""

import os
import sys
import json
from collections import Counter
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Tokenizer


def load_text_data(file_path):
    """
    加载文本数据
    @Author xiaomin.zhang
    """
    print(f"加载数据: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"  加载完成: {len(texts):,} 条")
    return texts


def build_vocab(texts, vocab_size=10000):
    """
    构建词表
    @Author xiaomin.zhang
    """
    print(f"\n构建词表 (目标大小: {vocab_size})...")
    
    # 统计字符频率
    char_counter = Counter()
    
    print("统计字符频率...")
    for text in tqdm(texts, desc="处理文本"):
        char_counter.update(text)
    
    # 获取最常见的字符
    total_chars = sum(char_counter.values())
    unique_chars = len(char_counter)
    
    print(f"\n字符统计:")
    print(f"  总字符数: {total_chars:,}")
    print(f"  唯一字符数: {unique_chars:,}")
    
    # 保留最常见的字符（减去特殊token的数量）
    special_tokens = 4  # <PAD>, <UNK>, <BOS>, <EOS>
    most_common = char_counter.most_common(vocab_size - special_tokens)
    
    # 计算覆盖率
    covered_chars = sum(count for _, count in most_common)
    coverage = covered_chars / total_chars * 100
    
    print(f"\n词表信息:")
    print(f"  词表大小: {len(most_common) + special_tokens}")
    print(f"  覆盖字符: {covered_chars:,} / {total_chars:,}")
    print(f"  覆盖率: {coverage:.2f}%")
    
    # 显示最常见的字符
    print(f"\n最常见字符 (Top 20):")
    for char, count in most_common[:20]:
        print(f"    '{char}': {count:,}")
    
    return most_common


def save_vocab(tokenizer, vocab_path):
    """
    保存词表
    @Author xiaomin.zhang
    """
    print(f"\n保存词表到: {vocab_path}")
    
    # 创建目录
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    
    # 保存词表
    tokenizer.save(vocab_path)
    
    print(f"  词表已保存")
    print(f"  词表大小: {len(tokenizer.word2idx)}")


def test_tokenizer(tokenizer, test_texts):
    """
    测试分词器
    @Author xiaomin.zhang
    """
    print("\n" + "=" * 60)
    print("测试分词器")
    print("=" * 60)
    
    for i, text in enumerate(test_texts[:3], 1):
        print(f"\n测试 {i}:")
        print(f"  原文: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # 编码
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"  Token数量: {len(token_ids)}")
        print(f"  Token IDs (前20个): {token_ids[:20]}")
        
        # 解码
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  解码: {decoded[:50]}{'...' if len(decoded) > 50 else ''}")
        
        # 检查是否一致
        if text == decoded:
            print(f"  [OK] 编码解码一致")
        else:
            print(f"  [ERROR] 编码解码不一致")


def analyze_data(texts):
    """
    分析数据统计信息
    @Author xiaomin.zhang
    """
    print("\n" + "=" * 60)
    print("数据统计分析")
    print("=" * 60)
    
    total_count = len(texts)
    total_chars = sum(len(t) for t in texts)
    avg_length = total_chars / total_count if total_count > 0 else 0
    
    print(f"\n总诗词数: {total_count:,}")
    print(f"总字符数: {total_chars:,}")
    print(f"平均长度: {avg_length:.1f} 字符/首")
    
    # 长度分布
    lengths = [len(t) for t in texts]
    print(f"\n长度分布:")
    print(f"  最短: {min(lengths)} 字符")
    print(f"  最长: {max(lengths)} 字符")
    print(f"  中位数: {sorted(lengths)[len(lengths)//2]} 字符")


def main():
    """主函数"""
    print("=" * 60)
    print("重建词表")
    print("=" * 60)
    
    # 定义路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(script_dir, "data", "processed")
    
    train_file = os.path.join(processed_dir, "train.txt")
    val_file = os.path.join(processed_dir, "val.txt")
    test_file = os.path.join(processed_dir, "test.txt")
    vocab_path = os.path.join(processed_dir, "vocab.json")
    
    # 检查文件是否存在
    if not os.path.exists(train_file):
        print(f"\n错误: 训练数据文件不存在: {train_file}")
        print("请先运行数据合并脚本: python scripts/merge_songci_data.py")
        return
    
    # 加载所有数据（用于构建词表）
    print("\n" + "=" * 60)
    print("第1步: 加载数据")
    print("=" * 60)
    
    train_texts = load_text_data(train_file)
    val_texts = load_text_data(val_file)
    test_texts = load_text_data(test_file)
    
    all_texts = train_texts + val_texts + test_texts
    print(f"\n总数据量: {len(all_texts):,} 条")
    
    # 分析数据
    analyze_data(all_texts)
    
    # 构建词表
    print("\n" + "=" * 60)
    print("第2步: 构建词表")
    print("=" * 60)
    
    vocab_size = 10000
    most_common = build_vocab(all_texts, vocab_size)
    
    # 创建分词器
    print("\n" + "=" * 60)
    print("第3步: 创建分词器")
    print("=" * 60)
    
    tokenizer = Tokenizer(vocab_size=vocab_size)
    
    # 构建词表
    tokenizer.word2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }
    
    for idx, (char, _) in enumerate(most_common, start=4):
        tokenizer.word2idx[char] = idx
    
    tokenizer.idx2word = {idx: word for word, idx in tokenizer.word2idx.items()}
    tokenizer.vocab_size = len(tokenizer.word2idx)
    
    print(f"分词器创建完成")
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  特殊token: <PAD>, <UNK>, <BOS>, <EOS>")
    
    # 保存词表
    print("\n" + "=" * 60)
    print("第4步: 保存词表")
    print("=" * 60)
    
    save_vocab(tokenizer, vocab_path)
    
    # 测试分词器
    test_tokenizer(tokenizer, train_texts)
    
    # 完成
    print("\n" + "=" * 60)
    print("词表重建完成！")
    print("=" * 60)
    
    print(f"\n词表文件: {vocab_path}")
    print(f"词表大小: {tokenizer.vocab_size}")
    
    print("\n下一步:")
    print("  开始训练: python train_poetry.py")


if __name__ == "__main__":
    main()
