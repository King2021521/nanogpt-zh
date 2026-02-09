"""
处理维基百科数据并转换为训练格式
@Author xiaomin.zhang
"""

import os
import json
import sys
import io
from tqdm import tqdm
from config import config
from utils import Tokenizer

# 设置标准输出编码为UTF-8（Windows兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def collect_wiki_texts(wiki_dir):
    """
    从WikiExtractor提取的文件中收集所有文本
    
    Args:
        wiki_dir: wiki数据目录
    
    Returns:
        texts: 文本列表
    """
    texts = []
    
    print(f"从 {wiki_dir} 收集文本...")
    
    # 遍历所有子目录和文件
    for root, dirs, files in os.walk(wiki_dir):
        for file in files:
            if file.startswith('wiki_'):
                file_path = os.path.join(root, file)
                print(f"处理文件: {file_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                # WikiExtractor的JSON格式
                                data = json.loads(line)
                                text = data.get('text', '').strip()
                                
                                if text and len(text) > 50:  # 过滤太短的文本
                                    # 按句子分割（简单处理）
                                    sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
                                    for sent in sentences:
                                        sent = sent.strip()
                                        if len(sent) > 20:  # 保留有效句子
                                            texts.append(sent)
                            except json.JSONDecodeError:
                                # 如果不是JSON格式，直接作为文本处理
                                if len(line) > 50:
                                    texts.append(line)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
                    continue
    
    print(f"总共收集到 {len(texts)} 条文本")
    return texts


def process_wiki_data():
    """
    处理维基百科数据
    """
    print("=" * 50)
    print("处理维基百科数据")
    print("=" * 50)
    
    wiki_dir = "data/raw/wiki"
    
    # 检查wiki目录是否存在
    if not os.path.exists(wiki_dir):
        print(f"\n错误: wiki数据目录不存在: {wiki_dir}")
        print("请先运行WikiExtractor提取数据：")
        print("python -m wikiextractor.WikiExtractor data/zhwiki-latest-pages-articles.xml.bz2 -o data/raw/wiki --json")
        return
    
    # 1. 收集所有文本
    print("\n1. 收集文本...")
    texts = collect_wiki_texts(wiki_dir)
    
    if len(texts) == 0:
        print("错误: 没有收集到任何文本！")
        return
    
    print(f"收集到 {len(texts)} 条文本")
    
    # 2. 数据清洗和过滤
    print("\n2. 数据清洗...")
    cleaned_texts = []
    for text in tqdm(texts, desc="清洗文本"):
        # 移除过短和过长的文本
        if 20 <= len(text) <= 500:
            # 移除特殊字符（保留中文、标点等）
            text = text.strip()
            if text:
                cleaned_texts.append(text)
    
    print(f"清洗后剩余 {len(cleaned_texts)} 条文本")
    
    # 3. 划分训练集和验证集
    print("\n3. 划分数据集...")
    split_idx = int(len(cleaned_texts) * 0.95)  # 95%训练，5%验证
    train_texts = cleaned_texts[:split_idx]
    val_texts = cleaned_texts[split_idx:]
    
    print(f"训练集: {len(train_texts)} 条")
    print(f"验证集: {len(val_texts)} 条")
    
    # 4. 保存数据
    print("\n4. 保存数据...")
    os.makedirs("data/processed", exist_ok=True)
    
    with open(config.train_data_path, 'w', encoding='utf-8') as f:
        for text in tqdm(train_texts, desc="保存训练数据"):
            f.write(text + '\n')
    
    with open(config.val_data_path, 'w', encoding='utf-8') as f:
        for text in tqdm(val_texts, desc="保存验证数据"):
            f.write(text + '\n')
    
    print(f"训练数据已保存: {config.train_data_path}")
    print(f"验证数据已保存: {config.val_data_path}")
    
    # 5. 构建词表
    print("\n5. 构建词表...")
    # 为了加速，只使用部分数据构建词表
    sample_size = min(50000, len(cleaned_texts))
    sample_texts = cleaned_texts[:sample_size]
    
    tokenizer = Tokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(sample_texts)
    tokenizer.save(config.vocab_path)
    
    # 6. 测试分词
    print("\n6. 测试分词...")
    test_text = train_texts[0] if train_texts else "测试文本"
    print(f"原始文本: {test_text[:100]}...")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token数量: {len(token_ids)}")
    
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码文本: {decoded_text[:100]}...")
    
    # 7. 统计信息
    print("\n" + "=" * 50)
    print("数据处理完成！")
    print("=" * 50)
    print(f"\n数据集统计:")
    print(f"  训练样本数: {len(train_texts):,}")
    print(f"  验证样本数: {len(val_texts):,}")
    print(f"  词表大小: {len(tokenizer.word2idx):,}")
    print(f"  平均文本长度: {sum(len(t) for t in train_texts) / len(train_texts):.1f} 字符")
    
    print(f"\n文件位置:")
    print(f"  训练数据: {config.train_data_path}")
    print(f"  验证数据: {config.val_data_path}")
    print(f"  词表: {config.vocab_path}")
    
    print("\n现在可以运行 'python train.py' 开始训练！")


if __name__ == "__main__":
    process_wiki_data()
