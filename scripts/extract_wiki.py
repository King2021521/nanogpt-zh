"""
直接从维基百科XML文件提取文本
@Author xiaomin.zhang
"""

import bz2
import re
import os
import sys
import io
from tqdm import tqdm
from config import config
from utils import Tokenizer

# 设置标准输出编码为UTF-8（Windows兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def clean_text(text):
    """
    清洗文本
    """
    # 移除XML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除Wiki标记
    text = re.sub(r'\[\[([^\]]+\|)?([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\{\{[^\}]+\}\}', '', text)
    text = re.sub(r"'''|''", '', text)
    # 移除URL
    text = re.sub(r'http[s]?://\S+', '', text)
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符（保留中文和常用标点）
    text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef0-9a-zA-Z，。！？；：、""''（）《》\s]', '', text)
    
    return text.strip()


def extract_from_xml(xml_path, max_articles=None):
    """
    从维基百科XML文件提取文本
    
    Args:
        xml_path: XML文件路径
        max_articles: 最大文章数（None表示全部）
    
    Returns:
        texts: 文本列表
    """
    texts = []
    current_text = []
    in_text = False
    article_count = 0
    
    print(f"开始提取: {xml_path}")
    print("这可能需要几分钟时间...")
    
    # 打开bz2压缩文件
    with bz2.open(xml_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="处理XML", unit=" lines"):
            line = line.strip()
            
            # 检测文本开始
            if '<text' in line:
                in_text = True
                # 提取当前行的文本部分
                start = line.find('>') + 1
                if start > 0:
                    current_text.append(line[start:])
                continue
            
            # 检测文本结束
            if '</text>' in line:
                in_text = False
                # 提取当前行的文本部分
                end = line.find('</text>')
                if end > 0:
                    current_text.append(line[:end])
                
                # 处理收集到的文本
                if current_text:
                    text = ' '.join(current_text)
                    text = clean_text(text)
                    
                    # 按句子分割
                    sentences = re.split(r'[。！？]', text)
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) >= 20:  # 保留有意义的句子
                            texts.append(sent)
                    
                    current_text = []
                    article_count += 1
                    
                    # 检查是否达到最大文章数
                    if max_articles and article_count >= max_articles:
                        print(f"\n已处理 {article_count} 篇文章，停止提取")
                        break
                
                continue
            
            # 收集文本内容
            if in_text:
                current_text.append(line)
    
    print(f"\n总共提取 {len(texts)} 条文本，来自 {article_count} 篇文章")
    return texts


def process_wiki_xml():
    """
    处理维基百科XML文件
    """
    print("=" * 50)
    print("提取维基百科数据")
    print("=" * 50)
    
    xml_path = "data/zhwiki-latest-pages-articles.xml.bz2"
    
    # 检查文件是否存在
    if not os.path.exists(xml_path):
        print(f"\n错误: 文件不存在: {xml_path}")
        return
    
    # 获取文件大小
    file_size = os.path.getsize(xml_path) / (1024 * 1024)
    print(f"文件大小: {file_size:.2f} MB")
    
    # 1. 提取文本（可以限制文章数量以加快处理）
    print("\n1. 提取文本...")
    print("提示: 如果想提取全部数据，请耐心等待。")
    print("      如果想快速测试，可以修改代码中的max_articles参数。")
    
    # 提取数据（这里设置max_articles=10000用于演示，删除此参数可提取全部）
    texts = extract_from_xml(xml_path)
    
    if len(texts) == 0:
        print("错误: 没有提取到任何文本！")
        return
    
    print(f"提取到 {len(texts)} 条文本")
    
    # 2. 划分训练集和验证集
    print("\n2. 划分数据集...")
    split_idx = int(len(texts) * 0.95)  # 95%训练，5%验证
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"训练集: {len(train_texts)} 条")
    print(f"验证集: {len(val_texts)} 条")
    
    # 3. 保存数据
    print("\n3. 保存数据...")
    os.makedirs("data/processed", exist_ok=True)
    
    with open(config.train_data_path, 'w', encoding='utf-8') as f:
        for text in tqdm(train_texts, desc="保存训练数据"):
            f.write(text + '\n')
    
    with open(config.val_data_path, 'w', encoding='utf-8') as f:
        for text in tqdm(val_texts, desc="保存验证数据"):
            f.write(text + '\n')
    
    print(f"训练数据已保存: {config.train_data_path}")
    print(f"验证数据已保存: {config.val_data_path}")
    
    # 4. 构建词表
    print("\n4. 构建词表...")
    # 使用部分数据构建词表以加速
    sample_size = min(20000, len(texts))
    sample_texts = texts[:sample_size]
    
    tokenizer = Tokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(sample_texts)
    tokenizer.save(config.vocab_path)
    
    # 5. 测试分词
    print("\n5. 测试分词...")
    test_text = train_texts[0] if train_texts else "测试文本"
    print(f"原始文本: {test_text[:80]}...")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token数量: {len(token_ids)}")
    
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码文本: {decoded_text[:80]}...")
    
    # 6. 统计信息
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
    process_wiki_xml()
