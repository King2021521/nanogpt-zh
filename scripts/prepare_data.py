"""
数据准备脚本
用于下载、处理和准备训练数据
@Author xiaomin.zhang
"""

import os
import requests
from tqdm import tqdm
from config import config
from utils import Tokenizer


def download_sample_data():
    """
    下载示例数据
    这里使用一些公开的中文文本作为示例
    实际使用时，您可以替换为自己的数据源
    """
    print("准备示例数据...")
    
    # 创建示例数据（实际项目中应该从真实数据源获取）
    sample_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "深度学习是机器学习的一个子领域，它基于人工神经网络的研究，特别是利用多层次的神经网络来进行学习和模式识别。",
        "自然语言处理是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言。",
        "Transformer是一种深度学习模型，主要用于自然语言处理领域。它在2017年由Google提出，彻底改变了NLP领域。",
        "PyTorch是一个开源的Python机器学习库，基于Torch，用于自然语言处理等应用程序。",
        "神经网络是一种模仿生物神经网络的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。",
        "机器学习是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。",
        "计算机视觉是一门研究如何使机器看的科学，更进一步的说，就是指用摄影机和计算机代替人眼对目标进行识别、跟踪和测量等。",
        "强化学习是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。",
        "卷积神经网络是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。",
        "循环神经网络是一类人工神经网络，其中节点之间的连接形成有向图。这使其能够展示时间动态行为。",
        "生成对抗网络是一种深度学习模型，是近年来复杂分布上无监督学习最具前景的方法之一。",
        "迁移学习是机器学习中的一个研究领域，它关注于将为一个任务开发的模型重新用于另一个相关任务的起点。",
        "注意力机制是深度学习中的一种技术，它允许模型在处理输入时动态地关注输入的不同部分。",
        "词嵌入是自然语言处理中的一组语言建模和特征学习技术的统称，其中来自词汇表的单词或短语被映射到实数的向量。",
        "今天天气很好，阳光明媚，适合出去散步。",
        "我喜欢在周末的时候读书，这让我感到很放松。",
        "学习新知识是一件令人兴奋的事情，它能够开阔我们的视野。",
        "科技的发展改变了我们的生活方式，让一切变得更加便捷。",
        "保持好奇心和学习的热情是成长的关键。"
    ]
    
    # 扩展数据（重复并添加变化）
    extended_texts = []
    for _ in range(50):  # 重复50次以增加数据量
        for text in sample_texts:
            extended_texts.append(text)
    
    return extended_texts


def prepare_data():
    """
    准备训练数据
    """
    print("=" * 50)
    print("数据准备")
    print("=" * 50)
    
    # 创建目录
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. 获取数据
    print("\n1. 获取数据...")
    texts = download_sample_data()
    print(f"总文本数: {len(texts)}")
    
    # 2. 划分训练集和验证集
    print("\n2. 划分数据集...")
    split_idx = int(len(texts) * 0.9)  # 90%训练，10%验证
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"训练集: {len(train_texts)} 条")
    print(f"验证集: {len(val_texts)} 条")
    
    # 3. 保存原始数据
    print("\n3. 保存数据...")
    with open(config.train_data_path, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text.strip() + '\n')
    
    with open(config.val_data_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text.strip() + '\n')
    
    print(f"训练数据已保存: {config.train_data_path}")
    print(f"验证数据已保存: {config.val_data_path}")
    
    # 4. 构建词表
    print("\n4. 构建词表...")
    tokenizer = Tokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(texts)
    
    # 5. 保存词表
    tokenizer.save(config.vocab_path)
    
    # 6. 测试分词
    print("\n5. 测试分词...")
    test_text = train_texts[0]
    print(f"原始文本: {test_text}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码文本: {decoded_text}")
    
    print("\n" + "=" * 50)
    print("数据准备完成！")
    print("=" * 50)
    print("\n提示：这是示例数据，数据量较小。")
    print("实际训练时，建议使用更大的数据集，例如：")
    print("- 中文维基百科")
    print("- 新闻语料库")
    print("- 小说、文章等文本数据")
    print("\n您可以将自己的数据放在 data/raw/ 目录下，")
    print("然后修改此脚本来处理您的数据。")


def download_wiki_data():
    """
    下载中文维基百科数据（可选）
    这是一个示例函数，展示如何下载真实数据
    """
    print("\n下载中文维基百科数据...")
    print("提示：这需要较长时间和较大存储空间")
    print("您可以从以下地址手动下载：")
    print("https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2")
    print("\n下载后，可以使用 WikiExtractor 工具提取文本：")
    print("pip install wikiextractor")
    print("python -m wikiextractor.WikiExtractor zhwiki-latest-pages-articles.xml.bz2 -o data/raw/wiki")


if __name__ == "__main__":
    prepare_data()
    
    # 如果需要下载真实数据，取消下面的注释
    #download_wiki_data()
