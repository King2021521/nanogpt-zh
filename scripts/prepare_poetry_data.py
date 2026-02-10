"""
中文诗词数据集预处理脚本
用于处理 chinese-poetry-collection 数据集
@Author xiaomin.zhang
"""

import os
import csv
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils import Tokenizer


def load_poetry_csv(csv_path):
    """
    从CSV文件加载诗词数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        诗词文本列表
    """
    texts = []
    print(f"正在加载: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过第一行（表头）
        for line in tqdm(lines[1:], desc="读取数据"):
            line = line.strip()
            if line:  # 非空行
                texts.append(line)
    
    return texts


def clean_poetry_text(text):
    """
    清洗诗词文本
    
    Args:
        text: 原始诗词文本
        
    Returns:
        清洗后的文本
    """
    # 移除多余空格
    text = ' '.join(text.split())
    
    # 过滤太短或太长的诗词
    if len(text) < 10 or len(text) > 500:
        return None
    
    return text


def prepare_poetry_data():
    """
    准备诗词训练数据
    """
    print("=" * 60)
    print("中文诗词数据集预处理")
    print("=" * 60)
    
    # 数据路径
    poetry_dir = "data/chinese-poetry-collection"
    train_csv = os.path.join(poetry_dir, "train.csv")
    test_csv = os.path.join(poetry_dir, "test.csv")
    
    # 检查数据是否存在
    if not os.path.exists(train_csv):
        print(f"\n错误: 找不到训练数据文件: {train_csv}")
        print("请先运行以下命令克隆数据集:")
        print("git clone http://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection.git data/chinese-poetry-collection")
        return
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_texts = load_poetry_csv(train_csv)
    test_texts = load_poetry_csv(test_csv)
    
    print(f"训练集原始数据: {len(train_texts)} 条")
    print(f"测试集原始数据: {len(test_texts)} 条")
    
    # 2. 数据清洗
    print("\n2. 数据清洗...")
    cleaned_train = []
    for text in tqdm(train_texts, desc="清洗训练数据"):
        cleaned = clean_poetry_text(text)
        if cleaned:
            cleaned_train.append(cleaned)
    
    cleaned_test = []
    for text in tqdm(test_texts, desc="清洗测试数据"):
        cleaned = clean_poetry_text(text)
        if cleaned:
            cleaned_test.append(cleaned)
    
    print(f"清洗后训练集: {len(cleaned_train)} 条")
    print(f"清洗后测试集: {len(cleaned_test)} 条")
    
    # 3. 划分训练集和验证集
    # 从训练集中取出5%作为验证集
    print("\n3. 划分数据集...")
    split_idx = int(len(cleaned_train) * 0.95)
    train_final = cleaned_train[:split_idx]
    val_final = cleaned_train[split_idx:]
    
    print(f"最终训练集: {len(train_final)} 条")
    print(f"最终验证集: {len(val_final)} 条")
    print(f"测试集: {len(cleaned_test)} 条")
    
    # 4. 保存处理后的数据
    print("\n4. 保存数据...")
    os.makedirs("data/processed", exist_ok=True)
    
    # 保存训练集
    with open(config.train_data_path, 'w', encoding='utf-8') as f:
        for text in tqdm(train_final, desc="保存训练数据"):
            f.write(text + '\n')
    print(f"训练数据已保存: {config.train_data_path}")
    
    # 保存验证集
    with open(config.val_data_path, 'w', encoding='utf-8') as f:
        for text in tqdm(val_final, desc="保存验证数据"):
            f.write(text + '\n')
    print(f"验证数据已保存: {config.val_data_path}")
    
    # 保存测试集
    test_data_path = "data/processed/test.txt"
    with open(test_data_path, 'w', encoding='utf-8') as f:
        for text in tqdm(cleaned_test, desc="保存测试数据"):
            f.write(text + '\n')
    print(f"测试数据已保存: {test_data_path}")
    
    # 5. 构建词表
    print("\n5. 构建词表...")
    all_texts = train_final + val_final
    
    tokenizer = Tokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(all_texts)
    
    # 6. 保存词表
    print("\n6. 保存词表...")
    tokenizer.save(config.vocab_path)
    print(f"词表已保存: {config.vocab_path}")
    print(f"词表大小: {len(tokenizer.word2idx)}")
    
    # 7. 数据统计
    print("\n7. 数据统计...")
    total_chars_train = sum(len(text) for text in train_final)
    total_chars_val = sum(len(text) for text in val_final)
    total_chars_test = sum(len(text) for text in cleaned_test)
    
    print(f"\n训练集统计:")
    print(f"  - 样本数: {len(train_final):,}")
    print(f"  - 总字符数: {total_chars_train:,}")
    print(f"  - 平均长度: {total_chars_train / len(train_final):.1f} 字符/样本")
    
    print(f"\n验证集统计:")
    print(f"  - 样本数: {len(val_final):,}")
    print(f"  - 总字符数: {total_chars_val:,}")
    print(f"  - 平均长度: {total_chars_val / len(val_final):.1f} 字符/样本")
    
    print(f"\n测试集统计:")
    print(f"  - 样本数: {len(cleaned_test):,}")
    print(f"  - 总字符数: {total_chars_test:,}")
    print(f"  - 平均长度: {total_chars_test / len(cleaned_test):.1f} 字符/样本")
    
    # 8. 测试分词
    print("\n8. 测试分词...")
    test_text = train_final[0]
    print(f"原始文本: {test_text}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token数量: {len(token_ids)}")
    print(f"Token IDs (前20个): {token_ids[:20]}")
    
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码文本: {decoded_text}")
    
    # 9. 保存数据集信息
    print("\n9. 保存数据集信息...")
    dataset_info = {
        "dataset_name": "chinese-poetry-collection",
        "source": "ModelScope",
        "train_samples": len(train_final),
        "val_samples": len(val_final),
        "test_samples": len(cleaned_test),
        "total_chars_train": total_chars_train,
        "total_chars_val": total_chars_val,
        "total_chars_test": total_chars_test,
        "vocab_size": len(tokenizer.word2idx),
        "avg_length_train": total_chars_train / len(train_final),
        "avg_length_val": total_chars_val / len(val_final),
        "avg_length_test": total_chars_test / len(cleaned_test),
    }
    
    info_path = "data/processed/dataset_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"数据集信息已保存: {info_path}")
    
    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)
    print("\n下一步: 运行 python train.py 开始训练")


if __name__ == "__main__":
    prepare_poetry_data()
