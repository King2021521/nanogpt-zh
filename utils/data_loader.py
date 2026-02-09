"""
数据加载器实现
@Author xiaomin.zhang
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os


class TextDataset(Dataset):
    """
    文本数据集
    用于加载和处理文本数据
    """
    def __init__(self, data_path, tokenizer, max_seq_len):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器对象
            max_seq_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 读取数据
        print(f"加载数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        
        print(f"数据量: {len(self.texts)} 条")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            input_ids: 输入token IDs
            target_ids: 目标token IDs（向右偏移一位）
        """
        text = self.texts[idx].strip()
        
        # 编码文本
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # 截断或填充到max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            # 填充到max_seq_len
            token_ids = token_ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(token_ids))
        
        # 转换为tensor
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # 输入是除了最后一个token的所有token
        # 目标是除了第一个token的所有token
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        
        return input_ids, target_ids


def create_dataloader(data_path, tokenizer, max_seq_len, batch_size, shuffle=True, num_workers=0):
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        tokenizer: 分词器对象
        max_seq_len: 最大序列长度
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
    
    Returns:
        dataloader: PyTorch DataLoader对象
    """
    dataset = TextDataset(data_path, tokenizer, max_seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 如果使用GPU，可以加速数据传输
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试代码
    from tokenizer import Tokenizer
    
    print("测试TextDataset和DataLoader...")
    
    # 创建测试数据
    test_data_path = "data/processed/test_data.txt"
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    
    test_texts = [
        "今天天气很好，适合出去玩。",
        "我喜欢学习人工智能和深度学习。",
        "PyTorch是一个很好的深度学习框架。",
        "Transformer模型改变了自然语言处理领域。",
        "深度学习在计算机视觉和自然语言处理中都有广泛应用。"
    ]
    
    with open(test_data_path, 'w', encoding='utf-8') as f:
        for text in test_texts:
            f.write(text + '\n')
    
    # 创建分词器
    tokenizer = Tokenizer(vocab_size=100)
    tokenizer.build_vocab(test_texts)
    
    # 创建数据加载器
    max_seq_len = 32
    batch_size = 2
    
    dataloader = create_dataloader(
        test_data_path,
        tokenizer,
        max_seq_len,
        batch_size,
        shuffle=True
    )
    
    # 测试加载一个batch
    print(f"\n测试加载数据...")
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Input IDs: {input_ids[0][:10]}...")  # 只打印前10个
        print(f"  Target IDs: {target_ids[0][:10]}...")
        
        if batch_idx == 0:  # 只测试第一个batch
            break
    
    print("\n✓ DataLoader测试通过！")
