"""
训练脚本
@Author xiaomin.zhang
"""

import torch
import os
from config import config
from models import GPTModel
from utils import Tokenizer, create_dataloader, Trainer


def main():
    """主训练函数"""
    print("=" * 50)
    print("中文GPT模型训练")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 1. 加载或创建分词器
    print("\n1. 加载分词器...")
    tokenizer = Tokenizer(vocab_size=config.vocab_size)
    
    if os.path.exists(config.vocab_path):
        tokenizer.load(config.vocab_path)
    else:
        print("词表文件不存在，请先运行 prepare_data.py 准备数据！")
        return
    
    # 2. 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader = create_dataloader(
        config.train_data_path,
        tokenizer,
        config.max_seq_len,
        config.batch_size,
        shuffle=True,
        num_workers=0  # Windows上建议设为0
    )
    
    val_loader = create_dataloader(
        config.val_data_path,
        tokenizer,
        config.max_seq_len,
        config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    model = GPTModel(config)
    print(config)
    
    # 4. 创建训练器
    print("\n4. 创建训练器...")
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 5. 开始训练
    print("\n5. 开始训练...")
    trainer.train()
    
    print("\n训练完成！")
    print(f"模型保存在: {config.checkpoint_dir}")
    print(f"日志保存在: {config.log_dir}")


if __name__ == "__main__":
    main()
