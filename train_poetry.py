"""
诗词模型训练脚本
使用针对诗词数据集优化的配置
@Author xiaomin.zhang
"""

import torch
import os
from config_poetry import config_poetry
from models import GPTModel
from utils import Tokenizer, create_dataloader, Trainer


def main():
    """主训练函数"""
    print("=" * 60)
    print("中文古诗词GPT模型训练")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 1. 加载分词器
    print("\n1. 加载分词器...")
    tokenizer = Tokenizer(vocab_size=config_poetry.vocab_size)
    
    if os.path.exists(config_poetry.vocab_path):
        tokenizer.load(config_poetry.vocab_path)
    else:
        print("词表文件不存在，请先运行 python scripts/prepare_poetry_data.py 准备数据！")
        return
    
    # 2. 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader = create_dataloader(
        config_poetry.train_data_path,
        tokenizer,
        config_poetry.max_seq_len,
        config_poetry.batch_size,
        shuffle=True,
        num_workers=0  # Windows上建议设为0
    )
    
    val_loader = create_dataloader(
        config_poetry.val_data_path,
        tokenizer,
        config_poetry.max_seq_len,
        config_poetry.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    model = GPTModel(config_poetry)
    
    # 计算实际参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 4. 显示配置
    print("\n4. 训练配置:")
    print(config_poetry)
    
    # 5. 创建训练器
    print("\n5. 创建训练器...")
    trainer = Trainer(model, train_loader, val_loader, config_poetry)
    
    # 6. 开始训练
    print("\n6. 开始训练...")
    print("=" * 60)
    trainer.train()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"模型保存在: {config_poetry.checkpoint_dir}")
    print(f"日志保存在: {config_poetry.log_dir}")
    print("\n下一步:")
    print("  1. 查看训练日志: tensorboard --logdir logs_poetry")
    print("  2. 测试模型: python generate_poetry.py")


if __name__ == "__main__":
    main()
