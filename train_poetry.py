"""
诗词模型训练脚本
使用针对诗词数据集优化的配置
@Author xiaomin.zhang
"""

import torch
import os
import glob
import re
from config_poetry import config_poetry
from models import GPTModel
from utils import Tokenizer, create_dataloader, Trainer


def find_latest_checkpoint(checkpoint_dir):
    """
    查找最新的检查点文件
    @Author xiaomin.zhang
    
    Args:
        checkpoint_dir: 检查点目录
    
    Returns:
        最新检查点的路径，如果没有找到则返回None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有检查点文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    
    if not checkpoint_files:
        return None
    
    # 提取步数并排序
    checkpoint_info = []
    for ckpt_file in checkpoint_files:
        # 从文件名中提取步数，例如: checkpoint_step_78000.pt -> 78000
        match = re.search(r'checkpoint_step_(\d+)\.pt', os.path.basename(ckpt_file))
        if match:
            step = int(match.group(1))
            checkpoint_info.append((step, ckpt_file))
    
    if not checkpoint_info:
        return None
    
    # 按步数排序，返回最新的
    checkpoint_info.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_file = checkpoint_info[0]
    
    print(f"\n找到最新检查点: {os.path.basename(latest_file)} (步数: {latest_step})")
    return latest_file


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
    
    # 5. 查找并加载最新检查点
    print("\n5. 检查是否有已保存的检查点...")
    latest_checkpoint = find_latest_checkpoint(config_poetry.checkpoint_dir)
    
    # 6. 创建训练器
    print("\n6. 创建训练器...")
    trainer = Trainer(model, train_loader, val_loader, config_poetry)
    
    # 7. 如果找到检查点，加载它
    if latest_checkpoint:
        print(f"\n正在从检查点恢复训练...")
        try:
            trainer.load_checkpoint(latest_checkpoint)
            print(f"[OK] 将从第 {trainer.epoch + 1} 轮，第 {trainer.global_step} 步继续训练")
        except Exception as e:
            print(f"[警告] 加载检查点失败: {e}")
            print("将从头开始训练...")
    else:
        print("未找到检查点，将从头开始训练")
    
    # 8. 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
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
