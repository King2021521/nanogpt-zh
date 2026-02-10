"""
快速测试脚本
用于快速验证训练好的诗词模型
@Author xiaomin.zhang
"""

import torch
import os
from config_poetry import config_poetry
from models import GPTModel
from utils import Tokenizer


def quick_test():
    """快速测试"""
    print("=" * 60)
    print("诗词模型快速测试")
    print("=" * 60)
    
    # 1. 选择检查点
    checkpoint_dir = "checkpoints_poetry"
    checkpoint_files = [
        "best_model.pt",
        "final_model.pt",
    ]
    
    checkpoint_path = None
    for ckpt in checkpoint_files:
        path = os.path.join(checkpoint_dir, ckpt)
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print(f"错误: 找不到模型检查点文件")
        print(f"请确保以下文件之一存在:")
        for ckpt in checkpoint_files:
            print(f"  - {os.path.join(checkpoint_dir, ckpt)}")
        return
    
    print(f"\n使用检查点: {checkpoint_path}")
    
    # 2. 加载分词器
    print("\n加载分词器...")
    tokenizer = Tokenizer(vocab_size=config_poetry.vocab_size)
    tokenizer.load(config_poetry.vocab_path)
    print(f"词表大小: {len(tokenizer.word2idx)}")
    
    # 3. 加载模型
    print("\n加载模型...")
    # 强制使用CPU（如果GPU不支持）
    try:
        device = torch.device(config_poetry.device)
        if device.type == 'cuda':
            # 测试CUDA是否真的可用
            torch.zeros(1).to(device)
    except:
        print("⚠️  GPU不可用或不支持，切换到CPU模式")
        device = torch.device('cpu')
    model = GPTModel(config_poetry)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"设备: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    if 'step' in checkpoint:
        print(f"训练步数: {checkpoint['step']}")
    if 'loss' in checkpoint:
        print(f"损失: {checkpoint['loss']:.4f}")
    
    # 4. 测试生成
    print("\n" + "=" * 60)
    print("开始生成测试")
    print("=" * 60)
    
    test_prompts = [
        "春江潮水连海平",
        "明月几时有",
        "床前明月光",
        "白日依山尽",
        "独在异乡为异客",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【测试 {i}】")
        print(f"提示: {prompt}")
        print("-" * 60)
        
        # 编码
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=64,
                temperature=0.8,
                top_k=40,
                top_p=0.85
            )
        
        # 解码
        generated_ids = generated_ids[0].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"生成: {generated_text}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n提示:")
    print("  - 运行完整测试: python test_poetry.py --mode all")
    print("  - 交互式生成: python test_poetry.py --mode interactive")
    print("  - 评估困惑度: python test_poetry.py --mode perplexity")


if __name__ == "__main__":
    quick_test()
