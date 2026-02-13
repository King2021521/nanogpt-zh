"""
快速诗词生成测试脚本
@Author xiaomin.zhang
"""

import torch
from config_poetry import config_poetry
from models import GPTModel
from utils import Tokenizer

def quick_generate(prompt, checkpoint_path="checkpoints_poetry/best_model.pt", 
                   temperature=1.0, top_k=80, top_p=0.9, max_length=128):
    """
    快速生成诗词
    
    Args:
        prompt: 诗词开头
        checkpoint_path: 模型检查点路径
        temperature: 采样温度（推荐1.0-1.2）
        top_k: Top-K采样（推荐80-100）
        top_p: Top-P采样（推荐0.9-0.95）
        max_length: 最大生成长度
    """
    # 设置设备
    device = torch.device("cpu")  # 强制使用CPU
    
    # 加载分词器
    tokenizer = Tokenizer(vocab_size=config_poetry.vocab_size)
    tokenizer.load(config_poetry.vocab_path)
    
    # 获取标点符号的token ID
    punct_tokens = ['，', '。', '、', '；', '：', '？', '！', ',', '.', '·', '…']
    punct_token_ids = []
    for p in punct_tokens:
        if p in tokenizer.word2idx:
            punct_token_ids.append(tokenizer.word2idx[p])
    
    # 加载模型
    model = GPTModel(config_poetry)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 编码输入（不添加特殊token）
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 生成
    print(f"\n提示: {prompt}")
    print(f"参数: temperature={temperature}, top_k={top_k}, top_p={top_p}, max_length={max_length}")
    print("-" * 60)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_id,
            suppress_eos_steps=int(max_length * 0.9),
            suppress_punct_tokens=punct_token_ids
        )
    
    # 解码
    generated_ids = generated_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"生成: {generated_text}")
    print("=" * 60)
    
    return generated_text


def test_different_temperatures():
    """测试不同温度参数的效果"""
    prompt = "春江潮水连海平"
    temperatures = [0.8, 1.0, 1.2, 1.5]
    
    print("\n" + "=" * 60)
    print("测试不同温度参数")
    print("=" * 60)
    
    for temp in temperatures:
        quick_generate(prompt, temperature=temp, top_k=80, top_p=0.9, max_length=64)
        print()


def test_different_top_k():
    """测试不同top_k参数的效果"""
    prompt = "明月几时有"
    top_k_values = [40, 60, 80, 100]
    
    print("\n" + "=" * 60)
    print("测试不同Top-K参数")
    print("=" * 60)
    
    for k in top_k_values:
        quick_generate(prompt, temperature=1.0, top_k=k, top_p=0.9, max_length=64)
        print()


def interactive_mode():
    """交互式生成"""
    print("\n" + "=" * 60)
    print("交互式诗词生成")
    print("=" * 60)
    print("\n输入诗词开头，模型将为您续写")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    # 默认参数
    temperature = 0.8
    top_k = 60
    top_p = 0.9
    max_length = 128
    
    # 预加载模型（避免每次都加载）
    device = torch.device("cpu")
    tokenizer = Tokenizer(vocab_size=config_poetry.vocab_size)
    tokenizer.load(config_poetry.vocab_path)
    
    # 获取标点符号的token ID
    punct_tokens = ['，', '。', '、', '；', '：', '？', '！', ',', '.', '·', '…']
    punct_token_ids = []
    for p in punct_tokens:
        if p in tokenizer.word2idx:
            punct_token_ids.append(tokenizer.word2idx[p])
    
    model = GPTModel(config_poetry)
    checkpoint = torch.load("checkpoints_poetry/final_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("\n模型加载完成！")
    print(f"已启用标点符号抑制机制（检测到 {len(punct_token_ids)} 个标点符号）")
    
    while True:
        try:
            prompt = input("\n请输入诗词开头: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break
            
            if not prompt:
                print("请输入有效的诗词开头")
                continue
            
            # 编码输入
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # 生成
            print("\n生成中...")
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=tokenizer.eos_id,
                    suppress_eos_steps=int(max_length * 0.9),
                    suppress_punct_tokens=punct_token_ids
                )
            
            # 解码
            generated_ids = generated_ids[0].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print("\n" + "=" * 60)
            print(f"生成结果: {generated_text}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n退出交互模式")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            # 交互模式
            interactive_mode()
        elif sys.argv[1] == "temp":
            # 测试温度
            test_different_temperatures()
        elif sys.argv[1] == "topk":
            # 测试top_k
            test_different_top_k()
        else:
            # 直接生成
            prompt = sys.argv[1]
            temp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 80
            quick_generate(prompt, temperature=temp, top_k=top_k)
    else:
        # 默认：测试几个常见的诗词开头
        print("\n" + "=" * 60)
        print("快速诗词生成测试")
        print("=" * 60)
        
        test_prompts = [
            "春江潮水连海平",
            "明月几时有",
            "床前明月光",
            "白日依山尽",
            "独在异乡为异客"
        ]
        
        for prompt in test_prompts:
            quick_generate(prompt, temperature=1.0, top_k=80, top_p=0.9, max_length=64)
            print()
        
        print("\n使用方法:")
        print("  1. 快速测试: python quick_generate.py")
        print("  2. 交互模式: python quick_generate.py interactive")
        print("  3. 测试温度: python quick_generate.py temp")
        print("  4. 测试top_k: python quick_generate.py topk")
        print("  5. 自定义生成: python quick_generate.py \"诗词开头\" 温度 top_k")
        print("     例如: python quick_generate.py \"春江潮水\" 1.2 100")
