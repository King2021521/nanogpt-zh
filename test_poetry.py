"""
诗词模型测试验证脚本
用于测试训练好的诗词生成模型
@Author xiaomin.zhang
"""

import torch
import os
import json
from config_poetry import config_poetry
from models import GPTModel
from utils import Tokenizer
from tqdm import tqdm


class PoetryTester:
    """诗词模型测试器"""
    
    def __init__(self, checkpoint_path, config):
        """
        初始化测试器
        
        Args:
            checkpoint_path: 模型检查点路径
            config: 配置对象
        """
        self.config = config
        self.device = torch.device(config.device)
        
        print("=" * 60)
        print("诗词模型测试器")
        print("=" * 60)
        
        # 1. 加载分词器
        print("\n1. 加载分词器...")
        self.tokenizer = Tokenizer(vocab_size=config.vocab_size)
        self.tokenizer.load(config.vocab_path)
        print(f"词表大小: {len(self.tokenizer.word2idx)}")
        
        # 2. 加载模型
        print("\n2. 加载模型...")
        self.model = GPTModel(config)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功: {checkpoint_path}")
        if 'step' in checkpoint:
            print(f"训练步数: {checkpoint['step']}")
        if 'loss' in checkpoint:
            print(f"损失: {checkpoint['loss']:.4f}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    def generate(self, prompt, max_length=64, temperature=0.8, top_k=40, top_p=0.85, num_samples=1):
        """
        生成诗词
        
        Args:
            prompt: 提示文本（诗词开头）
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P采样
            num_samples: 生成样本数
            
        Returns:
            生成的诗词列表
        """
        results = []
        
        for i in range(num_samples):
            # 编码输入
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            
            # 解码
            generated_ids = generated_ids[0].tolist()
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            results.append(generated_text)
        
        return results
    
    def test_single_generation(self, prompt, **kwargs):
        """测试单个生成"""
        print("\n" + "=" * 60)
        print(f"提示: {prompt}")
        print("-" * 60)
        
        results = self.generate(prompt, **kwargs)
        
        for i, text in enumerate(results, 1):
            if kwargs.get('num_samples', 1) > 1:
                print(f"\n样本 {i}:")
            print(text)
        
        print("=" * 60)
        
        return results
    
    def test_multiple_prompts(self, prompts, **kwargs):
        """测试多个提示"""
        print("\n" + "=" * 60)
        print("批量测试")
        print("=" * 60)
        
        all_results = []
        for prompt in prompts:
            print(f"\n提示: {prompt}")
            print("-" * 60)
            
            results = self.generate(prompt, num_samples=1, **kwargs)
            print(results[0])
            
            all_results.append({
                'prompt': prompt,
                'generated': results[0]
            })
        
        return all_results
    
    def test_temperature_effect(self, prompt, temperatures=[0.5, 0.8, 1.0, 1.2]):
        """测试温度参数的效果"""
        print("\n" + "=" * 60)
        print("温度参数测试")
        print("=" * 60)
        print(f"提示: {prompt}")
        print("=" * 60)
        
        results = []
        for temp in temperatures:
            print(f"\n温度 = {temp}")
            print("-" * 60)
            
            generated = self.generate(prompt, temperature=temp, num_samples=1)[0]
            print(generated)
            
            results.append({
                'temperature': temp,
                'generated': generated
            })
        
        return results
    
    def test_continuation(self, test_data_path, num_samples=10):
        """
        测试诗词续写能力
        从测试集中随机选择诗词，用前半部分作为提示，生成后半部分
        """
        print("\n" + "=" * 60)
        print("诗词续写测试")
        print("=" * 60)
        
        # 读取测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_poems = f.readlines()
        
        # 随机选择样本
        import random
        random.seed(42)
        selected_poems = random.sample(test_poems, min(num_samples, len(test_poems)))
        
        results = []
        for i, poem in enumerate(selected_poems, 1):
            poem = poem.strip()
            
            # 取前半部分作为提示
            mid_point = len(poem) // 2
            prompt = poem[:mid_point]
            ground_truth = poem
            
            print(f"\n样本 {i}:")
            print(f"提示: {prompt}")
            print(f"原文: {ground_truth}")
            print("-" * 60)
            
            # 生成
            generated = self.generate(prompt, max_length=len(poem)-mid_point+10, num_samples=1)[0]
            print(f"生成: {generated}")
            
            results.append({
                'prompt': prompt,
                'ground_truth': ground_truth,
                'generated': generated
            })
        
        return results
    
    def evaluate_perplexity(self, test_data_path, max_samples=100):
        """
        计算测试集上的困惑度（Perplexity）
        
        Args:
            test_data_path: 测试数据路径
            max_samples: 最大测试样本数
            
        Returns:
            平均困惑度
        """
        print("\n" + "=" * 60)
        print("困惑度评估")
        print("=" * 60)
        
        # 读取测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_poems = f.readlines()[:max_samples]
        
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for poem in tqdm(test_poems, desc="计算困惑度"):
                poem = poem.strip()
                
                # 编码
                token_ids = self.tokenizer.encode(poem, add_special_tokens=True)
                
                # 跳过太短或太长的序列
                if len(token_ids) < 5 or len(token_ids) > self.config.max_seq_len:
                    continue
                
                # 准备输入和目标
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long).to(self.device)
                targets = torch.tensor([token_ids[1:]], dtype=torch.long).to(self.device)
                
                # 前向传播
                logits, loss = self.model(input_ids, targets)
                
                if loss is not None:
                    total_loss += loss.item() * len(token_ids)
                    total_tokens += len(token_ids)
        
        # 计算平均损失和困惑度
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"\n测试样本数: {len(test_poems)}")
        print(f"平均损失: {avg_loss:.4f}")
        print(f"困惑度: {perplexity:.2f}")
        
        return perplexity
    
    def interactive_mode(self):
        """交互式生成模式"""
        print("\n" + "=" * 60)
        print("交互式诗词生成")
        print("=" * 60)
        print("\n输入诗词开头，模型将为您续写")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'config' 查看当前配置")
        print("=" * 60)
        
        # 默认参数
        temperature = 0.8
        top_k = 40
        top_p = 0.85
        max_length = 64
        
        while True:
            try:
                prompt = input("\n请输入诗词开头: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("退出交互模式")
                    break
                
                if prompt.lower() == 'config':
                    print(f"\n当前配置:")
                    print(f"  temperature: {temperature}")
                    print(f"  top_k: {top_k}")
                    print(f"  top_p: {top_p}")
                    print(f"  max_length: {max_length}")
                    continue
                
                if not prompt:
                    print("请输入有效的诗词开头")
                    continue
                
                # 生成
                print("\n生成中...")
                results = self.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=3  # 生成3个样本供选择
                )
                
                print("\n生成结果:")
                print("=" * 60)
                for i, text in enumerate(results, 1):
                    print(f"\n样本 {i}:")
                    print(text)
                    print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n退出交互模式")
                break
            except Exception as e:
                print(f"\n错误: {e}")
                continue


def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='诗词模型测试')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_poetry/best_model.pt',
                        help='模型检查点路径')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'single', 'batch', 'temperature', 'continuation', 'perplexity', 'interactive'],
                        help='测试模式')
    parser.add_argument('--prompt', type=str, default='春江潮水连海平',
                        help='测试提示（用于single和temperature模式）')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='采样温度')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-K采样')
    parser.add_argument('--top_p', type=float, default=0.85,
                        help='Top-P采样')
    parser.add_argument('--max_length', type=int, default=64,
                        help='最大生成长度')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = PoetryTester(args.checkpoint, config_poetry)
    
    # 根据模式执行测试
    if args.mode == 'interactive':
        # 交互式模式
        tester.interactive_mode()
    
    elif args.mode == 'single':
        # 单个生成测试
        tester.test_single_generation(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=3
        )
    
    elif args.mode == 'batch':
        # 批量测试
        prompts = [
            "春江潮水连海平",
            "明月几时有",
            "床前明月光",
            "白日依山尽",
            "独在异乡为异客",
            "空山新雨后",
            "大漠孤烟直",
            "飞流直下三千尺"
        ]
        tester.test_multiple_prompts(prompts, max_length=args.max_length)
    
    elif args.mode == 'temperature':
        # 温度测试
        tester.test_temperature_effect(args.prompt)
    
    elif args.mode == 'continuation':
        # 续写测试
        test_data_path = "data/processed/test.txt"
        if os.path.exists(test_data_path):
            tester.test_continuation(test_data_path, num_samples=10)
        else:
            print(f"测试数据不存在: {test_data_path}")
    
    elif args.mode == 'perplexity':
        # 困惑度评估
        test_data_path = "data/processed/test.txt"
        if os.path.exists(test_data_path):
            tester.evaluate_perplexity(test_data_path, max_samples=100)
        else:
            print(f"测试数据不存在: {test_data_path}")
    
    elif args.mode == 'all':
        # 完整测试
        print("\n" + "=" * 60)
        print("完整测试流程")
        print("=" * 60)
        
        # 1. 单个生成
        print("\n【1. 单个生成测试】")
        tester.test_single_generation(
            "春江潮水连海平",
            max_length=64,
            temperature=0.8,
            num_samples=3
        )
        
        # 2. 批量测试
        print("\n【2. 批量生成测试】")
        prompts = [
            "明月几时有",
            "床前明月光",
            "白日依山尽",
            "独在异乡为异客"
        ]
        tester.test_multiple_prompts(prompts, max_length=64)
        
        # 3. 温度测试
        print("\n【3. 温度参数测试】")
        tester.test_temperature_effect("空山新雨后")
        
        # 4. 续写测试
        print("\n【4. 诗词续写测试】")
        test_data_path = "data/processed/test.txt"
        if os.path.exists(test_data_path):
            tester.test_continuation(test_data_path, num_samples=5)
        
        # 5. 困惑度评估
        print("\n【5. 困惑度评估】")
        if os.path.exists(test_data_path):
            tester.evaluate_perplexity(test_data_path, max_samples=100)
        
        print("\n" + "=" * 60)
        print("完整测试完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()
