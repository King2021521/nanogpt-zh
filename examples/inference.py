"""
推理脚本 - 文本生成
@Author xiaomin.zhang
"""

import torch
import os
from config import config
from models import GPTModel
from utils import Tokenizer


class TextGenerator:
    """
    文本生成器
    """
    def __init__(self, model_path, config):
        """
        Args:
            model_path: 模型检查点路径
            config: 配置对象
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 加载分词器
        print("加载分词器...")
        self.tokenizer = Tokenizer(vocab_size=config.vocab_size)
        self.tokenizer.load(config.vocab_path)
        
        # 创建模型
        print("创建模型...")
        self.model = GPTModel(config)
        
        # 加载模型权重
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成！")
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P采样
        
        Returns:
            generated_text: 生成的文本
        """
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
        
        return generated_text
    
    def interactive_generate(self):
        """
        交互式生成
        """
        print("\n" + "=" * 50)
        print("交互式文本生成")
        print("=" * 50)
        print("输入提示文本，模型将继续生成。")
        print("输入 'quit' 或 'exit' 退出。")
        print("=" * 50 + "\n")
        
        while True:
            # 获取用户输入
            prompt = input("请输入提示文本: ").strip()
            
            if prompt.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not prompt:
                print("提示文本不能为空！")
                continue
            
            # 生成文本
            print("\n生成中...")
            try:
                generated_text = self.generate(
                    prompt,
                    max_length=self.config.max_gen_len,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                )
                
                print("\n" + "-" * 50)
                print("生成结果:")
                print(generated_text)
                print("-" * 50 + "\n")
                
            except Exception as e:
                print(f"生成失败: {e}")


def main():
    """主函数"""
    print("=" * 50)
    print("中文GPT文本生成")
    print("=" * 50)
    
    # 检查模型文件
    model_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    
    if not os.path.exists(model_path):
        print(f"\n错误: 模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型！")
        return
    
    # 创建生成器
    generator = TextGenerator(model_path, config)
    
    # 交互式生成
    generator.interactive_generate()


if __name__ == "__main__":
    main()
