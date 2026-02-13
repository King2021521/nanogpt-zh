# @Author xiaomin.zhang
"""
诗词创作指令微调模型测试脚本
用于测试训练好的指令微调模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class PoetryGenerator:
    """诗词生成器"""
    
    def __init__(self, model_path: str, use_lora: bool = False):
        """
        初始化生成器
        
        Args:
            model_path: 模型路径
            use_lora: 是否使用LoRA权重
        """
        self.model_path = model_path
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        logger.info(f"正在加载模型: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        if self.use_lora:
            try:
                from peft import PeftModel
                
                # 加载基础模型
                base_model_path = "Qwen/Qwen2-1.5B"  # 根据实际情况修改
                logger.info(f"加载基础模型: {base_model_path}")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # 加载LoRA权重
                lora_path = f"{self.model_path}/lora_weights"
                logger.info(f"加载LoRA权重: {lora_path}")
                
                self.model = PeftModel.from_pretrained(
                    base_model,
                    lora_path,
                    torch_dtype=torch.float16
                )
                
                logger.info("LoRA模型加载完成")
            except ImportError:
                logger.warning("未安装peft库，将加载完整模型")
                self.use_lora = False
        
        if not self.use_lora:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        self.model.eval()
        logger.info("模型加载完成")
    
    def generate_poem(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        生成诗词
        
        Args:
            instruction: 指令
            input_text: 输入（可选）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            
        Returns:
            生成的诗词
        """
        # 构建提示词
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 输出:\n"
        
        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取输出部分
        if "### 输出:" in result:
            result = result.split("### 输出:")[1].strip()
        
        return result
    
    def interactive_mode(self):
        """交互式模式"""
        print("\n" + "="*60)
        print("诗词创作交互式测试")
        print("="*60)
        print("输入指令来生成诗词，输入 'quit' 或 'exit' 退出")
        print("="*60 + "\n")
        
        while True:
            try:
                instruction = input("\n请输入指令: ").strip()
                
                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("退出程序")
                    break
                
                if not instruction:
                    print("指令不能为空")
                    continue
                
                # 可选输入
                input_text = input("输入额外信息（可选，直接回车跳过）: ").strip()
                
                print("\n生成中...")
                poem = self.generate_poem(instruction, input_text)
                
                print("\n" + "-"*60)
                print("生成结果:")
                print("-"*60)
                print(poem)
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\n程序被中断")
                break
            except Exception as e:
                logger.error(f"生成失败: {e}")


def run_test_cases(generator: PoetryGenerator):
    """运行测试用例"""
    test_cases = [
        {
            "instruction": "写一首描写春天的五言绝句",
            "input": ""
        },
        {
            "instruction": "创作一首表达思乡之情的七言绝句",
            "input": ""
        },
        {
            "instruction": "以'月'为主题，写一首诗",
            "input": ""
        },
        {
            "instruction": "写一首五言绝句，要求包含'风'、'雨'两个字",
            "input": "关键词：风、雨"
        },
        {
            "instruction": "模仿李白的风格，创作一首关于饮酒的诗",
            "input": "风格：李白"
        }
    ]
    
    print("\n" + "="*60)
    print("运行测试用例")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试用例 {i}/{len(test_cases)}")
        print(f"{'='*60}")
        print(f"指令: {test_case['instruction']}")
        if test_case['input']:
            print(f"输入: {test_case['input']}")
        print("-"*60)
        
        try:
            poem = generator.generate_poem(
                test_case['instruction'],
                test_case['input']
            )
            print("生成结果:")
            print(poem)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            print(f"错误: {e}")
        
        print("-"*60)


def compare_temperatures(generator: PoetryGenerator):
    """对比不同温度参数的效果"""
    instruction = "写一首描写春天的五言绝句"
    temperatures = [0.3, 0.7, 1.0]
    
    print("\n" + "="*60)
    print("温度参数对比测试")
    print("="*60)
    print(f"指令: {instruction}\n")
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Temperature = {temp}")
        print("-"*60)
        
        try:
            poem = generator.generate_poem(
                instruction,
                temperature=temp
            )
            print(poem)
        except Exception as e:
            logger.error(f"生成失败: {e}")
        
        print("-"*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="诗词创作模型测试")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/poetry_instruction_model",
        help="模型路径"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="是否使用LoRA权重"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "interactive", "compare"],
        help="运行模式: test(测试用例), interactive(交互式), compare(参数对比)"
    )
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = PoetryGenerator(
        model_path=args.model_path,
        use_lora=args.use_lora
    )
    
    # 根据模式运行
    if args.mode == "test":
        run_test_cases(generator)
    elif args.mode == "interactive":
        generator.interactive_mode()
    elif args.mode == "compare":
        compare_temperatures(generator)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
