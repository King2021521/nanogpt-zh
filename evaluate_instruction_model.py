# @Author xiaomin.zhang
"""
诗词创作指令微调模型评估脚本
用于评估训练好的模型性能
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List, Dict
import re
from collections import defaultdict

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class PoetryEvaluator:
    """诗词评估器"""
    
    def __init__(self, model_path: str, test_file: str):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            test_file: 测试数据文件
        """
        self.model_path = model_path
        self.test_file = test_file
        
        # 加载模型
        logger.info(f"正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()
        
        # 加载测试数据
        self.test_data = self.load_test_data()
        
        logger.info("评估器初始化完成")
    
    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        data = []
        logger.info(f"正在加载测试数据: {self.test_file}")
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"成功加载 {len(data)} 条测试数据")
        return data
    
    def generate_poem(self, instruction: str, input_text: str = "") -> str:
        """生成诗词"""
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
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取输出部分
        if "### 输出:" in result:
            result = result.split("### 输出:")[1].strip()
        
        return result
    
    def validate_poem_format(self, poem: str, poem_type: str) -> bool:
        """
        验证诗词格式
        
        Args:
            poem: 诗词内容
            poem_type: 诗词类型
            
        Returns:
            是否符合格律
        """
        lines = poem.strip().split('\n')
        
        # 移除标点符号后的字数
        clean_lines = [re.sub(r'[，。、；：！？]', '', line) for line in lines]
        
        try:
            if poem_type == "五言绝句":
                return len(lines) == 4 and all(len(line) == 5 for line in clean_lines)
            elif poem_type == "七言绝句":
                return len(lines) == 4 and all(len(line) == 7 for line in clean_lines)
            elif poem_type == "五言律诗":
                return len(lines) == 8 and all(len(line) == 5 for line in clean_lines)
            elif poem_type == "七言律诗":
                return len(lines) == 8 and all(len(line) == 7 for line in clean_lines)
        except:
            return False
        
        return True
    
    def calculate_bleu(self, reference: str, generated: str) -> float:
        """
        计算BLEU分数（简化版本）
        
        Args:
            reference: 参考文本
            generated: 生成文本
            
        Returns:
            BLEU分数
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            ref_chars = list(reference)
            gen_chars = list(generated)
            
            score = sentence_bleu([ref_chars], gen_chars)
            return score
        except ImportError:
            logger.warning("未安装nltk，跳过BLEU计算")
            return 0.0
    
    def calculate_char_accuracy(self, reference: str, generated: str) -> float:
        """
        计算字符准确率
        
        Args:
            reference: 参考文本
            generated: 生成文本
            
        Returns:
            准确率
        """
        ref_chars = set(reference)
        gen_chars = set(generated)
        
        if len(ref_chars) == 0:
            return 0.0
        
        intersection = ref_chars & gen_chars
        return len(intersection) / len(ref_chars)
    
    def evaluate(self) -> Dict:
        """
        执行评估
        
        Returns:
            评估结果
        """
        logger.info("="*60)
        logger.info("开始评估")
        logger.info("="*60)
        
        results = {
            'total': len(self.test_data),
            'format_correct': 0,
            'bleu_scores': [],
            'char_accuracies': [],
            'details': []
        }
        
        for i, item in enumerate(self.test_data, 1):
            logger.info(f"\n评估样本 {i}/{len(self.test_data)}")
            
            instruction = item['instruction']
            input_text = item.get('input', '')
            reference = item['output']
            metadata = item.get('metadata', {})
            poem_type = metadata.get('poem_type', '')
            
            # 生成诗词
            try:
                generated = self.generate_poem(instruction, input_text)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                generated = ""
            
            # 验证格式
            format_correct = False
            if poem_type:
                format_correct = self.validate_poem_format(generated, poem_type)
                if format_correct:
                    results['format_correct'] += 1
            
            # 计算BLEU
            bleu_score = self.calculate_bleu(reference, generated)
            results['bleu_scores'].append(bleu_score)
            
            # 计算字符准确率
            char_accuracy = self.calculate_char_accuracy(reference, generated)
            results['char_accuracies'].append(char_accuracy)
            
            # 保存详细结果
            detail = {
                'instruction': instruction,
                'reference': reference,
                'generated': generated,
                'poem_type': poem_type,
                'format_correct': format_correct,
                'bleu_score': bleu_score,
                'char_accuracy': char_accuracy
            }
            results['details'].append(detail)
            
            # 打印结果
            print(f"\n指令: {instruction}")
            print(f"参考: {reference}")
            print(f"生成: {generated}")
            print(f"格式正确: {format_correct}")
            print(f"BLEU: {bleu_score:.4f}")
            print(f"字符准确率: {char_accuracy:.4f}")
            print("-"*60)
        
        # 计算平均指标
        results['format_accuracy'] = results['format_correct'] / results['total']
        results['avg_bleu'] = sum(results['bleu_scores']) / len(results['bleu_scores']) if results['bleu_scores'] else 0
        results['avg_char_accuracy'] = sum(results['char_accuracies']) / len(results['char_accuracies']) if results['char_accuracies'] else 0
        
        return results
    
    def print_summary(self, results: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("评估摘要")
        print("="*60)
        print(f"总样本数: {results['total']}")
        print(f"格式正确数: {results['format_correct']}")
        print(f"格式准确率: {results['format_accuracy']:.2%}")
        print(f"平均BLEU分数: {results['avg_bleu']:.4f}")
        print(f"平均字符准确率: {results['avg_char_accuracy']:.2%}")
        print("="*60)
    
    def save_results(self, results: Dict, output_file: str = "evaluation_results.json"):
        """保存评估结果"""
        logger.info(f"正在保存评估结果到: {output_file}")
        
        # 移除details以减小文件大小（可选）
        save_data = {
            'total': results['total'],
            'format_correct': results['format_correct'],
            'format_accuracy': results['format_accuracy'],
            'avg_bleu': results['avg_bleu'],
            'avg_char_accuracy': results['avg_char_accuracy'],
            'details': results['details']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info("评估结果已保存")
    
    def generate_report(self, results: Dict, output_file: str = "evaluation_report.md"):
        """生成评估报告"""
        logger.info(f"正在生成评估报告: {output_file}")
        
        lines = []
        lines.append("# 诗词创作模型评估报告\n")
        lines.append(f"**模型路径**: {self.model_path}\n")
        lines.append(f"**测试数据**: {self.test_file}\n")
        lines.append(f"**测试样本数**: {results['total']}\n")
        
        lines.append("\n## 评估指标\n")
        lines.append(f"- **格式准确率**: {results['format_accuracy']:.2%} ({results['format_correct']}/{results['total']})")
        lines.append(f"- **平均BLEU分数**: {results['avg_bleu']:.4f}")
        lines.append(f"- **平均字符准确率**: {results['avg_char_accuracy']:.2%}\n")
        
        lines.append("\n## 详细结果\n")
        
        for i, detail in enumerate(results['details'], 1):
            lines.append(f"\n### 样本 {i}\n")
            lines.append(f"**指令**: {detail['instruction']}\n")
            lines.append(f"**参考输出**:")
            lines.append("```")
            lines.append(detail['reference'])
            lines.append("```\n")
            lines.append(f"**生成输出**:")
            lines.append("```")
            lines.append(detail['generated'])
            lines.append("```\n")
            lines.append(f"**诗词类型**: {detail['poem_type']}")
            lines.append(f"**格式正确**: {'✅' if detail['format_correct'] else '❌'}")
            lines.append(f"**BLEU分数**: {detail['bleu_score']:.4f}")
            lines.append(f"**字符准确率**: {detail['char_accuracy']:.2%}\n")
            lines.append("---\n")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info("评估报告已生成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="诗词创作模型评估")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/poetry_instruction_model",
        help="模型路径"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/poetry_instruction/test.jsonl",
        help="测试数据文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/evaluation",
        help="评估结果输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化评估器
    evaluator = PoetryEvaluator(
        model_path=args.model_path,
        test_file=args.test_file
    )
    
    # 执行评估
    results = evaluator.evaluate()
    
    # 打印摘要
    evaluator.print_summary(results)
    
    # 保存结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    evaluator.save_results(results, results_file)
    
    # 生成报告
    report_file = os.path.join(args.output_dir, "evaluation_report.md")
    evaluator.generate_report(results, report_file)
    
    print("\n" + "="*60)
    print("评估完成！")
    print(f"结果已保存到: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
