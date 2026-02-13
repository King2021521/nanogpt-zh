# @Author xiaomin.zhang
"""
诗词指令微调数据准备脚本
用于加载、验证、处理和分析诗词创作训练数据
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import re


class PoetryDataProcessor:
    """诗词数据处理器"""
    
    def __init__(self, data_dir: str = "data/poetry_instruction"):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """
        加载JSONL格式的数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据列表
        """
        data = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"警告: 文件 {file_path} 不存在")
            return data
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                    
        return data
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        """
        保存为JSONL格式
        
        Args:
            data: 数据列表
            file_path: 文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"已保存 {len(data)} 条数据到 {file_path}")
    
    def validate_poem_format(self, poem: str, poem_type: str) -> Tuple[bool, str]:
        """
        验证诗词格式是否符合要求
        
        Args:
            poem: 诗词内容
            poem_type: 诗词类型
            
        Returns:
            (是否有效, 错误信息)
        """
        lines = poem.strip().split('\n')
        
        # 移除标点符号后的字数
        clean_lines = [re.sub(r'[，。、；：！？]', '', line) for line in lines]
        
        if poem_type == "五言绝句":
            if len(lines) != 4:
                return False, f"五言绝句应有4句，实际有{len(lines)}句"
            for i, line in enumerate(clean_lines, 1):
                if len(line) != 5:
                    return False, f"第{i}句应为5字，实际{len(line)}字"
                    
        elif poem_type == "七言绝句":
            if len(lines) != 4:
                return False, f"七言绝句应有4句，实际有{len(lines)}句"
            for i, line in enumerate(clean_lines, 1):
                if len(line) != 7:
                    return False, f"第{i}句应为7字，实际{len(line)}字"
                    
        elif poem_type == "五言律诗":
            if len(lines) != 8:
                return False, f"五言律诗应有8句，实际有{len(lines)}句"
            for i, line in enumerate(clean_lines, 1):
                if len(line) != 5:
                    return False, f"第{i}句应为5字，实际{len(line)}字"
                    
        elif poem_type == "七言律诗":
            if len(lines) != 8:
                return False, f"七言律诗应有8句，实际有{len(lines)}句"
            for i, line in enumerate(clean_lines, 1):
                if len(line) != 7:
                    return False, f"第{i}句应为7字，实际{len(line)}字"
        
        return True, ""
    
    def validate_dataset(self, data: List[Dict]) -> Dict:
        """
        验证整个数据集
        
        Args:
            data: 数据列表
            
        Returns:
            验证结果统计
        """
        results = {
            'total': len(data),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        for i, item in enumerate(data):
            # 检查必需字段
            if 'instruction' not in item or 'output' not in item:
                results['invalid'] += 1
                results['errors'].append(f"样本 {i}: 缺少必需字段")
                continue
            
            # 检查诗词格式
            if 'metadata' in item and 'poem_type' in item['metadata']:
                poem_type = item['metadata']['poem_type']
                is_valid, error_msg = self.validate_poem_format(item['output'], poem_type)
                
                if not is_valid:
                    results['invalid'] += 1
                    results['errors'].append(f"样本 {i}: {error_msg}")
                    continue
            
            results['valid'] += 1
        
        return results
    
    def analyze_dataset(self, data: List[Dict]) -> Dict:
        """
        分析数据集统计信息
        
        Args:
            data: 数据列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_samples': len(data),
            'poem_types': Counter(),
            'styles': Counter(),
            'difficulties': Counter(),
            'themes': Counter(),
            'emotions': Counter(),
            'avg_instruction_length': 0,
            'avg_output_length': 0
        }
        
        instruction_lengths = []
        output_lengths = []
        
        for item in data:
            # 统计指令和输出长度
            instruction_lengths.append(len(item.get('instruction', '')))
            output_lengths.append(len(item.get('output', '')))
            
            # 统计元数据
            metadata = item.get('metadata', {})
            
            if 'poem_type' in metadata:
                stats['poem_types'][metadata['poem_type']] += 1
            
            if 'style' in metadata:
                stats['styles'][metadata['style']] += 1
            
            if 'difficulty' in metadata:
                stats['difficulties'][metadata['difficulty']] += 1
            
            if 'theme' in metadata:
                stats['themes'][metadata['theme']] += 1
            
            if 'emotion' in metadata:
                stats['emotions'][metadata['emotion']] += 1
        
        # 计算平均长度
        if instruction_lengths:
            stats['avg_instruction_length'] = sum(instruction_lengths) / len(instruction_lengths)
        if output_lengths:
            stats['avg_output_length'] = sum(output_lengths) / len(output_lengths)
        
        return stats
    
    def split_dataset(self, data: List[Dict], 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1,
                     seed: int = 42) -> Dict[str, List[Dict]]:
        """
        划分数据集
        
        Args:
            data: 数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
            
        Returns:
            包含train、val、test的字典
        """
        random.seed(seed)
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        n = len(data_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            'train': data_copy[:train_end],
            'val': data_copy[train_end:val_end],
            'test': data_copy[val_end:]
        }
    
    def format_for_training(self, data: List[Dict], format_type: str = "alpaca") -> List[Dict]:
        """
        将数据格式化为特定训练格式
        
        Args:
            data: 原始数据
            format_type: 格式类型 (alpaca, chatgpt, etc.)
            
        Returns:
            格式化后的数据
        """
        formatted_data = []
        
        for item in data:
            if format_type == "alpaca":
                # Alpaca格式
                formatted_item = {
                    "instruction": item['instruction'],
                    "input": item.get('input', ''),
                    "output": item['output']
                }
            elif format_type == "chatgpt":
                # ChatGPT对话格式
                formatted_item = {
                    "messages": [
                        {"role": "system", "content": "你是一个擅长创作古诗词的AI助手。"},
                        {"role": "user", "content": item['instruction'] + ('\n' + item['input'] if item.get('input') else '')},
                        {"role": "assistant", "content": item['output']}
                    ]
                }
            else:
                formatted_item = item
            
            formatted_data.append(formatted_item)
        
        return formatted_data
    
    def print_statistics(self, stats: Dict):
        """
        打印统计信息
        
        Args:
            stats: 统计信息字典
        """
        print("\n" + "="*50)
        print("数据集统计信息")
        print("="*50)
        print(f"总样本数: {stats['total_samples']}")
        print(f"平均指令长度: {stats['avg_instruction_length']:.2f} 字符")
        print(f"平均输出长度: {stats['avg_output_length']:.2f} 字符")
        
        if stats['poem_types']:
            print("\n诗词类型分布:")
            for poem_type, count in stats['poem_types'].most_common():
                print(f"  {poem_type}: {count} ({count/stats['total_samples']*100:.1f}%)")
        
        if stats['styles']:
            print("\n风格分布:")
            for style, count in stats['styles'].most_common():
                print(f"  {style}: {count} ({count/stats['total_samples']*100:.1f}%)")
        
        if stats['difficulties']:
            print("\n难度分布:")
            for difficulty, count in stats['difficulties'].most_common():
                print(f"  {difficulty}: {count} ({count/stats['total_samples']*100:.1f}%)")
        
        if stats['themes']:
            print("\n主题分布 (Top 10):")
            for theme, count in stats['themes'].most_common(10):
                print(f"  {theme}: {count}")
        
        if stats['emotions']:
            print("\n情感分布:")
            for emotion, count in stats['emotions'].most_common():
                print(f"  {emotion}: {count}")
        
        print("="*50 + "\n")


def main():
    """主函数"""
    # 初始化处理器
    processor = PoetryDataProcessor("data/poetry_instruction")
    
    print("正在加载数据集...")
    
    # 加载数据
    train_data = processor.load_jsonl(processor.data_dir / "train.jsonl")
    val_data = processor.load_jsonl(processor.data_dir / "val.jsonl")
    test_data = processor.load_jsonl(processor.data_dir / "test.jsonl")
    
    all_data = train_data + val_data + test_data
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    print(f"总计: {len(all_data)} 条")
    
    # 验证数据集
    print("\n正在验证数据集...")
    validation_results = processor.validate_dataset(all_data)
    
    print(f"有效样本: {validation_results['valid']}")
    print(f"无效样本: {validation_results['invalid']}")
    
    if validation_results['errors']:
        print("\n错误列表:")
        for error in validation_results['errors'][:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(validation_results['errors']) > 10:
            print(f"  ... 还有 {len(validation_results['errors']) - 10} 个错误")
    
    # 分析数据集
    print("\n正在分析数据集...")
    stats = processor.analyze_dataset(all_data)
    processor.print_statistics(stats)
    
    # 示例：格式化为不同训练格式
    print("生成不同格式的训练数据...")
    
    # Alpaca格式
    alpaca_train = processor.format_for_training(train_data, "alpaca")
    processor.save_jsonl(alpaca_train, processor.data_dir / "train_alpaca.jsonl")
    
    # ChatGPT格式
    chatgpt_train = processor.format_for_training(train_data, "chatgpt")
    processor.save_jsonl(chatgpt_train, processor.data_dir / "train_chatgpt.jsonl")
    
    print("\n数据处理完成！")
    print(f"数据目录: {processor.data_dir.absolute()}")


if __name__ == "__main__":
    main()
