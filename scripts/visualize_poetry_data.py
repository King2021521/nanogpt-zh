# @Author xiaomin.zhang
"""
诗词数据可视化分析脚本
生成数据集的统计图表和分析报告
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
import sys

# 尝试导入可视化库（可选）
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装matplotlib，将只生成文本报告")


class PoetryDataVisualizer:
    """诗词数据可视化器"""
    
    def __init__(self, data_dir: str = "data/poetry_instruction"):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL格式的数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def generate_text_report(self, data: List[Dict], split_name: str = "all"):
        """
        生成文本分析报告
        
        Args:
            data: 数据列表
            split_name: 数据集名称
        """
        report_lines = []
        report_lines.append(f"# {split_name.upper()} 数据集分析报告")
        report_lines.append(f"生成时间: 2026-02-13")
        report_lines.append(f"样本数量: {len(data)}\n")
        
        # 统计诗词类型
        poem_types = Counter()
        styles = Counter()
        difficulties = Counter()
        themes = Counter()
        emotions = Counter()
        
        instruction_lengths = []
        output_lengths = []
        
        for item in data:
            instruction_lengths.append(len(item.get('instruction', '')))
            output_lengths.append(len(item.get('output', '')))
            
            metadata = item.get('metadata', {})
            if 'poem_type' in metadata:
                poem_types[metadata['poem_type']] += 1
            if 'style' in metadata:
                styles[metadata['style']] += 1
            if 'difficulty' in metadata:
                difficulties[metadata['difficulty']] += 1
            if 'theme' in metadata:
                themes[metadata['theme']] += 1
            if 'emotion' in metadata:
                emotions[metadata['emotion']] += 1
        
        # 长度统计
        report_lines.append("## 长度统计")
        report_lines.append(f"- 指令平均长度: {sum(instruction_lengths)/len(instruction_lengths):.2f} 字符")
        report_lines.append(f"- 指令最短: {min(instruction_lengths)} 字符")
        report_lines.append(f"- 指令最长: {max(instruction_lengths)} 字符")
        report_lines.append(f"- 输出平均长度: {sum(output_lengths)/len(output_lengths):.2f} 字符")
        report_lines.append(f"- 输出最短: {min(output_lengths)} 字符")
        report_lines.append(f"- 输出最长: {max(output_lengths)} 字符\n")
        
        # 诗词类型分布
        if poem_types:
            report_lines.append("## 诗词类型分布")
            for poem_type, count in poem_types.most_common():
                percentage = count / len(data) * 100
                report_lines.append(f"- {poem_type}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 风格分布
        if styles:
            report_lines.append("## 风格分布")
            for style, count in styles.most_common():
                percentage = count / len(data) * 100
                report_lines.append(f"- {style}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 难度分布
        if difficulties:
            report_lines.append("## 难度分布")
            for difficulty, count in difficulties.most_common():
                percentage = count / len(data) * 100
                report_lines.append(f"- {difficulty}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 主题分布
        if themes:
            report_lines.append("## 主题分布 (Top 15)")
            for theme, count in themes.most_common(15):
                percentage = count / len(data) * 100
                report_lines.append(f"- {theme}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 情感分布
        if emotions:
            report_lines.append("## 情感分布")
            for emotion, count in emotions.most_common():
                percentage = count / len(data) * 100
                report_lines.append(f"- {emotion}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 示例展示
        report_lines.append("## 数据示例")
        for i, item in enumerate(data[:3], 1):
            report_lines.append(f"\n### 示例 {i}")
            report_lines.append(f"**指令**: {item.get('instruction', '')}")
            if item.get('input'):
                report_lines.append(f"**输入**: {item.get('input', '')}")
            report_lines.append(f"**输出**:")
            report_lines.append(f"```")
            report_lines.append(item.get('output', ''))
            report_lines.append(f"```")
            if 'metadata' in item:
                report_lines.append(f"**元数据**: {json.dumps(item['metadata'], ensure_ascii=False)}")
        
        # 保存报告
        report_path = self.output_dir / f"{split_name}_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"文本报告已保存到: {report_path}")
        
        return {
            'poem_types': poem_types,
            'styles': styles,
            'difficulties': difficulties,
            'themes': themes,
            'emotions': emotions,
            'instruction_lengths': instruction_lengths,
            'output_lengths': output_lengths
        }
    
    def plot_distribution(self, counter: Counter, title: str, filename: str):
        """
        绘制分布图
        
        Args:
            counter: 计数器对象
            title: 图表标题
            filename: 保存文件名
        """
        if not HAS_MATPLOTLIB or not counter:
            return
        
        labels = [item[0] for item in counter.most_common()]
        values = [item[1] for item in counter.most_common()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(labels)), values, color='skyblue')
        plt.xlabel('类别')
        plt.ylabel('数量')
        plt.title(title)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存到: {save_path}")
    
    def plot_pie_chart(self, counter: Counter, title: str, filename: str):
        """
        绘制饼图
        
        Args:
            counter: 计数器对象
            title: 图表标题
            filename: 保存文件名
        """
        if not HAS_MATPLOTLIB or not counter:
            return
        
        labels = [item[0] for item in counter.most_common()]
        values = [item[1] for item in counter.most_common()]
        
        plt.figure(figsize=(10, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"饼图已保存到: {save_path}")
    
    def plot_length_distribution(self, lengths: List[int], title: str, filename: str):
        """
        绘制长度分布直方图
        
        Args:
            lengths: 长度列表
            title: 图表标题
            filename: 保存文件名
        """
        if not HAS_MATPLOTLIB or not lengths:
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, color='lightgreen', edgecolor='black')
        plt.xlabel('长度（字符）')
        plt.ylabel('频数')
        plt.title(title)
        plt.axvline(sum(lengths)/len(lengths), color='red', linestyle='--', 
                   label=f'平均值: {sum(lengths)/len(lengths):.1f}')
        plt.legend()
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"直方图已保存到: {save_path}")
    
    def analyze_all_splits(self):
        """分析所有数据集划分"""
        splits = ['train', 'val', 'test']
        all_stats = {}
        
        for split in splits:
            file_path = self.data_dir / f"{split}.jsonl"
            if not file_path.exists():
                print(f"警告: {file_path} 不存在，跳过")
                continue
            
            print(f"\n正在分析 {split} 数据集...")
            data = self.load_jsonl(file_path)
            stats = self.generate_text_report(data, split)
            all_stats[split] = stats
            
            # 生成图表
            if HAS_MATPLOTLIB:
                self.plot_distribution(stats['poem_types'], 
                                     f'{split.upper()} - 诗词类型分布',
                                     f'{split}_poem_types.png')
                
                self.plot_pie_chart(stats['styles'],
                                  f'{split.upper()} - 风格分布',
                                  f'{split}_styles_pie.png')
                
                self.plot_distribution(stats['difficulties'],
                                     f'{split.upper()} - 难度分布',
                                     f'{split}_difficulties.png')
                
                self.plot_length_distribution(stats['instruction_lengths'],
                                            f'{split.upper()} - 指令长度分布',
                                            f'{split}_instruction_lengths.png')
                
                self.plot_length_distribution(stats['output_lengths'],
                                            f'{split.upper()} - 输出长度分布',
                                            f'{split}_output_lengths.png')
        
        # 生成综合对比报告
        self.generate_comparison_report(all_stats)
        
        print(f"\n所有分析结果已保存到: {self.output_dir.absolute()}")
    
    def generate_comparison_report(self, all_stats: Dict):
        """
        生成数据集对比报告
        
        Args:
            all_stats: 所有数据集的统计信息
        """
        report_lines = []
        report_lines.append("# 数据集对比分析报告\n")
        
        report_lines.append("## 样本数量对比")
        report_lines.append("| 数据集 | 样本数 |")
        report_lines.append("|--------|--------|")
        
        total = 0
        for split, stats in all_stats.items():
            count = len(stats['instruction_lengths'])
            total += count
            report_lines.append(f"| {split} | {count} |")
        report_lines.append(f"| **总计** | **{total}** |\n")
        
        report_lines.append("## 平均长度对比")
        report_lines.append("| 数据集 | 指令平均长度 | 输出平均长度 |")
        report_lines.append("|--------|--------------|--------------|")
        
        for split, stats in all_stats.items():
            avg_inst = sum(stats['instruction_lengths']) / len(stats['instruction_lengths'])
            avg_out = sum(stats['output_lengths']) / len(stats['output_lengths'])
            report_lines.append(f"| {split} | {avg_inst:.2f} | {avg_out:.2f} |")
        report_lines.append("")
        
        # 保存对比报告
        report_path = self.output_dir / "comparison_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"对比报告已保存到: {report_path}")


def main():
    """主函数"""
    print("诗词数据可视化分析工具")
    print("="*50)
    
    if not HAS_MATPLOTLIB:
        print("提示: 安装matplotlib可以生成图表")
        print("运行: pip install matplotlib\n")
    
    visualizer = PoetryDataVisualizer("data/poetry_instruction")
    visualizer.analyze_all_splits()
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
