"""
宋词数据清洗与合并脚本
将songci目录下的JSON数据清洗后与现有训练数据合并
@Author xiaomin.zhang
"""

import json
import os
import re
from collections import Counter

def clean_text(text):
    """
    清洗文本数据
    @Author xiaomin.zhang
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空格和换行
    text = text.strip()
    
    # 移除括号内的注释（如：(明明 一作：佼佼)）
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\（[^）]*\）', '', text)
    
    # 移除特殊标记
    text = re.sub(r'<[^>]*>', '', text)
    
    # 统一标点符号（全角）
    text = text.replace('(', '（').replace(')', '）')
    text = text.replace(',', '，').replace('.', '。')
    text = text.replace('!', '！').replace('?', '？')
    text = text.replace(':', '：').replace(';', '；')
    
    # 移除多余空格
    text = re.sub(r'\s+', '', text)
    
    return text

def is_valid_poetry(text, min_length=10, max_length=500):
    """
    验证诗词是否有效
    @Author xiaomin.zhang
    
    Args:
        text: 诗词文本
        min_length: 最小长度
        max_length: 最大长度
    
    Returns:
        是否有效
    """
    if not text:
        return False
    
    # 长度检查
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # 检查是否包含中文字符
    if not re.search(r'[\u4e00-\u9fff]', text):
        return False
    
    # 检查是否有过多的标点符号（超过50%）
    punct_count = len(re.findall(r'[，。、；：？！,.]', text))
    if punct_count / len(text) > 0.5:
        return False
    
    return True

def process_songci_format1(file_path):
    """
    处理格式1: ci.song.*.json (author, paragraphs, rhythmic)
    @Author xiaomin.zhang
    """
    print(f"处理文件: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    poems = []
    for item in data:
        if 'paragraphs' in item and item['paragraphs']:
            # 合并所有段落
            text = ''.join(item['paragraphs'])
            text = clean_text(text)
            
            if is_valid_poetry(text):
                poems.append(text)
    
    print(f"  提取有效诗词: {len(poems)} 首")
    return poems

def process_songci_format2(file_path):
    """
    处理格式2: poetrys.json (title, author, paragraphs, rhythmic, notes)
    @Author xiaomin.zhang
    """
    print(f"处理文件: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    poems = []
    for item in data:
        if 'paragraphs' in item and item['paragraphs']:
            # 合并所有段落
            text = ''.join(item['paragraphs'])
            text = clean_text(text)
            
            if is_valid_poetry(text):
                poems.append(text)
    
    print(f"  提取有效诗词: {len(poems)} 首")
    return poems

def process_songci_format3(file_path):
    """
    处理格式3: caocao.json (title, paragraphs)
    @Author xiaomin.zhang
    """
    print(f"处理文件: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    poems = []
    for item in data:
        if 'paragraphs' in item and item['paragraphs']:
            # 合并所有段落
            text = ''.join(item['paragraphs'])
            text = clean_text(text)
            
            if is_valid_poetry(text):
                poems.append(text)
    
    print(f"  提取有效诗词: {len(poems)} 首")
    return poems

def remove_duplicates(poems):
    """
    去除重复的诗词
    @Author xiaomin.zhang
    """
    print("\n去重处理...")
    original_count = len(poems)
    
    # 使用集合去重
    unique_poems = list(set(poems))
    
    # 按长度排序（可选）
    unique_poems.sort(key=len)
    
    removed = original_count - len(unique_poems)
    print(f"  原始数量: {original_count}")
    print(f"  去重后: {len(unique_poems)}")
    print(f"  移除重复: {removed}")
    
    return unique_poems

def load_existing_data(file_path):
    """
    加载现有的训练数据
    @Author xiaomin.zhang
    """
    print(f"\n加载现有数据: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"  文件不存在，将创建新文件")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        poems = [line.strip() for line in f if line.strip()]
    
    print(f"  现有诗词数量: {len(poems)}")
    return poems

def save_data(poems, file_path):
    """
    保存数据到文件
    @Author xiaomin.zhang
    """
    print(f"\n保存数据到: {file_path}")
    
    # 创建目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for poem in poems:
            f.write(poem + '\n')
    
    print(f"  保存完成: {len(poems)} 首诗词")

def analyze_data(poems):
    """
    分析数据统计信息
    @Author xiaomin.zhang
    """
    print("\n" + "=" * 60)
    print("数据统计分析")
    print("=" * 60)
    
    # 基本统计
    total_count = len(poems)
    total_chars = sum(len(p) for p in poems)
    avg_length = total_chars / total_count if total_count > 0 else 0
    
    print(f"\n总诗词数量: {total_count:,}")
    print(f"总字符数: {total_chars:,}")
    print(f"平均长度: {avg_length:.1f} 字符/首")
    
    # 长度分布
    lengths = [len(p) for p in poems]
    print(f"\n长度分布:")
    print(f"  最短: {min(lengths)} 字符")
    print(f"  最长: {max(lengths)} 字符")
    print(f"  中位数: {sorted(lengths)[len(lengths)//2]} 字符")
    
    # 长度区间统计
    length_ranges = {
        '0-20': 0,
        '21-40': 0,
        '41-60': 0,
        '61-80': 0,
        '81-100': 0,
        '100+': 0
    }
    
    for length in lengths:
        if length <= 20:
            length_ranges['0-20'] += 1
        elif length <= 40:
            length_ranges['21-40'] += 1
        elif length <= 60:
            length_ranges['41-60'] += 1
        elif length <= 80:
            length_ranges['61-80'] += 1
        elif length <= 100:
            length_ranges['81-100'] += 1
        else:
            length_ranges['100+'] += 1
    
    print(f"\n长度区间分布:")
    for range_name, count in length_ranges.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"  {range_name:10s}: {count:6,} ({percentage:5.1f}%)")
    
    # 字符统计
    all_chars = ''.join(poems)
    char_counter = Counter(all_chars)
    
    print(f"\n字符统计:")
    print(f"  唯一字符数: {len(char_counter)}")
    print(f"  最常见字符 (Top 20):")
    for char, count in char_counter.most_common(20):
        print(f"    '{char}': {count:,}")

def main():
    """主函数"""
    print("=" * 60)
    print("宋词数据清洗与合并")
    print("=" * 60)
    
    # 定义路径（使用绝对路径）
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    songci_dir = os.path.join(script_dir, "data", "songci")
    processed_dir = os.path.join(script_dir, "data", "processed")
    train_file = os.path.join(processed_dir, "train.txt")
    val_file = os.path.join(processed_dir, "val.txt")
    test_file = os.path.join(processed_dir, "test.txt")
    
    # 处理所有宋词文件
    all_new_poems = []
    
    print("\n" + "=" * 60)
    print("第1步: 处理宋词JSON文件")
    print("=" * 60)
    
    # 处理 ci.song.*.json 文件
    for filename in os.listdir(songci_dir):
        if filename.startswith('ci.song.') and filename.endswith('.json'):
            file_path = os.path.join(songci_dir, filename)
            poems = process_songci_format1(file_path)
            all_new_poems.extend(poems)
    
    # 处理 poetrys.json
    poetrys_file = os.path.join(songci_dir, "poetrys.json")
    if os.path.exists(poetrys_file):
        poems = process_songci_format2(poetrys_file)
        all_new_poems.extend(poems)
    
    # 处理 caocao.json
    caocao_file = os.path.join(songci_dir, "caocao.json")
    if os.path.exists(caocao_file):
        poems = process_songci_format3(caocao_file)
        all_new_poems.extend(poems)
    
    print(f"\n宋词数据提取完成，共 {len(all_new_poems)} 首")
    
    # 去重
    print("\n" + "=" * 60)
    print("第2步: 去重处理")
    print("=" * 60)
    all_new_poems = remove_duplicates(all_new_poems)
    
    # 加载现有数据
    print("\n" + "=" * 60)
    print("第3步: 加载现有训练数据")
    print("=" * 60)
    existing_train = load_existing_data(train_file)
    existing_val = load_existing_data(val_file)
    existing_test = load_existing_data(test_file)
    
    # 合并所有现有数据
    all_existing = existing_train + existing_val + existing_test
    print(f"\n现有数据总量: {len(all_existing)}")
    
    # 与现有数据去重
    print("\n" + "=" * 60)
    print("第4步: 与现有数据去重")
    print("=" * 60)
    
    existing_set = set(all_existing)
    new_poems_only = [p for p in all_new_poems if p not in existing_set]
    
    print(f"  宋词数据: {len(all_new_poems)}")
    print(f"  与现有数据重复: {len(all_new_poems) - len(new_poems_only)}")
    print(f"  新增唯一诗词: {len(new_poems_only)}")
    
    # 合并数据
    print("\n" + "=" * 60)
    print("第5步: 合并数据")
    print("=" * 60)
    
    all_poems = all_existing + new_poems_only
    print(f"  合并后总量: {len(all_poems)}")
    
    # 最终去重（确保没有重复）
    all_poems = list(set(all_poems))
    print(f"  最终去重后: {len(all_poems)}")
    
    # 重新划分训练集、验证集、测试集
    print("\n" + "=" * 60)
    print("第6步: 重新划分数据集")
    print("=" * 60)
    
    import random
    random.seed(42)
    random.shuffle(all_poems)
    
    # 划分比例: 95% 训练, 2.5% 验证, 2.5% 测试
    total = len(all_poems)
    train_size = int(total * 0.95)
    val_size = int(total * 0.025)
    
    new_train = all_poems[:train_size]
    new_val = all_poems[train_size:train_size + val_size]
    new_test = all_poems[train_size + val_size:]
    
    print(f"  训练集: {len(new_train):,} ({len(new_train)/total*100:.1f}%)")
    print(f"  验证集: {len(new_val):,} ({len(new_val)/total*100:.1f}%)")
    print(f"  测试集: {len(new_test):,} ({len(new_test)/total*100:.1f}%)")
    
    # 备份原始数据
    print("\n" + "=" * 60)
    print("第7步: 备份原始数据")
    print("=" * 60)
    
    import shutil
    from datetime import datetime
    
    backup_dir = os.path.join(processed_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(backup_dir, exist_ok=True)
    
    if os.path.exists(train_file):
        shutil.copy(train_file, os.path.join(backup_dir, "train.txt"))
        print(f"  备份训练集到: {backup_dir}")
    if os.path.exists(val_file):
        shutil.copy(val_file, os.path.join(backup_dir, "val.txt"))
        print(f"  备份验证集到: {backup_dir}")
    if os.path.exists(test_file):
        shutil.copy(test_file, os.path.join(backup_dir, "test.txt"))
        print(f"  备份测试集到: {backup_dir}")
    
    # 保存新数据
    print("\n" + "=" * 60)
    print("第8步: 保存合并后的数据")
    print("=" * 60)
    
    save_data(new_train, train_file)
    save_data(new_val, val_file)
    save_data(new_test, test_file)
    
    # 数据分析
    analyze_data(all_poems)
    
    # 显示样本
    print("\n" + "=" * 60)
    print("数据样本 (随机抽取10首)")
    print("=" * 60)
    
    samples = random.sample(all_poems, min(10, len(all_poems)))
    for i, poem in enumerate(samples, 1):
        print(f"\n样本 {i} (长度: {len(poem)}):")
        print(f"  {poem[:100]}{'...' if len(poem) > 100 else ''}")
    
    print("\n" + "=" * 60)
    print("数据合并完成！")
    print("=" * 60)
    print(f"\n新增诗词: {len(new_poems_only):,} 首")
    print(f"总诗词数: {len(all_poems):,} 首")
    print(f"增长率: {len(new_poems_only)/len(all_existing)*100:.1f}%")
    
    print("\n下一步:")
    print("  1. 重新构建词表: python scripts/prepare_poetry_data.py")
    print("  2. 开始训练: python train_poetry.py")

if __name__ == "__main__":
    main()
