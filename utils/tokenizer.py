"""
中文分词器实现
@Author xiaomin.zhang
"""

import json
import os
from collections import Counter
import jieba


class Tokenizer:
    """
    中文分词器
    支持构建词表、编码、解码等功能
    """
    def __init__(self, vocab_size=10000):
        """
        Args:
            vocab_size: 词表大小
        """
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
        # 特殊token
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        # 特殊token的ID
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # 字符级词表标记：为 True 时按字符编码/解码（与 rebuild_vocab 字符级词表一致）
        # @Author xiaomin.zhang
        self._char_level = False
    
    def build_vocab(self, texts):
        """
        从文本列表构建词表
        
        Args:
            texts: 文本列表
        """
        print("开始构建词表...")
        
        # 统计词频
        word_counts = Counter()
        for text in texts:
            words = list(jieba.cut(text))
            word_counts.update(words)
        
        print(f"总词数: {len(word_counts)}")
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # 构建词表
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        for idx, (word, count) in enumerate(most_common):
            self.word2idx[word] = idx + len(self.special_tokens)
        
        # 构建反向映射
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"词表大小: {len(self.word2idx)}")
        print(f"覆盖率: {sum([count for word, count in most_common]) / sum(word_counts.values()) * 100:.2f}%")
    
    def encode(self, text, add_special_tokens=True):
        """
        将文本编码为token IDs
        若词表为字符级（如 rebuild_vocab 构建），则按字符切分；否则使用 jieba 分词。
        @Author xiaomin.zhang
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token（BOS和EOS）
        
        Returns:
            token_ids: token ID列表
        """
        if self._char_level:
            # 字符级词表：逐字编码，避免 jieba 切词导致大量 UNK
            units = list(text)
        else:
            units = list(jieba.cut(text))
        token_ids = [self.word2idx.get(u, self.unk_id) for u in units]
        
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        将token IDs解码为文本
        
        Args:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊token
        
        Returns:
            text: 解码后的文本
        """
        words = []
        for idx in token_ids:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if skip_special_tokens and word in self.special_tokens:
                    continue
                words.append(word)
        
        return "".join(words)
    
    def save(self, save_path):
        """
        保存词表到文件
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        vocab_data = {
            "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
            "special_tokens": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"词表已保存到: {save_path}")
    
    def load(self, load_path):
        """
        从文件加载词表
        
        Args:
            load_path: 加载路径
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data["vocab_size"]
        self.word2idx = vocab_data["word2idx"]
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
        
        special_tokens = vocab_data["special_tokens"]
        self.pad_token = special_tokens["pad_token"]
        self.unk_token = special_tokens["unk_token"]
        self.bos_token = special_tokens["bos_token"]
        self.eos_token = special_tokens["eos_token"]
        
        self.pad_id = self.word2idx[self.pad_token]
        self.unk_id = self.word2idx[self.unk_token]
        self.bos_id = self.word2idx[self.bos_token]
        self.eos_id = self.word2idx[self.eos_token]
        
        # 检测是否为字符级词表（如 scripts/rebuild_vocab.py 构建）：非特殊 token 中绝大多数为单字
        # @Author xiaomin.zhang
        non_special = [k for k in self.word2idx if k not in self.special_tokens]
        if non_special:
            single_char_ratio = sum(1 for k in non_special if len(k) == 1) / len(non_special)
            self._char_level = single_char_ratio >= 0.9
            if self._char_level:
                print(f"检测到字符级词表，已启用按字符编码/解码")
        
        print(f"词表已加载: {load_path}")
        print(f"词表大小: {len(self.word2idx)}")


if __name__ == "__main__":
    # 测试代码
    print("测试Tokenizer...")
    
    # 示例文本
    texts = [
        "今天天气很好，适合出去玩。",
        "我喜欢学习人工智能和深度学习。",
        "PyTorch是一个很好的深度学习框架。",
        "Transformer模型改变了自然语言处理领域。"
    ]
    
    # 创建分词器并构建词表
    tokenizer = Tokenizer(vocab_size=100)
    tokenizer.build_vocab(texts)
    
    # 测试编码
    test_text = "今天天气很好"
    print(f"\n原始文本: {test_text}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"编码结果: {token_ids}")
    
    # 测试解码
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码结果: {decoded_text}")
    
    # 测试保存和加载
    save_path = "data/processed/test_vocab.json"
    tokenizer.save(save_path)
    
    new_tokenizer = Tokenizer()
    new_tokenizer.load(save_path)
    
    # 验证加载的分词器
    token_ids2 = new_tokenizer.encode(test_text)
    print(f"\n加载后编码结果: {token_ids2}")
    print(f"编码结果是否一致: {token_ids == token_ids2}")
    
    print("\n✓ Tokenizer测试通过！")
