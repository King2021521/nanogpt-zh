"""
GPT模型实现
@Author xiaomin.zhang
"""

import torch
import torch.nn as nn
from .embedding import TokenEmbedding, PositionalEncoding
from .transformer import TransformerBlock


class GPTModel(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) 模型
    用于自回归文本生成
    """
    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含所有超参数
        """
        super().__init__()
        self.config = config
        
        # Token嵌入层
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.d_model, 
            config.max_seq_len, 
            config.dropout
        )
        
        # Transformer解码器层堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # 最后的Layer Normalization
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # 输出层（语言模型头）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享：Token嵌入和输出层共享权重（可选，但能减少参数量）
        self.lm_head.weight = self.token_embedding.embedding.weight
        
        # 初始化参数
        self.apply(self._init_weights)
        
        # 打印模型信息
        print(f"GPT模型初始化完成")
        print(f"参数量: {self.get_num_params() / 1e6:.2f}M")
    
    def _init_weights(self, module):
        """
        初始化模型权重
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_num_params(self):
        """
        获取模型参数量
        """
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_ids, targets=None):
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            targets: (batch_size, seq_len) 目标token IDs（训练时使用）
        
        Returns:
            如果targets为None（推理模式）:
                logits: (batch_size, seq_len, vocab_size) 预测的logits
            如果targets不为None（训练模式）:
                logits: (batch_size, seq_len, vocab_size)
                loss: 交叉熵损失
        """
        batch_size, seq_len = input_ids.size()
        
        # Token嵌入
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = self.pos_encoding(x)  # (batch_size, seq_len, d_model)
        
        # 通过Transformer层
        for block in self.transformer_blocks:
            x = block(x)  # (batch_size, seq_len, d_model)
        
        # 最后的Layer Normalization
        x = self.ln_f(x)  # (batch_size, seq_len, d_model)
        
        # 输出层：映射到词表
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # 如果提供了targets，计算损失
        loss = None
        if targets is not None:
            # 重塑logits和targets以计算交叉熵损失
            # logits: (batch_size * seq_len, vocab_size)
            # targets: (batch_size * seq_len)
            # ignore_index 需与 tokenizer.pad_id 一致（通常为 0），避免对 padding 位置计算损失
            # @Author xiaomin.zhang
            pad_id = getattr(self.config, 'pad_id', 0)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_id
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, 
                 eos_token_id=None, suppress_eos_steps=None, suppress_punct_tokens=None):
        """
        自回归生成文本
        @Author xiaomin.zhang
        
        Args:
            input_ids: (batch_size, seq_len) 初始token IDs
            max_new_tokens: 最大生成token数量
            temperature: 采样温度（越高越随机）
            top_k: Top-K采样
            top_p: Top-P (nucleus) 采样
            eos_token_id: EOS token的ID，如果提供则会降低其概率避免过早结束
            suppress_eos_steps: 在前N步抑制EOS token，默认为max_new_tokens-5
            suppress_punct_tokens: 标点符号token ID列表，用于抑制连续标点符号
        
        Returns:
            generated: (batch_size, seq_len + max_new_tokens) 生成的token IDs
        """
        self.eval()  # 设置为评估模式
        
        # 如果没有指定suppress_eos_steps，默认在前90%的步数中抑制EOS
        if suppress_eos_steps is None:
            suppress_eos_steps = max(int(max_new_tokens * 0.9), max_new_tokens - 5)
        
        # 记录最近生成的token，用于检测连续标点符号
        recent_tokens = []
        
        for step in range(max_new_tokens):
            # 如果序列太长，截断到max_seq_len
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else \
                            input_ids[:, -self.config.max_seq_len:]
            
            # 前向传播
            logits, _ = self(input_ids_cond)
            
            # 只取最后一个位置的logits
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # 应用温度
            logits = logits / temperature
            
            # 在生成前期大幅降低EOS token的概率，避免过早结束
            if eos_token_id is not None and step < suppress_eos_steps:
                logits[:, eos_token_id] = logits[:, eos_token_id] - 10.0  # 大幅降低EOS概率
            
            # 抑制连续标点符号：如果最近生成的2-3个token都是标点，大幅降低标点概率
            if suppress_punct_tokens is not None and len(recent_tokens) >= 2:
                # 检查最近2个token是否都是标点
                recent_punct_count = sum(1 for t in recent_tokens[-2:] if t in suppress_punct_tokens)
                if recent_punct_count >= 2:
                    # 如果连续2个都是标点，大幅降低所有标点的概率
                    for punct_id in suppress_punct_tokens:
                        logits[:, punct_id] = logits[:, punct_id] - 15.0
                elif recent_punct_count >= 1:
                    # 如果最近有1个标点，适度降低标点概率
                    for punct_id in suppress_punct_tokens:
                        logits[:, punct_id] = logits[:, punct_id] - 5.0
            
            # Top-K采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-P (nucleus) 采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 计算概率分布
            probs = torch.softmax(logits, dim=-1)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # 记录最近生成的token（用于下一轮检测）
            next_token_id = next_token.item()
            recent_tokens.append(next_token_id)
            if len(recent_tokens) > 5:  # 只保留最近5个token
                recent_tokens.pop(0)
            
            # 拼接到序列
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


if __name__ == "__main__":
    # 测试代码
    from config import config
    
    print("测试GPT模型...")
    print(config)
    
    # 创建模型
    model = GPTModel(config)
    
    # 创建随机输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n输入shape: {input_ids.shape}")
    
    # 测试前向传播（训练模式）
    logits, loss = model(input_ids, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 测试生成（推理模式）
    print("\n测试文本生成...")
    generated = model.generate(input_ids, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"生成的序列shape: {generated.shape}")
    
    print("\n✓ GPT模型测试通过！")
