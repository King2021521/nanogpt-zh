"""
NanoGPT-ZH 配置文件
定义模型和训练的所有超参数

@Author xiaomin.zhang
"""

class Config:
    """
    NanoGPT-ZH 模型和训练配置
    
    模型规模: ~7.3M 参数
    架构: Decoder-Only Transformer (GPT风格)
    """
    
    # ============ 项目信息 ============
    project_name = "NanoGPT-ZH"
    version = "1.0.0"
    description = "A Tiny GPT Implementation from Scratch for Chinese"
    
    # ============ 模型架构参数 ============
    vocab_size = 10000          # 词表大小
    max_seq_len = 256           # 最大序列长度
    d_model = 256               # 模型维度（嵌入维度）
    n_layers = 6                # Transformer层数
    n_heads = 8                 # 注意力头数
    d_ff = 1024                 # 前馈网络维度（通常是d_model的4倍）
    dropout = 0.1               # Dropout概率
    
    # ============ 训练参数 ============
    batch_size = 32             # 批次大小
    learning_rate = 3e-4        # 学习率
    weight_decay = 0.01         # 权重衰减
    max_epochs = 10             # 最大训练轮数
    warmup_steps = 1000         # 学习率预热步数
    max_steps = 50000           # 最大训练步数
    
    # ============ 优化器参数 ============
    betas = (0.9, 0.95)         # Adam优化器的beta参数
    grad_clip = 1.0             # 梯度裁剪阈值
    
    # ============ 数据参数 ============
    train_data_path = "data/processed/train.txt"
    val_data_path = "data/processed/val.txt"
    vocab_path = "data/processed/vocab.json"
    
    # ============ 检查点和日志 ============
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    save_interval = 1000        # 每N步保存一次模型
    eval_interval = 500         # 每N步评估一次
    log_interval = 100          # 每N步记录一次日志
    
    # ============ 推理参数 ============
    temperature = 1.0           # 采样温度
    top_k = 50                  # Top-K采样
    top_p = 0.9                 # Top-P (nucleus) 采样
    max_gen_len = 100           # 最大生成长度
    
    # ============ 特殊token ============
    pad_token = "<PAD>"
    unk_token = "<UNK>"
    bos_token = "<BOS>"         # Begin of sequence
    eos_token = "<EOS>"         # End of sequence
    
    @property
    def device(self):
        """自动检测设备"""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def __repr__(self):
        """打印配置信息"""
        config_str = "=" * 50 + "\n"
        config_str += f"{self.project_name} v{self.version}\n"
        config_str += f"{self.description}\n"
        config_str += "=" * 50 + "\n"
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config_str += f"{key:20s}: {value}\n"
        config_str += "=" * 50
        return config_str


# 创建全局配置实例
config = Config()


if __name__ == "__main__":
    # 测试配置
    print(config)
    print(f"\n使用设备: {config.device}")
