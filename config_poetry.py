"""
NanoGPT-ZH 诗词训练配置文件
针对中文古诗词数据集优化的超参数配置
@Author xiaomin.zhang
"""

class PoetryConfig:
    """
    NanoGPT-ZH 诗词模型配置 - RTX 3080优化版本
    
    数据集规模（更新后，含宋词）:
    - 训练样本: 372,911 条 (+5,015)
    - 验证样本: 9,813 条
    - 测试样本: 9,814 条
    - 总字符数: ~1471万字符 (+106万)
    - 平均长度: 37.5字符/样本
    - 唯一字符: 9,833
    
    模型规模: ~50M 参数（理想配置）
    架构: Decoder-Only Transformer (GPT风格)
    硬件: RTX 3080 (10GB VRAM)
    """
    
    # ============ 项目信息 ============
    project_name = "NanoGPT-ZH-Poetry"
    version = "1.0.0"
    description = "Chinese Poetry Generation Model"
    
    # ============ 模型架构参数 ============
    # 针对诗词生成任务优化 - 50M参数规模（RTX 3080）
    vocab_size = 10000          # 词表大小（与预处理保持一致）
    max_seq_len = 128           # 最大序列长度（增加以支持长诗词）
    d_model = 512               # 模型维度（增大至768以提升表达能力）
    n_layers = 8               # Transformer层数（12层深度网络）
    n_heads = 8                # 注意力头数（与d_model匹配）
    d_ff = 2048                 # 前馈网络维度（d_model的4倍）
    dropout = 0.1               # Dropout概率
    
    # ============ 训练参数 ============
    # 基于36.7万样本的训练策略 - RTX 3080优化配置
    batch_size = 48             # 批次大小（3080显存充足，增大批次）
    learning_rate = 1e-4        # 学习率（标准GPT学习率）
    weight_decay = 0.01         # 权重衰减
    max_epochs = 25             # 最大训练轮数（充分训练）
    warmup_steps = 2000         # 学习率预热步数（更平滑的预热）
    max_steps = 300000          # 最大训练步数（充分训练以达到最佳效果）
    
    # ============ 优化器参数 ============
    betas = (0.9, 0.95)         # Adam优化器的beta参数
    grad_clip = 1.0             # 梯度裁剪阈值
    
    # ============ 数据参数 ============
    train_data_path = "data/processed/train.txt"
    val_data_path = "data/processed/val.txt"
    vocab_path = "data/processed/vocab.json"
    
    # ============ 检查点和日志 ============
    checkpoint_dir = "checkpoints_poetry"
    log_dir = "logs_poetry"
    save_interval = 5000        # 每5000步保存一次模型（减少IO开销）
    eval_interval = 2000        # 每2000步评估一次
    log_interval = 500          # 每500步记录一次日志
    
    # ============ 推理参数 ============
    # 针对诗词生成优化
    temperature = 1.0           # 采样温度（平衡多样性和连贯性）
    top_k = 80                  # Top-K采样（增加多样性）
    top_p = 0.9                 # Top-P采样（标准设置）
    max_gen_len = 128           # 最大生成长度（支持长诗词）
    
    # ============ 特殊token ============
    pad_token = "<PAD>"
    pad_id = 0                  # 与 Tokenizer 一致，损失计算时忽略 padding
    unk_token = "<UNK>"
    bos_token = "<BOS>"         # Begin of sequence
    eos_token = "<EOS>"         # End of sequence
    
    @property
    def device(self):
        """自动检测设备"""
        import torch
        # RTX 3080服务器，使用CUDA加速
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_training_stats(self):
        """计算训练统计信息"""
        # 训练样本数（更新后）
        train_samples = 372911
        
        # 每个epoch的步数
        steps_per_epoch = train_samples // self.batch_size
        
        # 总步数（取max_steps和max_epochs * steps_per_epoch的最小值）
        total_steps = min(self.max_steps, self.max_epochs * steps_per_epoch)
        
        # 预计训练时间（假设每步0.5秒）
        estimated_time_hours = (total_steps * 0.5) / 3600
        
        return {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "estimated_time_hours": estimated_time_hours,
            "checkpoints_saved": total_steps // self.save_interval,
            "evaluations": total_steps // self.eval_interval
        }
    
    def __repr__(self):
        """打印配置信息"""
        config_str = "=" * 60 + "\n"
        config_str += f"{self.project_name} v{self.version}\n"
        config_str += f"{self.description}\n"
        config_str += "=" * 60 + "\n\n"
        
        config_str += "【模型架构】\n"
        config_str += f"  词表大小:        {self.vocab_size:,}\n"
        config_str += f"  最大序列长度:    {self.max_seq_len}\n"
        config_str += f"  模型维度:        {self.d_model}\n"
        config_str += f"  Transformer层数: {self.n_layers}\n"
        config_str += f"  注意力头数:      {self.n_heads}\n"
        config_str += f"  前馈网络维度:    {self.d_ff}\n"
        
        # 计算参数量
        params = (
            self.vocab_size * self.d_model +  # 词嵌入
            self.max_seq_len * self.d_model +  # 位置编码
            self.n_layers * (
                4 * self.d_model * self.d_model +  # 注意力层
                2 * self.d_model * self.d_ff  # 前馈层
            ) +
            self.vocab_size * self.d_model  # 输出层
        )
        config_str += f"  预计参数量:      {params/1e6:.2f}M\n"
        
        config_str += "\n【训练参数】\n"
        config_str += f"  批次大小:        {self.batch_size}\n"
        config_str += f"  学习率:          {self.learning_rate}\n"
        config_str += f"  最大轮数:        {self.max_epochs}\n"
        config_str += f"  最大步数:        {self.max_steps:,}\n"
        config_str += f"  预热步数:        {self.warmup_steps:,}\n"
        
        stats = self.get_training_stats()
        config_str += "\n【训练统计】\n"
        config_str += f"  每轮步数:        {stats['steps_per_epoch']:,}\n"
        config_str += f"  预计总步数:      {stats['total_steps']:,}\n"
        config_str += f"  预计训练时间:    {stats['estimated_time_hours']:.1f} 小时\n"
        config_str += f"  保存检查点次数:  {stats['checkpoints_saved']}\n"
        config_str += f"  评估次数:        {stats['evaluations']}\n"
        
        config_str += "\n【数据路径】\n"
        config_str += f"  训练数据:        {self.train_data_path}\n"
        config_str += f"  验证数据:        {self.val_data_path}\n"
        config_str += f"  词表:            {self.vocab_path}\n"
        
        config_str += "\n【输出路径】\n"
        config_str += f"  检查点目录:      {self.checkpoint_dir}\n"
        config_str += f"  日志目录:        {self.log_dir}\n"
        
        config_str += "\n" + "=" * 60
        return config_str


# 创建全局配置实例
config_poetry = PoetryConfig()


if __name__ == "__main__":
    # 测试配置
    print(config_poetry)
    print(f"\n使用设备: {config_poetry.device}")
    
    # 显示详细的训练计划
    print("\n" + "=" * 60)
    print("训练计划详情")
    print("=" * 60)
    stats = config_poetry.get_training_stats()
    
    print(f"\n数据集规模:")
    print(f"  - 训练样本: 372,911 条 (+5,015，含宋词)")
    print(f"  - 验证样本: 9,813 条")
    print(f"  - 测试样本: 9,814 条")
    print(f"  - 平均长度: 37.5 字符/样本")
    print(f"  - 唯一字符: 9,833")
    
    print(f"\n训练策略:")
    print(f"  - 批次大小: {config_poetry.batch_size}")
    print(f"  - 每轮步数: {stats['steps_per_epoch']:,}")
    print(f"  - 训练轮数: {config_poetry.max_epochs}")
    print(f"  - 总训练步数: {stats['total_steps']:,}")
    
    print(f"\n检查点策略:")
    print(f"  - 保存间隔: 每 {config_poetry.save_interval} 步")
    print(f"  - 评估间隔: 每 {config_poetry.eval_interval} 步")
    print(f"  - 预计保存 {stats['checkpoints_saved']} 个检查点")
    
    print(f"\n预计训练时间:")
    print(f"  - GPU训练: ~{stats['estimated_time_hours']:.1f} 小时")
    print(f"  - CPU训练: ~{stats['estimated_time_hours'] * 10:.1f} 小时（约10倍GPU时间）")
    
    print("\n建议:")
    print("  1. 使用GPU训练以获得最佳性能")
    print("  2. 定期检查验证集损失，避免过拟合")
    print("  3. 可以在训练过程中调整temperature参数以优化生成质量")
    print("  4. 如果显存不足，可以减小batch_size或d_model")
