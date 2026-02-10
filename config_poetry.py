"""
NanoGPT-ZH 诗词训练配置文件
针对中文古诗词数据集优化的超参数配置
@Author xiaomin.zhang
"""

class PoetryConfig:
    """
    NanoGPT-ZH 诗词模型配置
    
    数据集规模:
    - 训练样本: 367,896 条
    - 验证样本: 19,363 条
    - 总字符数: ~1365万字符
    - 平均长度: 37字符/样本
    
    模型规模: ~12.8M 参数
    架构: Decoder-Only Transformer (GPT风格)
    """
    
    # ============ 项目信息 ============
    project_name = "NanoGPT-ZH-Poetry"
    version = "1.0.0"
    description = "Chinese Poetry Generation Model"
    
    # ============ 模型架构参数 ============
    # 针对诗词生成任务优化
    vocab_size = 10000          # 词表大小（与预处理保持一致）
    max_seq_len = 64           # 最大序列长度（诗词平均37字符，128足够）
    d_model = 384               # 模型维度（增大以提升表达能力）
    n_layers = 8                # Transformer层数（增加层数学习诗词结构）
    n_heads = 8                 # 注意力头数
    d_ff = 1536                 # 前馈网络维度（d_model的4倍）
    dropout = 0.1               # Dropout概率
    
    # ============ 训练参数 ============
    # 基于36.7万样本的训练策略
    batch_size = 32             # 批次大小（增大以加速训练）
    learning_rate = 3e-4        # 学习率（稍微提高）
    weight_decay = 0.01         # 权重衰减
    max_epochs = 8             # 最大训练轮数（数据量大，可以多训练几轮）
    warmup_steps = 1000         # 学习率预热步数（增加预热步数）
    max_steps = 80000          # 最大训练步数
    
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
    save_interval = 2000        # 每2000步保存一次模型
    eval_interval = 1000        # 每1000步评估一次
    log_interval = 1000          # 每100步记录一次日志
    
    # ============ 推理参数 ============
    # 针对诗词生成优化
    temperature = 0.8           # 采样温度（降低以生成更连贯的诗词）
    top_k = 40                  # Top-K采样（减小以提高质量）
    top_p = 0.85                # Top-P采样（稍微降低）
    max_gen_len = 128            # 最大生成长度（诗词一般不会太长）
    
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
    
    def get_training_stats(self):
        """计算训练统计信息"""
        # 训练样本数
        train_samples = 367896
        
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
    print(f"  - 训练样本: 367,896 条")
    print(f"  - 验证样本: 19,363 条")
    print(f"  - 平均长度: 37 字符/样本")
    
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
