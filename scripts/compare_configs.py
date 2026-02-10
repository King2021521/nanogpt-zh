"""
配置对比脚本
对比原始配置和诗词优化配置的差异
@Author xiaomin.zhang
"""

import sys
import os

# 设置UTF-8编码（Windows兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from config_poetry import config_poetry


def compare_configs():
    """对比两个配置"""
    print("=" * 80)
    print("配置对比: 原始配置 vs 诗词优化配置")
    print("=" * 80)
    
    # 模型架构参数
    print("\n【模型架构参数】")
    print("-" * 80)
    print(f"{'参数':<20} {'原始配置':<20} {'诗词配置':<20} {'变化':<20}")
    print("-" * 80)
    
    arch_params = [
        ('vocab_size', '词表大小'),
        ('max_seq_len', '最大序列长度'),
        ('d_model', '模型维度'),
        ('n_layers', 'Transformer层数'),
        ('n_heads', '注意力头数'),
        ('d_ff', '前馈网络维度'),
        ('dropout', 'Dropout概率'),
    ]
    
    for param, name in arch_params:
        val1 = getattr(config, param)
        val2 = getattr(config_poetry, param)
        
        if val1 == val2:
            change = "✓ 相同"
        elif val2 > val1:
            change = f"⬆️ +{((val2-val1)/val1*100):.1f}%"
        else:
            change = f"⬇️ {((val2-val1)/val1*100):.1f}%"
        
        print(f"{name:<20} {str(val1):<20} {str(val2):<20} {change:<20}")
    
    # 计算参数量
    def calc_params(cfg):
        return (
            cfg.vocab_size * cfg.d_model +
            cfg.max_seq_len * cfg.d_model +
            cfg.n_layers * (
                4 * cfg.d_model * cfg.d_model +
                2 * cfg.d_model * cfg.d_ff
            ) +
            cfg.vocab_size * cfg.d_model
        )
    
    params1 = calc_params(config)
    params2 = calc_params(config_poetry)
    print("-" * 80)
    print(f"{'预计参数量':<20} {f'{params1/1e6:.2f}M':<20} {f'{params2/1e6:.2f}M':<20} {f'⬆️ {params2/params1:.1f}x':<20}")
    
    # 训练参数
    print("\n【训练参数】")
    print("-" * 80)
    print(f"{'参数':<20} {'原始配置':<20} {'诗词配置':<20} {'变化':<20}")
    print("-" * 80)
    
    train_params = [
        ('batch_size', '批次大小'),
        ('learning_rate', '学习率'),
        ('max_epochs', '最大轮数'),
        ('warmup_steps', '预热步数'),
        ('max_steps', '最大步数'),
        ('weight_decay', '权重衰减'),
    ]
    
    for param, name in train_params:
        val1 = getattr(config, param)
        val2 = getattr(config_poetry, param)
        
        if val1 == val2:
            change = "✓ 相同"
        elif val2 > val1:
            change = f"⬆️ +{((val2-val1)/val1*100):.1f}%"
        else:
            change = f"⬇️ {((val2-val1)/val1*100):.1f}%"
        
        print(f"{name:<20} {str(val1):<20} {str(val2):<20} {change:<20}")
    
    # 训练统计
    print("\n【训练统计】")
    print("-" * 80)
    
    # 假设训练样本数
    train_samples = 367896
    
    steps_per_epoch1 = train_samples // config.batch_size
    steps_per_epoch2 = train_samples // config_poetry.batch_size
    
    total_steps1 = min(config.max_steps, config.max_epochs * steps_per_epoch1)
    total_steps2 = min(config_poetry.max_steps, config_poetry.max_epochs * steps_per_epoch2)
    
    time1 = (total_steps1 * 0.5) / 3600
    time2 = (total_steps2 * 0.8) / 3600  # 诗词配置模型更大，每步耗时更长
    
    print(f"{'指标':<20} {'原始配置':<20} {'诗词配置':<20}")
    print("-" * 80)
    print(f"{'每轮步数':<20} {f'{steps_per_epoch1:,}':<20} {f'{steps_per_epoch2:,}':<20}")
    print(f"{'预计总步数':<20} {f'{total_steps1:,}':<20} {f'{total_steps2:,}':<20}")
    print(f"{'预计训练时间':<20} {f'{time1:.1f}小时':<20} {f'{time2:.1f}小时':<20}")
    
    # 推理参数
    print("\n【推理参数】")
    print("-" * 80)
    print(f"{'参数':<20} {'原始配置':<20} {'诗词配置':<20} {'变化':<20}")
    print("-" * 80)
    
    infer_params = [
        ('temperature', '采样温度'),
        ('top_k', 'Top-K采样'),
        ('top_p', 'Top-P采样'),
        ('max_gen_len', '最大生成长度'),
    ]
    
    for param, name in infer_params:
        val1 = getattr(config, param)
        val2 = getattr(config_poetry, param)
        
        if val1 == val2:
            change = "✓ 相同"
        elif val2 > val1:
            change = f"⬆️ +{((val2-val1)/val1*100):.1f}%"
        else:
            change = f"⬇️ {((val2-val1)/val1*100):.1f}%"
        
        print(f"{name:<20} {str(val1):<20} {str(val2):<20} {change:<20}")
    
    # 检查点策略
    print("\n【检查点策略】")
    print("-" * 80)
    print(f"{'参数':<20} {'原始配置':<20} {'诗词配置':<20} {'变化':<20}")
    print("-" * 80)
    
    checkpoint_params = [
        ('save_interval', '保存间隔'),
        ('eval_interval', '评估间隔'),
        ('log_interval', '日志间隔'),
    ]
    
    for param, name in checkpoint_params:
        val1 = getattr(config, param)
        val2 = getattr(config_poetry, param)
        
        if val1 == val2:
            change = "✓ 相同"
        elif val2 > val1:
            change = f"⬆️ +{((val2-val1)/val1*100):.1f}%"
        else:
            change = f"⬇️ {((val2-val1)/val1*100):.1f}%"
        
        print(f"{name:<20} {str(val1):<20} {str(val2):<20} {change:<20}")
    
    checkpoints1 = total_steps1 // config.save_interval
    checkpoints2 = total_steps2 // config_poetry.save_interval
    
    print("-" * 80)
    print(f"{'预计保存次数':<20} {checkpoints1:<20} {checkpoints2:<20}")
    
    # 总结
    print("\n【总结】")
    print("=" * 80)
    print("\n诗词配置的主要优化:")
    print("  ✓ 模型容量增大3倍 (7.3M → 21.9M 参数)")
    print("  ✓ 序列长度减半以节省显存 (256 → 128)")
    print("  ✓ 批次大小翻倍以加速训练 (32 → 64)")
    print("  ✓ 训练步数翻倍以充分学习 (50K → 100K)")
    print("  ✓ 生成参数优化以提高诗词质量")
    
    print("\n适用场景:")
    print("  • 原始配置: 适合快速实验、资源受限的环境")
    print("  • 诗词配置: 适合正式训练、追求高质量的诗词生成")
    
    print("\n硬件要求:")
    print("  • 原始配置: 2-4GB 显存即可")
    print("  • 诗词配置: 建议 6-8GB 显存 (最低4GB)")
    
    print("\n推荐:")
    print("  🎯 使用诗词配置 (config_poetry.py) 进行训练")
    print("  📝 运行: python train_poetry.py")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_configs()
