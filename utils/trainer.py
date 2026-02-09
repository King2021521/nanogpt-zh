"""
训练器实现
@Author xiaomin.zhang
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import math


class Trainer:
    """
    模型训练器
    负责训练循环、验证、保存检查点等
    """
    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: 模型对象
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置对象
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设置设备
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器（带预热的余弦退火）
        self.scheduler = self.get_scheduler()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
    
    def get_scheduler(self):
        """
        创建学习率调度器（带预热的余弦退火）
        """
        def lr_lambda(step):
            # 预热阶段：线性增长
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            
            # 余弦退火
            progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            # 数据移到设备
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # 前向传播
            logits, loss = self.model(input_ids, target_ids)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # 记录到TensorBoard
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                self.writer.add_scalar('train/perplexity', math.exp(loss.item()), self.global_step)
            
            # 验证
            if self.global_step % self.config.eval_interval == 0:
                val_loss = self.validate()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                self.writer.add_scalar('val/perplexity', math.exp(val_loss), self.global_step)
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"\n✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
                
                self.model.train()
            
            # 定期保存检查点
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
            
            # 达到最大步数
            if self.global_step >= self.config.max_steps:
                break
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        
        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, loss = self.model(input_ids, target_ids)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """
        完整训练流程
        """
        print("\n" + "=" * 50)
        print("开始训练")
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 打印epoch统计
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):.2f}")
            print(f"  Val Loss: {val_loss:.4f} | Val PPL: {math.exp(val_loss):.2f}")
            print(f"  Time: {elapsed / 60:.2f} min")
            
            # 达到最大步数
            if self.global_step >= self.config.max_steps:
                print(f"\n达到最大训练步数 {self.config.max_steps}")
                break
        
        # 保存最终模型
        self.save_checkpoint('final_model.pt')
        
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"训练完成！总时间: {total_time / 60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print("=" * 50)
        
        self.writer.close()
    
    def save_checkpoint(self, filename):
        """
        保存检查点
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"检查点已加载: {checkpoint_path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Global Step: {self.global_step}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    print("Trainer模块 - 需要完整的模型和数据才能测试")
    print("请在train.py中使用此模块")
