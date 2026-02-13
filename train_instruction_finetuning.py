# @Author xiaomin.zhang
"""
诗词创作指令微调训练脚本
基于Transformers库实现的完整训练流程
"""

import json
import os
import torch
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from datasets import Dataset
import logging

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型参数配置"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2-1.5B",
        metadata={"help": "预训练模型路径或名称"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA微调"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )


@dataclass
class DataArguments:
    """数据参数配置"""
    train_file: str = field(
        default="data/poetry_instruction/train.jsonl",
        metadata={"help": "训练数据文件路径"}
    )
    val_file: str = field(
        default="data/poetry_instruction/val.jsonl",
        metadata={"help": "验证数据文件路径"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    prompt_template: str = field(
        default="alpaca",
        metadata={"help": "提示词模板类型: alpaca, chatgpt, custom"}
    )


class InstructionDataProcessor:
    """指令数据处理器"""
    
    def __init__(self, tokenizer, max_length: int = 512, template: str = "alpaca"):
        """
        初始化数据处理器
        
        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
            template: 提示词模板类型
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        
    def load_data(self, file_path: str) -> List[Dict]:
        """
        加载JSONL格式数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据列表
        """
        data = []
        logger.info(f"正在加载数据: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行JSON解析失败: {e}")
        
        logger.info(f"成功加载 {len(data)} 条数据")
        return data
    
    def format_alpaca_prompt(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """
        Alpaca格式提示词
        
        Args:
            instruction: 指令
            input_text: 输入（可选）
            output: 输出
            
        Returns:
            格式化后的提示词
        """
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n{output}"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 输出:\n{output}"
        return prompt
    
    def format_chatgpt_prompt(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """
        ChatGPT对话格式提示词
        
        Args:
            instruction: 指令
            input_text: 输入（可选）
            output: 输出
            
        Returns:
            格式化后的提示词
        """
        system_msg = "你是一个擅长创作古诗词的AI助手。"
        user_msg = instruction
        if input_text:
            user_msg += f"\n{input_text}"
        
        prompt = f"System: {system_msg}\nUser: {user_msg}\nAssistant: {output}"
        return prompt
    
    def format_instruction(self, example: Dict) -> str:
        """
        格式化单条指令数据
        
        Args:
            example: 数据样本
            
        Returns:
            格式化后的文本
        """
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']
        
        if self.template == "alpaca":
            return self.format_alpaca_prompt(instruction, input_text, output)
        elif self.template == "chatgpt":
            return self.format_chatgpt_prompt(instruction, input_text, output)
        else:
            # 自定义格式
            if input_text:
                return f"{instruction}\n{input_text}\n{output}"
            else:
                return f"{instruction}\n{output}"
    
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """
        准备训练数据集
        
        Args:
            data: 原始数据列表
            
        Returns:
            处理后的Dataset对象
        """
        # 格式化文本
        texts = [self.format_instruction(item) for item in data]
        
        # 创建Dataset
        dataset = Dataset.from_dict({'text': texts})
        
        # 分词
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors=None
            )
        
        logger.info("正在进行分词...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset


class PoetryInstructionTrainer:
    """诗词指令微调训练器"""
    
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments
    ):
        """
        初始化训练器
        
        Args:
            model_args: 模型参数
            data_args: 数据参数
            training_args: 训练参数
        """
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # 设置随机种子
        set_seed(training_args.seed)
        
        # 初始化tokenizer和model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"正在加载模型: {self.model_args.model_name_or_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.training_args.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 启用梯度检查点（节省内存）
        if self.training_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("模型加载完成")
        
        # 如果使用LoRA
        if self.model_args.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """配置LoRA"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            logger.info("正在配置LoRA...")
            
            lora_config = LoraConfig(
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("LoRA配置完成")
        except ImportError:
            logger.warning("未安装peft库，无法使用LoRA。请运行: pip install peft")
            self.model_args.use_lora = False
    
    def prepare_data(self):
        """准备训练数据"""
        # 初始化数据处理器
        processor = InstructionDataProcessor(
            self.tokenizer,
            max_length=self.data_args.max_length,
            template=self.data_args.prompt_template
        )
        
        # 加载数据
        train_data = processor.load_data(self.data_args.train_file)
        val_data = processor.load_data(self.data_args.val_file)
        
        # 准备数据集
        train_dataset = processor.prepare_dataset(train_data)
        val_dataset = processor.prepare_dataset(val_data)
        
        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self):
        """执行训练"""
        # 加载模型
        self.load_model_and_tokenizer()
        
        # 准备数据
        train_dataset, val_dataset = self.prepare_data()
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 初始化Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        logger.info("="*50)
        logger.info("开始训练...")
        logger.info("="*50)
        
        train_result = self.trainer.train()
        
        # 保存模型
        self.save_model()
        
        # 保存训练指标
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("="*50)
        logger.info("训练完成！")
        logger.info("="*50)
    
    def save_model(self):
        """保存模型"""
        output_dir = self.training_args.output_dir
        logger.info(f"正在保存模型到: {output_dir}")
        
        # 保存模型
        self.trainer.save_model(output_dir)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 如果使用LoRA，保存LoRA权重
        if self.model_args.use_lora:
            lora_dir = os.path.join(output_dir, "lora_weights")
            self.model.save_pretrained(lora_dir)
            logger.info(f"LoRA权重已保存到: {lora_dir}")
        
        logger.info("模型保存完成")
    
    def evaluate(self):
        """评估模型"""
        if self.trainer is None:
            logger.error("请先执行训练")
            return
        
        logger.info("正在评估模型...")
        metrics = self.trainer.evaluate()
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        logger.info("评估完成")
        return metrics


def main():
    """主函数"""
    # 模型参数
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen2-1.5B",  # 可以改为其他模型
        use_lora=True,  # 是否使用LoRA（推荐，节省内存）
        lora_r=8,
        lora_alpha=32
    )
    
    # 数据参数
    data_args = DataArguments(
        train_file="data/poetry_instruction/train.jsonl",
        val_file="data/poetry_instruction/val.jsonl",
        max_length=512,
        prompt_template="alpaca"  # alpaca, chatgpt, custom
    )
    
    # 训练参数
    training_args = TrainingArguments(
        # 输出目录
        output_dir="./output/poetry_instruction_model",
        
        # 训练参数
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        
        # 学习率
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # 优化器
        optim="adamw_torch",
        weight_decay=0.01,
        
        # 日志和保存
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        
        # 评估
        eval_steps=100,
        evaluation_strategy="steps",
        
        # 精度
        fp16=torch.cuda.is_available(),
        
        # 其他
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        
        # TensorBoard
        report_to="tensorboard",
        logging_dir="./output/poetry_instruction_model/logs",
        
        # 加载最佳模型
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # 创建训练器
    trainer = PoetryInstructionTrainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    
    # 执行训练
    trainer.train()
    
    # 评估模型
    trainer.evaluate()
    
    logger.info("="*50)
    logger.info("所有任务完成！")
    logger.info(f"模型已保存到: {training_args.output_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
