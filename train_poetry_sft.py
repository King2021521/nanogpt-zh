"""
Instruction fine-tuning for the local NanoGPT-ZH poetry model.
@Author xiaomin.zhang
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config_poetry import config_poetry
from models import GPTModel
from utils import Tokenizer


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EarlyStopping:
    patience: int = 2
    min_delta: float = 0.002
    best_loss: float = float("inf")
    bad_checks: int = 0

    def update(self, val_loss: float) -> bool:
        if self.patience <= 0:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_checks = 0
            return False
        self.bad_checks += 1
        return self.bad_checks >= self.patience


def format_sft_text(example: Dict) -> str:
    """Compact format that fits the current 128-token character-level context."""
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output = example["output"].strip()
    if input_text:
        return f"要求：{instruction}\n补充：{input_text}\n作品：{output}"
    return f"要求：{instruction}\n作品：{output}"


def format_prompt_text(example: Dict) -> str:
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    if input_text:
        return f"要求：{instruction}\n补充：{input_text}\n作品："
    return f"要求：{instruction}\n作品："


class PoetrySFTDataset(Dataset):
    """JSONL SFT dataset with labels masked over the prompt tokens."""

    def __init__(self, data_path: str | Path, tokenizer: Tokenizer, max_seq_len: int):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load_examples()
        if not self.examples:
            raise ValueError(f"No usable SFT samples found in {self.data_path}")

    def _load_examples(self) -> List[Dict]:
        examples: List[Dict] = []
        with self.data_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("instruction") or not item.get("output"):
                    continue
                prompt_ids = self.tokenizer.encode(format_prompt_text(item), add_special_tokens=False)
                output_ids = self.tokenizer.encode(item["output"].strip(), add_special_tokens=False)
                full_len = 1 + len(prompt_ids) + len(output_ids) + 1
                if full_len <= self.max_seq_len:
                    examples.append(item)
                else:
                    print(f"[skip] {self.data_path}:{line_number} exceeds max_seq_len ({full_len}>{self.max_seq_len})")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        prompt_ids = self.tokenizer.encode(format_prompt_text(example), add_special_tokens=False)
        output_ids = self.tokenizer.encode(example["output"].strip(), add_special_tokens=False)

        output_start = 1 + len(prompt_ids)
        token_ids = [self.tokenizer.bos_id] + prompt_ids + output_ids + [self.tokenizer.eos_id]
        token_ids = token_ids[:self.max_seq_len]
        if len(token_ids) < self.max_seq_len:
            token_ids += [self.tokenizer.pad_id] * (self.max_seq_len - len(token_ids))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)

        ignore_until = max(output_start - 1, 0)
        target_ids[:ignore_until] = self.tokenizer.pad_id
        return input_ids, target_ids


@torch.no_grad()
def evaluate(model: GPTModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        _, loss = model(input_ids, target_ids)
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def save_checkpoint(
    output_dir: Path,
    filename: str,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config_poetry,
            "sft_args": vars(args),
        },
        path,
    )
    print(f"checkpoint saved: {path}")


def build_dataloaders(args: argparse.Namespace, tokenizer: Tokenizer) -> Tuple[DataLoader, DataLoader]:
    train_dataset = PoetrySFTDataset(args.train_file, tokenizer, config_poetry.max_seq_len)
    val_dataset = PoetrySFTDataset(args.val_file, tokenizer, config_poetry.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    return train_loader, val_loader


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = select_device()
    print(f"device: {device}")

    tokenizer = Tokenizer(vocab_size=config_poetry.vocab_size)
    tokenizer.load(config_poetry.vocab_path)
    train_loader, val_loader = build_dataloaders(args, tokenizer)

    model = GPTModel(config_poetry)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"loaded init checkpoint: {args.init_checkpoint} "
            f"(step={checkpoint.get('global_step')}, best_val_loss={checkpoint.get('best_val_loss')})"
        )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=config_poetry.betas,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    best_val_loss = float("inf")
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )
    global_step = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    stop_training = False

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{args.epochs}")
        for batch_index, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config_poetry.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                if writer and global_step % args.log_interval == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/perplexity", math.exp(min(loss.item(), 20)), global_step)

                if global_step % args.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"\nstep {global_step}: val_loss={val_loss:.4f}")
                    if writer:
                        writer.add_scalar("val/loss", val_loss, global_step)
                        writer.add_scalar("val/perplexity", math.exp(min(val_loss, 20)), global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if not args.no_save:
                            save_checkpoint(output_dir, "best_model.pt", model, optimizer, global_step, epoch, best_val_loss, args)
                    if early_stopping.update(val_loss):
                        print(
                            f"early stopping triggered at step {global_step}: "
                            f"early_stop_best={early_stopping.best_loss:.4f}, "
                            f"checkpoint_best={best_val_loss:.4f}, "
                            f"bad_checks={early_stopping.bad_checks}, "
                            f"patience={early_stopping.patience}"
                        )
                        stop_training = True
                        break
                    model.train()

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    if not args.no_save:
                        save_checkpoint(
                            output_dir,
                            f"checkpoint_step_{global_step}.pt",
                            model,
                            optimizer,
                            global_step,
                            epoch,
                            best_val_loss,
                            args,
                        )

                if args.max_steps and global_step >= args.max_steps:
                    break

        if stop_training:
            break

        val_loss = evaluate(model, val_loader, device)
        print(f"\nepoch {epoch + 1}: val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not args.no_save:
                save_checkpoint(output_dir, "best_model.pt", model, optimizer, global_step, epoch, best_val_loss, args)
        if early_stopping.update(val_loss):
            print(
                f"early stopping triggered after epoch {epoch + 1}: "
                f"early_stop_best={early_stopping.best_loss:.4f}, "
                f"checkpoint_best={best_val_loss:.4f}, "
                f"bad_checks={early_stopping.bad_checks}, "
                f"patience={early_stopping.patience}"
            )
            break

        if args.max_steps and global_step >= args.max_steps:
            break

    if not args.no_save:
        save_checkpoint(output_dir, "final_model.pt", model, optimizer, global_step, epoch, best_val_loss, args)
    if writer:
        writer.close()
    elapsed = (time.time() - start_time) / 60
    print(f"SFT complete: steps={global_step}, best_val_loss={best_val_loss:.4f}, minutes={elapsed:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune NanoGPT-ZH poetry model on instruction data.")
    parser.add_argument("--train_file", default="data/poetry_sft_v2_10000/train.jsonl")
    parser.add_argument("--val_file", default="data/poetry_sft_v2_10000/val.jsonl")
    parser.add_argument("--output_dir", default="checkpoints_poetry_sft_v2")
    parser.add_argument("--log_dir", default="logs_poetry_sft_v2")
    parser.add_argument("--init_checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.002)
    parser.add_argument("--no_save", action="store_true", help="Run training without writing checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
