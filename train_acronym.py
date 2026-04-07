"""
train_acronym.py - Cross-Encoder Acronym Disambiguation Trainer.

Architecture: Binary scoring per (context, candidate) pair.
  Input:  "[CLS] context_with_<e>acronym</e> [SEP] candidate_expansion [SEP]"
  Output: scalar logit → BCEWithLogitsLoss

Usage (Colab):
    !python train_acronym.py \
        --data_dir data/acrDrAid \
        --output_dir saved_models/acronym_wsd \
        --epochs 10 \
        --batch_size 8 \
        --lr 2e-5
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from eval_utils import evaluate_cross_encoder

from data_loader import (
    AcronymDataLoader,
    AcronymDataset,
    acronym_eval_collate_fn,
    acronym_train_collate_fn,
)


# ============================================================
# 📋 TRAINING CONFIG
# ============================================================

@dataclass
class TrainingConfig:
    """Hyperparameters cho Cross-Encoder Acronym WSD."""
    model_name: str = "demdecuong/vihealthbert-base-syllable"
    data_dir: str = "data/acrDrAid"
    output_dir: str = "saved_models/acronym_wsd"
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    seed: int = 42
    fp16: bool = True
    patience: int = 3  # early stopping patience (reduced from 5)
    label_smoothing: float = 0.1  # smooth 0/1 labels → 0.05/0.95


# ============================================================
# 🏋️ TRAINER
# ============================================================

class AcronymTrainer:
    """
    Custom Trainer cho Cross-Encoder Acronym Disambiguation.
    
    Loss: BCEWithLogitsLoss
    Eval: Accuracy (overall, seen, unseen), MRR
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device}")

        # Seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Load data
        self.data_loader = AcronymDataLoader(
            data_dir=config.data_dir,
            tokenizer_name=config.model_name,
            max_length=config.max_length,
        )
        self.train_ds, self.dev_ds, self.test_ds = self.data_loader.get_datasets()

        # Build seen acronyms set for eval
        self.train_acronyms = set(s["acronym"] for s in self.train_ds.processed)

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=1
        )
        # Resize embeddings for <e>, </e> tokens
        self.model.resize_token_embeddings(len(self.data_loader.tokenizer))
        self.model.to(self.device)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: differential LR (lower for encoder, higher for classifier head)
        encoder_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        self.optimizer = AdamW([
            {"params": encoder_params, "lr": config.learning_rate},
            {"params": head_params, "lr": config.learning_rate * 5},
        ], weight_decay=config.weight_decay)

        # DataLoader
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=acronym_train_collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        # Scheduler
        total_steps = (len(self.train_loader) // config.gradient_accumulation_steps) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # FP16
        self.scaler = torch.amp.GradScaler("cuda") if config.fp16 and self.device.type == "cuda" else None

        # History
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Label smoothing: 0.0/1.0 → 0.05/0.95
            if self.config.label_smoothing > 0:
                labels = labels * (1 - self.config.label_smoothing) + self.config.label_smoothing / 2

            # Forward
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_samples += labels.size(0)

            # Binary accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()

            # Log every 50 steps
            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                acc = correct / total_samples * 100
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Step {step+1}/{len(self.train_loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | LR: {lr:.2e}")

        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_binary_acc": correct / total_samples * 100,
        }

    def evaluate(self, name: str = "dev") -> Dict[str, float]:
        """
        Evaluate using shared eval_utils module.
        Uses raw samples stored by data_loader for consistency.
        """
        raw_samples = {
            "dev": self.data_loader.raw_dev,
            "test": self.data_loader.raw_test,
        }.get(name, self.data_loader.raw_dev)

        _, metrics = evaluate_cross_encoder(
            self.model,
            self.data_loader.tokenizer,
            self.data_loader.acronym_dict,
            raw_samples,
            self.device,
            train_acronyms=self.train_acronyms,
            max_length=self.config.max_length,
            name=name,
        )
        return metrics

    def train(self) -> None:
        """Full training loop with early stopping."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best_accuracy = 0.0
        patience_counter = 0

        print("\n" + "=" * 60)
        print("🏥 TRAINING: Acronym WSD — Cross-Encoder")
        print(f"   Base Model:      {self.config.model_name}")
        print(f"   Epochs:          {self.config.epochs}")
        print(f"   Batch Size:      {self.config.batch_size} × {self.config.gradient_accumulation_steps} = {self.config.batch_size * self.config.gradient_accumulation_steps} (effective)")
        print(f"   LR:              {self.config.learning_rate}")
        print(f"   Loss:            BCEWithLogitsLoss")
        print(f"   Train samples:   {len(self.train_ds)}")
        print(f"   Dev samples:     {len(self.dev_ds)}")
        print(f"   Output Dir:      {self.config.output_dir}")
        print("=" * 60)

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n📌 Epoch {epoch}/{self.config.epochs}")
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate on dev
            dev_metrics = self.evaluate(name="dev")

            elapsed = time.time() - start_time
            epoch_results = {**train_metrics, **dev_metrics, "epoch": epoch, "elapsed_s": elapsed}
            self.history.append(epoch_results)

            # Print results
            print(f"\n  📊 Epoch {epoch} results ({elapsed:.0f}s):")
            print(f"     Train Loss:     {train_metrics['train_loss']:.4f}")
            print(f"     Train Bin Acc:  {train_metrics['train_binary_acc']:.1f}%")
            print(f"     Dev Accuracy:   {dev_metrics['dev_accuracy']:.2f}%")
            print(f"     Dev MRR:        {dev_metrics['dev_mrr']:.4f}")
            print(f"     Dev Seen Acc:   {dev_metrics['dev_seen_acc']:.2f}% ({dev_metrics['dev_seen_total']})")
            print(f"     Dev Unseen Acc: {dev_metrics['dev_unseen_acc']:.2f}% ({dev_metrics['dev_unseen_total']})")

            # Best model checkpoint — combined metric (seen + unseen + MRR)
            if dev_metrics["dev_unseen_total"] > 0:
                current_score = (
                    0.5 * dev_metrics["dev_seen_acc"]
                    + 0.3 * dev_metrics["dev_unseen_acc"]
                    + 0.2 * dev_metrics["dev_mrr"] * 100
                )
            else:
                current_score = dev_metrics["dev_accuracy"]
            if current_score > best_accuracy:
                best_accuracy = current_score
                patience_counter = 0

                # Save model
                self.model.save_pretrained(output_dir)
                self.data_loader.tokenizer.save_pretrained(output_dir)
                self.data_loader.save_dictionary(output_dir)

                print(f"     ✅ NEW BEST! Saved to {output_dir}")
            else:
                patience_counter += 1
                print(f"     ⏳ No improvement ({patience_counter}/{self.config.patience})")

            if patience_counter >= self.config.patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        # Save training history
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        # Final evaluation on test set
        print("\n" + "=" * 60)
        print("📊 FINAL TEST SET EVALUATION")
        print("=" * 60)
        
        # Reload best model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(output_dir), num_labels=1
        )
        self.model.to(self.device)
        
        test_metrics = self.evaluate(name="test")
        print(f"   Test Accuracy:   {test_metrics['test_accuracy']:.2f}%")
        print(f"   Test MRR:        {test_metrics['test_mrr']:.4f}")
        print(f"   Test Seen Acc:   {test_metrics['test_seen_acc']:.2f}% ({test_metrics['test_seen_total']})")
        print(f"   Test Unseen Acc: {test_metrics['test_unseen_acc']:.2f}% ({test_metrics['test_unseen_total']})")

        with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 Training complete! Best dev accuracy: {best_accuracy:.2f}%")


# ============================================================
# 🚀 MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Cross-Encoder Acronym WSD")
    parser.add_argument("--model_name", type=str, default="demdecuong/vihealthbert-base-syllable")
    parser.add_argument("--data_dir", type=str, default="data/acrDrAid")
    parser.add_argument("--output_dir", type=str, default="saved_models/acronym_wsd")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        seed=args.seed,
        fp16=args.fp16 and not args.no_fp16,
        patience=args.patience,
    )

    trainer = AcronymTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()