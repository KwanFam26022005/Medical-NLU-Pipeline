"""
train_intent.py - Script huấn luyện model Multi-label Intent.
Trạm 2C: Fine-tune ViHealthBERT + Asymmetric Loss (ASL) + Dynamic Thresholding.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    INTENT_ID2LABEL,
    INTENT_LABELS,
    INTENT_MODEL_DIR,
    INTENT_MODEL_NAME,
    INTENT_NUM_LABELS,
    TRAIN_CONFIG,
)
from data_loader import IntentDataLoader

# ============================================================
# 🔧 SOTA: ASYMMETRIC LOSS CHO MULTI-LABEL
# ============================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Cắt bỏ nhiễu từ các nhãn negative quá dễ (ví dụ: Diagnosis áp đảo)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        loss *= one_sided_w
        return -loss.mean()

class ASLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 📊 METRICS THÔNG THƯỜNG (Dùng threshold = 0.5 để log)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits)) # Sigmoid
    predictions = (probs >= 0.5).astype(int)
    
    return {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, predictions, average="micro", zero_division=0),
    }

# ============================================================
# 🔍 THUẬT TOÁN DYNAMIC THRESHOLDING
# ============================================================
def optimize_thresholds(trainer, eval_dataset, output_dir):
    print("\n🔍 Đang tìm Threshold tối ưu cho từng Intent...")
    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    probs = 1 / (1 + np.exp(-logits))
    
    best_thresholds = {}
    
    for i in range(INTENT_NUM_LABELS):
        best_t = 0.5
        best_f1 = 0.0
        # Quét ngưỡng từ 0.1 đến 0.9
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                
        label_name = INTENT_ID2LABEL[i]
        best_thresholds[label_name] = round(best_t, 2)
        print(f"  - {label_name}: Tối ưu tại ngưỡng {best_t:.2f} (F1: {best_f1:.4f})")
        
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"✅ Đã lưu file thresholds.json vào thư mục model.")

# ============================================================
# 🚀 MAIN SCRIPT
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(INTENT_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    print("=" * 60)
    print("🎯 TRAINING SOTA: Multi-label Intent (ASL + Dynamic Threshold)")
    print("=" * 60)

    loader = IntentDataLoader(tokenizer_name=INTENT_MODEL_NAME)
    datasets, _ = loader.prepare_datasets(
        train_path=Path(args.train_data), 
        val_path=Path(args.val_data)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_MODEL_NAME, 
        num_labels=INTENT_NUM_LABELS,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        save_total_limit=2,
    )

    trainer = ASLTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    
    print("\n📈 Đánh giá model tốt nhất (ngưỡng 0.5):")
    print(trainer.evaluate())

    # Tối ưu hóa ngưỡng trước khi lưu
    optimize_thresholds(trainer, datasets["validation"], args.output_dir)

    trainer.save_model(args.output_dir)
    loader.tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()