"""
train_intent.py - Script huấn luyện Intent Classification (Multi-label).
Trạm 2C: Fine-tune ViHealthBERT-syllable trên ViMQ Intent dataset.

⚠️ XỬ LÝ IMBALANCED DATA:
- Loss Function: BCEWithLogitsLoss (thay vì CrossEntropyLoss)
- Class Weights: Tính pos_weight tự động từ phân bố data
  -> Phạt nặng khi model miss class thiểu số (Severity, Cause)

Cách chạy:
    python train_intent.py \
        --train_data data/intent_train.json \
        --val_data data/intent_val.json \
        --output_dir saved_models/intent_classification \
        --epochs 15 --batch_size 16
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    INTENT_ID2LABEL,
    INTENT_LABEL2ID,
    INTENT_LABELS,
    INTENT_MODEL_DIR,
    INTENT_MODEL_NAME,
    INTENT_NUM_LABELS,
    TRAIN_CONFIG,
)
from data_loader import IntentDataLoader


# ============================================================
# 🔧 CUSTOM TRAINER: Thay đổi Loss Function
# ============================================================

class MultiLabelTrainer(Trainer):
    """
    Custom Trainer cho bài toán Multi-label Classification.
    
    THAY ĐỔI QUAN TRỌNG so với Trainer mặc định:
    1. Đổi loss từ CrossEntropyLoss -> BCEWithLogitsLoss
       - CrossEntropy giả định labels loại trừ nhau (single-label)
       - BCEWithLogits cho phép nhiều labels đồng thời (multi-label)
    
    2. Truyền pos_weight vào BCEWithLogitsLoss
       - pos_weight[i] = N_negative[i] / N_positive[i]
       - Class thiểu số có weight cao -> loss cao khi predict sai
       - Giúp model học tốt hơn trên class ít data (vd: Severity)
    
    Lý do BẮT BUỘC đổi loss:
    - Dữ liệu y tế thường: Diagnosis rất nhiều, Severity rất ít
    - Nếu dùng CrossEntropy default -> model bias nặng về Diagnosis
    - BCEWithLogits + pos_weight -> cân bằng lại, tránh nhầm Diagnosis/Treatment
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Override compute_loss để dùng BCEWithLogitsLoss + pos_weight.
        
        Flow:
        1. Forward pass qua model -> lấy logits (chưa qua sigmoid)
        2. BCEWithLogitsLoss tự apply sigmoid rồi tính binary cross entropy
        3. pos_weight scale loss theo class weight -> phạt nặng class thiểu số
        """
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        # Chuyển pos_weight sang đúng device (GPU/CPU)
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
        else:
            pw = None

        # BCEWithLogitsLoss = Sigmoid + BinaryCrossEntropy (numerically stable)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        # labels phải là float tensor cho BCE
        loss = loss_fn(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 📊 METRICS
# ============================================================

def compute_multilabel_metrics(eval_pred) -> dict:
    """
    Tính metrics cho multi-label classification.
    
    Khác single-label:
    - Dùng sigmoid threshold = 0.5 (không phải argmax)
    - Tính sample-average và macro-average F1
    """
    logits, labels = eval_pred

    # Sigmoid + threshold cho multi-label
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    predictions = (probs >= 0.5).astype(int)

    # Macro F1: trung bình F1 của từng class (coi trọng class thiểu số)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    # Micro F1: tính trên tổng TP/FP/FN (coi trọng class đa số)
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    # Sample F1: trung bình F1 của từng sample
    sample_f1 = f1_score(labels, predictions, average="samples", zero_division=0)

    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)

    return {
        "f1": macro_f1,  # metric_for_best_model
        "micro_f1": micro_f1,
        "sample_f1": sample_f1,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# 🚀 MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Intent Classification (Multi-label)")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Đường dẫn file train (JSON/JSONL/CSV)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Đường dẫn file validation (optional)")
    parser.add_argument("--output_dir", type=str, default=str(INTENT_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold cho sigmoid prediction")
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 TRAINING: Intent Classification (Multi-label)")
    print(f"   Base Model: {INTENT_MODEL_NAME}")
    print(f"   Labels: {INTENT_LABELS}")
    print(f"   Loss: BCEWithLogitsLoss + pos_weight (Imbalanced Handling)")
    print(f"   Output Dir: {args.output_dir}")
    print("=" * 60)

    # 1. Load & prepare data (bao gồm tính pos_weight)
    loader = IntentDataLoader(tokenizer_name=INTENT_MODEL_NAME)
    datasets, pos_weight = loader.prepare_datasets(
        train_path=Path(args.train_data),
        val_path=Path(args.val_data) if args.val_data else None,
    )

    print(f"\n📊 Dataset stats:")
    print(f"   Train: {len(datasets['train'])} samples")
    print(f"   Val:   {len(datasets['validation'])} samples")
    print(f"\n⚖️  pos_weight (class weights cho BCEWithLogitsLoss):")
    for i, label in enumerate(INTENT_LABELS):
        print(f"   {label}: {pos_weight[i]:.2f}")

    # 2. Load model (problem_type = multi_label_classification)
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_MODEL_NAME,
        num_labels=INTENT_NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=INTENT_ID2LABEL,
        label2id=INTENT_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
        learning_rate=args.learning_rate,
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=TRAIN_CONFIG["logging_steps"],
        seed=TRAIN_CONFIG["seed"],
        fp16=TRAIN_CONFIG["fp16"],
        report_to="none",
        save_total_limit=3,
    )

    # 4. Custom Trainer với BCEWithLogitsLoss + pos_weight
    trainer = MultiLabelTrainer(
        pos_weight=pos_weight,  # ← CRITICAL: class weights cho imbalanced data
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_multilabel_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 5. Train
    print("\n🚀 Bắt đầu training với BCEWithLogitsLoss + pos_weight...")
    trainer.train()

    # 6. Evaluation chi tiết
    print("\n📈 Evaluation trên tập validation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")

    # Classification report theo từng intent label
    predictions_output = trainer.predict(datasets["validation"])
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= args.threshold).astype(int)

    print(f"\n📋 Classification Report (threshold={args.threshold}):")
    print(classification_report(
        labels, preds,
        target_names=INTENT_LABELS,
        zero_division=0,
    ))

    # 7. Save model
    output_path = Path(args.output_dir)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n✅ Model đã lưu tại: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
