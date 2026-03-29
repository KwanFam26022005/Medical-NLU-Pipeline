"""
train_topic.py - Script huấn luyện Topic Classification.
Trạm 2B: Fine-tune ViHealthBERT-syllable phân loại Khoa y tế.

✅ TRẠNG THÁI: SẴN SÀNG - Đọc từ JSON đã xử lý sẵn.
  - data/topic_train.json  (tạo bởi build_topic_dataset.py)
  - data/topic_val.json

Cách chạy:
    1. python build_topic_dataset.py   # (chỉ cần chạy 1 lần)
    2. python train_topic.py --output_dir saved_models/topic_classification
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    TOPIC_DATASET_READY,
    TOPIC_MODEL_DIR,
    TOPIC_MODEL_NAME,
    TOPIC_TRAIN_JSON,
    TOPIC_VAL_JSON,
    TRAIN_CONFIG,
)
from data_loader import TopicDataLoader


def compute_metrics(eval_pred) -> dict:
    """Tính Accuracy và Macro-F1 cho Topic Classification."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main() -> None:
    # ================================================================
    # 🚨 GATE CHECK: Chặn chạy nếu dataset chưa sẵn sàng
    # ================================================================
    if not TOPIC_DATASET_READY:
        print("=" * 60)
        print("⚠️  TOPIC TRAINING - ĐANG CHỜ HOÀN THIỆN DATASET")
        print("=" * 60)
        print("Hãy chạy: python build_topic_dataset.py trước!")
        sys.exit(0)

    # ================================================================
    # Data sẵn sàng -> Chạy training
    # ================================================================
    parser = argparse.ArgumentParser(description="Train Topic Classification")
    parser.add_argument("--output_dir", type=str, default=str(TOPIC_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_train_epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"])
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 TRAINING: Topic Classification (Phân loại Khoa)")
    print(f"   Base Model:  {TOPIC_MODEL_NAME}")
    print(f"   Train JSON:  {TOPIC_TRAIN_JSON.name}")
    print(f"   Val JSON:    {TOPIC_VAL_JSON.name}")
    print(f"   Output Dir:  {args.output_dir}")
    print("=" * 60)

    # 1. Load & prepare data (từ JSON đã xử lý sẵn)
    loader = TopicDataLoader(tokenizer_name=TOPIC_MODEL_NAME)
    datasets = loader.prepare_datasets(
        train_path=TOPIC_TRAIN_JSON,
        val_path=TOPIC_VAL_JSON,
    )

    num_labels = len(loader.label2id)
    print(f"\n📊 Dataset stats:")
    print(f"   Train: {len(datasets['train'])} samples")
    print(f"   Val:   {len(datasets['validation'])} samples")
    print(f"   Labels: {num_labels} topic classes")

    # 2. Load model
    tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TOPIC_MODEL_NAME,
        num_labels=num_labels,
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
        eval_strategy="epoch",  # ← FIX: transformers v5+ dùng eval_strategy
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

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 5. Train
    print("\n🚀 Bắt đầu training...")
    trainer.train()

    # 6. Evaluate
    print("\n📈 Evaluation trên tập validation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")

    # 7. Save model + label mapping
    output_path = Path(args.output_dir)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    loader.save_label_mapping(output_path)  # ← Dùng method mới của TopicDataLoader

    print(f"\n✅ Model đã lưu tại: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
