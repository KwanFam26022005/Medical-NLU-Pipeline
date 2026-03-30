"""
train_topic.py - Script huấn luyện Topic Classification (Trạm 2B).

Dữ liệu: preprocess_topic.py → data/topic_{train,val,test}.json + topic_label_map.json
Kỹ thuật: dynamic padding (DataCollatorWithPadding), weighted CrossEntropyLoss, F1 theo từng class.

Cách chạy:
    python preprocess_topic.py ...   # tiền xử lý
    python train_topic.py --output_dir saved_models/topic_classification
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    TOPIC_LABEL_MAP_JSON,
    TOPIC_MODEL_DIR,
    TOPIC_MODEL_NAME,
    TOPIC_TEST_JSON,
    TOPIC_TRAIN_JSON,
    TOPIC_VAL_JSON,
    TRAIN_CONFIG,
)
from data_loader import TopicDataLoader


def _topic_files_ready() -> bool:
    paths = [
        TOPIC_TRAIN_JSON,
        TOPIC_VAL_JSON,
        TOPIC_TEST_JSON,
        TOPIC_LABEL_MAP_JSON,
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("⚠️  Thiếu file dữ liệu topic:")
        for p in missing:
            print(f"    - {p}")
        return False
    return True


def build_compute_metrics(
    id2label: Dict[int, str],
    num_labels: int,
) -> Callable[..., Dict[str, float]]:
    """Accuracy, macro F1, và F1 từng class (vd: f1_cardiology)."""

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        label_ids = list(range(num_labels))
        f1_per_class = f1_score(
            labels,
            predictions,
            labels=label_ids,
            average=None,
            zero_division=0,
        )
        out: Dict[str, float] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
        }
        for i in range(num_labels):
            name = id2label.get(i, f"class_{i}")
            out[f"f1_{name}"] = float(f1_per_class[i])
        return out

    return compute_metrics


class WeightedTrainer(Trainer):
    """Trainer với CrossEntropyLoss(weight=class_weights) để xử lý imbalance."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.class_weights = (
            class_weights.float().cpu() if class_weights is not None else None
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        labels = inputs.get("labels")
        if labels is not None:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else None
            return (loss, outputs) if return_outputs else loss
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=w)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    if not _topic_files_ready():
        print("=" * 60)
        print("⚠️  TOPIC TRAINING — chưa có đủ JSON trong data/")
        print("    Chạy preprocess_topic.py trước.")
        print("=" * 60)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train Topic Classification")
    parser.add_argument("--output_dir", type=str, default=str(TOPIC_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_train_epochs"])
    parser.add_argument(
        "--batch_size", type=int, default=TRAIN_CONFIG["per_device_train_batch_size"]
    )
    parser.add_argument(
        "--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"]
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 TRAINING: Topic Classification (Trạm 2B)")
    print(f"   Base Model:   {TOPIC_MODEL_NAME}")
    print(f"   Train:        {TOPIC_TRAIN_JSON.name}")
    print(f"   Val:          {TOPIC_VAL_JSON.name}")
    print(f"   Test (holdout): {TOPIC_TEST_JSON.name}")
    print(f"   Output Dir:   {args.output_dir}")
    print("=" * 60)

    loader = TopicDataLoader(tokenizer_name=TOPIC_MODEL_NAME)
    datasets, class_weights = loader.prepare_datasets()

    num_labels = len(loader.label2id)
    print(f"\n📊 Labels: {num_labels} topic classes")

    tokenizer = loader.tokenizer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        TOPIC_MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    compute_metrics = build_compute_metrics(loader.id2label, num_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
        learning_rate=args.learning_rate,
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=TRAIN_CONFIG["logging_steps"],
        seed=TRAIN_CONFIG["seed"],
        fp16=TRAIN_CONFIG["fp16"],
        report_to="none",
        save_total_limit=3,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n🚀 Bắt đầu training (train) — best model theo validation (f1_macro)...")
    trainer.train()

    print("\n📈 Đánh giá trên validation (checkpoint tốt nhất):")
    val_results = trainer.evaluate(eval_dataset=datasets["validation"])
    for key, value in val_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n📈 Hold-out test (tập test độc lập):")
    test_results = trainer.evaluate(eval_dataset=datasets["test"])
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    output_path = Path(args.output_dir)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    loader.save_label_mapping(output_path)

    print(f"\n✅ Model đã lưu tại: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
