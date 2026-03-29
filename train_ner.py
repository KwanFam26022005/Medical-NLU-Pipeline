"""
train_ner.py - Script huấn luyện model Medical NER (Token Classification).
Trạm 2A: Fine-tune ViHealthBERT-word trên dataset ViMQ NER (BIO format).

Nhãn NER: SYMPTOM_AND_DISEASE, MEDICAL_PROCEDURE, MEDICINE.

Cách chạy:
    python train_ner.py \
        --train_data data/ner_train.conll \
        --val_data data/ner_val.conll \
        --output_dir saved_models/medical_ner \
        --epochs 15 --batch_size 16
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    NER_ID2LABEL,
    NER_LABEL2ID,
    NER_LABELS,
    NER_MODEL_DIR,
    NER_MODEL_NAME,
    TRAIN_CONFIG,
)
from data_loader import NERDataLoader


def compute_ner_metrics(eval_pred) -> dict:
    """
    Tính metrics cho NER task dùng seqeval.
    
    Xử lý đặc biệt:
    - Bỏ qua token có label = -100 (padding/special tokens)
    - Chuyển id -> label string cho seqeval
    - Trả về entity-level Precision, Recall, F1
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Chuyển numeric id -> label string, bỏ qua -100
    true_labels: List[List[str]] = []
    true_preds: List[List[str]] = []

    for pred_seq, label_seq in zip(predictions, labels):
        seq_labels: List[str] = []
        seq_preds: List[str] = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue  # Bỏ qua padding/special tokens
            seq_labels.append(NER_ID2LABEL.get(label_id, "O"))
            seq_preds.append(NER_ID2LABEL.get(pred_id, "O"))

        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": seqeval_precision(true_labels, true_preds, zero_division=0),
        "recall": seqeval_recall(true_labels, true_preds, zero_division=0),
        "f1": seqeval_f1(true_labels, true_preds, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Medical NER Model")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Đường dẫn file train (CoNLL format)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Đường dẫn file validation (optional)")
    parser.add_argument("--output_dir", type=str, default=str(NER_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"])
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 TRAINING: Medical NER (Token Classification)")
    print(f"   Base Model: {NER_MODEL_NAME}")
    print(f"   Labels: {NER_LABELS}")
    print(f"   Output Dir: {args.output_dir}")
    print("=" * 60)

    # 1. Load & prepare data
    loader = NERDataLoader(tokenizer_name=NER_MODEL_NAME)
    datasets = loader.prepare_datasets(
        train_path=Path(args.train_data),
        val_path=Path(args.val_data) if args.val_data else None,
    )

    print(f"\n📊 Dataset stats:")
    print(f"   Train: {len(datasets['train'])} samples")
    print(f"   Val:   {len(datasets['validation'])} samples")

    # 2. Load model
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_NAME,
        num_labels=len(NER_LABELS),
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # 3. Data collator cho Token Classification
    # Tự động pad labels theo batch, dùng -100 cho padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # 4. Training arguments
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

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 6. Train
    print("\n🚀 Bắt đầu training...")
    trainer.train()

    # 7. Detailed evaluation
    print("\n📈 Đánh giá chi tiết trên tập validation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")

    # In classification report chi tiết theo từng entity type
    predictions_output = trainer.predict(datasets["validation"])
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    true_labels: List[List[str]] = []
    true_preds: List[List[str]] = []
    for pred_seq, label_seq in zip(preds, labels):
        seq_l, seq_p = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_l.append(NER_ID2LABEL.get(l, "O"))
            seq_p.append(NER_ID2LABEL.get(p, "O"))
        true_labels.append(seq_l)
        true_preds.append(seq_p)

    print("\n📋 Classification Report (Entity-level):")
    print(classification_report(true_labels, true_preds, zero_division=0))

    # 8. Save
    output_path = Path(args.output_dir)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n✅ Model đã lưu tại: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
