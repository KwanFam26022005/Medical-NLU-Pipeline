"""
train_ner.py - Script huấn luyện model Medical NER (Token Classification).
Trạm 2A: Fine-tune ViHealthBERT-word + CRF trên dataset ViMQ.
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import (
    classification_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    NER_ID2LABEL,
    NER_LABELS,
    NER_MODEL_DIR,
    NER_MODEL_NAME,
    TRAIN_CONFIG,
)
from custom_models import ViHealthBertCRF

NER_LABEL2ID = {v: k for k, v in NER_ID2LABEL.items()}

# ============================================================
# 🔧 CUSTOM TRAINER CHO CRF
# ============================================================
class CRFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 📊 METRICS & DATA BYPASS
# ============================================================
def compute_ner_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        seq_l, seq_p = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100: continue
            seq_l.append(NER_ID2LABEL.get(l, "O"))
            seq_p.append(NER_ID2LABEL.get(p, "O"))
        true_labels.append(seq_l)
        true_preds.append(seq_p)

    return {
        "precision": seqeval_precision(true_labels, true_preds, zero_division=0),
        "recall": seqeval_recall(true_labels, true_preds, zero_division=0),
        "f1": seqeval_f1(true_labels, true_preds, zero_division=0),
    }

def read_conll(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                parts = line.split()
                words.append(parts[0])
                tags.append(parts[-1])
        if words:
            sentences.append(words)
            labels.append(tags)
    return sentences, labels

def prepare_phobert_dataset(sentences, tags, tokenizer, max_len=256):
    data = {"input_ids": [], "attention_mask": [], "labels": []}
    for words, lbls in zip(sentences, tags):
        input_ids = [tokenizer.cls_token_id]
        label_ids = [-100]
        for word, label in zip(words, lbls):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens: continue
            w_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            input_ids.extend(w_ids)
            label_ids.append(NER_LABEL2ID.get(label, 0))
            label_ids.extend([-100] * (len(w_ids) - 1))
        input_ids.append(tokenizer.sep_token_id)
        label_ids.append(-100)
        
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len-1] + [tokenizer.sep_token_id]
            label_ids = label_ids[:max_len-1] + [-100]
            
        data["input_ids"].append(input_ids)
        data["attention_mask"].append([1] * len(input_ids))
        data["labels"].append(label_ids)
    return Dataset.from_dict(data)

# ============================================================
# 🚀 MAIN SCRIPT
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train SOTA Medical NER (ViHealthBERT + CRF)")
    parser.add_argument("--train_data", type=str, required=True, help="Đường dẫn file train (.conll)")
    parser.add_argument("--val_data", type=str, required=True, help="Đường dẫn file val (.conll)")
    parser.add_argument("--output_dir", type=str, default=str(NER_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 TRAINING SOTA: Medical NER (ViHealthBERT + CRF)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    train_words, train_tags = read_conll(args.train_data)
    val_words, val_tags = read_conll(args.val_data)

    datasets = DatasetDict({
        "train": prepare_phobert_dataset(train_words, train_tags, tokenizer),
        "validation": prepare_phobert_dataset(val_words, val_tags, tokenizer)
    })

    model = ViHealthBertCRF(model_name=NER_MODEL_NAME, num_labels=len(NER_LABELS))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

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
        load_best_model_at_end=True, # Load model xịn nhất để tránh overfitting
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=TRAIN_CONFIG["logging_steps"],
        seed=TRAIN_CONFIG["seed"],
        fp16=TRAIN_CONFIG["fp16"],
        report_to="none",
        save_total_limit=2,
    )

    trainer = CRFTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    print("\n📈 Đánh giá chi tiết model tốt nhất:")
    print(trainer.evaluate())

    # Lưu Model PyTorch thủ công vì đây là Custom Module
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Model đã được lưu gọn gàng tại: {args.output_dir}")

if __name__ == "__main__":
    main()