"""
train_joint.py — Script huấn luyện KEHN (Knowledge-Enhanced Hierarchical Network).

Hỗ trợ:
- Curriculum Learning 4 phases
- Các experiment configs (E1-E7 baselines & ablations)
- Early stopping trên topic_macro_f1
- Logging & checkpoint saving

CHANGES:
  [+TR]  Gọi model.constrain_crf_transitions() sau mỗi optimizer.step()
         để giữ vững BIO constraints trong CRF transitions.
  [+CWL] Truyền batch["ner_confidence"] vào model.forward() cho
         Confidence-Weighted NER Loss.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .config_joint import (
    MODEL_CONFIG, TRAIN_CONFIG, JOINT_DATA_DIR, JOINT_OUTPUT_DIR,
    TOPIC_LABELS, INTENT_LABELS, NER_TAGS, ID2NER, N_NER_TAG,
)
from .data_loader_joint import create_dataloaders
from .curriculum import CurriculumScheduler
from .metrics import compute_all_metrics


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(backbone_name: str, topic_class_weights=None, device="cpu"):
    from .model.kehn_model import KEHN

    weights_tensor = None
    if topic_class_weights is not None:
        weights_tensor = torch.tensor(topic_class_weights, dtype=torch.float32).to(device)

    model = KEHN(
        backbone_name=backbone_name,
        n_topic=MODEL_CONFIG["n_topic"],
        n_intent=MODEL_CONFIG["n_intent"],
        n_ner_tag=MODEL_CONFIG["n_ner_tag"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_co_blocks=MODEL_CONFIG["num_co_interactive_blocks"],
        dropout=MODEL_CONFIG["hidden_dropout"],
        use_bilstm=MODEL_CONFIG["use_bilstm"],
        topic_class_weights=weights_tensor,
    )
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────

def evaluate(model, dataloader, device, phase="full"):
    """Evaluate model trên 1 split, trả về metrics dict."""
    model.eval()

    all_topic_preds, all_topic_labels = [], []
    all_intent_preds, all_intent_labels = [], []
    all_ner_pred_tags, all_ner_true_tags = [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            topic_labels   = batch["topic_labels"].to(device)
            intent_labels  = batch["intent_labels"].to(device)
            ner_labels     = batch["ner_labels"].to(device)
            ner_confidence = batch.get("ner_confidence", None)
            if ner_confidence is not None:
                ner_confidence = ner_confidence.to(device)

            token_intent_ids = batch.get("token_intent_ids", None)
            if token_intent_ids is not None:
                token_intent_ids = token_intent_ids.to(device)

            output = model(
                input_ids, attention_mask,
                topic_labels=topic_labels,
                intent_labels=intent_labels,
                ner_labels=ner_labels,
                phase=phase,
                token_intent_ids=token_intent_ids,
                ner_confidence=ner_confidence,   # [+CWL]
            )

            if "loss" in output:
                total_loss += output["loss"].item()
            n_batches += 1

            topic_preds = output["logits_topic"].argmax(dim=-1).cpu().numpy()
            all_topic_preds.extend(topic_preds)
            all_topic_labels.extend(topic_labels.cpu().numpy())

            intent_preds = output["logits_intent"].argmax(dim=-1).cpu().numpy()
            all_intent_preds.extend(intent_preds)
            all_intent_labels.extend(intent_labels.cpu().numpy())

            # NER predictions (CRF Viterbi — BIO-constrained)
            ner_preds = model.predict_ner(output["logits_ner"], attention_mask)
            for i in range(len(ner_preds)):
                pred_tags, true_tags = [], []
                ner_label_seq = ner_labels[i].cpu().numpy()
                pred_seq      = ner_preds[i]
                for j in range(len(pred_seq)):
                    if j < len(ner_label_seq) and ner_label_seq[j] != -100:
                        pred_tags.append(ID2NER.get(pred_seq[j], "O"))
                        true_tags.append(ID2NER.get(ner_label_seq[j], "O"))
                all_ner_pred_tags.append(pred_tags)
                all_ner_true_tags.append(true_tags)

    metrics = compute_all_metrics(
        np.array(all_topic_preds), np.array(all_topic_labels),
        np.array(all_intent_preds), np.array(all_intent_labels),
        all_ner_pred_tags, all_ner_true_tags,
        topic_label_names=TOPIC_LABELS,
    )
    metrics["eval_loss"] = total_loss / max(n_batches, 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────

def train(args):
    """Main training loop."""
    set_seed(TRAIN_CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # 1. Backbone path
    backbone_path = MODEL_CONFIG.get(args.backbone, MODEL_CONFIG["phobert"])

    # 2. Hidden dim cho XLM-R Large
    if args.backbone == "xlmr_large":
        MODEL_CONFIG["hidden_dim"] = 1024
        print("✨ Detected XLM-R Large: Setting hidden_dim to 1024")
    else:
        MODEL_CONFIG["hidden_dim"] = 768

    print("\n📂 Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer_name=backbone_path,
        batch_size=args.batch_size,
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
    )

    # Class weights cho Topic loss
    meta_path = JOINT_DATA_DIR / "metadata.json"
    topic_class_weights = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        topic_class_weights = meta.get("topic_class_weights")
    else:
        print("   metadata.json not found. Calculating weights from train_loader...")
        from collections import Counter
        topic_counts = Counter()
        for batch in train_loader:
            topic_counts.update(batch["topic_labels"].tolist())
        n_topics = MODEL_CONFIG["n_topic"]
        total    = sum(topic_counts.values())
        weights  = [total / topic_counts.get(i, 1) for i in range(n_topics)]
        min_w    = min(weights)
        topic_class_weights = [w / min_w for w in weights]
        print(f"   Calculated Topic Weights: {topic_class_weights}")

    print(f"🤖 Building KEHN with backbone: {backbone_path}")
    model = build_model(backbone_path, topic_class_weights, device)

    # Log CRF constraint summary
    stats = model.get_illegal_transition_stats()
    print(
        f"   [+TR] CRF constraints: "
        f"{stats['n_illegal_start']} illegal start tags, "
        f"{stats['n_illegal_trans']} illegal transitions → set to −1e9"
    )

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {n_params:,} | Trainable: {n_trainable:,}")

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    accum_steps = TRAIN_CONFIG.get("gradient_accumulation_steps", 1)
    use_amp     = device.type == "cuda" and TRAIN_CONFIG.get("fp16", False)
    scaler      = torch.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps
    total_steps     = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps, pct_start=TRAIN_CONFIG["warmup_ratio"],
    )

    curriculum       = CurriculumScheduler()
    best_metric      = 0.0
    patience_counter = 0
    best_epoch       = 0

    output_dir = JOINT_OUTPUT_DIR / f"{args.exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🚀 Training KEHN — Experiment: {args.exp_name}")
    print(f"   Backbone: {backbone_path}")
    print(f"   Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"   FP16: {use_amp}, Grad Accum: {accum_steps}x")
    print(f"   [+TR]  BIO CRF Constraints: ENABLED")
    print(f"   [+CWL] Confidence-Weighted NER Loss: ENABLED")
    print(f"{'=' * 60}\n")

    history = []

    for epoch in range(1, args.epochs + 1):
        phase = curriculum.get_phase(epoch)
        curriculum.apply_freeze(model, epoch)
        phase_info = curriculum.get_phase_info(epoch)

        print(f"\n📅 Epoch {epoch}/{args.epochs} | {phase_info}")

        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            topic_labels   = batch["topic_labels"].to(device)
            intent_labels  = batch["intent_labels"].to(device)
            ner_labels     = batch["ner_labels"].to(device)

            # [+CWL] ner_confidence: [B] float, per-sample weight
            ner_confidence = batch.get("ner_confidence", None)
            if ner_confidence is not None:
                ner_confidence = ner_confidence.to(device)

            token_intent_ids = batch.get("token_intent_ids", None)
            if token_intent_ids is not None:
                token_intent_ids = token_intent_ids.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(
                    input_ids, attention_mask,
                    topic_labels=topic_labels,
                    intent_labels=intent_labels,
                    ner_labels=ner_labels,
                    phase=phase,
                    token_intent_ids=token_intent_ids,
                    ner_confidence=ner_confidence,   # [+CWL]
                )
                loss = output["loss"] / accum_steps
            # Thêm kiểm tra loss để debug nếu cần
            if torch.isnan(loss) or loss.item() > 10000:
                print(f"⚠️ Cảnh báo: Loss bất thường tại step {step}: {loss.item()}")
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # [+TR] Re-clamp illegal CRF transitions sau optimizer step.
                # Vì optimizer đã cập nhật tất cả nn.Parameter (kể cả
                # crf.transitions), cần restore các ô illegal về -1e9.
                model.constrain_crf_transitions()

            epoch_loss += loss.item() * accum_steps
            n_batches  += 1

            if (step + 1) % 50 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"   Step {step + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = evaluate(model, val_loader, device, phase)
        topic_f1    = val_metrics["topic_macro_f1"]
        ner_f1      = val_metrics["ner_f1"]
        intent_acc  = val_metrics["intent_accuracy"]

        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(
            f"   Val — Topic F1: {topic_f1:.4f} | "
            f"Intent Acc: {intent_acc:.4f} | NER F1: {ner_f1:.4f}"
        )

        # [+TR] Log CRF constraint health mỗi epoch
        stats = model.get_illegal_transition_stats()
        if stats["max_illegal_trans_val"] > -1e8:
            print(
                f"   ⚠️  CRF constraint drift detected! "
                f"max_illegal_trans={stats['max_illegal_trans_val']:.1f}"
            )

        epoch_record = {
            "epoch": epoch, "phase": phase,
            "train_loss": avg_train_loss, **val_metrics,
        }
        history.append(epoch_record)

        # Early stopping (on topic_macro_f1)
        if topic_f1 > best_metric:
            best_metric      = topic_f1
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"   💾 New best! Topic F1={topic_f1:.4f} (saved)")
            
            # [Added] Backup checkpoint to Google Drive for large models on Colab
            if args.backbone == "xlmr_large":
                drive_dir = Path("/content/drive/MyDrive/Medical_NLU_Checkpoints")
                if Path("/content/drive/MyDrive").exists():
                    drive_dir.mkdir(parents=True, exist_ok=True)
                    drive_file = drive_dir / f"{args.exp_name}_best.pt"
                    torch.save(model.state_dict(), drive_file)
                    print(f"   ☁️ Backup XLM-R Large checkpoint to GDrive: {drive_file}")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG["patience"] and epoch > 10:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

    # Final evaluation
    print(f"\n{'=' * 60}")
    print(f"📊 Final Evaluation (best model from epoch {best_epoch})")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    test_metrics = evaluate(model, test_loader, device, "full")
    print(f"   Topic Macro-F1:    {test_metrics['topic_macro_f1']:.4f}")
    print(f"   Intent Accuracy:   {test_metrics['intent_accuracy']:.4f}")
    print(f"   NER F1:            {test_metrics['ner_f1']:.4f}")
    print(f"   Semantic Accuracy: {test_metrics['semantic_accuracy']:.4f}")

    results = {
        "experiment":         args.exp_name,
        "backbone":           backbone_path,
        "best_epoch":         best_epoch,
        "best_val_topic_f1":  best_metric,
        "test_metrics":       test_metrics,
        "history":            history,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Training complete! Results saved to {output_dir}")
    return results


# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train KEHN Joint Model")
    parser.add_argument("--exp_name",   type=str, default="E4_kehn_phobert")
    parser.add_argument("--backbone",   type=str, default="phobert",
                        choices=["phobert", "vihealthbert", "xlmr_large"])
    parser.add_argument("--epochs",     type=int, default=TRAIN_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--max_samples",type=int, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()