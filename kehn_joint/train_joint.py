"""
train_joint.py — Script huấn luyện KEHN (Knowledge-Enhanced Hierarchical Network).

Hỗ trợ:
- Curriculum Learning 4 phases
- Các experiment configs (E1-E7 baselines & ablations)
- Early stopping trên topic_macro_f1
- Logging & checkpoint saving
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
    """Reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(backbone_name: str, topic_class_weights=None, device="cpu"):
    """Build KEHN model."""
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


def evaluate(model, dataloader, device, phase="full"):
    """Evaluate model trên 1 split, trả về metrics dict."""
    model.eval()

    all_topic_preds, all_topic_labels = [], []
    all_intent_preds, all_intent_labels = [], []
    all_ner_pred_tags, all_ner_true_tags = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            topic_labels = batch["topic_labels"].to(device)
            intent_labels = batch["intent_labels"].to(device)
            ner_labels = batch["ner_labels"].to(device)
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
            )

            if "loss" in output:
                total_loss += output["loss"].item()
            n_batches += 1

            # Topic predictions
            topic_preds = output["logits_topic"].argmax(dim=-1).cpu().numpy()
            all_topic_preds.extend(topic_preds)
            all_topic_labels.extend(topic_labels.cpu().numpy())

            # Intent predictions (sentence-level)
            intent_preds = output["logits_intent"].argmax(dim=-1).cpu().numpy()
            all_intent_preds.extend(intent_preds)
            all_intent_labels.extend(intent_labels.cpu().numpy())

            # NER predictions (CRF decode)
            ner_preds = model.predict_ner(output["logits_ner"], attention_mask)
            for i in range(len(ner_preds)):
                pred_tags = []
                true_tags = []
                ner_label_seq = ner_labels[i].cpu().numpy()
                pred_seq = ner_preds[i]

                for j in range(len(pred_seq)):
                    if j < len(ner_label_seq) and ner_label_seq[j] != -100:
                        pred_tags.append(ID2NER.get(pred_seq[j], "O"))
                        true_tags.append(ID2NER.get(ner_label_seq[j], "O"))

                all_ner_pred_tags.append(pred_tags)
                all_ner_true_tags.append(true_tags)

    # Compute all metrics
    metrics = compute_all_metrics(
        np.array(all_topic_preds), np.array(all_topic_labels),
        np.array(all_intent_preds), np.array(all_intent_labels),
        all_ner_pred_tags, all_ner_true_tags,
        topic_label_names=TOPIC_LABELS,
    )
    metrics["eval_loss"] = total_loss / max(n_batches, 1)

    return metrics


def train(args):
    """Main training loop."""
    set_seed(TRAIN_CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # DataLoaders
    print(f"\n📂 Loading data...")
    backbone = MODEL_CONFIG.get(args.backbone, MODEL_CONFIG["phobert"])
    
    # Logic xác định hidden_dim động
    current_hidden_dim = MODEL_CONFIG["hidden_dim"]
    if "large" in backbone:
        current_hidden_dim = MODEL_CONFIG.get("xlmr_hidden_dim", 1024)
        print(f"✨ Detecting Large model, setting hidden_dim to {current_hidden_dim}")

    # Cập nhật MODEL_CONFIG tạm thời để build_model sử dụng đúng dimension
    MODEL_CONFIG["hidden_dim"] = current_hidden_dim

    # DataLoaders (truyền backbone vào để tokenizer đồng nhất)
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer_name=backbone,
        batch_size=args.batch_size,
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
    )

    # Load metadata cho class weights (hoặc tính tự động từ tập train)
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
        weights = []
        total = sum(topic_counts.values())
        for i in range(n_topics):
            count = topic_counts.get(i, 0)
            if count > 0:
                weights.append(total / count)
            else:
                weights.append(1.0)
        
        min_w = min(weights)
        topic_class_weights = [w / min_w for w in weights]
        print(f"   Calculated Topic Weights: {topic_class_weights}")

    # Build model
    print(f"🤖 Building KEHN with backbone: {backbone}")
    model = build_model(backbone, topic_class_weights, device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {n_params:,}")
    print(f"   Trainable:    {n_trainable:,}")

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    # FP16 Mixed Precision & Gradient Accumulation
    accum_steps = TRAIN_CONFIG.get("gradient_accumulation_steps", 1)
    use_amp = device.type == "cuda" and TRAIN_CONFIG.get("fp16", False)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps, pct_start=TRAIN_CONFIG["warmup_ratio"],
    )

    # Curriculum
    curriculum = CurriculumScheduler()

    # Early stopping
    best_metric = 0.0
    patience_counter = 0
    best_epoch = 0

    # Output directory
    output_dir = JOINT_OUTPUT_DIR / f"{args.exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🚀 Training KEHN — Experiment: {args.exp_name}")
    print(f"   Backbone: {backbone}")
    print(f"   Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"   FP16: {use_amp}, Grad Accum: {accum_steps}x (effective batch={args.batch_size * accum_steps})")
    print(f"{'=' * 60}\n")

    history = []

    for epoch in range(1, args.epochs + 1):
        phase = curriculum.get_phase(epoch)
        curriculum.apply_freeze(model, epoch)
        phase_info = curriculum.get_phase_info(epoch)

        print(f"\n📅 Epoch {epoch}/{args.epochs} | {phase_info}")

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            topic_labels = batch["topic_labels"].to(device)
            intent_labels = batch["intent_labels"].to(device)
            ner_labels = batch["ner_labels"].to(device)
            token_intent_ids = batch.get("token_intent_ids", None)
            if token_intent_ids is not None:
                token_intent_ids = token_intent_ids.to(device)

            # Forward with AMP autocast
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(
                    input_ids, attention_mask,
                    topic_labels=topic_labels,
                    intent_labels=intent_labels,
                    ner_labels=ner_labels,
                    phase=phase,
                    token_intent_ids=token_intent_ids,
                )
                loss = output["loss"] / accum_steps

            # Backward with GradScaler
            scaler.scale(loss).backward()

            # Gradient accumulation: step only every accum_steps
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            n_batches += 1

            if (step + 1) % 50 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"   Step {step + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = evaluate(model, val_loader, device, phase)
        topic_f1 = val_metrics["topic_macro_f1"]
        ner_f1 = val_metrics["ner_f1"]
        intent_acc = val_metrics["intent_accuracy"]

        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val — Topic F1: {topic_f1:.4f} | Intent Acc: {intent_acc:.4f} | NER F1: {ner_f1:.4f}")

        epoch_record = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": avg_train_loss,
            **val_metrics,
        }
        history.append(epoch_record)

        # Early stopping (on topic_macro_f1)
        if topic_f1 > best_metric:
            best_metric = topic_f1
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"   💾 New best! Topic F1={topic_f1:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG["patience"] and epoch > 10:
                print(f"\n⏹️ Early stopping at epoch {epoch} (patience={TRAIN_CONFIG['patience']})")
                break

    # Load best model and evaluate on test set
    print(f"\n{'=' * 60}")
    print(f"📊 Final Evaluation (best model from epoch {best_epoch})")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    test_metrics = evaluate(model, test_loader, device, "full")
    print(f"   Topic Macro-F1:    {test_metrics['topic_macro_f1']:.4f}")
    print(f"   Intent Accuracy:   {test_metrics['intent_accuracy']:.4f}")
    print(f"   NER F1:            {test_metrics['ner_f1']:.4f}")
    print(f"   Semantic Accuracy: {test_metrics['semantic_accuracy']:.4f}")

    # Save results
    results = {
        "experiment": args.exp_name,
        "backbone": backbone,
        "best_epoch": best_epoch,
        "best_val_topic_f1": best_metric,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Training complete! Results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train KEHN Joint Model")
    parser.add_argument("--exp_name", type=str, default="E4_kehn_phobert",
                        help="Experiment name")
    parser.add_argument("--backbone", type=str, default="phobert",
                        choices=["phobert", "vihealthbert"],
                        help="Backbone model key")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size for smoke testing")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
