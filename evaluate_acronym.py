"""
evaluate_acronym.py - Evaluation script for Cross-Encoder Acronym WSD.

Reads test/data.json, runs inference, outputs predictions.json.

Usage (Colab):
    !python evaluate_acronym.py \
        --model_dir saved_models/acronym_wsd \
        --data_dir data/acrDrAid \
        --output predictions.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval_utils import evaluate_cross_encoder


def load_model_and_dict(model_dir: str, device: torch.device):
    """Load trained model, tokenizer, and dictionary."""
    model_path = Path(model_dir)
    is_local = model_path.exists()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=1
    )
    model.to(device)
    model.eval()

    # Load dictionary
    if is_local:
        dict_path = model_path / "acronym_dict.json"
        if not dict_path.exists():
            raise FileNotFoundError(f"No acronym_dict.json in {model_dir}")
    else:
        from huggingface_hub import hf_hub_download
        dict_path = hf_hub_download(repo_id=model_dir, filename="acronym_dict.json")

    with open(dict_path, "r", encoding="utf-8") as f:
        acronym_dict = json.load(f)

    return model, tokenizer, acronym_dict


@torch.no_grad()
def predict_sample(
    model, tokenizer, acronym_dict, sample: Dict, device: torch.device, max_length: int = 128
) -> Tuple[str, List[str], List[float]]:
    """Predict the best expansion for a single sample.

    Returns:
        (predicted_expansion, candidates, scores) — scores reusable for MRR.
    """
    text = sample["text"]
    start = sample["start_char_idx"]
    length = sample["length_acronym"]
    acronym = text[start: start + length]

    # Entity marking 
    marked_text = text[:start] + "<e>" + acronym + "</e>" + text[start + length:]

    candidates = acronym_dict.get(acronym, [])
    if not candidates:
        return acronym, [], []
    if len(candidates) == 1:
        return candidates[0], candidates, [1.0]

    # Encode all candidates — single forward pass
    encodings = tokenizer(
        [marked_text] * len(candidates),
        candidates,
        max_length=max_length,
        padding=True,
        truncation="only_first",
        return_tensors="pt",
    ).to(device)

    logits = model(**encodings).logits.squeeze(-1)
    scores = logits.cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx], candidates, scores


# evaluate() is now provided by eval_utils.evaluate_cross_encoder()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Encoder Acronym WSD")
    parser.add_argument("--model_dir", type=str, default="saved_models/acronym_wsd")
    parser.add_argument("--data_dir", type=str, default="data/acrDrAid")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Load model
    model, tokenizer, acronym_dict = load_model_and_dict(args.model_dir, device)
    print(f"✅ Model loaded from {args.model_dir}")
    print(f"   Dictionary: {len(acronym_dict)} acronyms")

    # Load data
    data_dir = Path(args.data_dir)
    with open(data_dir / args.split / "data.json", "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    print(f"   {args.split.capitalize()} samples: {len(test_samples)}")

    # Load train acronyms for seen/unseen split
    train_acronyms = set()
    train_path = data_dir / "train" / "data.json"
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_samples = json.load(f)
        for s in train_samples:
            acr = s["text"][s["start_char_idx"]: s["start_char_idx"] + s["length_acronym"]]
            train_acronyms.add(acr)
        print(f"   Train acronyms (seen): {len(train_acronyms)}")

    # Evaluate using shared module
    print("\n📊 Running evaluation...")
    start = time.time()
    predictions, raw_metrics = evaluate_cross_encoder(
        model, tokenizer, acronym_dict, test_samples, device, train_acronyms
    )
    elapsed = time.time() - start

    # Remap keys for backward-compatible output format
    metrics = {
        "accuracy": raw_metrics["accuracy"],
        "mrr": raw_metrics["mrr"],
        "total": raw_metrics["total"],
        "seen_accuracy": raw_metrics["seen_acc"] if raw_metrics["seen_total"] > 0 else None,
        "unseen_accuracy": raw_metrics["unseen_acc"] if raw_metrics["unseen_total"] > 0 else None,
        "seen_total": raw_metrics["seen_total"],
        "unseen_total": raw_metrics["unseen_total"],
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"📊 RESULTS on {args.split.upper()} set ({elapsed:.1f}s)")
    print(f"{'='*50}")
    print(f"   Accuracy:       {metrics['accuracy']:.2f}%")
    print(f"   MRR:            {metrics['mrr']:.4f}")
    print(f"   Total:          {metrics['total']}")
    if metrics['seen_accuracy'] is not None:
        print(f"   Seen Acc:       {metrics['seen_accuracy']:.2f}% ({metrics['seen_total']})")
    if metrics['unseen_accuracy'] is not None:
        print(f"   Unseen Acc:     {metrics['unseen_accuracy']:.2f}% ({metrics['unseen_total']})")
    # Save predictions
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Predictions saved to {args.output}")

    # Save metrics
    metrics_path = Path(args.output).stem + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 Metrics saved to {metrics_path}")

    # ── P4: Error Analysis ──────────────────────────────────
    from collections import defaultdict

    errors = []
    per_acronym = defaultdict(lambda: {"total": 0, "correct": 0})

    for idx, sample in enumerate(test_samples):
        text = sample["text"]
        s = sample["start_char_idx"]
        ln = sample["length_acronym"]
        acronym = text[s: s + ln]
        gold = sample["expansion"]
        pred_str = predictions[str(idx)]

        per_acronym[acronym]["total"] += 1
        if pred_str == gold:
            per_acronym[acronym]["correct"] += 1
        else:
            # Get scores for error details
            _, cands, scores = predict_sample(
                model, tokenizer, acronym_dict, sample, device
            )
            top_scores = dict(zip(cands, scores)) if cands and scores else {}
            sorted_scores = sorted(scores, reverse=True) if scores else []
            margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else None

            errors.append({
                "idx": idx,
                "acronym": acronym,
                "gold": gold,
                "predicted": pred_str,
                "is_seen": acronym in train_acronyms if train_acronyms else None,
                "top_scores": top_scores,
                "margin": margin,
            })

    # Per-acronym accuracy
    per_acronym_list = {}
    for acr, stats in sorted(per_acronym.items(), key=lambda x: x[1]["correct"] / max(x[1]["total"], 1)):
        stats["accuracy"] = round(stats["correct"] / max(stats["total"], 1) * 100, 2)
        per_acronym_list[acr] = stats

    error_analysis = {
        "total_errors": len(errors),
        "total_samples": len(test_samples),
        "error_rate": round(len(errors) / max(len(test_samples), 1) * 100, 2),
        "errors": errors,
        "per_acronym": per_acronym_list,
    }

    ea_path = Path(args.output).stem + "_error_analysis.json"
    with open(ea_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    print(f"🔍 Error analysis saved to {ea_path} ({len(errors)} errors)")


if __name__ == "__main__":
    main()
