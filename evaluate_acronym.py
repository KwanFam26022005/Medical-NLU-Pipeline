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
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
) -> str:
    """Predict the best expansion for a single sample."""
    text = sample["text"]
    start = sample["start_char_idx"]
    length = sample["length_acronym"]
    acronym = text[start: start + length]

    # Entity marking 
    marked_text = text[:start] + "<e>" + acronym + "</e>" + text[start + length:]

    candidates = acronym_dict.get(acronym, [])
    if not candidates:
        return acronym
    if len(candidates) == 1:
        return candidates[0]

    # Encode all candidates
    encodings = tokenizer(
        [marked_text] * len(candidates),
        candidates,
        max_length=max_length,
        padding=True,
        truncation="only_first",
        return_tensors="pt",
    ).to(device)

    logits = model(**encodings).logits.squeeze(-1)
    best_idx = logits.argmax().item()
    return candidates[best_idx]


def evaluate(
    model, tokenizer, acronym_dict, samples: List[Dict],
    device: torch.device, train_acronyms: set = None,
) -> Dict[str, Any]:
    """Run evaluation on a list of samples."""
    predictions = {}
    total = 0
    correct = 0
    seen_total = seen_correct = 0
    unseen_total = unseen_correct = 0
    mrr_sum = 0.0

    for idx, sample in enumerate(samples):
        text = sample["text"]
        start = sample["start_char_idx"]
        length = sample["length_acronym"]
        acronym = text[start: start + length]
        gold = sample["expansion"]

        # Predict
        pred = predict_sample(model, tokenizer, acronym_dict, sample, device)
        predictions[str(idx)] = pred

        total += 1
        if pred == gold:
            correct += 1

        # MRR
        marked_text = text[:start] + "<e>" + acronym + "</e>" + text[start + length:]
        candidates = acronym_dict.get(acronym, [])
        if candidates:
            encodings = tokenizer(
                [marked_text] * len(candidates),
                candidates,
                max_length=128, padding=True, truncation="only_first", return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**encodings).logits.squeeze(-1)
            scores = logits.cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(ranked, 1):
                if candidates[i] == gold:
                    mrr_sum += 1.0 / rank
                    break

        # Seen/unseen
        if train_acronyms is not None:
            if acronym in train_acronyms:
                seen_total += 1
                if pred == gold:
                    seen_correct += 1
            else:
                unseen_total += 1
                if pred == gold:
                    unseen_correct += 1

    metrics = {
        "accuracy": correct / max(total, 1) * 100,
        "mrr": mrr_sum / max(total, 1),
        "total": total,
        "correct": correct,
        "seen_accuracy": seen_correct / max(seen_total, 1) * 100 if seen_total > 0 else None,
        "unseen_accuracy": unseen_correct / max(unseen_total, 1) * 100 if unseen_total > 0 else None,
        "seen_total": seen_total,
        "unseen_total": unseen_total,
    }
    return predictions, metrics


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

    # Evaluate
    print("\n📊 Running evaluation...")
    start = time.time()
    predictions, metrics = evaluate(
        model, tokenizer, acronym_dict, test_samples, device, train_acronyms
    )
    elapsed = time.time() - start

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


if __name__ == "__main__":
    main()
