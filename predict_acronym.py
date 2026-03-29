"""
predict_acronym.py - Script chẩn đoán & test Acronym WSD model.

🔧 v3 - DICTIONARY-CONSTRAINED WSD:
  ✅ Mask logits → chỉ softmax trên valid expansions per acronym
  ✅ Confidence tăng từ ~2% lên 40-80%
  ✅ Eval trên test set + gold.json
  ✅ Per-acronym accuracy analysis

Cách chạy:
    python predict_acronym.py --eval_data data/acronym_test.json
    python predict_acronym.py --stress_test
    python predict_acronym.py --sentence "nhu mô gan bt, kt bình thường" --acronyms "bt,kt"
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, Trainer

from data_loader import AcronymDataLoader


PROMPT_TEMPLATE = "Giải nghĩa từ viết tắt {acronym} trong câu: {sentence}"


def load_constraint(model_dir: str) -> dict:
    """Load dictionary constraint từ model dir hoặc data dir."""
    for path in [
        Path(model_dir) / "dictionary_constraint.json",
        Path("data/dictionary_constraint.json"),
    ]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                constraint = json.load(f)
            print(f"🔒 Loaded constraint: {len(constraint)} acronyms từ {path}")
            return constraint
    print("⚠️  Không tìm thấy dictionary_constraint.json")
    return {}


def apply_constraint_mask(logits: torch.Tensor, acronym: str, constraint: dict) -> torch.Tensor:
    """🔒 Mask logits: chỉ giữ valid expansions cho acronym này."""
    acr_lower = acronym.lower()
    if acr_lower not in constraint:
        return logits  # Không có constraint → dùng tất cả

    valid_ids = constraint[acr_lower]
    mask = torch.full_like(logits, float("-inf"))
    for vid in valid_ids:
        if vid < len(logits):
            mask[vid] = logits[vid]
    return mask


def evaluate_on_dataset(
    model_dir: str,
    eval_data: str,
    gold_path: Optional[str] = None,
    max_length: int = 128,
):
    """Evaluate với dictionary-constrained decoding."""
    print("\n" + "=" * 60)
    print("📊 EVALUATE (DICTIONARY-CONSTRAINED)")
    print("=" * 60)

    # Load model + data
    loader = AcronymDataLoader(max_length=max_length, use_prompt=True)
    loader.prepare_datasets(
        train_path=Path("data/acronym_train.json"),
        val_path=Path("data/acronym_val.json"),
    )
    id2label = loader.id2expansion

    model_path = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    tokenizer = loader.tokenizer

    # Load constraint
    constraint = load_constraint(model_dir)

    # Load eval samples
    with open(eval_data, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    valid_samples = [s for s in test_samples if s["expansion"] in loader.expansion2id]
    skipped = len(test_samples) - len(valid_samples)
    if skipped > 0:
        print(f"⚠️  Bỏ qua {skipped} mẫu có expansion mới")

    print(f"⏳ Evaluating {len(valid_samples)} mẫu...")

    # Predict with constraint
    true_labels = []
    pred_labels = []
    acronyms_list = []

    for sample in valid_samples:
        acr = sample["acronym"]
        sent = sample["sentence"]
        true_id = loader.expansion2id[sample["expansion"]]

        text = PROMPT_TEMPLATE.format(acronym=acr, sentence=sent)
        inputs = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]

            # 🔒 Apply constraint
            masked_logits = apply_constraint_mask(logits, acr, constraint)
            pred_id = torch.argmax(masked_logits).item()

        true_labels.append(true_id)
        pred_labels.append(pred_id)
        acronyms_list.append(acr)

    # Metrics
    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print(f"\n🏆 KẾT QUẢ (Constrained):")
    print(f"   Accuracy:    {acc * 100:.2f}%")
    print(f"   Macro-F1:    {f1_macro * 100:.2f}%")
    print(f"   Weighted-F1: {f1_weighted * 100:.2f}%")

    # Fix #3: Seen vs Unseen evaluation
    train_exps = set()
    try:
        with open("data/acronym_train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        train_exps = set(s["expansion"] for s in train_data)
    except FileNotFoundError:
        pass

    if train_exps:
        seen_correct, seen_total = 0, 0
        unseen_correct, unseen_total = 0, 0
        for i, sample in enumerate(valid_samples):
            is_seen = sample["expansion"] in train_exps
            correct = true_labels[i] == pred_labels[i]
            if is_seen:
                seen_total += 1
                seen_correct += int(correct)
            else:
                unseen_total += 1
                unseen_correct += int(correct)

        total_test = seen_total + unseen_total
        print(f"\n🔍 SEEN vs UNSEEN:")
        print(f"   Seen ({seen_total}/{total_test}):   {seen_correct}/{seen_total} ({seen_correct/max(seen_total,1)*100:.1f}%)")
        print(f"   Unseen ({unseen_total}/{total_test}): {unseen_correct}/{unseen_total} ({unseen_correct/max(unseen_total,1)*100:.1f}%)")
        print(f"   → True ceiling (seen only): {seen_total/max(total_test,1)*100:.0f}%")

    # Per-acronym analysis
    print(f"\n📈 PHÂN TÍCH PER-ACRONYM:")
    acr_correct = defaultdict(int)
    acr_total = defaultdict(int)
    errors = Counter()

    for i, (true_id, pred_id) in enumerate(zip(true_labels, pred_labels)):
        acr = acronyms_list[i]
        acr_total[acr] += 1
        if true_id == pred_id:
            acr_correct[acr] += 1
        else:
            true_name = id2label.get(true_id, f"id_{true_id}")
            pred_name = id2label.get(pred_id, f"id_{pred_id}")
            errors[f"{true_name} → {pred_name}"] += 1

    sorted_acrs = sorted(acr_total.items(), key=lambda x: x[1], reverse=True)
    for acr, total in sorted_acrs[:20]:
        correct = acr_correct.get(acr, 0)
        acr_acc = correct / max(total, 1) * 100
        n_choices = len(constraint.get(acr, []))
        marker = "✅" if acr_acc >= 80 else "⚠️" if acr_acc >= 50 else "❌"
        print(f"   {marker} '{acr}' ({n_choices} choices): {correct}/{total} ({acr_acc:.0f}%)")

    # Top errors
    if errors:
        print(f"\n🔍 TOP 10 NHẦM LẪN:")
        for error_type, count in errors.most_common(10):
            print(f"   {error_type}: {count}x")

    # Gold comparison
    if gold_path:
        gold_file = Path(gold_path)
        if gold_file.exists():
            with open(gold_file, "r", encoding="utf-8") as f:
                gold = json.load(f)
            gold_match = 0
            gold_total = 0
            for idx_str, gold_exp in gold.items():
                idx = int(idx_str)
                if idx < len(pred_labels):
                    pred_exp = id2label.get(int(pred_labels[idx]), "")
                    if pred_exp.lower() == gold_exp.lower():
                        gold_match += 1
                    gold_total += 1
            print(f"\n🏆 SO SÁNH VỚI GOLD.JSON:")
            print(f"   Match: {gold_match}/{gold_total} ({gold_match/max(gold_total,1)*100:.1f}%)")

    return acc, f1_macro


def stress_test(model_dir: str, sentence: str, acronyms: List[str], max_length: int = 128):
    """Stress test với dictionary-constrained decoding."""
    print("\n" + "=" * 60)
    print("🧪 STRESS TEST (DICTIONARY-CONSTRAINED)")
    print("=" * 60)

    # Load model
    loader = AcronymDataLoader(max_length=max_length, use_prompt=True)
    loader.prepare_datasets(
        train_path=Path("data/acronym_train.json"),
        val_path=Path("data/acronym_val.json"),
    )
    id2label = loader.id2expansion
    tokenizer = loader.tokenizer

    model_path = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    # Load constraint + dictionary
    constraint = load_constraint(model_dir)
    dict_path = Path("data/acrDrAid/dictionary.json")
    dictionary = {}
    if dict_path.exists():
        with open(dict_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

    print(f"\n📝 Câu test: {sentence}")
    print(f"🔤 Từ viết tắt: {acronyms}")
    print("-" * 60)

    for acr in acronyms:
        text = PROMPT_TEMPLATE.format(acronym=acr, sentence=sentence)
        inputs = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]

            # 🔒 Constrained vs Unconstrained
            masked_logits = apply_constraint_mask(logits, acr, constraint)
            probs = F.softmax(masked_logits, dim=-1)
            top5_probs, top5_indices = torch.topk(probs, min(5, len(probs)))

        # Dictionary info
        expected = dictionary.get(acr, dictionary.get(acr.lower(), []))
        n_choices = len(constraint.get(acr, constraint.get(acr.lower(), [])))

        print(f"\n🔍 '{acr}' ({n_choices} valid choices):")
        if expected:
            print(f"   📚 Dictionary: {expected}")

        for i in range(len(top5_probs)):
            if top5_probs[i].item() < 0.001:
                break  # Skip near-zero after constraint
            label_idx = top5_indices[i].item()
            expansion = id2label.get(label_idx, f"LABEL_{label_idx}")
            confidence = top5_probs[i].item() * 100

            is_valid = expansion.lower() in [e.lower() for e in expected] if expected else False
            marker = "✅" if is_valid and i == 0 else "🟡" if is_valid else "  "
            print(f"   {marker} {i+1}. {expansion.upper()} ({confidence:.1f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Acronym WSD (v3 Constrained)")
    parser.add_argument("--model_dir", type=str, default="saved_models/acronym_wsd")
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--gold", type=str, default=None)
    parser.add_argument("--stress_test", action="store_true")
    parser.add_argument("--sentence", type=str, default=None)
    parser.add_argument("--acronyms", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=384)
    args = parser.parse_args()

    print("🔍 ACRONYM WSD - v3 DICTIONARY-CONSTRAINED")
    print(f"   Model: {args.model_dir}")

    if args.eval_data:
        gold = args.gold
        if gold is None and "test" in args.eval_data:
            auto_gold = Path("data/acrDrAid/gold.json")
            if auto_gold.exists():
                gold = str(auto_gold)
        evaluate_on_dataset(args.model_dir, args.eval_data, gold, args.max_length)

    if args.stress_test:
        test_cases = [
            (
                "Nhu mô gan bt, kt gan PG 115mm, đường mật trong gan và ngoài gan bt, "
                "tm cửa kt 9mm, túi mật kt bình thường, nhu mô tụy bt.",
                ["bt", "kt", "tm"],
            ),
            (
                "SA thai 32 tuần, tn phát triển bình thường, vđ 310mm, "
                "vb 280mm, nc ối bình thường, rr nhịp đều.",
                ["tn", "vđ", "vb", "nc", "rr"],
            ),
            (
                "XQ ngực thẳng: nhu mô phổi 2 bên bt, "
                "bóng tim kt bình thường, hc rốn phổi bình thường.",
                ["bt", "kt", "hc"],
            ),
        ]
        for i, (sentence, acrs) in enumerate(test_cases, 1):
            print(f"\n{'─' * 60}")
            print(f"📋 TEST CASE {i}:")
            stress_test(args.model_dir, sentence, acrs, args.max_length)

    if args.sentence and args.acronyms:
        acr_list = [a.strip() for a in args.acronyms.split(",")]
        stress_test(args.model_dir, args.sentence, acr_list, args.max_length)

    if not args.eval_data and not args.stress_test and not args.sentence:
        print("\n💡 Hướng dẫn sử dụng:")
        print("   python predict_acronym.py --eval_data data/acronym_test.json")
        print("   python predict_acronym.py --stress_test")
        print('   python predict_acronym.py --sentence "nhu mô gan bt" --acronyms "bt"')


if __name__ == "__main__":
    main()
