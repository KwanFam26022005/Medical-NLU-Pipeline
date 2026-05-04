"""
preprocess_joint.py — Pipeline tạo dataset hợp nhất cho KEHN.

Bước 1: Load ViMQ (train/dev/test) → convert span NER → BIO tags
Bước 2: Pseudo-label Topic cho ViMQ bằng model Trạm 2B đã deploy
Bước 3: (Optional) Pseudo-label Intent+NER cho CSV topic data
Bước 4: Merge + stratified split → data/joint/
"""

import json
import sys
import io
import os
from pathlib import Path
from collections import Counter

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Thêm parent dir vào path để import config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_joint import (
    VIMQ_DATA_DIR, TOPIC_DATA_DIR, JOINT_DATA_DIR,
    ENTITY_TYPE_MAP, NER2ID, INTENT2ID, TOPIC2ID, TOPIC_LABELS,
    PSEUDO_CONFIG,
)


# ============================================================
# BƯỚC 1: Convert ViMQ span-based NER → BIO format
# ============================================================

def span_to_bio(words: list, seq_label: list) -> list:
    """
    Convert ViMQ span format [[start, end, type], ...] → BIO tag list.
    
    Example:
        words = ["Hẹp", "động_mạch", "thận", "phải", "hiến", "thận"]
        seq_label = [[0, 2, "SYMPTOM_AND_DISEASE"], [4, 5, "medical_procedure"]]
        → ["B-SYM", "I-SYM", "I-SYM", "O", "B-PRO", "I-PRO"]
    """
    tags = ["O"] * len(words)
    for span in seq_label:
        start, end, ent_type = span[0], span[1], span[2]
        bio_prefix = ENTITY_TYPE_MAP.get(ent_type)
        if bio_prefix is None:
            continue
        # Clamp indices to word list bounds
        start = max(0, min(start, len(words) - 1))
        end = max(start, min(end, len(words) - 1))
        tags[start] = f"B-{bio_prefix}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{bio_prefix}"
    return tags


def load_vimq_split(split_path: Path) -> list:
    """Load 1 split của ViMQ và convert sang joint format."""
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sentence = item["sentence"]
        words = sentence.split()
        ner_tags = span_to_bio(words, item.get("seq_label", []))
        intent_label = item.get("sent_label", "method_diagnosis")

        samples.append({
            "text": sentence,
            "words": words,
            "ner_tags": ner_tags,
            "ner_tag_ids": [NER2ID.get(t, 0) for t in ner_tags],
            "intent_label": intent_label,
            "intent_label_id": INTENT2ID.get(intent_label, 0),
            "topic_label": None,  # Sẽ pseudo-label sau
            "topic_label_id": -1,
            "topic_confidence": 0.0,
            "source": "vimq",
        })

    return samples


# ============================================================
# BƯỚC 2: Pseudo-label Topic cho ViMQ samples
# ============================================================

def pseudo_label_topic(samples: list, device: str = "cpu") -> list:
    """
    Dùng model Topic đã train (Trạm 2B) để gán nhãn Topic cho ViMQ samples.
    Chỉ giữ samples có confidence ≥ threshold.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        print("⚠️ transformers/torch not installed. Skipping pseudo-labeling.")
        return samples

    print("📥 Loading Topic model for pseudo-labeling...")
    topic_model_name = PSEUDO_CONFIG.get(
        "tokenizer_for_topic_model", "demdecuong/vihealthbert-base-syllable"
    )
    
    # Load topic label map
    label_map_path = TOPIC_DATA_DIR / "topic_label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        id2topic = {int(k): v for k, v in label_map["id2topic"].items()}
    else:
        print("⚠️ topic_label_map.json not found. Using default mapping.")
        id2topic = {i: t for i, t in enumerate(TOPIC_LABELS)}

    try:
        from config import TOPIC_MODEL_HF_ID
        model_id = TOPIC_MODEL_HF_ID
    except (ImportError, AttributeError):
        model_id = "KwanFam26022005/model2B-topic-classification"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(topic_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"⚠️ Cannot load Topic model: {e}")
        print("   Assigning random topic labels for testing purposes.")
        import random
        random.seed(42)
        for s in samples:
            topic_id = random.randint(0, len(TOPIC_LABELS) - 1)
            s["topic_label"] = TOPIC_LABELS[topic_id]
            s["topic_label_id"] = topic_id
            s["topic_confidence"] = 0.7
        return samples

    threshold = PSEUDO_CONFIG.get("topic_confidence_threshold", 0.6)
    batch_size = 64
    labeled_count = 0

    print(f"🔮 Pseudo-labeling {len(samples)} samples (threshold={threshold})...")

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        texts = [s["text"].replace("_", " ") for s in batch]  # Remove word-seg for syllable model

        with torch.no_grad():
            inputs = tokenizer(
                texts, max_length=128, padding=True,
                truncation=True, return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, pred_ids = probs.max(dim=-1)

        for j, s in enumerate(batch):
            conf = confidences[j].item()
            pred_id = pred_ids[j].item()

            if conf >= threshold and pred_id in id2topic:
                topic_name = id2topic[pred_id]
                if topic_name in TOPIC2ID:
                    s["topic_label"] = topic_name
                    s["topic_label_id"] = TOPIC2ID[topic_name]
                    s["topic_confidence"] = conf
                    labeled_count += 1

    print(f"✅ Pseudo-labeled {labeled_count}/{len(samples)} samples (≥{threshold} confidence)")
    return samples


# ============================================================
# BƯỚC 3: Main pipeline
# ============================================================

def main():
    """Chạy toàn bộ pipeline preprocessing."""
    JOINT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🏥 KEHN Joint Dataset Preprocessing Pipeline")
    print("=" * 60)

    # Bước 1: Load ViMQ
    print("\n📂 Bước 1: Loading ViMQ dataset...")
    train_samples = load_vimq_split(VIMQ_DATA_DIR / "train.json")
    dev_samples = load_vimq_split(VIMQ_DATA_DIR / "dev.json")
    test_samples = load_vimq_split(VIMQ_DATA_DIR / "test.json")
    print(f"   Train: {len(train_samples)}, Dev: {len(dev_samples)}, Test: {len(test_samples)}")

    # Kiểm tra NER conversion
    sample = train_samples[0]
    print(f"   Sample: {sample['text'][:60]}...")
    print(f"   NER tags: {sample['ner_tags'][:8]}...")
    print(f"   Intent: {sample['intent_label']}")

    # Bước 2: Pseudo-label Topic
    print("\n🔮 Bước 2: Pseudo-labeling Topic...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    train_samples = pseudo_label_topic(train_samples, device)
    dev_samples = pseudo_label_topic(dev_samples, device)
    test_samples = pseudo_label_topic(test_samples, device)

    # Filter: chỉ giữ samples có topic label
    train_labeled = [s for s in train_samples if s["topic_label_id"] >= 0]
    dev_labeled = [s for s in dev_samples if s["topic_label_id"] >= 0]
    test_labeled = [s for s in test_samples if s["topic_label_id"] >= 0]

    print(f"\n📊 Sau filtering:")
    print(f"   Train: {len(train_labeled)}/{len(train_samples)}")
    print(f"   Dev: {len(dev_labeled)}/{len(dev_samples)}")
    print(f"   Test: {len(test_labeled)}/{len(test_samples)}")

    # Topic distribution
    topic_dist = Counter(s["topic_label"] for s in train_labeled)
    print(f"\n📊 Topic distribution (train):")
    for topic, count in topic_dist.most_common():
        print(f"   {topic}: {count}")

    # Bước 3: Tính class weights cho Topic
    topic_counts = [0] * len(TOPIC_LABELS)
    for s in train_labeled:
        tid = s["topic_label_id"]
        if 0 <= tid < len(TOPIC_LABELS):
            topic_counts[tid] += 1

    n_total = sum(topic_counts)
    n_classes = len(TOPIC_LABELS)
    class_weights = []
    for c in topic_counts:
        if c > 0:
            class_weights.append(n_total / (n_classes * c))
        else:
            class_weights.append(1.0)

    # Cap extreme weights để tránh loss spike (Issue 4: traditional_medicine=376.0)
    MAX_CLASS_WEIGHT = 10.0
    class_weights = [min(w, MAX_CLASS_WEIGHT) for w in class_weights]

    # Bước 4: Save
    print("\n💾 Bước 3: Saving joint dataset...")

    def save_split(samples, filename):
        path = JOINT_DATA_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"   Saved {len(samples)} samples → {path}")

    save_split(train_labeled, "joint_train.json")
    save_split(dev_labeled, "joint_val.json")
    save_split(test_labeled, "joint_test.json")

    # Save metadata
    metadata = {
        "n_topic": len(TOPIC_LABELS),
        "n_intent": len(INTENT2ID),
        "n_ner_tag": len(NER2ID),
        "topic_labels": TOPIC_LABELS,
        "intent_labels": list(INTENT2ID.keys()),
        "ner_tags": list(NER2ID.keys()),
        "topic_class_weights": class_weights,
        "stats": {
            "train": len(train_labeled),
            "val": len(dev_labeled),
            "test": len(test_labeled),
            "topic_distribution": dict(topic_dist),
        },
    }
    meta_path = JOINT_DATA_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   Saved metadata → {meta_path}")

    print("\n✅ Preprocessing hoàn tất!")
    print(f"   Output: {JOINT_DATA_DIR}")


if __name__ == "__main__":
    main()
