"""
pseudo_cross_source.py — Pseudo-label Intent + NER cho hospital data (single-task sources).

Output: kehn_joint/data/pseudo_new/  (NEVER merged into existing pipeline data)

Workflow:
  1. Load single-task topic training data (human-labeled topics)
  2. Pseudo-label Intent using vihealthbert-asl-medical-intent
  3. Pseudo-label NER using vihealthbert-crf-medical-ner
  4. Filter: only keep samples where ALL 3 labels have confidence >= threshold
  5. Apply TOPIC_REMAP / TOPIC_DROP from config
  6. Save to pseudo_new/ directory
"""

import json
import sys
import io
from pathlib import Path
from collections import Counter

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config_joint import (
    TOPIC_DATA_DIR, JOINT_DATA_DIR,
    INTENT_LABELS, INTENT2ID, NER2ID, NER_TAGS,
    TOPIC2ID, TOPIC_LABELS, TOPIC_REMAP, TOPIC_DROP,
    INTENT_MODEL_HF, NER_MODEL_HF,
)

# Output directory — isolated from existing pipeline
PSEUDO_NEW_DIR = JOINT_DATA_DIR / "pseudo_new"

# Confidence threshold for ALL 3 tasks
CONFIDENCE_THRESHOLD = 0.85


def load_hospital_topic_data(path: Path) -> list:
    """Load single-task topic_train.json (human-labeled topics)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        text = item.get("text", "").strip()
        topic = item.get("topic", "")
        if isinstance(topic, int):
            # 'topic' field is sometimes int ID — skip, use label map
            continue
        topic = str(topic).strip()

        if not text or not topic:
            continue

        # Apply remap/drop
        if topic in TOPIC_DROP:
            continue
        if topic in TOPIC_REMAP:
            topic = TOPIC_REMAP[topic]
        if topic not in TOPIC2ID:
            continue

        samples.append({
            "text": text,
            "topic_label": topic,
            "topic_label_id": TOPIC2ID[topic],
            "topic_confidence": 1.0,  # human label
            "source": "hospital",
        })

    return samples


def pseudo_label_intent(samples: list, device: str = "cpu") -> list:
    """Pseudo-label Intent using the trained intent model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        print("⚠️ transformers/torch not installed.")
        return samples

    print(f"📥 Loading Intent model: {INTENT_MODEL_HF}")
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_HF)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_HF)
    model.to(device).eval()

    # Detect if model is multi-label (sigmoid) or single-label (softmax)
    n_labels = model.config.num_labels
    print(f"   Model has {n_labels} labels")

    batch_size = 64
    labeled = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        texts = [s["text"] for s in batch]

        with torch.no_grad():
            inputs = tokenizer(
                texts, max_length=128, padding=True,
                truncation=True, return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)

            if n_labels == len(INTENT_LABELS):
                # Single-label intent
                probs = torch.softmax(outputs.logits, dim=-1)
                confidences, pred_ids = probs.max(dim=-1)
                for j, s in enumerate(batch):
                    conf = confidences[j].item()
                    pred_id = pred_ids[j].item()
                    if pred_id < len(INTENT_LABELS):
                        s["intent_label"] = INTENT_LABELS[pred_id]
                        s["intent_label_id"] = pred_id
                        s["intent_confidence"] = conf
                        labeled += 1
            else:
                # Multi-label intent (ASL model) — take argmax for joint format
                probs = torch.sigmoid(outputs.logits)
                confidences, pred_ids = probs.max(dim=-1)
                for j, s in enumerate(batch):
                    conf = confidences[j].item()
                    pred_id = pred_ids[j].item()
                    if pred_id < len(INTENT_LABELS):
                        s["intent_label"] = INTENT_LABELS[pred_id]
                        s["intent_label_id"] = pred_id
                        s["intent_confidence"] = conf
                        labeled += 1

    print(f"✅ Intent pseudo-labeled: {labeled}/{len(samples)}")
    return samples


def pseudo_label_ner(samples: list, device: str = "cpu") -> list:
    """Pseudo-label NER using the trained NER model (ViHealthBERT + CRF)."""
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        import torch
        from custom_models import ViHealthBertCRF
    except ImportError:
        print("⚠️ transformers/torch/custom_models not installed.")
        return samples

    print(f"📥 Loading NER model: {NER_MODEL_HF}")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_HF)
    
    # Init custom architecture
    model = ViHealthBertCRF(num_labels=len(NER_TAGS))
    # Download and load state_dict
    ckpt_path = hf_hub_download(repo_id=NER_MODEL_HF, filename="pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Build id2tag from NER_TAGS
    id2tag = {i: t for i, t in enumerate(NER_TAGS)}
    print(f"   NER tags: {list(id2tag.values())[:7]}...")

    batch_size = 32
    labeled = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        texts = [s["text"] for s in batch]

        with torch.no_grad():
            inputs = tokenizer(
                texts, max_length=128, padding=True,
                truncation=True, return_tensors="pt",
                return_offsets_mapping=True,
            )
            offset_mapping = inputs.pop("offset_mapping")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # ViHealthBertCRF without labels returns decoded tags directly (list of lists of ints)
            batch_preds = model(**inputs)

        for j, s in enumerate(batch):
            words = s["text"].split()
            n_words = len(words)

            offsets = offset_mapping[j].tolist()
            token_preds = batch_preds[j]  # List of predicted tag IDs
            token_confs = [1.0] * len(token_preds) # CRF decode doesn't give probabilities natively

            word_tags = ["O"] * n_words
            word_confs = [1.0] * n_words

            # Simple alignment: assign first subword's prediction to each word
            char_pos = 0
            word_idx = 0
            for k, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:
                    continue  # special token
                if word_idx >= n_words:
                    break
                # Check if this subword starts a new word
                word_start = sum(len(w) + 1 for w in words[:word_idx])
                if start >= word_start and word_idx < n_words:
                    tag_id = token_preds[k]
                    tag_str = id2tag.get(tag_id, id2tag.get(str(tag_id), "O"))
                    word_tags[word_idx] = tag_str
                    word_confs[word_idx] = token_confs[k]
                    if end >= word_start + len(words[word_idx]):
                        word_idx += 1

            # Map to our NER tag set
            mapped_tags = []
            for tag in word_tags:
                if tag in NER2ID:
                    mapped_tags.append(tag)
                else:
                    mapped_tags.append("O")

            min_conf = min(word_confs) if word_confs else 0.0

            s["words"] = words
            s["ner_tags"] = mapped_tags
            s["ner_tag_ids"] = [NER2ID.get(t, 0) for t in mapped_tags]
            s["ner_confidence"] = min_conf
            labeled += 1

    print(f"✅ NER pseudo-labeled: {labeled}/{len(samples)}")
    return samples


def main():
    PSEUDO_NEW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔬 Cross-Source Pseudo-Labeling Pipeline")
    print(f"   Output: {PSEUDO_NEW_DIR}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 60)

    # Step 1: Load hospital data
    topic_train_path = TOPIC_DATA_DIR / "topic_train.json"
    if not topic_train_path.exists():
        print(f"❌ Not found: {topic_train_path}")
        return

    print(f"\n📂 Loading hospital topic data: {topic_train_path}")
    new_pseudo_samples = load_hospital_topic_data(topic_train_path)
    print(f"   Loaded {len(new_pseudo_samples)} samples (after remap/drop)")

    # Step 2: Pseudo-label Intent
    print("\n🎯 Pseudo-labeling Intent...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    new_pseudo_samples = pseudo_label_intent(new_pseudo_samples, device)

    # Step 3: Pseudo-label NER
    print("\n🏷️ Pseudo-labeling NER...")
    new_pseudo_samples = pseudo_label_ner(new_pseudo_samples, device)

    # Step 4: Filter by confidence threshold on ALL 3 tasks
    print(f"\n🔍 Filtering (all 3 tasks confidence >= {CONFIDENCE_THRESHOLD})...")
    new_pseudo_filtered = []
    for s in new_pseudo_samples:
        topic_conf = s.get("topic_confidence", 0)
        intent_conf = s.get("intent_confidence", 0)
        ner_conf = s.get("ner_confidence", 0)

        if (topic_conf >= CONFIDENCE_THRESHOLD and
            intent_conf >= CONFIDENCE_THRESHOLD and
            ner_conf >= CONFIDENCE_THRESHOLD):
            new_pseudo_filtered.append(s)

    print(f"   Passed filter: {len(new_pseudo_filtered)}/{len(new_pseudo_samples)}")

    # Stats
    topic_dist = Counter(s["topic_label"] for s in new_pseudo_filtered)
    print(f"\n📊 Topic distribution (pseudo_new):")
    for topic, count in topic_dist.most_common():
        print(f"   {topic}: {count}")

    # Step 5: Save to pseudo_new/ directory
    out_path = PSEUDO_NEW_DIR / "pseudo_new_joint.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_pseudo_filtered, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved {len(new_pseudo_filtered)} samples → {out_path}")

    # Save metadata for this pseudo batch
    meta = {
        "source": "hospital_topic_data_pseudo_labeled",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_input": len(new_pseudo_samples),
        "total_passed": len(new_pseudo_filtered),
        "topic_distribution": dict(topic_dist),
        "intent_model": INTENT_MODEL_HF,
        "ner_model": NER_MODEL_HF,
    }
    meta_path = PSEUDO_NEW_DIR / "pseudo_new_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   Saved metadata → {meta_path}")

    print("\n✅ Cross-source pseudo-labeling complete!")
    print(f"   ⚠️ Data is in {PSEUDO_NEW_DIR} — NOT merged into main pipeline.")


if __name__ == "__main__":
    main()
