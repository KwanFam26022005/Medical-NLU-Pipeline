"""
Fuse hospital extraction pipeline for KEHN architecture.

Goals:
- Input from data/topic_train.json (human-labeled topic)
- Lightly compact repetitive text (rule-based, deterministic)
- Pseudo-label intent (reuse pseudo_cross_source logic)
- Relabel NER using ViMQ joint model decode (reuse relabel_hospital_vimq logic)
- Canonicalize labels for current KEHN config space
- Export hospital fused jsonl and merged jsonl with vimq_kehn.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

try:
    from pyvi import ViTokenizer  # type: ignore[reportMissingImports]
except ImportError:
    ViTokenizer = None

from config_joint import TOPIC2ID, TOPIC_DROP, TOPIC_REMAP, NER2ID
from pseudo_cross_source import pseudo_label_intent
from relabel_hospital_vimq import load_vimq_model, preprocess_text, span_decode

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "ViMQ-main" / "ViMQ-main" / "src"),
)
from utils import spacy_to_iob  # type: ignore[reportMissingImports]


ROOT_DIR = Path(__file__).resolve().parent.parent
TOPIC_TRAIN_PATH = ROOT_DIR / "data" / "topic_train.json"
PSEUDO_DIR = Path(__file__).resolve().parent / "data" / "pseduo_kehn"
VIMQ_KEHN_PATH = PSEUDO_DIR / "vimq_kehn.jsonl"
OUT_HOSPITAL_PATH = PSEUDO_DIR / "hospital_kehn_fused.jsonl"
OUT_MERGED_PATH = PSEUDO_DIR / "merged_kehn_fused.jsonl"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def compact_repetitive_text(text: str, similarity_threshold: float = 0.92) -> str:
    # Exact duplicate removal at sentence-level (common in topic_train)
    sentences = split_sentences(text)
    if not sentences:
        return text.strip()

    kept: List[str] = []
    seen_norm: List[str] = []
    for sent in sentences:
        norm = re.sub(r"\s+", " ", sent).strip().lower()
        # Fast exact dedup
        if norm in seen_norm:
            continue
        # Simple near-dup guard for paraphrased repeats
        is_near_dup = False
        for prev in seen_norm:
            # Token-overlap ratio avoids expensive sequence matcher
            a = set(norm.split())
            b = set(prev.split())
            overlap = len(a & b) / max(1, len(a | b))
            if overlap >= similarity_threshold:
                is_near_dup = True
                break
        if not is_near_dup:
            kept.append(sent)
            seen_norm.append(norm)
    return " ".join(kept).strip()


def tokenize_text(text: str) -> List[str]:
    punct_spaced = re.sub(r'([.,?!;:()\[\]{}"\'])', r" \1 ", text)
    punct_spaced = re.sub(r"\s+", " ", punct_spaced).strip()
    if ViTokenizer is not None:
        return ViTokenizer.tokenize(punct_spaced).split()
    return punct_spaced.split()


def remap_topic(topic: str) -> str | None:
    if topic in TOPIC_DROP:
        return None
    return TOPIC_REMAP.get(topic, topic)


def normalize_ner_tag(tag: str) -> str:
    map_exact = {
        "B-SYMPTOM_AND_DISEASE": "B-SYM",
        "I-SYMPTOM_AND_DISEASE": "I-SYM",
        "B-medical_procedure": "B-PRO",
        "I-medical_procedure": "I-PRO",
        "B-drug": "B-DRU",
        "I-drug": "I-DRU",
    }
    if tag in map_exact:
        return map_exact[tag]
    if tag in NER2ID:
        return tag
    return "O"


def load_topic_samples() -> List[Dict]:
    with open(TOPIC_TRAIN_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)

    samples: List[Dict] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        topic = str(row.get("topic", "")).strip()
        if not text or not topic:
            continue
        topic = remap_topic(topic)
        if topic is None or topic not in TOPIC2ID:
            continue
        compact = compact_repetitive_text(text)
        words = tokenize_text(compact)
        if not words:
            continue
        samples.append(
            {
                "source": "hospital",
                "text": " ".join(words),
                "words": words,
                "topic_label": topic,
                "topic_label_id": TOPIC2ID[topic],
                "topic_confidence": 1.0,
            }
        )
    return samples


def relabel_ner_vimq(samples: List[Dict]) -> List[Dict]:
    model, tokenizer, args, index2label, char_vocab, device = load_vimq_model()

    for item in samples:
        words = item["words"]
        seq_len = len(words)
        input_ids, attention_mask, first_subword, char_ids = preprocess_text(
            tokenizer, words, char_vocab, args
        )

        import torch
        import numpy as np

        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "first_subword": first_subword.to(device),
            "seq_len": torch.tensor([seq_len]).to(device),
            "char_ids": char_ids.to(device),
            "label": None,
        }

        with torch.no_grad():
            score, _ = model(**inputs)
        preds = np.argmax(score.detach().cpu().numpy(), axis=-1)[0]
        spans = span_decode(preds, index2label)
        raw_tags = spacy_to_iob(spans, seq_len)
        ner_tags = [normalize_ner_tag(t) for t in raw_tags]

        item["ner_tags"] = ner_tags
        item["ner_tag_ids"] = [NER2ID.get(t, 0) for t in ner_tags]
        item["ner_confidence"] = 1.0
    return samples


def finalize_intent_fields(samples: List[Dict]) -> List[Dict]:
    for item in samples:
        intent_id = int(item.get("intent_label_id", 0))
        words = item["words"]
        item["token_intent_ids"] = [intent_id] * len(words)
    return samples


def validate_sample(item: Dict) -> bool:
    words = item.get("words", [])
    tags = item.get("ner_tags", [])
    tag_ids = item.get("ner_tag_ids", [])
    tok_intent = item.get("token_intent_ids", [])
    if not (len(words) == len(tags) == len(tag_ids) == len(tok_intent)):
        return False
    if item.get("topic_label") not in TOPIC2ID:
        return False
    if item.get("topic_label_id") != TOPIC2ID[item["topic_label"]]:
        return False
    return True


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    print("1) Load + compact topic_train...")
    hospital = load_topic_samples()
    print(f"   Loaded {len(hospital)} hospital samples")

    print("2) Pseudo-label intent...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    hospital = pseudo_label_intent(hospital, device=device)

    print("3) Relabel NER with ViMQ...")
    hospital = relabel_ner_vimq(hospital)

    print("4) Finalize fields + validate...")
    hospital = finalize_intent_fields(hospital)
    hospital = [s for s in hospital if validate_sample(s)]
    print(f"   Valid fused hospital samples: {len(hospital)}")

    print("5) Save hospital fused dataset...")
    write_jsonl(OUT_HOSPITAL_PATH, hospital)
    print(f"   -> {OUT_HOSPITAL_PATH}")

    print("6) Merge with vimq_kehn and shuffle...")
    vimq = load_jsonl(VIMQ_KEHN_PATH)
    merged = vimq + hospital
    random.seed(42)
    random.shuffle(merged)
    write_jsonl(OUT_MERGED_PATH, merged)
    print(f"   -> {OUT_MERGED_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()

