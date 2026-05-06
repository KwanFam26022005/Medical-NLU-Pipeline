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

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

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

# Confidence gates for hospital pseudo data
MIN_TOPIC_CONFIDENCE = 1.0      # topic is human-labeled from topic_train
MIN_INTENT_CONFIDENCE = 0.85    # from intent pseudo model
# ViMQ relabel uses span probabilities from decode_spans_from_score().
# Keep the gating thresholds consistent with the decode min_prob to avoid
# filtering everything due to averaging/softmax range.
NER_SPAN_MIN_PROB = 0.85
MIN_NER_CONFIDENCE = NER_SPAN_MIN_PROB


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_text_surface(text: str) -> str:
    # Lowercase for stable text style and reduced sparsity
    text = text.lower().strip()

    # Normalize repeated punctuation and spacing
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{2,}", ".", text)

    # Expand frequent colloquial pattern
    text = re.sub(r"\btầm\b", "khoảng", text)

    # Remove brackets around dosage chunks, keep content
    text = re.sub(r"\(\s*([^()]*?)\s*\)", r"\1", text)

    # Normalize spaces around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    # Keep phrase connectors with underscore when they look medical terms.
    # But normalize some noisy connective forms that are usually not entities.
    text = re.sub(r"\bcó_ăn\b", "có ăn", text)
    text = re.sub(r"\bbác_sĩ\b", "bác_sĩ", text)
    punct_spaced = re.sub(r'([.,?!;:()\[\]{}"\'])', r" \1 ", text)
    punct_spaced = re.sub(r"\s+", " ", punct_spaced).strip()
    if ViTokenizer is not None:
        return ViTokenizer.tokenize(punct_spaced).split()
    return punct_spaced.split()


def chunk_long_words(words: List[str], max_words: int = 120) -> List[str]:
    """
    Split long token sequences by punctuation boundaries to reduce truncation risk.
    If no good boundary exists, hard-cut by max_words.
    """
    if len(words) <= max_words:
        return words

    boundary_tokens = {".", "!", "?", ";", ":"}
    chunks: List[List[str]] = []
    cur: List[str] = []

    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            # try nearest right boundary within current chunk
            cut_at = -1
            for i in range(len(cur) - 1, max(-1, len(cur) - 25), -1):
                if cur[i] in boundary_tokens:
                    cut_at = i
                    break
            if cut_at >= 0:
                chunks.append(cur[: cut_at + 1])
                cur = cur[cut_at + 1 :]
            else:
                chunks.append(cur[:max_words])
                cur = cur[max_words:]

    if cur:
        chunks.append(cur)

    # Keep at most first 2 chunks to avoid over-long narratives dominating.
    merged = []
    for ch in chunks[:2]:
        merged.extend(ch)
    return merged[: max_words * 2]


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


def repair_bio_tags(tags: List[str]) -> List[str]:
    """Fix invalid BIO transitions produced by span-to-IOB conversion noise."""
    repaired: List[str] = []
    prev_type = None
    for tag in tags:
        if tag == "O":
            repaired.append("O")
            prev_type = None
            continue
        if "-" not in tag:
            repaired.append("O")
            prev_type = None
            continue
        prefix, ent_type = tag.split("-", 1)
        if prefix == "B":
            repaired.append(tag)
            prev_type = ent_type
        elif prefix == "I":
            if prev_type == ent_type:
                repaired.append(tag)
            else:
                repaired.append(f"B-{ent_type}")
            prev_type = ent_type
        else:
            repaired.append("O")
            prev_type = None
    return repaired


def decode_spans_from_score(
    score_4d: np.ndarray,
    index2label: Dict[int, str],
    seq_len: int,
    min_prob: float = 0.85,
    max_span_len: int = 8,
) -> List[List]:
    """
    Decode spans from ViMQ score tensor with confidence gating + non-overlap.
    This avoids dense noisy spans produced by plain argmax over all i,j cells.
    """
    # ViMQ forward thường trả về shape (B, L, L, C) nhưng trong một số trường hợp
    # có thể là (L, L, C) (không có batch). Chuẩn hóa về (L_i, L_j, C).
    if score_4d.ndim == 4:
        score = score_4d[0]
    elif score_4d.ndim == 3:
        score = score_4d
    else:
        raise ValueError(f"Unexpected ViMQ score shape: {score_4d.shape}")

    # score expected: (L_i, L_j, C)
    l_i, l_j = score.shape[0], score.shape[1]

    # Inference output can be capped by model max_seq_len (e.g., 256),
    # while raw sample seq_len may be longer.
    effective_len = min(int(seq_len), int(l_i), int(l_j))
    candidates = []

    # Candidate extraction
    for i in range(effective_len):
        # Clamp theo kích thước thật của trục j để tránh out-of-bounds
        j_max = min(int(l_j - 1), int(effective_len - 1), i + max_span_len - 1)
        if j_max < i:
            continue
        for j in range(i, j_max + 1):
            logits = score[i, j]
            # Stable softmax
            z = logits - np.max(logits)
            p = np.exp(z)
            p = p / np.sum(p)
            cls = int(np.argmax(p))
            conf = float(p[cls])
            if cls <= 0 or conf < min_prob:
                continue
            label = index2label.get(cls, "UNK")
            if label == "UNK":
                continue
            candidates.append((conf, i, j, label))

    # Prefer high-confidence, then shorter spans (usually cleaner)
    candidates.sort(key=lambda x: (-x[0], (x[2] - x[1])))

    occupied = [False] * seq_len
    selected = []
    for conf, i, j, label in candidates:
        if any(occupied[k] for k in range(i, j + 1)):
            continue
        for k in range(i, j + 1):
            occupied[k] = True
        selected.append([i, j, label, conf])

    # sort by start index for downstream iob conversion
    selected.sort(key=lambda x: (x[0], x[1]))
    return selected


def load_topic_samples(max_words: int = 120) -> List[Dict]:
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
        cleaned = normalize_text_surface(text)
        compact = compact_repetitive_text(cleaned)
        words = tokenize_text(compact)
        words = chunk_long_words(words, max_words=max_words)
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
        score_np = score.detach().cpu().numpy()
        spans_with_conf = decode_spans_from_score(
            score_np, index2label, seq_len, min_prob=NER_SPAN_MIN_PROB, max_span_len=8
        )
        spans = [[s, e, t] for s, e, t, _ in spans_with_conf]
        raw_tags = spacy_to_iob(spans, seq_len)
        ner_tags = [normalize_ner_tag(t) for t in raw_tags]
        ner_tags = repair_bio_tags(ner_tags)

        item["ner_tags"] = ner_tags
        item["ner_tag_ids"] = [NER2ID.get(t, 0) for t in ner_tags]
        if spans_with_conf:
            item["ner_confidence"] = float(
                sum(conf for *_rest, conf in spans_with_conf) / len(spans_with_conf)
            )
        else:
            item["ner_confidence"] = 0.0
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


def pass_confidence_filter(item: Dict) -> bool:
    topic_conf = float(item.get("topic_confidence", 0.0))
    intent_conf = float(item.get("intent_confidence", 0.0))
    ner_conf = float(item.get("ner_confidence", 0.0))
    return (
        topic_conf >= MIN_TOPIC_CONFIDENCE
        and intent_conf >= MIN_INTENT_CONFIDENCE
        and ner_conf >= MIN_NER_CONFIDENCE
    )


def pass_quality_filter(item: Dict, min_entity_ratio: float = 0.01) -> bool:
    """
    Extra quality gate:
    - keep samples containing at least one entity
    - avoid too-sparse NER outputs (often decode failures on long/noisy text)
    """
    tags = item.get("ner_tags", [])
    words = item.get("words", [])
    if not tags or not words:
        return False
    if len(tags) != len(words):
        return False
    non_o = sum(1 for t in tags if t != "O")
    ratio = non_o / max(1, len(tags))
    return non_o > 0 and ratio >= min_entity_ratio


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
    parser = argparse.ArgumentParser(description="Fuse hospital extracts into KEHN format")
    parser.add_argument("--smoke_limit", type=int, default=None, help="Limit number of hospital samples")
    parser.add_argument("--max_words", type=int, default=120, help="Max words before chunking long samples")
    parser.add_argument(
        "--min_entity_ratio",
        type=float,
        default=0.01,
        help="Minimum ratio of non-O tags to keep sample",
    )
    args = parser.parse_args()

    print("1) Load + compact topic_train...")
    hospital = load_topic_samples(max_words=args.max_words)
    if args.smoke_limit is not None:
        hospital = hospital[: args.smoke_limit]
    print(f"   Loaded {len(hospital)} hospital samples")

    print("2) Pseudo-label intent...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    hospital = pseudo_label_intent(hospital, device=device)

    print("3) Relabel NER with ViMQ...")
    hospital = relabel_ner_vimq(hospital)

    print("4) Finalize fields + validate...")
    hospital = finalize_intent_fields(hospital)
    before_filter = len(hospital)
    hospital = [s for s in hospital if validate_sample(s)]
    after_validate = len(hospital)
    hospital = [s for s in hospital if pass_confidence_filter(s)]
    after_conf = len(hospital)
    hospital = [s for s in hospital if pass_quality_filter(s, min_entity_ratio=args.min_entity_ratio)]
    print(f"   After schema validation: {after_validate}/{before_filter}")
    print(
        "   After confidence filter "
        f"(topic>={MIN_TOPIC_CONFIDENCE}, intent>={MIN_INTENT_CONFIDENCE}, ner>={MIN_NER_CONFIDENCE}): "
        f"{after_conf}/{after_validate}"
    )
    print(f"   After quality filter (min_entity_ratio>={args.min_entity_ratio}): {len(hospital)}/{after_conf}")

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

