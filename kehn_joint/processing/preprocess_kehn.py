"""
preprocess_kehn.py

This script preprocesses, normalizes, and merges Vietnamese medical datasets
(ViMQ and Hospital) into a strict JSONL format for the KEHN architecture.

Core Steps:
1. Punctuation Separation & Word Segmentation using PyVi.
2. Character-Level Index Tracking for NER Alignment.
3. Generate Token-level Intent IDs.
4. Local Deduplication (Hospital Dataset Only).
5. Fill Missing Confidences.
"""

import argparse
import json
import re
import random
import difflib
import logging
import subprocess
from pathlib import Path

try:
    from pyvi import ViTokenizer
except ImportError:
    raise ImportError("Please install pyvi: pip install pyvi")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Immutable output schema requirements
# {
#   "source": "vimq|hospital",
#   "text": "...",
#   "words": ["..."],
#   "topic_label": "...",
#   "topic_label_id": 0,
#   "topic_confidence": 1.0,
#   "intent_label": "...",
#   "intent_label_id": 0,
#   "intent_confidence": 1.0,
#   "token_intent_ids": [0, 0, 0],
#   "ner_tags": ["B-SYM", "I-SYM", "O"],
#   "ner_tag_ids": [1, 2, 0],
#   "ner_confidence": 1.0
# }

NER2ID = {
    "O": 0,
    "B-SYM": 1,
    "I-SYM": 2,
    "B-PRO": 3,
    "I-PRO": 4,
    "B-DRU": 5,
    "I-DRU": 6
}

PUNCT_TOKEN_RE = re.compile(r"^[^\w\s]+$", re.UNICODE)
SPLIT_PUNCT_RE = re.compile(r'([.,?!;:()[\]{}"\'/\\])')


def clean_and_segment(text: str) -> list:
    """Detach punctuation to standalone tokens, then apply ViTokenizer."""
    cleaned = SPLIT_PUNCT_RE.sub(r" \1 ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    segmented = ViTokenizer.tokenize(cleaned)
    return segmented.split()


def normalize_for_compare(text: str) -> str:
    """Normalize for duplicate detection only (not for training text)."""
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", "", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def remove_repeated_opening_clause(text: str, min_words: int = 4, ratio: float = 0.9) -> str:
    """
    Remove duplicated opening fragment: 'A A?' or 'A A ...'.
    Keeps original content when confidence is low.
    """
    tokens = text.split()
    if len(tokens) < min_words * 2:
        return text
    max_probe = min(30, len(tokens) // 2)
    for n in range(max_probe, min_words - 1, -1):
        first = " ".join(tokens[:n])
        second = " ".join(tokens[n:2 * n])
        if not second:
            continue
        score = difflib.SequenceMatcher(None, normalize_for_compare(first), normalize_for_compare(second)).ratio()
        if score >= ratio:
            return " ".join(tokens[n:]).strip()
    return text


def repair_bio(tags: list) -> list:
    """Repair invalid BIO transitions deterministically."""
    repaired = []
    prev_type = None
    for tag in tags:
        if tag == "O" or "-" not in tag:
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


def is_punctuation_token(token: str) -> bool:
    return bool(PUNCT_TOKEN_RE.match(token))

def deduplicate_tokens(words: list, tags: list, intent_ids: list):
    """
    Detects and removes repetitive sentences (like "A B C A B C").
    Splits by end-of-sentence punctuation and compares string similarity.
    """
    sentences = []
    cur_words, cur_tags, cur_intents = [], [], []
    
    for w, t, i in zip(words, tags, intent_ids):
        cur_words.append(w)
        cur_tags.append(t)
        cur_intents.append(i)
        # End of sentence markers
        if w in [".", "?", "!"]:
            sentences.append((cur_words, cur_tags, cur_intents))
            cur_words, cur_tags, cur_intents = [], [], []
            
    if cur_words:
        sentences.append((cur_words, cur_tags, cur_intents))
        
    deduped_words, deduped_tags, deduped_intents = [], [], []
    seen_texts = []
    
    for w_list, t_list, i_list in sentences:
        sent_text = " ".join(w_list).lower()
        # Do not deduplicate very short natural communication phrases
        if len(w_list) <= 5:
            deduped_words.extend(w_list)
            deduped_tags.extend(t_list)
            deduped_intents.extend(i_list)
            seen_texts.append(sent_text)
            continue
            
        is_dup = False
        for seen in seen_texts:
            seq = difflib.SequenceMatcher(None, sent_text, seen)
            # Threshold set to 90% as requested
            if seq.ratio() > 0.90:
                is_dup = True
                break
                
        if not is_dup:
            deduped_words.extend(w_list)
            deduped_tags.extend(t_list)
            deduped_intents.extend(i_list)
            seen_texts.append(sent_text)
            
    return deduped_words, deduped_tags, deduped_intents


def process_sample(sample: dict, source_name: str) -> dict:
    orig_words = sample.get("words", [])
    orig_tags = sample.get("ner_tags", [])

    # 1) Start from text field (if present), then remove duplicated opening title/body.
    raw_text = sample.get("text", "").strip() or " ".join(orig_words)
    raw_text = remove_repeated_opening_clause(raw_text)
    raw_text = re.sub(r'\s+([.,?!;:()[\]{}"\'/\\])', r"\1", raw_text)
    new_words = clean_and_segment(raw_text)

    # 2) Character-level NER realignment using only alnum chars.
    # Punctuation never contributes to entity span and stays O.
    char_to_tag = []
    for w, t in zip(orig_words, orig_tags):
        pure_w = w.replace("_", "").replace(" ", "")
        for ch in pure_w:
            if ch.isalnum():
                char_to_tag.append(t)

    new_tags = []
    char_idx = 0
    for nw in new_words:
        pure_nw = nw.replace("_", "").replace(" ", "")
        if is_punctuation_token(pure_nw):
            new_tags.append("O")
            continue

        alnum_chars = [ch for ch in pure_nw if ch.isalnum()]
        if not alnum_chars:
            new_tags.append("O")
            continue

        tags_for_word = []
        for _ in range(len(alnum_chars)):
            if char_idx < len(char_to_tag):
                tags_for_word.append(char_to_tag[char_idx])
            else:
                tags_for_word.append("O")
            char_idx += 1

        has_b = [t for t in tags_for_word if t.startswith("B-")]
        has_i = [t for t in tags_for_word if t.startswith("I-")]
        if has_b:
            new_tags.append(has_b[0])
        elif has_i:
            new_tags.append(has_i[0])
        else:
            new_tags.append("O")

    # 3) BIO guardrail and punctuation rule.
    for i, w in enumerate(new_words):
        if is_punctuation_token(w):
            new_tags[i] = "O"
    new_tags = repair_bio(new_tags)

    # 4) Generate Token-level Intent IDs
    intent_label = sample.get("intent_label", "unknown")
    intent_label_id = sample.get("intent_label_id", 0)
    token_intent_ids = [intent_label_id] * len(new_words)

    # 5) Local sentence deduplication (Hospital only).
    if source_name == "hospital":
        new_words, new_tags, token_intent_ids = deduplicate_tokens(new_words, new_tags, token_intent_ids)

    new_tag_ids = [NER2ID.get(t, 0) for t in new_tags]
    if not (len(new_words) == len(new_tags) == len(new_tag_ids) == len(token_intent_ids)):
        logging.warning(f"Length mismatch in sample. Words: {len(new_words)}, Tags: {len(new_tags)}. Skipping.")
        return None

    # 6) Fill confidences.
    topic_conf = sample.get("topic_confidence", 1.0)
    intent_conf = sample.get("intent_confidence", 1.0)
    ner_conf = sample.get("ner_confidence", 1.0)
    final_text = " ".join(new_words)

    output = {
        "source": source_name,
        "text": final_text,
        "words": new_words,
        "topic_label": sample.get("topic_label", "unknown"),
        "topic_label_id": sample.get("topic_label_id", 0),
        "topic_confidence": topic_conf,
        "intent_label": intent_label,
        "intent_label_id": intent_label_id,
        "intent_confidence": intent_conf,
        "token_intent_ids": token_intent_ids,
        "ner_tags": new_tags,
        "ner_tag_ids": new_tag_ids,
        "ner_confidence": ner_conf
    }
    return output


def evaluate_sample_quality(sample: dict) -> dict:
    words = sample.get("words", [])
    tags = sample.get("ner_tags", [])
    stuck = 0
    punct_in_entity = 0
    bio_invalid = 0
    for i, token in enumerate(words):
        has_alnum = any(ch.isalnum() for ch in token)
        has_punct = any((not ch.isalnum()) and (not ch.isspace()) for ch in token)
        if has_alnum and has_punct:
            stuck += 1
        tag = tags[i] if i < len(tags) else "O"
        if tag != "O" and has_punct:
            punct_in_entity += 1
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            if i == 0 or tags[i - 1] == "O" or tags[i - 1][2:] != tag[2:]:
                bio_invalid += 1
    reasons = []
    if stuck > 0:
        reasons.append("stuck_punctuation")
    if punct_in_entity > 0:
        reasons.append("punctuation_inside_entity")
    if bio_invalid > 0:
        reasons.append("invalid_bio_transition")
    return {
        "stuck_punctuation_tokens": stuck,
        "punctuation_inside_entity_tokens": punct_in_entity,
        "bio_invalid_count": bio_invalid,
        "flagged": len(reasons) > 0,
        "reasons": reasons,
    }

def process_file(input_path: str, source_name: str, output_path: str, report_path: str = None) -> list:
    results = []
    quality_summary = {
        "samples": 0,
        "tokens": 0,
        "stuck_punctuation_tokens": 0,
        "punctuation_inside_entity_tokens": 0,
        "bio_invalid_count": 0,
        "flagged_samples": 0,
    }
    flagged_rows = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Could not read {input_path}: {e}")
        return results
        
    valid_count = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for sample in data:
            processed = process_sample(sample, source_name)
            if processed:
                quality = evaluate_sample_quality(processed)
                quality_summary["samples"] += 1
                quality_summary["tokens"] += len(processed["words"])
                quality_summary["stuck_punctuation_tokens"] += quality["stuck_punctuation_tokens"]
                quality_summary["punctuation_inside_entity_tokens"] += quality["punctuation_inside_entity_tokens"]
                quality_summary["bio_invalid_count"] += quality["bio_invalid_count"]
                if quality["flagged"]:
                    quality_summary["flagged_samples"] += 1
                    flagged_rows.append({"sample": processed, "reasons": quality["reasons"]})
                results.append(processed)
                out_f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                valid_count += 1
                
    logging.info(f"Processed {source_name}: {valid_count}/{len(data)} valid samples.")
    if report_path:
        report = {
            "source": source_name,
            "input_path": input_path,
            "output_path": output_path,
            "quality_summary": quality_summary,
            "flagged_samples_preview": flagged_rows[:100],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logging.info(f"Quality report saved: {report_path}")
    return results


def maybe_relabel_flagged(
    flagged_jsonl: str,
    output_jsonl: str,
    relabel_model: str,
    enabled: bool,
) -> None:
    """Optional fallback relabeling by calling relabel_ner_openai.py."""
    if not enabled:
        return
    script_path = Path(__file__).resolve().parent / "relabel_ner_openai.py"
    if not script_path.exists():
        logging.warning("relabel_ner_openai.py not found, skip fallback relabel.")
        return
    api_key = Path().cwd()
    _ = api_key  # Keep linter quiet for environments without OPENAI_API_KEY checks.
    cmd = [
        "python",
        str(script_path),
        "--input",
        flagged_jsonl,
        "--output",
        output_jsonl,
        "--model",
        relabel_model,
        "--fallback_to_existing",
    ]
    logging.info("Running fallback NER relabel on flagged samples...")
    subprocess.run(cmd, check=False)

def main():
    parser = argparse.ArgumentParser(description="Preprocess KEHN data with cleanup and quality gate")
    parser.add_argument("--vimq_input", default="data/joint_train.json")
    parser.add_argument("--hospital_input", default="data/pseudo_new/pseudo_new_joint.json")
    parser.add_argument("--vimq_output", default="vimq_kehn.jsonl")
    parser.add_argument("--hospital_output", default="hospital_kehn.jsonl")
    parser.add_argument("--merged_output", default="merged_kehn.jsonl")
    parser.add_argument("--vimq_report", default="vimq_quality_report.json")
    parser.add_argument("--hospital_report", default="hospital_quality_report.json")
    parser.add_argument("--relabel_flagged", action="store_true")
    parser.add_argument("--relabel_model", default="gpt-4.1-mini")
    args = parser.parse_args()
    
    logging.info("Starting processing for ViMQ...")
    vimq_data = process_file(args.vimq_input, "vimq", args.vimq_output, args.vimq_report)
    
    logging.info("Starting processing for Hospital...")
    hospital_data = process_file(args.hospital_input, "hospital", args.hospital_output, args.hospital_report)

    if args.relabel_flagged:
        flagged_jsonl = "hospital_flagged_for_relabel.jsonl"
        relabel_output = "hospital_flagged_relabelled.jsonl"
        with open(args.hospital_output, "r", encoding="utf-8") as rf, open(flagged_jsonl, "w", encoding="utf-8") as wf:
            for line in rf:
                row = json.loads(line)
                quality = evaluate_sample_quality(row)
                if quality["flagged"]:
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        maybe_relabel_flagged(flagged_jsonl, relabel_output, args.relabel_model, True)
    
    # Merge and Shuffle
    merged_data = vimq_data + hospital_data
    random.seed(42)
    random.shuffle(merged_data)
    
    with open(args.merged_output, 'w', encoding='utf-8') as out_f:
        for item in merged_data:
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    logging.info(f"Successfully created {args.merged_output} with {len(merged_data)} samples.")

if __name__ == "__main__":
    main()
