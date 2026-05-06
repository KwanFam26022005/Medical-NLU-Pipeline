"""
Relabel NER tags for KEHN jsonl data using OpenAI LLM.

Expected input row fields:
- words: List[str]
- ner_tags / ner_tag_ids (optional, used as fallback)

Output:
- ner_tags (BIO in KEHN label space)
- ner_tag_ids
- ner_confidence (LLM self-reported confidence in [0,1], fallback 0.0)
- ner_source = "openai_llm"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from config_joint import NER2ID


ALLOWED_TAGS = ["O", "B-SYM", "I-SYM", "B-PRO", "I-PRO", "B-DRU", "I-DRU"]
ENTITY_TYPES = {"SYM", "PRO", "DRU"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        x = float(value)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x
    except Exception:
        return default


def repair_bio(tags: List[str]) -> List[str]:
    """Repair invalid tags and BIO transitions deterministically."""
    repaired: List[str] = []
    prev_type = None

    for raw in tags:
        tag = raw if raw in ALLOWED_TAGS else "O"
        if tag == "O":
            repaired.append("O")
            prev_type = None
            continue

        prefix, ent_type = tag.split("-", 1)
        if ent_type not in ENTITY_TYPES:
            repaired.append("O")
            prev_type = None
            continue

        if prefix == "B":
            repaired.append(tag)
            prev_type = ent_type
            continue

        # I-xxx is only valid after B-xxx or I-xxx of same type.
        if prefix == "I":
            if prev_type == ent_type:
                repaired.append(tag)
            else:
                repaired.append(f"B-{ent_type}")
            prev_type = ent_type
            continue

        repaired.append("O")
        prev_type = None

    return repaired


def build_prompt(words: List[str]) -> str:
    words_json = json.dumps(words, ensure_ascii=False)
    allowed_json = json.dumps(ALLOWED_TAGS, ensure_ascii=False)
    return (
        "Bạn là chuyên gia gán nhãn NER y khoa tiếng Việt.\n"
        "Nhiệm vụ: gán nhãn BIO cho từng token trong mảng `words`.\n"
        "Chỉ dùng đúng các nhãn trong danh sách cho phép.\n"
        "Không thêm/bớt token, không đổi thứ tự.\n\n"
        f"Allowed tags: {allowed_json}\n"
        f"Words: {words_json}\n\n"
        "Trả về JSON object duy nhất theo format:\n"
        "{\"ner_tags\": [...], \"confidence\": 0.0-1.0}\n"
        "- ner_tags phải có cùng độ dài với words.\n"
        "- confidence là độ tin cậy tổng quát cho output.\n"
    )


def request_ner_tags(
    client: OpenAI,
    model: str,
    words: List[str],
    max_retries: int = 2,
    sleep_seconds: float = 1.0,
) -> Tuple[List[str], float]:
    """Call OpenAI and return (validated_tags, confidence)."""
    prompt = build_prompt(words)
    last_error = None

    for _ in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": "You must output valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.output_text or "").strip()
            data = json.loads(text)
            tags = data.get("ner_tags", [])
            conf = _safe_float(data.get("confidence", 0.0), default=0.0)

            if not isinstance(tags, list):
                raise ValueError("ner_tags is not a list")
            if len(tags) != len(words):
                raise ValueError(
                    f"length mismatch: tags={len(tags)} vs words={len(words)}"
                )

            repaired = repair_bio([str(t) for t in tags])
            return repaired, conf
        except Exception as e:
            last_error = e
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel NER with OpenAI LLM")
    parser.add_argument("--input", required=True, help="Input jsonl path")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke test")
    parser.add_argument(
        "--fallback_to_existing",
        action="store_true",
        help="Fallback to existing ner_tags when LLM output is invalid",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = load_jsonl(in_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    client = OpenAI(api_key=api_key)
    success = 0
    fallback = 0

    for i, row in enumerate(rows):
        words = row.get("words", [])
        if not isinstance(words, list) or not words:
            # keep row unchanged if schema is invalid
            row["ner_confidence"] = 0.0
            row["ner_source"] = "openai_llm_invalid_words"
            continue

        try:
            tags, conf = request_ner_tags(client, args.model, [str(w) for w in words])
            row["ner_tags"] = tags
            row["ner_tag_ids"] = [NER2ID.get(t, 0) for t in tags]
            row["ner_confidence"] = conf
            row["ner_source"] = "openai_llm"
            success += 1
        except Exception:
            if args.fallback_to_existing and isinstance(row.get("ner_tags"), list):
                prev_tags = row["ner_tags"]
                if len(prev_tags) == len(words):
                    repaired = repair_bio([str(t) for t in prev_tags])
                    row["ner_tags"] = repaired
                    row["ner_tag_ids"] = [NER2ID.get(t, 0) for t in repaired]
                    row["ner_confidence"] = 0.0
                    row["ner_source"] = "fallback_existing"
                    fallback += 1
                    continue
            row["ner_tags"] = ["O"] * len(words)
            row["ner_tag_ids"] = [0] * len(words)
            row["ner_confidence"] = 0.0
            row["ner_source"] = "fallback_all_o"
            fallback += 1

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(rows)} rows...")

    write_jsonl(out_path, rows)
    print(f"Saved: {out_path}")
    print(f"Rows: {len(rows)} | LLM success: {success} | fallback: {fallback}")


if __name__ == "__main__":
    main()

