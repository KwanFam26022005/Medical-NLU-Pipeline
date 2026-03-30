#!/usr/bin/env python3
"""
FAQ Data Augmentation — pseudo-label silver data cho các khoa hiếm (Trạm 2B).

Đọc FAQ thô, làm sạch, inference bằng model Hub, lọc confidence >= 0.95 & target classes,
gộp vào topic_train.json → topic_train_augmented.json.

Chạy trên Colab (GPU khuyến nghị):
    python scripts/augment_faq_topics.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Các khoa cần cứu — không thêm pediatrics, obstetrics_gynecology, orthopedics, cardiology
TARGET_CLASSES = [
    "dentistry",
    "dermatology",
    "endocrinology",
    "ent",
    "gastroenterology",
    "internal_medicine",
    "neurology",
    "nutrition",
    "ophthalmology",
    "rheumatology",
    "traditional_medicine",
    "urology",
]
TARGET_SET = frozenset(TARGET_CLASSES)

# Cụm rác cần xóa (ưu tiên chuỗi dài trước)
_BOILERPLATE_PHRASES = [
    "Câu hỏi khách hàng ẩn danh",
    "Bác sĩ cho em hỏi",
    "Xin chào bác sĩ",
    "Chào bác sĩ",
    "Thưa bác sĩ",
    "Cho em hỏi",
    "Xin cảm ơn",
]

# Mẫu: Tên ( Năm ) Trả lời :
_NAME_YEAR_REPLY = re.compile(
    r"[\w\s\u00C0-\u1EF9]{1,120}?\(\s*\d{2,4}\s*\)\s*Trả lời\s*:\s*",
    re.UNICODE | re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """
    Desegmentation & cleaning: _, boilerplate, chữ ký [Tên] (năm) Trả lời :,
    sau đó chuẩn hóa khoảng trắng.
    """
    if not text or not isinstance(text, str):
        return ""
    t = text.replace("_", " ")
    for phrase in sorted(_BOILERPLATE_PHRASES, key=len, reverse=True):
        t = re.sub(re.escape(phrase), " ", t, flags=re.IGNORECASE)
    t = _NAME_YEAR_REPLY.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def load_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file FAQ: {path}")
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines()]
    return [ln for ln in lines if ln]


def build_id2label_from_model(model) -> Dict[int, str]:
    cfg = model.config
    raw = getattr(cfg, "id2label", None) or {}
    out: Dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def load_topic2id_from_map(map_path: Path) -> Dict[str, int]:
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data["topic2id"].items()}


@torch.inference_mode()
def predict_batches(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[List[int], List[float], List[str]]:
    """Trả về (pred_ids, confidences, pred_topics) theo cùng thứ tự texts."""
    id2label = build_id2label_from_model(model)
    pred_ids: List[int] = []
    confidences: List[float] = []
    pred_topics: List[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        conf, ids = probs.max(dim=-1)
        for j in range(ids.size(0)):
            pid = int(ids[j].item())
            cf = float(conf[j].item())
            pred_ids.append(pid)
            confidences.append(cf)
            pred_topics.append(id2label.get(pid, f"topic_{pid}"))
    return pred_ids, confidences, pred_topics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAQ pseudo-labeling + merge topic_train (silver augmentation)"
    )
    parser.add_argument(
        "--faq_src",
        type=Path,
        default=Path("data/FAQ_summarization/train.txt.src"),
        help="File FAQ nguồn (mỗi dòng một câu/cụm).",
    )
    parser.add_argument(
        "--topic_train",
        type=Path,
        default=Path("data/topic_train.json"),
    )
    parser.add_argument(
        "--topic_label_map",
        type=Path,
        default=Path("data/topic_label_map.json"),
        help="Dùng để gán trường label (id) nhất quán với pipeline train.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/topic_train_augmented.json"),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="KwanFam26022005/vihealthbert-topic-classification",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--min_confidence", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --- Bước 1: đọc & làm sạch ---
    lines = load_lines(args.faq_src)
    cleaned: List[str] = []
    for ln in lines:
        c = clean_text(ln)
        if word_count(c) < 10:
            continue
        cleaned.append(c)

    print(f"[Bước 1] FAQ dòng đọc được: {len(lines)} → sau clean & >=10 từ: {len(cleaned)}")
    if not cleaned:
        print("Không còn mẫu nào sau bước làm sạch. Thoát.")
        sys.exit(1)

    # --- Bước 2: model ---
    print(f"[Bước 2] Load model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    topic2id = load_topic2id_from_map(args.topic_label_map)

    pred_ids, confs, topics = predict_batches(
        cleaned,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    silver: List[Dict[str, Any]] = []
    silver_seq = 0
    for text, _pid, conf, top in zip(cleaned, pred_ids, confs, topics):
        if conf < args.min_confidence:
            continue
        if top not in TARGET_SET:
            continue
        if top not in topic2id:
            print(f"  ⚠️ Bỏ qua: topic '{top}' không có trong topic_label_map.json")
            continue
        label_id = topic2id[top]
        silver.append(
            {
                "id": f"faq_pseudo_{silver_seq:07d}",
                "text": text,
                "label": label_id,
                "topic": top,
                "source": "faq_pseudo_silver",
            }
        )
        silver_seq += 1

    print(f"[Bước 2] Silver sau lọc (conf>={args.min_confidence} & target class): {len(silver)}")

    # Thống kê theo khoa hiếm (target)
    counts = Counter(s["topic"] for s in silver)
    print("\n" + "=" * 60)
    print("Số câu vớt được theo từng khoa (TARGET_CLASSES):")
    print("=" * 60)
    for cls in TARGET_CLASSES:
        n = counts.get(cls, 0)
        print(f"  {cls:28s}  {n:6d}")
    print("=" * 60)
    print(f"  {'TỔNG':28s}  {len(silver):6d}")

    # --- Bước 3: gộp train ---
    if not args.topic_train.exists():
        raise FileNotFoundError(f"Không có {args.topic_train}")
    with open(args.topic_train, "r", encoding="utf-8") as f:
        original = json.load(f)
    if not isinstance(original, list):
        raise ValueError("topic_train.json phải là mảng JSON.")

    merged = original + silver
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[Bước 3] Train gốc: {len(original)}  +  silver: {len(silver)}  =  {len(merged)}")
    print(f"Đã ghi: {args.output.resolve()}")


if __name__ == "__main__":
    main()
