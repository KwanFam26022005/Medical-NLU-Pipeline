"""
preprocess_topic.py — Pipeline tiền xử lý dữ liệu topic classification (phân loại khoa).

Nguồn dữ liệu: alobacsi, tamanh (Tâm Anh), train_ml (Vinmec)
Output:
  data/topic_train.json   — training set
  data/topic_val.json     — validation set
  data/topic_test.json    — test set
  data/topic_label_map.json
  data/topic_report.txt   — EDA + imbalance report

Cách chạy:
  python preprocess_topic.py \
      --alobacsi  alobacsi_processed.csv \
      --tamanh    ml_training_data_tamanh.csv \
      --train_ml  train_ml.csv \
      --output_dir data/
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ─────────────────────────────────────────────────────────────
# 1. CANONICAL TOPIC MAPPING
#    Lý do merge:
#      - neurosurgery     → neurology      : đều liên quan bệnh thần kinh
#      - hepatology       → gastroenterology: gan mật thuộc hệ tiêu hoá
#      - andrology        → urology        : nam khoa thuộc tiết niệu
#      - neonatology      → pediatrics     : sơ sinh thuộc nhi khoa
#      - otolaryngology   → ent            : cùng khoa TMH
#      - rehabilitation   → orthopedics    : phục hồi chức năng chủ yếu cơ xương
#      - sports_medicine  → orthopedics    : y học thể thao ≈ cơ xương khớp
#      - cardiovascular_thoracic → cardiology: tim mạch - lồng ngực
#      - general_practice → internal_medicine: đa khoa ≈ nội khoa tổng quát
#      - regenerative_medicine → traditional_medicine: cùng hướng thay thế
#      - breast_health    → obstetrics_gynecology: vú thuộc sản phụ khoa
#      - women_health     → obstetrics_gynecology
#      - reproductive_health → obstetrics_gynecology
#      - obstetrics       → obstetrics_gynecology
#    Lý do DROP:
#      - radiology (4 mẫu), laboratory (1), research (3), general_surgery (2):
#        quá ít và không thuộc nhóm khoa lâm sàng rõ ràng
# ─────────────────────────────────────────────────────────────

TOPIC_MAPPING: dict[str, str] = {
    # ── Cardiology ──
    "cardiology":                    "cardiology",
    "cardiovascular_thoracic":       "cardiology",

    # ── Orthopedics ──
    "orthopedics":                   "orthopedics",
    "orthopedics_and_sports_medicine": "orthopedics",
    "sports_medicine":               "orthopedics",
    "rehabilitation":                "orthopedics",

    # ── OB/GYN ──
    "obstetrics_gynecology":         "obstetrics_gynecology",
    "obstetrics":                    "obstetrics_gynecology",
    "reproductive_health":           "obstetrics_gynecology",
    "women_health":                  "obstetrics_gynecology",
    "breast_health":                 "obstetrics_gynecology",

    # ── Neurology ──
    "neurology":                     "neurology",
    "neurosurgery":                  "neurology",

    # ── Gastroenterology ──
    "gastroenterology":              "gastroenterology",
    "hepatology":                    "gastroenterology",

    # ── Urology ──
    "urology":                       "urology",
    "andrology":                     "urology",

    # ── Pediatrics ──
    "pediatrics":                    "pediatrics",
    "neonatology":                   "pediatrics",

    # ── ENT ──
    "ent":                           "ent",
    "otolaryngology":                "ent",

    # ── Internal medicine ──
    "internal_medicine":             "internal_medicine",
    "general_practice":              "internal_medicine",

    # ── Traditional medicine ──
    "traditional_medicine":          "traditional_medicine",
    "regenerative_medicine":         "traditional_medicine",

    # ── 1-to-1 pass-through ──
    "oncology":                      "oncology",
    "reproductive_endocrinology":    "reproductive_endocrinology",
    "endocrinology":                 "endocrinology",
    "rheumatology":                  "rheumatology",
    "dermatology":                   "dermatology",
    "ophthalmology":                 "ophthalmology",
    "nutrition":                     "nutrition",
    "nephrology":                    "nephrology",
    "pulmonology":                   "pulmonology",
    "dentistry":                     "dentistry",

    # ── DROP ──
    "radiology":                     "DROP",
    "laboratory":                    "DROP",
    "research":                      "DROP",
    "general_surgery":               "DROP",
}

# Minimum samples để giữ một class trong final dataset
MIN_CLASS_SAMPLES = 20


# ─────────────────────────────────────────────────────────────
# 2. TEXT CLEANING
# ─────────────────────────────────────────────────────────────

# Greeting boilerplate patterns
_BOILERPLATE = re.compile(
    r"(Chào\s+bác\s+sĩ[,.]?\s*|Thưa\s+bác\s+sĩ[,.]?\s*|Kính\s+thưa\s+bác\s+sĩ[,.]?\s*|"
    r"Khách\s+hàng\s+ẩn\s+danh\.?\s*|"
    r"Chuyên\s+gia\s+tư\s+vấn\s*:\s*)",
    flags=re.IGNORECASE,
)

# Unicode ellipsis → 3 dots, then strip trailing
_ELLIPSIS_UNICODE = re.compile(r"…")
# Trailing "..." at end of sentence (truncated title)
_TRAILING_ELLIPSIS = re.compile(r"\s*\.{2,}\s*$")
# Repeated whitespace/newlines
_WHITESPACE = re.compile(r"\s+")


def extract_title_body(text: str) -> tuple[str, str]:
    """
    Tách tiêu đề và nội dung câu hỏi.

    Định dạng phổ biến trong cả 3 nguồn:
        "Tiêu đề câu hỏi?. Nội dung chi tiết..."
        "Tiêu đề…. Nội dung chi tiết..."

    Quy tắc:
      - Split tại dấu "?. " hoặc ". " sau dấu "…"
      - Nếu không tìm thấy separator: cả text là body, title rỗng
    """
    # Replace unicode ellipsis
    text = _ELLIPSIS_UNICODE.sub("...", text)

    # Try splitting: "Title?. Body" or "Title.... Body" or "Title. Body" (if title ends with ?)
    m = re.split(r"(?<=[?!])\.\s+|\.{2,3}\s+", text, maxsplit=1)
    if len(m) == 2:
        title = m[0].rstrip(".").strip()
        body  = m[1].strip()
    else:
        # No clear split: use first sentence as title
        first_sent = re.split(r"[.?!]\s", text, maxsplit=1)
        title = first_sent[0].strip() if len(first_sent) > 1 else ""
        body  = text.strip()

    return title, body


def clean_text(text: str, use_title_only: bool = False) -> str:
    """
    Pipeline làm sạch văn bản.

    Args:
        text: raw input text
        use_title_only: nếu True, chỉ dùng tiêu đề (hữu ích cho model ngắn)
                        nếu False, dùng cả title + body (tốt hơn cho BERT)

    Bước thực hiện:
        1. Unicode normalise (NFC)
        2. Thay unicode ellipsis "…" → "..."
        3. Tách title / body
        4. Xoá boilerplate greetings
        5. Xoá khoảng trắng thừa
        6. Strip trailing ellipsis (title bị cắt)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Normalise unicode
    text = unicodedata.normalize("NFC", text)

    # 2 & 3. Extract parts
    title, body = extract_title_body(text)

    # 4. Remove boilerplate
    body = _BOILERPLATE.sub(" ", body).strip()
    title = _BOILERPLATE.sub(" ", title).strip()
    title = _TRAILING_ELLIPSIS.sub("", title).strip()

    if use_title_only:
        cleaned = title if title else body
    else:
        # SOTA models (BERT, XLM-R) xử lý tốt nhất khi có đủ context
        # Format: "Tiêu đề: <title> Nội dung: <body>"
        # → giúp model phân biệt rõ title signal vs detail
        if title and body and title.lower() not in body.lower()[:80]:
            cleaned = f"{title} {body}"
        elif body:
            cleaned = body
        else:
            cleaned = title

    # 5. Whitespace
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    return cleaned


def text_format_for_model(title: str, body: str, mode: str = "concat") -> str:
    """
    Định dạng text cho các loại model khác nhau.

    Modes:
      "concat"   → "{title} {body}"   — dùng cho PhoBERT, ViHealthBERT
      "sep"      → "{title} [SEP] {body}"  — dùng khi muốn explicit separator
      "prompt"   → "Phân loại khoa cho câu hỏi: {title} {body}"
                   — dùng cho generative models (ViT5, BARTPho)
      "title_only" → "{title}"        — baseline nhanh
    """
    if mode == "concat":
        return f"{title} {body}".strip()
    elif mode == "sep":
        return f"{title} [SEP] {body}".strip() if body else title
    elif mode == "prompt":
        return f"Phân loại khoa cho câu hỏi sau: {title} {body}".strip()
    elif mode == "title_only":
        return title
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ─────────────────────────────────────────────────────────────
# 3. LOAD + MERGE
# ─────────────────────────────────────────────────────────────

def load_and_merge(
    alobacsi_path: Path,
    tamanh_path: Path,
    train_ml_path: Path,
) -> pd.DataFrame:
    """Load 3 CSV files và merge thành DataFrame chuẩn."""

    dfs = []
    for path in [alobacsi_path, tamanh_path, train_ml_path]:
        if not Path(path).exists():
            print(f"  ⚠️  Bỏ qua (không tìm thấy): {path}")
            continue
        df = pd.read_csv(path)
        # Ensure required columns exist
        for col in ["id", "text", "topic", "source"]:
            if col not in df.columns:
                raise ValueError(f"File {path} thiếu cột '{col}'")
        dfs.append(df[["id", "text", "topic", "source"]])
        print(f"  Loaded {path}: {len(df)} rows")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  Tổng sau merge: {len(merged)} rows")
    return merged


# ─────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    alobacsi_path: Path,
    tamanh_path: Path,
    train_ml_path: Path,
    output_dir: Path,
    min_samples: int = MIN_CLASS_SAMPLES,
    val_size: float = 0.1,
    test_size: float = 0.1,
    text_mode: str = "concat",
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []
    log = lambda s: (print(s), report_lines.append(s))

    log("=" * 65)
    log("TOPIC CLASSIFICATION — PREPROCESSING PIPELINE")
    log("=" * 65)

    # ── Step 1: Load ──────────────────────────────────────────
    log("\n[1/7] Load & Merge files")
    df = load_and_merge(alobacsi_path, tamanh_path, train_ml_path)

    # ── Step 2: Drop rows with missing values ─────────────────
    log("\n[2/7] Remove rows with null text / topic")
    before = len(df)
    df = df.dropna(subset=["text", "topic"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    log(f"  Dropped (null/empty): {before - len(df)} rows → {len(df)} remain")

    # ── Step 3: Canonical topic mapping ──────────────────────
    log("\n[3/7] Canonical topic mapping")
    unknown_topics = set(df["topic"].unique()) - set(TOPIC_MAPPING.keys())
    if unknown_topics:
        log(f"  ⚠️  Unknown topics (will be kept as-is): {sorted(unknown_topics)}")
        for t in unknown_topics:
            TOPIC_MAPPING[t] = t

    df["topic_canonical"] = df["topic"].map(TOPIC_MAPPING)
    drop_count = (df["topic_canonical"] == "DROP").sum()
    log(f"  Dropped (topic=DROP): {drop_count} rows")
    df = df[df["topic_canonical"] != "DROP"].copy()
    df["topic"] = df["topic_canonical"]
    df.drop(columns=["topic_canonical"], inplace=True)
    log(f"  → {len(df)} rows, {df['topic'].nunique()} canonical topics")

    # ── Step 4: Drop rare classes ────────────────────────────
    log(f"\n[4/7] Drop classes with < {min_samples} samples")
    class_counts = df["topic"].value_counts()
    rare = class_counts[class_counts < min_samples].index.tolist()
    if rare:
        log(f"  Classes to drop: {rare}")
        df = df[~df["topic"].isin(rare)]
    log(f"  → {len(df)} rows, {df['topic'].nunique()} classes after dropping rare")

    # ── Step 5: Deduplicate ──────────────────────────────────
    log("\n[5/7] Deduplicate")
    before = len(df)
    df["text_norm"] = df["text"].str.lower().str.strip()
    df = df.drop_duplicates(subset="text_norm").copy()
    df.drop(columns=["text_norm"], inplace=True)
    log(f"  Removed {before - len(df)} duplicates → {len(df)} rows")

    # ── Step 6: Clean text ───────────────────────────────────
    log(f"\n[6/7] Clean text (mode='{text_mode}')")
    df["title"] = ""
    df["body"]  = ""
    cleaned_texts = []
    titles, bodies = [], []

    for raw in df["text"]:
        t, b = extract_title_body(
            unicodedata.normalize("NFC", str(raw))
        )
        t = _BOILERPLATE.sub(" ", t).strip()
        t = _TRAILING_ELLIPSIS.sub("", t).strip()
        b = _BOILERPLATE.sub(" ", b).strip()
        b = _WHITESPACE.sub(" ", b).strip()
        t = _WHITESPACE.sub(" ", t).strip()
        titles.append(t)
        bodies.append(b)
        cleaned_texts.append(text_format_for_model(t, b, mode=text_mode))

    df["title"] = titles
    df["body"]  = bodies
    df["text_clean"] = cleaned_texts

    # Quality check: empty cleaned texts
    empty_clean = (df["text_clean"].str.strip() == "").sum()
    if empty_clean:
        log(f"  ⚠️  {empty_clean} rows have empty cleaned text — dropped")
        df = df[df["text_clean"].str.strip() != ""]

    # Text length stats after cleaning
    df["text_len"] = df["text_clean"].str.len()
    log(f"  Text length after clean — min:{df['text_len'].min()} "
        f"median:{df['text_len'].median():.0f} "
        f"max:{df['text_len'].max()} "
        f"mean:{df['text_len'].mean():.0f}")

    # ── Step 7: Build label map ──────────────────────────────
    log("\n[7/8] Build label mapping")
    unique_topics = sorted(df["topic"].unique())
    topic2id = {t: i for i, t in enumerate(unique_topics)}
    id2topic  = {i: t for t, i in topic2id.items()}
    df["label"] = df["topic"].map(topic2id)
    log(f"  {len(topic2id)} classes: {unique_topics}")

    # ── Step 8: Stratified split ─────────────────────────────
    log(f"\n[8/8] Stratified split (train/val/test = "
        f"{1-val_size-test_size:.0%}/{val_size:.0%}/{test_size:.0%})")

    # First split off test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(df, df["label"]))
    df_trainval = df.iloc[train_val_idx].copy()
    df_test     = df.iloc[test_idx].copy()

    # Then split val from train
    val_ratio_adj = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=seed)
    train_idx, val_idx = next(sss2.split(df_trainval, df_trainval["label"]))
    df_train = df_trainval.iloc[train_idx].copy()
    df_val   = df_trainval.iloc[val_idx].copy()

    log(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # ── Save outputs ─────────────────────────────────────────
    def to_records(df_split: pd.DataFrame) -> list[dict]:
        return [
            {
                "id":       row["id"],
                "text":     row["text_clean"],
                "title":    row["title"],
                "body":     row["body"],
                "label":    int(row["label"]),
                "topic":    row["topic"],
                "source":   row["source"],
            }
            for _, row in df_split.iterrows()
        ]

    for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out_path = output_dir / f"topic_{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_records(df_split), f, ensure_ascii=False, indent=2)
        log(f"  → Saved {split_name}: {out_path}")

    label_map_path = output_dir / "topic_label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"topic2id": topic2id, "id2topic": id2topic}, f, ensure_ascii=False, indent=2)
    log(f"  → Saved label map: {label_map_path}")

    # ── EDA Report ───────────────────────────────────────────
    log("\n" + "=" * 65)
    log("CLASS DISTRIBUTION (train set)")
    log("=" * 65)
    train_counts = df_train["topic"].value_counts()
    max_c = train_counts.max()
    for topic, count in train_counts.items():
        bar = "█" * int(count / max_c * 30)
        log(f"  {topic:35s} {count:5d}  {bar}")

    log("\n" + "=" * 65)
    log("IMBALANCE ANALYSIS")
    log("=" * 65)
    ratio = train_counts.max() / train_counts.min()
    log(f"  Max class: {train_counts.idxmax()} ({train_counts.max()})")
    log(f"  Min class: {train_counts.idxmin()} ({train_counts.min()})")
    log(f"  Imbalance ratio: {ratio:.1f}x")

    if ratio > 10:
        log("\n  ⚠️  Imbalance ratio > 10x — khuyến nghị:")
        log("     - Dùng class_weight='balanced' hoặc focal loss")
        log("     - Oversampling (SMOTE text / back-translation) cho class hiếm")
        log("     - Undersample class dominant (reproductive_endocrinology, pediatrics)")

    log("\n" + "=" * 65)
    log("SOURCE DISTRIBUTION (train)")
    log("=" * 65)
    log(df_train["source"].value_counts().to_string())

    # Save report
    report_path = output_dir / "topic_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  → Report saved: {report_path}")
    print(f"\n✅ Pipeline hoàn tất! Xem {output_dir}/")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alobacsi",   type=str, required=True)
    parser.add_argument("--tamanh",     type=str, required=True)
    parser.add_argument("--train_ml",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--min_samples", type=int, default=20)
    parser.add_argument("--val_size",   type=float, default=0.1)
    parser.add_argument("--test_size",  type=float, default=0.1)
    parser.add_argument("--text_mode",  type=str, default="concat",
                        choices=["concat", "sep", "prompt", "title_only"])
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        alobacsi_path  = Path(args.alobacsi),
        tamanh_path    = Path(args.tamanh),
        train_ml_path  = Path(args.train_ml),
        output_dir     = Path(args.output_dir),
        min_samples    = args.min_samples,
        val_size       = args.val_size,
        test_size      = args.test_size,
        text_mode      = args.text_mode,
        seed           = args.seed,
    )
