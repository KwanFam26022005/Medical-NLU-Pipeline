"""
build_topic_dataset.py - Pipeline xây dựng dataset Topic Classification.
Trạm 2B: Gộp dữ liệu từ Vinmec, Tâm Anh, AloBacsi + FAQ → JSON.

v2 - DEEP CLEANING + MINORITY PROTECTION:
  ✅ html.unescape + strip HTML tags
  ✅ Emoji / special char removal
  ✅ Viết tắt y khoa → full form
  ✅ Word-count filter (≥6 words)
  ✅ Downsampling (>1500) + Oversampling (<200)
  ✅ Stratified dedup (dedup BEFORE balancing)
  ✅ N-gram dedup (near-duplicate removal)
  ✅ Quality scoring + low-quality rejection

Cách chạy:
    python build_topic_dataset.py
"""

import html
import re
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. TEXT CLEANING - DEEP CLEANING v2
# ============================================================

# Viết tắt y khoa phổ biến → full form
MEDICAL_ABBREVIATIONS = {
    r"\bbs\b": "bác sĩ",
    r"\bbn\b": "bệnh nhân",
    r"\bbv\b": "bệnh viện",
    r"\bsa\b": "siêu âm",
    r"\bxn\b": "xét nghiệm",
    r"\bhp\b": "helicobacter pylori",
    r"\bha\b": "huyết áp",
    r"\btmh\b": "tai mũi họng",
    r"\bđtđ\b": "đái tháo đường",
    r"\btvđđ\b": "thoát vị đĩa đệm",
    r"\bcts\b": "cột sống",
    r"\bntm\b": "nhồi máu cơ tim",
    r"\bxhdd\b": "xuất huyết dạ dày",
    r"\bvp\b": "viêm phổi",
    r"\bvg\b": "viêm gan",
    r"\bck\b": "chuyên khoa",
    r"\bpk\b": "phòng khám",
}

# Stopword patterns (lời chào / kết thúc / quảng cáo)
NOISE_PATTERNS = [
    r"(Chào bác sĩ|Thưa bác sĩ|Xin chào bác sĩ|Bác sĩ cho tôi hỏi"
    r"|Tôi xin cảm ơn|Khách hàng ẩn danh|Câu hỏi ẩn danh"
    r"|Chào bác_sĩ|Thưa bác_sĩ|Xin chào bác_sĩ)[.,:;!]*",
    r"(Xin chào|Cảm ơn bác sĩ|Cám ơn bác sĩ|Mong bác sĩ giải đáp"
    r"|Em cảm ơn|Cảm ơn nhiều|Xin cảm ơn)[.,:;!]*",
    r"(Vinmec|Tâm Anh|AloBacsi|Từ Dũ|Chợ Rẫy|Bạch Mai|108)",
    r"_[A-Za-z\s]+_$",
]


def clean_medical_text(text: str) -> str:
    """
    Deep cleaning cho văn bản y khoa.
    
    Pipeline:
    1. HTML unescape + strip tags
    2. Remove emoji & special chars
    3. Noise patterns (greeting, hospital names)
    4. Expand medical abbreviations
    5. Normalize whitespace
    6. Lowercase (BẮT BUỘC cuối cùng)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # --- 1. HTML Processing ---
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)  # Strip ALL HTML tags

    # --- 2. Emoji & Special Characters ---
    # Giữ lại: chữ cái (bao gồm tiếng Việt), số, dấu câu cơ bản, khoảng trắng
    text = re.sub(
        r"[^\w\s.,;:!?()%/\-–—àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄ"
        r"ÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]",
        " ", text
    )

    # --- 3. Noise Patterns ---
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # --- 4. Medical Abbreviations ---
    for abbr, full in MEDICAL_ABBREVIATIONS.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)

    # --- 5. Normalize Whitespace ---
    text = re.sub(r"\s+", " ", text).strip()

    # --- 6. Lowercase (BẮT BUỘC) ---
    text = text.lower()

    return text


def compute_text_quality_score(text: str) -> float:
    """
    Tính điểm chất lượng text (0.0 → 1.0).
    
    Criteria:
    - Word count (longer = better, up to a point)
    - Unique word ratio (diversity)
    - Has Vietnamese medical keywords
    - Not too repetitive
    """
    if not text:
        return 0.0

    words = text.split()
    n_words = len(words)

    # Word count score (0-0.3): optimal 15-200 words
    if n_words < 6:
        wc_score = 0.0
    elif n_words < 15:
        wc_score = 0.15
    elif n_words <= 200:
        wc_score = 0.3
    else:
        wc_score = 0.2  # Too long, might be noisy

    # Unique word ratio (0-0.3)
    unique_ratio = len(set(words)) / max(n_words, 1)
    uw_score = min(unique_ratio * 0.4, 0.3)

    # Medical keyword presence (0-0.2)
    medical_kws = [
        "bệnh", "triệu chứng", "điều trị", "thuốc", "bác sĩ",
        "khám", "xét nghiệm", "siêu âm", "viêm", "đau",
        "phẫu thuật", "chẩn đoán", "sức khỏe", "bệnh viện",
    ]
    kw_count = sum(1 for kw in medical_kws if kw in text)
    med_score = min(kw_count * 0.05, 0.2)

    # Repetition penalty (0-0.2)
    if n_words > 5:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        bigram_unique = len(set(bigrams)) / max(len(bigrams), 1)
        rep_score = min(bigram_unique * 0.25, 0.2)
    else:
        rep_score = 0.0

    return wc_score + uw_score + med_score + rep_score


def get_text_fingerprint(text: str, n: int = 3) -> str:
    """
    Tạo fingerprint từ n-gram để phát hiện near-duplicates.
    Dùng sorted character trigrams → hash.
    """
    words = text.split()
    if len(words) < n:
        return hashlib.md5(text.encode()).hexdigest()

    ngrams = set()
    for i in range(len(words) - n + 1):
        ngrams.add(" ".join(words[i : i + n]))

    # Sort để đảm bảo thứ tự giống nhau cho near-duplicates
    sorted_ngrams = sorted(ngrams)
    fingerprint = " ".join(sorted_ngrams[:20])  # Lấy 20 ngram đầu
    return hashlib.md5(fingerprint.encode()).hexdigest()


# ============================================================
# 2. TOPIC MAPPING - từ nhãn gốc (CSV) → nhãn hợp nhất
# ============================================================
TOPIC_MAPPING = {
    # Nội khoa & các chuyên khoa nội
    "internal_medicine":            "internal_medicine",
    "gastroenterology":             "gastroenterology",
    "cardiology":                   "cardiology",
    "endocrinology":                "endocrinology",
    "respiratory":                  "respiratory",
    "nephrology":                   "internal_medicine",
    "rheumatology":                 "internal_medicine",
    "hematology":                   "internal_medicine",
    "allergy":                      "internal_medicine",
    "infectious_disease":           "internal_medicine",
    "nutrition":                    "internal_medicine",
    "immunology":                   "internal_medicine",

    # Sản phụ khoa & sức khoẻ sinh sản nữ
    "obstetrics_and_gynecology":    "obstetrics_and_gynecology",
    "reproductive_endocrinology":   "obstetrics_and_gynecology",
    "reproductive_health":          "obstetrics_and_gynecology",
    "gynecology":                   "obstetrics_and_gynecology",
    "obstetrics":                   "obstetrics_and_gynecology",
    "maternal_health":              "obstetrics_and_gynecology",

    # Nam khoa & tiết niệu
    "andrology":                    "urology_andrology",
    "urology":                      "urology_andrology",

    # Nhi khoa
    "pediatrics":                   "pediatrics",

    # Thần kinh
    "neurology":                    "neurology",
    "psychiatry":                   "psychiatry",
    "psychology":                   "psychiatry",

    # Ung thư
    "oncology":                     "oncology",

    # Da liễu
    "dermatology":                  "dermatology",

    # Xương khớp
    "orthopedics_and_sports_medicine": "orthopedics",
    "orthopedics":                  "orthopedics",
    "rehabilitation":               "orthopedics",
    "traditional_medicine":         "orthopedics",

    # Tai Mũi Họng
    "otolaryngology":               "ent",

    # Mắt
    "ophthalmology":                "ophthalmology",

    # Răng Hàm Mặt
    "odontology":                   "odontology",
    "dentistry":                    "odontology",

    # Ngoại khoa
    "surgery":                      "surgery",
    "neurosurgery":                 "surgery",
    "cardiac_surgery":              "surgery",
    "plastic_surgery":              "surgery",
    "vascular_surgery":             "surgery",

    # Khác có thể map
    "medical_genetics":             "pediatrics",
    "health_information":           "internal_medicine",
}


# ============================================================
# 3. FAQ KEYWORD CLASSIFIER
# ============================================================

FAQ_KEYWORD_RULES = [
    # --- Nhi khoa (ưu tiên cao nhất vì phong phú) ---
    ("pediatrics", [
        "bé", "trẻ sơ sinh", "trẻ nhỏ", "trẻ em", "con", "cháu",
        "tuổi", "tháng tuổi", "đứa bé", "sơ sinh", "bú sữa", "bú mẹ",
        "chích ngừa", "tiêm phòng", "vacxin", "vaccine", "vắc.xin",
        "trẻ bú", "bú bình", "sữa mẹ", "sữa công thức",
        "mọc răng", "tăng cân chậm", "biếng ăn", "rốn rụng",
        "vàng da trẻ", "trẻ sốt", "bé sốt", "bé ho",
    ]),
    # --- Sản phụ khoa ---
    ("obstetrics_and_gynecology", [
        "mang thai", "thai nhi", "thai kỳ", "thai phụ", "mẹ bầu",
        "sinh mổ", "sinh thường", "sinh nở", "hậu sản",
        "kinh nguyệt", "kinh nguyệt không đều", "khí hư", "âm đạo",
        "tử cung", "buồng trứng", "cổ tử cung", "niêm mạc tử cung",
        "thuốc tránh thai", "que thử thai", "siêu âm thai",
        "tuần thai", "thai ngoài tử cung", "sảy thai",
        "lạc nội mạc", "u xơ tử cung", "u nang buồng trứng",
        "cho con bú", "sau sinh", "huyết trắng",
    ]),
    # --- Tiết niệu & Nam khoa ---
    ("urology_andrology", [
        "tiểu rắt", "tiểu buốt", "đi tiểu", "nước tiểu", "tiết niệu",
        "sỏi thận", "thận", "bàng quang", "niệu đạo", "niệu quản",
        "dương vật", "bao quy đầu", "tinh hoàn", "tinh trùng",
        "xuất tinh", "cương dương", "nam khoa", "sùi mào gà",
        "viêm tuyến tiền liệt", "dãn tĩnh mạch tinh",
    ]),
    # --- Tiêu hoá / Gastroenterology ---
    ("gastroenterology", [
        "dạ dày", "ruột", "đại tràng", "trực tràng", "gan", "mật",
        "tụy", "viêm dạ dày", "loét dạ dày", "trào ngược",
        "viêm gan", "xơ gan", "hp", "helicobacter",
        "táo bón", "tiêu chảy", "đi ngoài", "phân",
        "buồn nôn", "nôn", "ợ hơi", "đầy bụng", "đau bụng",
        "polyp", "sỏi mật", "túi mật", "viêm tụy",
    ]),
    # --- Tim mạch ---
    ("cardiology", [
        "tim", "nhịp tim", "huyết áp", "cao huyết áp", "tăng huyết áp",
        "hạ huyết áp", "mạch", "động mạch", "tĩnh mạch",
        "nhồi máu cơ tim", "suy tim", "đau ngực", "đánh trống ngực",
        "rối loạn nhịp tim", "điện tâm đồ", "stent",
    ]),
    # --- Hô hấp ---
    ("respiratory", [
        "phổi", "ho", "đờm", "khò khè", "khó thở", "tức ngực",
        "viêm phổi", "viêm phế quản", "hen suyễn", "hen phế quản",
        "lao phổi", "tràn khí màng phổi", "viêm tiểu phế quản",
        "hô hấp", "thở", "sổ mũi nghẹt mũi viêm mũi",
    ]),
    # --- Nội tiết ---
    ("endocrinology", [
        "tiểu đường", "đái tháo đường", "đường huyết",
        "tuyến giáp", "bướu cổ", "cường giáp", "suy giáp",
        "insulin", "triglycerid", "cholesterol", "rối loạn chuyển hoá",
        "hormone", "nội tiết tố", "tuyến thượng thận",
    ]),
    # --- Thần kinh ---
    ("neurology", [
        "đau đầu", "chóng mặt", "đột quỵ", "tai biến", "nhồi máu não",
        "xuất huyết não", "động kinh", "co giật", "tê liệt",
        "thần kinh", "đau thần kinh", "parkinson", "alzheimer",
        "mất trí nhớ", "u não", "màng não",
    ]),
    # --- Tâm thần / Sức khoẻ tâm thần ---
    ("psychiatry", [
        "trầm cảm", "lo âu", "rối loạn lo âu", "mất ngủ",
        "tâm thần", "tâm lý", "stress", "căng thẳng",
        "rối loạn tâm thần", "nghiện", "hoảng sợ",
    ]),
    # --- Ung thư ---
    ("oncology", [
        "ung thư", "u ác tính", "khối u", "ung bướu",
        "hóa trị", "xạ trị", "sinh thiết", "di căn",
        "tế bào ung thư", "u lympho", "leukemia",
    ]),
    # --- Da liễu ---
    ("dermatology", [
        "mụn", "da", "viêm da", "chàm", "eczema",
        "nổi mẩn", "ngứa da", "vảy nến", "phát ban",
        "rụng tóc", "tóc", "nấm da", "ghẻ",
    ]),
    # --- Xương khớp ---
    ("orthopedics", [
        "xương", "khớp", "gãy xương", "đau lưng", "cột sống",
        "thoát vị đĩa đệm", "viêm khớp", "loãng xương",
        "đau cổ", "vai gáy", "đầu gối", "khớp háng",
        "dây chằng", "gân", "cơ bắp", "chấn thương",
    ]),
    # --- Tai Mũi Họng ---
    ("ent", [
        "tai", "mũi", "họng", "viêm họng", "viêm amidan",
        "viêm xoang", "ù tai", "điếc", "khản tiếng",
        "lệch vách ngăn", "polyp mũi", "viêm tai giữa",
        "amidan", "hạch cổ",
    ]),
    # --- Mắt ---
    ("ophthalmology", [
        "mắt", "thị lực", "cận thị", "viễn thị", "loạn thị",
        "đau mắt", "đỏ mắt", "ghèn mắt", "mờ mắt",
        "đục thuỷ tinh thể", "glaucom", "võng mạc",
    ]),
    # --- Răng Hàm Mặt ---
    ("odontology", [
        "răng", "lợi", "nướu", "hàm", "sâu răng",
        "nhổ răng", "niềng răng", "implant", "răng sữa",
        "mọc răng khôn", "viêm nướu", "quai hàm",
    ]),
    # --- Ngoại khoa ---
    ("surgery", [
        "phẫu thuật", "mổ", "nội soi", "gây mê", "gây tê",
        "vết mổ", "vết thương", "khâu", "u lành",
        "ruột thừa", "thoát vị", "sỏi", "cắt bỏ",
    ]),
]


def classify_faq_text(text: str) -> Optional[str]:
    """Classify FAQ text by keyword matching. Returns topic string or None."""
    if not isinstance(text, str):
        return None
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for topic, keywords in FAQ_KEYWORD_RULES:
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score

    if not scores:
        return None

    best = max(scores, key=scores.get)
    # Require at least 2 keyword matches for confidence
    if scores[best] < 2:
        return None
    return best


# ============================================================
# 4. LOAD FAQ DATASET
# ============================================================
def load_faq_dataset(faq_dir: Path) -> pd.DataFrame:
    """Load FAQ_summarization dataset and auto-assign topics via keyword matching."""
    records = []
    for split in ["train", "dev", "test"]:
        src_file = faq_dir / split / f"{split}.txt.src"
        if not src_file.exists():
            alt_name = "val" if split == "dev" else split
            src_file = faq_dir / split / f"{alt_name}.txt.src"
        if not src_file.exists():
            continue

        with open(src_file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            topic = classify_faq_text(line)
            if topic:
                records.append({"text": line, "topic": topic, "source": "faq"})

    if records:
        df = pd.DataFrame(records)
        print(f"  [+] FAQ dataset: {len(df)} mẫu có nhãn / {len(records)} total")
        return df
    return pd.DataFrame()


# ============================================================
# 5. MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("🏥 BUILD TOPIC DATASET - v2 DEEP CLEANING")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # 5.1 Load all CSV sources
    # ──────────────────────────────────────────────
    print("\n⏳ Đang đọc và gộp dữ liệu từ 3 bệnh viện...")
    csv_files = ["train_ml.csv", "alobacsi_processed.csv", "ml_training_data_tamanh.csv"]

    dfs = []
    for file in csv_files:
        p = Path(file) if Path(file).exists() else Path("data") / file
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  [+] Load {p}: {len(df)} dòng.")
        else:
            print(f"  [!] Cảnh báo: Không tìm thấy {p}")

    # Load FAQ dataset
    faq_dir = Path("data") / "FAQ_summarization"
    if not faq_dir.exists():
        faq_dir = Path("data/FAQ_summarization")
    if faq_dir.exists():
        print("\n📚 Đang load FAQ_summarization dataset...")
        faq_df = load_faq_dataset(faq_dir)
        if not faq_df.empty:
            dfs.append(faq_df)
    else:
        print("  [!] Không tìm thấy data/FAQ_summarization/")

    if not dfs:
        print("❌ Không có dữ liệu để xử lý!")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["text"])
    print(f"\n📊 Tổng dữ liệu thô: {len(full_df)} mẫu")

    # ──────────────────────────────────────────────
    # 5.2 Exact dedup TRƯỚC KHI clean (nhanh hơn)
    # ──────────────────────────────────────────────
    before_dedup = len(full_df)
    full_df = full_df.drop_duplicates(subset=["text"])
    print(f"   Exact dedup: loại {before_dedup - len(full_df)} bản sao → {len(full_df)} mẫu")

    # ──────────────────────────────────────────────
    # 5.3 Deep Cleaning
    # ──────────────────────────────────────────────
    print("\n🧹 Đang Deep Clean văn bản...")
    full_df["clean_text"] = full_df["text"].apply(clean_medical_text)

    # ──────────────────────────────────────────────
    # 5.4 Word-count filter (≥6 words)
    # ──────────────────────────────────────────────
    before_wc = len(full_df)
    full_df = full_df[full_df["clean_text"].apply(lambda x: len(str(x).split()) >= 6)]
    print(f"   Word-count filter (≥6 words): loại {before_wc - len(full_df)} mẫu ngắn → {len(full_df)} mẫu")

    # ──────────────────────────────────────────────
    # 5.5 Near-duplicate removal (n-gram fingerprint)
    # ──────────────────────────────────────────────
    before_near = len(full_df)
    full_df["_fingerprint"] = full_df["clean_text"].apply(get_text_fingerprint)
    full_df = full_df.drop_duplicates(subset=["_fingerprint"])
    full_df = full_df.drop(columns=["_fingerprint"])
    print(f"   Near-dedup (n-gram): loại {before_near - len(full_df)} near-duplicates → {len(full_df)} mẫu")

    # ──────────────────────────────────────────────
    # 5.6 Quality scoring + filtering
    # ──────────────────────────────────────────────
    full_df["_quality"] = full_df["clean_text"].apply(compute_text_quality_score)
    quality_threshold = 0.15
    before_quality = len(full_df)
    full_df = full_df[full_df["_quality"] >= quality_threshold]
    full_df = full_df.drop(columns=["_quality"])
    print(f"   Quality filter (≥{quality_threshold}): loại {before_quality - len(full_df)} mẫu kém → {len(full_df)} mẫu")

    # ──────────────────────────────────────────────
    # 5.7 Topic Mapping
    # ──────────────────────────────────────────────
    def get_unified_topic(row):
        if row.get("source") == "faq":
            return row["topic"]
        raw_topic = row.get("topic", "")
        if isinstance(raw_topic, str):
            return TOPIC_MAPPING.get(raw_topic.strip(), "others")
        return "others"

    full_df["unified_topic"] = full_df.apply(get_unified_topic, axis=1)

    # Remove "others"
    before_others = len(full_df)
    full_df = full_df[full_df["unified_topic"] != "others"]
    print(f"   Loại bỏ {before_others - len(full_df)} mẫu nhãn 'others'")

    # Remove very rare topics (<30 samples)
    label_counts = full_df["unified_topic"].value_counts()
    valid_labels = label_counts[label_counts >= 30].index
    removed_rare = len(full_df) - len(full_df[full_df["unified_topic"].isin(valid_labels)])
    full_df = full_df[full_df["unified_topic"].isin(valid_labels)]
    if removed_rare > 0:
        print(f"   Loại bỏ {removed_rare} mẫu thuộc nhãn có <30 samples")

    print(f"\n📈 PHÂN BỐ NHÃN TRƯỚC KHI CÂN BẰNG ({len(full_df)} mẫu):")
    dist_before = full_df["unified_topic"].value_counts()
    print(dist_before.to_string())

    # ──────────────────────────────────────────────
    # 5.8 Balanced Sampling: Downsampling + Oversampling
    # ──────────────────────────────────────────────
    MAX_PER_CLASS = 1500
    MIN_PER_CLASS = 200

    balanced_dfs = []
    for label, group in full_df.groupby("unified_topic"):
        n = len(group)
        if n > MAX_PER_CLASS:
            # Downsampling: lấy MAX_PER_CLASS mẫu
            group = group.sample(MAX_PER_CLASS, random_state=42)
            print(f"   ⬇️  {label}: {n} → {MAX_PER_CLASS} (downsample)")
        elif n < MIN_PER_CLASS:
            # Oversampling: nhân đôi dữ liệu thiểu số
            repeats = (MIN_PER_CLASS // n) + 1
            group = pd.concat([group] * repeats, ignore_index=True).head(MIN_PER_CLASS)
            print(f"   ⬆️  {label}: {n} → {MIN_PER_CLASS} (oversample ×{repeats})")
        balanced_dfs.append(group)

    final_df = pd.concat(balanced_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n📊 PHÂN BỐ NHÃN SAU KHI CÂN BẰNG ({len(final_df)} mẫu):")
    dist_after = final_df["unified_topic"].value_counts()
    print(dist_after.to_string())

    # Tính imbalance ratio
    max_c = dist_after.max()
    min_c = dist_after.min()
    print(f"   Imbalance ratio: {max_c / max(min_c, 1):.1f}x (max={max_c}, min={min_c})")

    # ──────────────────────────────────────────────
    # 5.9 Train/Val Split (stratified)
    # ──────────────────────────────────────────────
    train_df, val_df = train_test_split(
        final_df,
        test_size=0.15,
        stratify=final_df["unified_topic"],
        random_state=42,
    )

    # ──────────────────────────────────────────────
    # 5.10 Save as JSON
    # ──────────────────────────────────────────────
    def df_to_json(df: pd.DataFrame, output_path: str):
        json_data = [
            {"text": row["clean_text"], "label": row["unified_topic"]}
            for _, row in df.iterrows()
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    Path("data").mkdir(exist_ok=True)
    df_to_json(train_df, "data/topic_train.json")
    df_to_json(val_df, "data/topic_val.json")

    # Save label list
    label_list = sorted(final_df["unified_topic"].unique().tolist())
    with open("data/topic_labels.json", "w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────
    # 5.11 Summary report
    # ──────────────────────────────────────────────
    print(f"\n✅ Số nhãn: {len(label_list)}")
    print(f"   {label_list}")
    print(f"\n🎉 HOÀN TẤT!")
    print(f"   Train: {len(train_df)} mẫu")
    print(f"   Val:   {len(val_df)} mẫu")
    print(f"   Đã lưu: data/topic_train.json, data/topic_val.json, data/topic_labels.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
