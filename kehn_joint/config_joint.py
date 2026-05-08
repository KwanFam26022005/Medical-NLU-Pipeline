"""
config_joint.py — Cấu hình trung tâm cho KEHN (Knowledge-Enhanced Hierarchical Network).
Chứa paths, label mappings, và hyperparameters cho joint Topic + Intent + NER.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # Chatbot-Y tế/
KEHN_DIR = Path(__file__).resolve().parent         # kehn_joint/

_DATA_ROOT = BASE_DIR
VIMQ_DATA_DIR = _DATA_ROOT / "ViMQ-main" / "ViMQ-main" / "data"
TOPIC_DATA_DIR = _DATA_ROOT / "data"

JOINT_DATA_DIR = KEHN_DIR / "data"
JOINT_OUTPUT_DIR = KEHN_DIR / "outputs"

# Existing model directories for pseudo-labeling
TOPIC_MODEL_HF = "KwanFam26022005/model2B-topic-classification"
NER_MODEL_HF = "hoangkhang1628/vihealthbert-crf-medical-ner"
INTENT_MODEL_HF = "hoangkhang1628/vihealthbert-asl-medical-intent"

# ============================================================
# 🏷️ LABEL DEFINITIONS
# ============================================================

# ── Intent labels ────────────────────────────────────────────
# BUG FIX: Đây là thứ tự THỰC TẾ trong JSONL (kiểm chứng từ data).
# method_diagnosis=0, treatment=1, severity=2, cause=3.
# (metadata.json dùng thứ tự alphabet khác — không phải nguồn truth cho IDs.)
INTENT_LABELS = ["method_diagnosis", "treatment", "severity", "cause"]
INTENT2ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ID2INTENT  = {idx: label for idx, label in enumerate(INTENT_LABELS)}
N_INTENT   = len(INTENT_LABELS)

# ── NER tags ─────────────────────────────────────────────────
# BUG FIX [CRITICAL]: I-SEV bị THIẾU trong config cũ (chỉ có 10 tags).
# Dataset thực tế có 11 tags; I-SEV xuất hiện 33 lần trong JSONL.
# Nếu thiếu I-SEV → NER2ID.get("I-SEV", 0) → map thành O, âm thầm sai.
# Thứ tự giữ nguyên (O=0..B-SEV=9) để không làm vỡ ner_tag_ids đã lưu;
# chỉ thêm I-SEV=10 vào cuối — khớp với ner_tag_ids trong JSONL (giá trị 10).
NER_TAGS = [
    "O",       # 0
    "B-SYM",   # 1
    "I-SYM",   # 2
    "B-PRO",   # 3
    "I-PRO",   # 4
    "B-DRU",   # 5
    "I-DRU",   # 6
    "B-DUR",   # 7
    "I-DUR",   # 8
    "B-SEV",   # 9
    "I-SEV",   # 10  ← thêm mới
]
NER2ID     = {tag: idx for idx, tag in enumerate(NER_TAGS)}
ID2NER     = {idx: tag for idx, tag in enumerate(NER_TAGS)}
N_NER_TAG  = len(NER_TAGS)  # 11

# Entity type set (dùng để build BIO constraint mask trong KEHN)
NER_ENTITY_TYPES = ["SYM", "PRO", "DRU", "DUR", "SEV"]

# ── Topic labels ─────────────────────────────────────────────
# BUG FIX [CRITICAL]: JSONL có 2 lỗi encoding topic:
#   1. oncology (300 mẫu) và ophthalmology (51 mẫu) cùng được gán id=10
#      do quá trình remap bị lỗi khi tạo data.
#   2. urology (609 mẫu) bị gán id=17 (out-of-range).
# Fix: data_loader_joint.py luôn dùng chuỗi `topic_label` → TOPIC2ID
#      thay vì đọc `topic_label_id` trực tiếp từ JSONL.
#
# Quyết định giữ oncology là class riêng (không drop/remap) vì:
#   - 300 mẫu đủ để học (>1.6% tổng data)
#   - Dropping làm class ophthalmology/urology collide với oncology slot
# → N_TOPIC = 17
TOPIC_LABELS = [
    "cardiology",               # 0
    "dentistry",                # 1
    "dermatology",              # 2
    "endocrinology",            # 3
    "ent",                      # 4
    "gastroenterology",         # 5
    "internal_medicine",        # 6
    "neurology",                # 7
    "nutrition",                # 8
    "obstetrics_gynecology",    # 9
    "oncology",                 # 10
    "ophthalmology",            # 11
    "orthopedics",              # 12
    "pediatrics",               # 13
    "reproductive_endocrinology",# 14
    "rheumatology",             # 15
    "urology",                  # 16
]
TOPIC_REMAP = {}   # Không còn remap; data_loader dùng string label
TOPIC_DROP  = set()
TOPIC2ID    = {label: idx for idx, label in enumerate(TOPIC_LABELS)}
ID2TOPIC    = {idx: label for idx, label in enumerate(TOPIC_LABELS)}
N_TOPIC     = len(TOPIC_LABELS)  # 17

# ============================================================
# 🤖 MODEL CONFIG
# ============================================================
MODEL_CONFIG = {
    "phobert":      "vinai/phobert-base-v2",
    "vihealthbert": "demdecuong/vihealthbert-base-word",
    "xlmr_base":   "FacebookAI/xlm-roberta-base",

    "hidden_dim":              768,
    "num_co_interactive_blocks": 2,
    "num_attention_heads":     8,
    "attention_dropout":       0.1,
    "hidden_dropout":          0.1,
    "use_bilstm":              True,
    "use_crf":                 True,

    "n_topic":    N_TOPIC,    # 17
    "n_intent":   N_INTENT,   # 4
    "n_ner_tag":  N_NER_TAG,  # 11
}

# ============================================================
# 🏋️ TRAINING CONFIG
# ============================================================
TRAIN_CONFIG = {
    "max_seq_len": 128,
    "batch_size":  32,
    "learning_rate": 3e-5,
    "weight_decay":  0.01,
    "warmup_ratio":  0.1,
    "num_epochs":    30,
    "gradient_accumulation_steps": 2,
    "fp16":   True,
    "seed":   42,

    "patience": 5,
    "metric_for_best_model": "topic_macro_f1",

    # Curriculum Learning phases (epoch ranges, inclusive)
    "phase_topic_only":    (1,  3),
    "phase_mining_only":   (4,  6),
    "phase_joint_no_prop": (7,  10),
    "phase_full":          (11, 30),

    # Loss weights per phase
    "loss_weights": {
        "topic_only":    {"topic": 1.0, "intent": 0.0, "ner": 0.0},
        "mining_only":   {"topic": 0.1, "intent": 1.0, "ner": 1.0},
        "joint_no_prop": {"topic": 0.5, "intent": 0.3, "ner": 0.2},
        "full":          {"topic": 0.5, "intent": 0.3, "ner": 0.2},
    },
}

# ============================================================
# 📊 PSEUDO-LABELING CONFIG
# ============================================================
PSEUDO_CONFIG = {
    "topic_confidence_threshold": 0.90,
    "tokenizer_for_topic_model":  "demdecuong/vihealthbert-base-syllable",
}