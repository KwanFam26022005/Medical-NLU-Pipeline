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

# Data sources (portable — works on both Windows local and Colab)
_DATA_ROOT = BASE_DIR
VIMQ_DATA_DIR = _DATA_ROOT / "ViMQ-main" / "ViMQ-main" / "data"
TOPIC_DATA_DIR = _DATA_ROOT / "data"

# Data & output dirs (relative — portable giữa local Windows và Colab)
JOINT_DATA_DIR = KEHN_DIR / "splits"
JOINT_OUTPUT_DIR = KEHN_DIR / "outputs"

# Existing model directories for pseudo-labeling
TOPIC_MODEL_HF = "KwanFam26022005/model2B-topic-classification"
NER_MODEL_HF = "hoangkhang1628/vihealthbert-crf-medical-ner"
INTENT_MODEL_HF = "hoangkhang1628/vihealthbert-asl-medical-intent"

# ============================================================
# 🏷️ LABEL DEFINITIONS
# ============================================================

# Intent labels (single-label, from ViMQ sent_label)
INTENT_LABELS = ["method_diagnosis", "treatment", "severity", "cause"]
INTENT2ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ID2INTENT = {idx: label for idx, label in enumerate(INTENT_LABELS)}
N_INTENT = len(INTENT_LABELS)

# NER labels (BIO format, from ViMQ entity_set.txt)
NER_TAGS = ["O", "B-SYM", "I-SYM", "B-PRO", "I-PRO", "B-DRU", "I-DRU"]
NER2ID = {tag: idx for idx, tag in enumerate(NER_TAGS)}
ID2NER = {idx: tag for idx, tag in enumerate(NER_TAGS)}
N_NER_TAG = len(NER_TAGS)

# ViMQ entity type → BIO prefix mapping
ENTITY_TYPE_MAP = {
    "SYMPTOM_AND_DISEASE": "SYM",
    "medical_procedure": "PRO",
    "drug": "DRU",
}

# Topic labels (16 classes — dropped traditional_medicine [1 sample],
# merged oncology → internal_medicine [17 samples])
TOPIC_LABELS = [
    "cardiology", "dentistry", "dermatology", "endocrinology",
    "ent", "gastroenterology", "internal_medicine", "neurology",
    "nutrition", "obstetrics_gynecology", "ophthalmology",
    "orthopedics", "pediatrics", "reproductive_endocrinology",
    "rheumatology", "urology",
]

# Remap rules applied during preprocessing (oncology → internal_medicine)
TOPIC_REMAP = {"oncology": "internal_medicine"}
TOPIC_DROP = {"traditional_medicine"}
TOPIC2ID = {label: idx for idx, label in enumerate(TOPIC_LABELS)}
ID2TOPIC = {idx: label for idx, label in enumerate(TOPIC_LABELS)}
N_TOPIC = len(TOPIC_LABELS)

# ============================================================
# 🤖 MODEL CONFIG
# ============================================================
MODEL_CONFIG = {
    # Backbone (benchmark cả 2)
    "phobert": "vinai/phobert-base-v2",
    "vihealthbert": "demdecuong/vihealthbert-base-word",
    "xlmr_large": "FacebookAI/xlm-roberta-large", # Thêm backbone mới
    # Architecture
    "hidden_dim": 768,
    "num_co_interactive_blocks": 2,
    "num_attention_heads": 8,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.1,
    "use_bilstm": True,
    "use_crf": True,

    # Task heads
    "n_topic": N_TOPIC,
    "n_intent": N_INTENT,
    "n_ner_tag": N_NER_TAG,
}

# ============================================================
# 🏋️ TRAINING CONFIG
# ============================================================
TRAIN_CONFIG = {
    "max_seq_len": 128,
    "batch_size": 32,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "num_epochs": 30,
    "gradient_accumulation_steps": 2,
    "fp16": True,
    "seed": 42,

    # Early stopping
    "patience": 5,
    "metric_for_best_model": "topic_macro_f1",

    # Curriculum Learning phases (epoch ranges, inclusive)
    "phase_topic_only": (1, 3),
    "phase_mining_only": (4, 6),
    "phase_joint_no_prop": (7, 10),
    "phase_full": (11, 30),

    # Loss weights per phase
    "loss_weights": {
        "topic_only":    {"topic": 1.0, "intent": 0.0, "ner": 0.0},
        "mining_only":   {"topic": 0.0, "intent": 1.0, "ner": 1.0},
        "joint_no_prop": {"topic": 0.5, "intent": 0.3, "ner": 0.2},
        "full":          {"topic": 0.5, "intent": 0.3, "ner": 0.2},
    },
}

# ============================================================
# 📊 PSEUDO-LABELING CONFIG
# ============================================================
PSEUDO_CONFIG = {
    "topic_confidence_threshold": 0.90,
    "tokenizer_for_topic_model": "demdecuong/vihealthbert-base-syllable",
}
