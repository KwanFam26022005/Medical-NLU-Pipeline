"""
config.py - Cấu hình trung tâm cho Medical NLU Pipeline.
Chứa tất cả các hằng số, đường dẫn model, nhãn, và hyperparameters.
"""

import os
from pathlib import Path

# ============================================================
# 📁 ĐƯỜNG DẪN THƯ MỤC GỐC
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"

# ============================================================
# 🤖 PRETRAINED MODEL NAMES (HuggingFace Hub)
# ============================================================
# vihealthbert-base-word: yêu cầu word-segmentation trước khi tokenize
VIHEALTHBERT_WORD = "demdecuong/vihealthbert-base-word"
# vihealthbert-base-syllable: tokenize trực tiếp trên âm tiết
VIHEALTHBERT_SYLLABLE = "demdecuong/vihealthbert-base-syllable"

# ============================================================
# 📍 TRẠM 1: ACRONYM DISAMBIGUATION (WSD)
# ============================================================
ACRONYM_MODEL_NAME = VIHEALTHBERT_SYLLABLE
ACRONYM_MODEL_DIR = "KwanFam26022005/model1-acronym-wsd"
ACRONYM_DATA_DIR = DATA_DIR / "acrDrAid"
ACRONYM_DICT_PATH = ACRONYM_DATA_DIR / "dictionary.json"

# ============================================================
# 📍 TRẠM 2A: MEDICAL NER
# ============================================================
NER_MODEL_NAME = VIHEALTHBERT_WORD
NER_MODEL_DIR = MODEL_DIR / "medical_ner"
# Nhãn NER theo BIO schema - ViMQ dataset
NER_LABELS = [
    "O",
    "B-SYMPTOM_AND_DISEASE",
    "I-SYMPTOM_AND_DISEASE",
    "B-MEDICAL_PROCEDURE",
    "I-MEDICAL_PROCEDURE",
    "B-MEDICINE",
    "I-MEDICINE",
]
NER_LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
NER_ID2LABEL = {idx: label for idx, label in enumerate(NER_LABELS)}

# ============================================================
# 📍 TRẠM 2B: TOPIC CLASSIFICATION (PENDING DATA)
# ============================================================
TOPIC_MODEL_NAME = VIHEALTHBERT_SYLLABLE
TOPIC_MODEL_DIR = MODEL_DIR / "topic_classification"
TOPIC_DATASET_READY = True   # ✅ Dataset đã sẵn sàng (JSON đã được tạo bởi build_topic_dataset.py)
TOPIC_CSV_FILES = [
    DATA_DIR / "train_ml.csv",
    DATA_DIR / "alobacsi_processed.csv",
    DATA_DIR / "ml_training_data_tamanh.csv",
]
# ✅ JSON đã xử lý sẵn (preprocess_topic.py) — TopicDataLoader đọc train/val/test + label map
TOPIC_TRAIN_JSON = DATA_DIR / "topic_train.json"
TOPIC_VAL_JSON = DATA_DIR / "topic_val.json"
TOPIC_TEST_JSON = DATA_DIR / "topic_test.json"
TOPIC_LABEL_MAP_JSON = DATA_DIR / "topic_label_map.json"


# ============================================================
# 📍 TRẠM 2C: INTENT CLASSIFICATION
# ============================================================
INTENT_MODEL_NAME = VIHEALTHBERT_SYLLABLE
INTENT_MODEL_DIR = MODEL_DIR / "intent_classification"
INTENT_LABELS = ["Diagnosis", "Treatment", "Severity", "Cause"]
INTENT_NUM_LABELS = len(INTENT_LABELS)
INTENT_LABEL2ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
INTENT_ID2LABEL = {idx: label for idx, label in enumerate(INTENT_LABELS)}

# ============================================================
# 🏋️ TRAINING HYPERPARAMETERS
# ============================================================
TRAIN_CONFIG = {
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "logging_steps": 50,
    "seed": 42,
    "fp16": True,
}

# ============================================================
# 🌐 API CONFIG
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_INPUT_LENGTH = 512
