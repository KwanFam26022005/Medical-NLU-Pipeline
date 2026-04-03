"""
conftest.py — Shared fixtures cho toàn bộ test suite Medical NLU Pipeline.

Cung cấp:
  - Mock models (không cần GPU/HF Hub)
  - Sample data thực tế cho từng trạm
  - FastAPI TestClient
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Thêm project root vào sys.path để import models, config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 🔧 SAMPLE DATA — Input/Output thực tế cho mỗi trạm
# ============================================================

@pytest.fixture
def sample_raw_text():
    """Câu hỏi y tế thô — có viết tắt, chưa qua NLU."""
    return "bs ơi e bị đau dd có cần mổ kt k ạ?"


@pytest.fixture
def sample_clean_text():
    """Câu hỏi đã qua Trạm 1 (giải viết tắt)."""
    return "bác sĩ ơi em bị đau dạ dày có cần mổ kích thước không ạ?"


@pytest.fixture
def sample_ner_bio_output():
    """Output thực tế từ MedicalNER.predict() — format BIO tokens."""
    return [
        {"word": "bác_sĩ", "label": "O"},
        {"word": "ơi", "label": "O"},
        {"word": "em", "label": "O"},
        {"word": "bị", "label": "O"},
        {"word": "đau", "label": "B-SYMPTOM_AND_DISEASE"},
        {"word": "dạ_dày", "label": "I-SYMPTOM_AND_DISEASE"},
        {"word": "có", "label": "O"},
        {"word": "cần", "label": "O"},
        {"word": "mổ", "label": "B-MEDICAL_PROCEDURE"},
        {"word": "nội_soi", "label": "I-MEDICAL_PROCEDURE"},
        {"word": "không", "label": "O"},
    ]


@pytest.fixture
def sample_ner_aggregated_entities():
    """Expected output sau BIO aggregation — List[str]."""
    return ["đau dạ dày", "mổ nội soi"]


@pytest.fixture
def sample_topic_output():
    """Output chuẩn từ TopicClassifier.predict()."""
    return {
        "topic": "gastroenterology",
        "confidence": 0.9134,
        "is_reliable": True,
    }


@pytest.fixture
def sample_intent_raw_output():
    """Output thực tế từ IntentClassifier.predict() — List[Dict]."""
    return [
        {"intent": "Treatment", "score": 0.87},
        {"intent": "Diagnosis", "score": 0.62},
    ]


@pytest.fixture
def sample_intent_normalized_output():
    """Expected output sau intent normalization — Dict chuẩn cho main.py."""
    return {
        "intents": ["Treatment", "Diagnosis"],
        "scores": {"Treatment": 0.87, "Diagnosis": 0.62},
        "primary_intent": "Treatment",
    }


@pytest.fixture
def valid_ner_labels():
    """Tập nhãn NER hợp lệ theo BIO schema."""
    return [
        "O",
        "B-SYMPTOM_AND_DISEASE",
        "I-SYMPTOM_AND_DISEASE",
        "B-MEDICAL_PROCEDURE",
        "I-MEDICAL_PROCEDURE",
        "B-MEDICINE",
        "I-MEDICINE",
    ]


@pytest.fixture
def valid_intent_labels():
    """Tập nhãn Intent hợp lệ."""
    return ["Diagnosis", "Treatment", "Severity", "Cause"]


@pytest.fixture
def sample_acronym_dict():
    """Dictionary viết tắt y tế mẫu cho test Trạm 1."""
    return {
        "dd": ["dạ dày", "da dày", "đường dẫn"],
        "bs": ["bác sĩ"],
        "kt": ["kích thước", "kiểm tra", "kỹ thuật"],
        "bn": ["bệnh nhân", "bình nguyên"],
    }


# ============================================================
# 🧪 MOCK OBJECTS — Không cần GPU / HuggingFace
# ============================================================

@pytest.fixture
def mock_acronym_resolver(sample_acronym_dict):
    """Mock AcronymCrossEncoder — trả về clean text cố định."""
    resolver = MagicMock()
    resolver.acronym_dict = sample_acronym_dict
    resolver.predict.return_value = "bác sĩ ơi em bị đau dạ dày có cần mổ kích thước không ạ?"
    resolver._is_loaded = True
    return resolver


@pytest.fixture
def mock_medical_ner(sample_ner_bio_output):
    """Mock MedicalNER — trả về BIO output cố định."""
    ner = MagicMock()
    ner.predict.return_value = sample_ner_bio_output
    ner.load_model.return_value = None
    ner.id2label = {
        0: "O",
        1: "B-SYMPTOM_AND_DISEASE", 2: "I-SYMPTOM_AND_DISEASE",
        3: "B-MEDICAL_PROCEDURE", 4: "I-MEDICAL_PROCEDURE",
        5: "B-MEDICINE", 6: "I-MEDICINE",
    }
    return ner


@pytest.fixture
def mock_topic_classifier(sample_topic_output):
    """Mock TopicClassifier — trả về topic cố định."""
    classifier = MagicMock()
    classifier.predict.return_value = sample_topic_output
    classifier._is_loaded = True
    classifier._is_ready = True
    return classifier


@pytest.fixture
def mock_intent_classifier(sample_intent_raw_output):
    """Mock IntentClassifier — trả về intent list cố định."""
    classifier = MagicMock()
    classifier.predict.return_value = sample_intent_raw_output
    classifier.load_model.return_value = None
    classifier.thresholds = {
        "Diagnosis": 0.5,
        "Treatment": 0.5,
        "Severity": 0.5,
        "Cause": 0.5,
    }
    classifier.id2label = {0: "Diagnosis", 1: "Treatment", 2: "Severity", 3: "Cause"}
    return classifier
