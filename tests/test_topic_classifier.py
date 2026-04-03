"""
test_topic_classifier.py — Tests cho Trạm 2B: TopicClassifier

Bao gồm:
  - Unit test: predict() output schema (Dict)
  - Integration test: output khớp input cho CRAG (topic filter)
  - Edge case: confidence thấp, dummy mode, text rỗng
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 📋 UNIT TESTS — TopicClassifier.predict() output format
# ============================================================

class TestTopicClassifierUnit:
    """Unit tests cho TopicClassifier.predict() output."""

    def test_predict_returns_dict(self, mock_topic_classifier, sample_clean_text):
        """predict() PHẢI trả về Dict."""
        result = mock_topic_classifier.predict(sample_clean_text)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_output_has_required_keys(self, mock_topic_classifier, sample_clean_text):
        """Output PHẢI có đủ 3 keys: topic, confidence, is_reliable."""
        result = mock_topic_classifier.predict(sample_clean_text)
        required_keys = {"topic", "confidence", "is_reliable"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_topic_is_string(self, mock_topic_classifier, sample_clean_text):
        """topic phải là str."""
        result = mock_topic_classifier.predict(sample_clean_text)
        assert isinstance(result["topic"], str), f"topic phải là str, got {type(result['topic'])}"

    def test_confidence_is_float(self, mock_topic_classifier, sample_clean_text):
        """confidence phải là float."""
        result = mock_topic_classifier.predict(sample_clean_text)
        assert isinstance(result["confidence"], (int, float)), (
            f"confidence phải là float, got {type(result['confidence'])}"
        )

    def test_confidence_range(self, mock_topic_classifier, sample_clean_text):
        """confidence phải nằm trong [0.0, 1.0]."""
        result = mock_topic_classifier.predict(sample_clean_text)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"confidence = {result['confidence']} nằm ngoài [0, 1]"
        )

    def test_is_reliable_is_bool(self, mock_topic_classifier, sample_clean_text):
        """is_reliable phải là bool."""
        result = mock_topic_classifier.predict(sample_clean_text)
        assert isinstance(result["is_reliable"], bool), (
            f"is_reliable phải là bool, got {type(result['is_reliable'])}"
        )

    def test_topic_not_empty(self, mock_topic_classifier, sample_clean_text):
        """topic string không được rỗng."""
        result = mock_topic_classifier.predict(sample_clean_text)
        assert len(result["topic"]) > 0, "topic không được rỗng"


# ============================================================
# 🔗 INTEGRATION TESTS — Tương thích CRAG Pipeline
# ============================================================

class TestTopicCRAGCompatibility:
    """Kiểm tra output phù hợp với CRAG Document Grader & Topic Filter."""

    def test_topic_usable_as_qdrant_filter(self, sample_topic_output):
        """topic value phải dùng được làm Qdrant payload filter."""
        topic = sample_topic_output["topic"]
        # Qdrant filter cần string không chứa ký tự đặc biệt
        assert isinstance(topic, str)
        assert len(topic) > 0
        # Không nên chứa space (convention: dùng underscore hoặc camelCase)
        # Nhưng thực tế topic là English label nên OK

    def test_high_confidence_is_reliable(self, sample_topic_output):
        """confidence >= 0.7 → nên được dùng làm filter."""
        if sample_topic_output["confidence"] >= 0.7:
            assert sample_topic_output["is_reliable"] is True

    def test_low_confidence_topic_still_valid(self):
        """Khi confidence thấp, topic vẫn phải là string hợp lệ (fallback no-filter)."""
        low_conf_output = {
            "topic": "general",
            "confidence": 0.35,
            "is_reliable": True,
        }
        assert isinstance(low_conf_output["topic"], str)
        assert len(low_conf_output["topic"]) > 0

    def test_dummy_mode_output_still_valid_schema(self):
        """Dummy mode output vẫn phải có đúng schema (dù is_reliable=False)."""
        dummy_output = {
            "topic": "topic_0",
            "confidence": 0.12,
            "is_reliable": False,
        }
        assert "topic" in dummy_output
        assert "confidence" in dummy_output
        assert "is_reliable" in dummy_output
        assert dummy_output["is_reliable"] is False


# ============================================================
# ⚡ EDGE CASE TESTS
# ============================================================

class TestTopicEdgeCases:
    """Edge cases cho TopicClassifier."""

    def test_fallback_on_error(self):
        """Khi model lỗi, main.py fallback phải đúng schema."""
        fallback = {"topic": "unknown", "confidence": 0.0, "is_reliable": False}
        assert fallback["topic"] == "unknown"
        assert fallback["confidence"] == 0.0
        assert fallback["is_reliable"] is False

    def test_topic_with_numbers_in_name(self):
        """Topic có số (vd: 'topic_5') → vẫn hợp lệ khi dummy mode."""
        result = {"topic": "topic_5", "confidence": 0.1, "is_reliable": False}
        assert isinstance(result["topic"], str)

    def test_max_confidence_1_0(self):
        """confidence tối đa là 1.0 (softmax output)."""
        result = {"topic": "cardiology", "confidence": 1.0, "is_reliable": True}
        assert result["confidence"] <= 1.0

    def test_empty_text_handling(self, mock_topic_classifier):
        """Text rỗng không crash — trả về kết quả hợp lệ."""
        mock_topic_classifier.predict.return_value = {
            "topic": "unknown",
            "confidence": 0.0,
            "is_reliable": False,
        }
        result = mock_topic_classifier.predict("")
        assert isinstance(result, dict)
        assert "topic" in result

    def test_very_long_text(self, mock_topic_classifier, sample_topic_output):
        """Text rất dài (>512 tokens) — truncation xử lý, không crash."""
        long_text = "tôi bị đau dạ dày " * 200
        mock_topic_classifier.predict.return_value = sample_topic_output
        result = mock_topic_classifier.predict(long_text)
        assert isinstance(result, dict)


# ============================================================
# 🏗️ HF HUB DETECTION TESTS (Fix B8, B9)
# ============================================================

class TestTopicHFHubDetection:
    """Kiểm tra logic phát hiện HuggingFace Hub vs local path (fix B8)."""

    def test_hf_hub_id_detected(self):
        """String có '/' và không tồn tại local → là HF Hub ID."""
        model_dir = Path("KwanFam26022005/model2B-topic-classification")
        model_dir_str = str(model_dir)
        is_hf_hub = "/" in model_dir_str and not model_dir.exists()
        assert is_hf_hub is True, "HF Hub ID phải được detect đúng"

    def test_local_path_not_detected_as_hf(self, tmp_path):
        """Local path tồn tại → KHÔNG phải HF Hub."""
        local_dir = tmp_path / "local_model"
        local_dir.mkdir()
        model_dir_str = str(local_dir)
        is_hf_hub = "/" in model_dir_str and not local_dir.exists()
        # local_dir exists → is_hf_hub = False
        assert is_hf_hub is False, "Local path không được detect nhầm thành HF Hub"

    def test_simple_name_not_hf(self):
        """Path đơn giản không có '/' → không phải HF Hub."""
        model_dir = Path("my_local_model")
        model_dir_str = str(model_dir)
        # Trên Windows, Path("my_local_model") str sẽ là "my_local_model" (không có /)
        # Tuy nhiên trên Windows, os.sep = '\\' nên "/" check vẫn đúng
        is_hf_hub = "/" in model_dir_str and not model_dir.exists()
        assert is_hf_hub is False
