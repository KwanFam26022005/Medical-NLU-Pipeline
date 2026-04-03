"""
test_intent_classifier.py — Tests cho Trạm 2C: IntentClassifier

Bao gồm:
  - Unit test: predict() output format (List[Dict])
  - Integration test: Intent normalization adapter (List[Dict] → Dict chuẩn)
  - Edge case: không intent nào vượt ngưỡng, text rỗng, multi-label
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 📋 UNIT TESTS — IntentClassifier.predict() output format
# ============================================================

class TestIntentClassifierUnit:
    """Unit tests cho IntentClassifier.predict() — raw output."""

    def test_predict_returns_list(self, mock_intent_classifier, sample_clean_text):
        """predict() PHẢI trả về List."""
        result = mock_intent_classifier.predict(sample_clean_text)
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_predict_returns_list_of_dicts(self, mock_intent_classifier, sample_clean_text):
        """Mỗi phần tử PHẢI là Dict với keys 'intent' và 'score'."""
        result = mock_intent_classifier.predict(sample_clean_text)
        for item in result:
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"
            assert "intent" in item, f"Missing key 'intent' in {item}"
            assert "score" in item, f"Missing key 'score' in {item}"

    def test_intent_labels_are_valid(self, mock_intent_classifier, sample_clean_text, valid_intent_labels):
        """Tất cả intent labels phải thuộc tập hợp lệ."""
        result = mock_intent_classifier.predict(sample_clean_text)
        for item in result:
            assert item["intent"] in valid_intent_labels, (
                f"Intent '{item['intent']}' không hợp lệ. Cho phép: {valid_intent_labels}"
            )

    def test_scores_are_float_in_range(self, mock_intent_classifier, sample_clean_text):
        """Score phải là float trong [0.0, 1.0] (sigmoid output)."""
        result = mock_intent_classifier.predict(sample_clean_text)
        for item in result:
            score = item["score"]
            assert isinstance(score, (int, float)), f"Score phải là float, got {type(score)}"
            assert 0.0 <= score <= 1.0, f"Score {score} ngoài phạm vi [0, 1]"

    def test_at_least_one_intent_returned(self, mock_intent_classifier, sample_clean_text):
        """Luôn trả về ít nhất 1 intent (do fallback logic)."""
        result = mock_intent_classifier.predict(sample_clean_text)
        assert len(result) >= 1, "Phải có ít nhất 1 intent (fallback)"

    def test_load_model_wrapper_exists(self, mock_intent_classifier):
        """load_model() wrapper phải tồn tại (fix B4)."""
        mock_intent_classifier.load_model()  # Không raise AttributeError


# ============================================================
# 🔗 INTEGRATION TESTS — Intent Normalization Adapter
# ============================================================

def normalize_intent_output(raw_intents):
    """
    Replica chính xác của normalization logic trong main.py run_intent().
    Dùng để test adapter tách biệt khỏi FastAPI.
    """
    if not raw_intents:
        return {"intents": [], "scores": {}, "primary_intent": "unknown"}

    sorted_intents = sorted(raw_intents, key=lambda x: x["score"], reverse=True)
    primary = sorted_intents[0]["intent"]
    scores = {item["intent"]: round(item["score"], 4) for item in sorted_intents}
    return {
        "intents": [item["intent"] for item in sorted_intents],
        "scores": scores,
        "primary_intent": primary,
    }


class TestIntentNormalizationAdapter:
    """Integration tests: raw intent output → normalized Dict cho CRAG."""

    def test_normalize_correct_primary_intent(self, sample_intent_raw_output):
        """primary_intent = intent có score cao nhất."""
        result = normalize_intent_output(sample_intent_raw_output)
        assert result["primary_intent"] == "Treatment", (
            f"Expected Treatment (0.87), got {result['primary_intent']}"
        )

    def test_normalize_intents_sorted_by_score(self, sample_intent_raw_output):
        """intents list phải sorted descending theo score."""
        result = normalize_intent_output(sample_intent_raw_output)
        assert result["intents"] == ["Treatment", "Diagnosis"], (
            f"Expected ['Treatment', 'Diagnosis'], got {result['intents']}"
        )

    def test_normalize_scores_dict_format(self, sample_intent_raw_output):
        """scores phải là Dict[str, float]."""
        result = normalize_intent_output(sample_intent_raw_output)
        assert isinstance(result["scores"], dict)
        for intent, score in result["scores"].items():
            assert isinstance(intent, str)
            assert isinstance(score, float)

    def test_normalize_output_has_required_keys(self, sample_intent_raw_output):
        """Output PHẢI có đúng 3 keys: intents, scores, primary_intent."""
        result = normalize_intent_output(sample_intent_raw_output)
        required_keys = {"intents", "scores", "primary_intent"}
        assert required_keys == result.keys(), f"Keys mismatch: {result.keys()}"

    def test_normalize_single_intent(self):
        """Chỉ 1 intent → primary = chính nó."""
        raw = [{"intent": "Severity", "score": 0.92}]
        result = normalize_intent_output(raw)
        assert result["primary_intent"] == "Severity"
        assert result["intents"] == ["Severity"]
        assert result["scores"] == {"Severity": 0.92}

    def test_normalize_all_four_intents(self):
        """Tất cả 4 intents vượt ngưỡng — multi-label scenario."""
        raw = [
            {"intent": "Diagnosis", "score": 0.88},
            {"intent": "Treatment", "score": 0.72},
            {"intent": "Severity", "score": 0.65},
            {"intent": "Cause", "score": 0.51},
        ]
        result = normalize_intent_output(raw)
        assert result["primary_intent"] == "Diagnosis"
        assert len(result["intents"]) == 4
        # Verify sorted order
        scores_list = [result["scores"][i] for i in result["intents"]]
        assert scores_list == sorted(scores_list, reverse=True)


# ============================================================
# ⚡ EDGE CASE TESTS
# ============================================================

class TestIntentEdgeCases:
    """Edge cases cho IntentClassifier."""

    def test_empty_list_fallback(self):
        """Input rỗng → fallback output chuẩn."""
        result = normalize_intent_output([])
        assert result == {"intents": [], "scores": {}, "primary_intent": "unknown"}

    def test_equal_scores(self):
        """Hai intents có cùng score → primary là cái đầu tiên sau sort (stable)."""
        raw = [
            {"intent": "Diagnosis", "score": 0.75},
            {"intent": "Treatment", "score": 0.75},
        ]
        result = normalize_intent_output(raw)
        assert result["primary_intent"] in ["Diagnosis", "Treatment"]
        assert len(result["intents"]) == 2

    def test_very_low_scores(self):
        """Score rất thấp (fallback triggered) → vẫn trả kết quả hợp lệ."""
        raw = [{"intent": "Cause", "score": 0.12}]
        result = normalize_intent_output(raw)
        assert result["primary_intent"] == "Cause"
        assert result["scores"]["Cause"] == 0.12

    def test_score_rounding_precision(self):
        """Scores được round đến 4 decimal places."""
        raw = [{"intent": "Treatment", "score": 0.876543}]
        result = normalize_intent_output(raw)
        assert result["scores"]["Treatment"] == 0.8765

    def test_primary_intent_usable_for_crag_prompt(self):
        """primary_intent phải map được vào INTENT_PROMPTS dict (CRAG)."""
        intent_prompts = {
            "Diagnosis": "Tập trung vào triệu chứng...",
            "Treatment": "Tập trung vào phương pháp điều trị...",
            "Severity": "Tập trung đánh giá mức độ nguy hiểm...",
            "Cause": "Tập trung vào nguyên nhân gây bệnh...",
        }
        raw = [{"intent": "Treatment", "score": 0.9}]
        result = normalize_intent_output(raw)
        assert result["primary_intent"] in intent_prompts, (
            f"primary_intent '{result['primary_intent']}' không có trong INTENT_PROMPTS"
        )
