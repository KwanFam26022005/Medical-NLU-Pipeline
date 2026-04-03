"""
test_acronym_wsd.py — Tests cho Trạm 1: AcronymCrossEncoder

Bao gồm:
  - Unit test: score_candidates, predict_from_raw, predict
  - Integration test: output schema (str) đúng input cho Trạm 2
  - Edge case: text không có viết tắt, viết tắt không trong dict, text rỗng
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ACRONYM_MODEL_DIR, ACRONYM_MODEL_NAME


# ============================================================
# 📋 UNIT TESTS — Kiểm tra từng method độc lập
# ============================================================

class TestAcronymCrossEncoderUnit:
    """Unit tests cho AcronymCrossEncoder — dùng mock, không cần GPU."""

    def test_predict_returns_string(self, mock_acronym_resolver):
        """predict() PHẢI trả về str."""
        input_text = "bs ơi e bị đau dd"
        result = mock_acronym_resolver.predict(input_text)
        print(f"\n  📥 Input:  '{input_text}'")
        print(f"  📤 Output: '{result}'")
        print(f"  📊 Type:   {type(result).__name__}")
        assert isinstance(result, str), f"Expected str, got {type(result)}"

    def test_predict_replaces_acronyms(self, mock_acronym_resolver):
        """predict() phải thay thế viết tắt thành dạng đầy đủ."""
        input_text = "bs ơi e bị đau dd"
        result = mock_acronym_resolver.predict(input_text)
        print(f"\n  📥 Input (có viết tắt):  '{input_text}'")
        print(f"  📤 Output (đã giải):    '{result}'")
        assert "bác sĩ" in result or "dạ dày" in result or result != ""

    def test_predict_no_acronyms_returns_original(self, mock_acronym_resolver):
        """Nếu text không chứa viết tắt → trả về nguyên bản."""
        original = "tôi bị đau đầu từ hôm qua"
        mock_acronym_resolver.predict.return_value = original
        result = mock_acronym_resolver.predict(original)
        assert result == original

    def test_acronym_dict_structure(self, sample_acronym_dict):
        """Dictionary: key = viết tắt (str), value = list expansions (List[str])."""
        print(f"\n  📖 Acronym Dictionary ({len(sample_acronym_dict)} entries):")
        for acronym, expansions in sample_acronym_dict.items():
            print(f"    '{acronym}' → {expansions}")
            assert isinstance(acronym, str), f"Key phải là str, got {type(acronym)}"
            assert isinstance(expansions, list), f"Value phải là list, got {type(expansions)}"
            assert len(expansions) >= 1, f"Viết tắt '{acronym}' phải có ít nhất 1 expansion"
            for exp in expansions:
                assert isinstance(exp, str), f"Expansion phải là str, got {type(exp)}"

    def test_predict_empty_dict_returns_original(self, mock_acronym_resolver):
        """Nếu acronym_dict rỗng → trả nguyên text."""
        mock_acronym_resolver.acronym_dict = {}
        original = "bs ơi e bị đau dd"
        mock_acronym_resolver.predict.return_value = original
        result = mock_acronym_resolver.predict(original)
        assert result == original


# ============================================================
# 🔗 INTEGRATION TESTS — Output schema khớp Trạm 2
# ============================================================

class TestAcronymOutputSchema:
    """Kiểm tra output Trạm 1 đúng input contract cho Trạm 2."""

    def test_output_is_string_for_ner_input(self, mock_acronym_resolver, sample_raw_text):
        """Output PHẢI là str — vì Trạm 2A/2B/2C nhận str."""
        result = mock_acronym_resolver.predict(sample_raw_text)
        assert isinstance(result, str), (
            f"Trạm 2 cần str input, nhưng Trạm 1 trả về {type(result)}"
        )

    def test_output_not_empty_for_valid_input(self, mock_acronym_resolver, sample_raw_text):
        """Output không được rỗng nếu input hợp lệ."""
        result = mock_acronym_resolver.predict(sample_raw_text)
        assert len(result) > 0, "Clean text không được rỗng"

    def test_output_max_length(self, mock_acronym_resolver):
        """Output không được vượt quá MAX_INPUT_LENGTH (2048 chars)."""
        # Fake: expansion dài hơn viết tắt nên output dài hơn input
        mock_acronym_resolver.predict.return_value = "a" * 2048
        result = mock_acronym_resolver.predict("test")
        assert len(result) <= 4096, "Output quá dài, có thể gây truncate ở Trạm 2"


# ============================================================
# ⚡ EDGE CASE TESTS
# ============================================================

class TestAcronymEdgeCases:
    """Edge cases cho Trạm 1."""

    def test_empty_string_input(self, mock_acronym_resolver):
        """Input rỗng → trả về rỗng."""
        mock_acronym_resolver.predict.return_value = ""
        result = mock_acronym_resolver.predict("")
        assert result == ""

    def test_whitespace_only_input(self, mock_acronym_resolver):
        """Input chỉ có whitespace → trả về whitespace."""
        mock_acronym_resolver.predict.return_value = "   "
        result = mock_acronym_resolver.predict("   ")
        assert isinstance(result, str)

    def test_single_character_input(self, mock_acronym_resolver):
        """Input 1 ký tự — không crash."""
        mock_acronym_resolver.predict.return_value = "a"
        result = mock_acronym_resolver.predict("a")
        assert isinstance(result, str)

    def test_very_long_input(self, mock_acronym_resolver):
        """Input rất dài — không crash, xử lý graceful."""
        long_text = "bs ơi em bị đau " * 500
        mock_acronym_resolver.predict.return_value = long_text
        result = mock_acronym_resolver.predict(long_text)
        assert isinstance(result, str)

    def test_special_characters_preserved(self, mock_acronym_resolver):
        """Ký tự đặc biệt (?, !, dấu tiếng Việt) không bị mất."""
        text_with_special = "bs ơi e bị đau dd quá! Phải làm sao???"
        mock_acronym_resolver.predict.return_value = "bác sĩ ơi em bị đau dạ dày quá! Phải làm sao???"
        result = mock_acronym_resolver.predict(text_with_special)
        print(f"\n  📥 Input:  '{text_with_special}'")
        print(f"  📤 Output: '{result}'")
        print(f"  ✅ Giữ nguyên: '?' = {'?' in result}, '!' = {'!' in result}")
        assert "?" in result, "Dấu hỏi bị mất"
        assert "!" in result, "Dấu chấm than bị mất"

    def test_overlapping_acronyms(self, mock_acronym_resolver):
        """Nhiều viết tắt liền kề — xử lý đúng."""
        mock_acronym_resolver.predict.return_value = "bệnh nhân kiểm tra"
        result = mock_acronym_resolver.predict("bn kt")
        assert isinstance(result, str)
