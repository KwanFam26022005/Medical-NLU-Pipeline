"""
test_medical_ner.py — Tests cho Trạm 2A: MedicalNER

Bao gồm:
  - Unit test: predict() output format (List[Dict])
  - Integration test: BIO aggregation adapter (List[Dict] → List[str])
  - Edge case: text rỗng, text không có entity, text toàn entity
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 📋 UNIT TESTS — NER predict() output format
# ============================================================

class TestMedicalNERUnit:
    """Unit tests cho MedicalNER.predict() — raw BIO output."""

    def test_predict_returns_list(self, mock_medical_ner, sample_clean_text):
        """predict() PHẢI trả về List."""
        result = mock_medical_ner.predict(sample_clean_text)
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_predict_returns_list_of_dicts(self, mock_medical_ner, sample_clean_text):
        """Mỗi phần tử trong list PHẢI là Dict với keys 'word' và 'label'."""
        result = mock_medical_ner.predict(sample_clean_text)
        print(f"\n  📥 Input text: '{sample_clean_text}'")
        print(f"  📤 NER raw output ({len(result)} tokens):")
        for i, item in enumerate(result):
            label_icon = "🟢" if item["label"] == "O" else "🟡" if item["label"].startswith("B-") else "🔵"
            print(f"    {label_icon} [{i}] word='{item['word']}' label='{item['label']}'")
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"
            assert "word" in item, f"Missing key 'word' in {item}"
            assert "label" in item, f"Missing key 'label' in {item}"

    def test_labels_are_valid_bio(self, mock_medical_ner, sample_clean_text, valid_ner_labels):
        """Tất cả labels phải thuộc BIO schema hợp lệ."""
        result = mock_medical_ner.predict(sample_clean_text)
        for item in result:
            assert item["label"] in valid_ner_labels, (
                f"Label '{item['label']}' không hợp lệ. Cho phép: {valid_ner_labels}"
            )

    def test_bio_sequence_validity(self, sample_ner_bio_output):
        """BIO sequence: I-X không được xuất hiện nếu trước đó không có B-X."""
        prev_type = None
        for token in sample_ner_bio_output:
            label = token["label"]
            if label.startswith("I-"):
                current_type = label[2:]
                assert prev_type == current_type, (
                    f"BIO violation: I-{current_type} xuất hiện sau "
                    f"{'B-' + prev_type if prev_type else 'O'}"
                )
            elif label.startswith("B-"):
                prev_type = label[2:]
            else:
                prev_type = None

    def test_words_are_nonempty_strings(self, sample_ner_bio_output):
        """Mỗi 'word' phải là string không rỗng."""
        for item in sample_ner_bio_output:
            assert isinstance(item["word"], str), f"Word phải là str"
            assert len(item["word"]) > 0, f"Word không được rỗng"

    def test_load_model_wrapper_exists(self, mock_medical_ner):
        """load_model() wrapper phải tồn tại (fix B3)."""
        mock_medical_ner.load_model()  # Không raise AttributeError


# ============================================================
# 🔗 INTEGRATION TESTS — BIO Aggregation Adapter
# ============================================================

def aggregate_bio_entities(ner_output):
    """
    Replica chính xác của BIO aggregation logic trong main.py run_ner().
    Dùng để test adapter tách biệt khỏi FastAPI.
    """
    entities = []
    current_entity = []
    for token in ner_output:
        label = token.get("label", "O")
        word = token.get("word", "").replace("_", " ")
        if label.startswith("B-"):
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = [word]
        elif label.startswith("I-") and current_entity:
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    if current_entity:
        entities.append(" ".join(current_entity))
    return entities


class TestBIOAggregationAdapter:
    """Integration tests: NER BIO output → entity strings cho RAG."""

    def test_aggregate_produces_correct_entities(
        self, sample_ner_bio_output, sample_ner_aggregated_entities
    ):
        """Aggregate BIO → đúng danh sách entity strings."""
        result = aggregate_bio_entities(sample_ner_bio_output)
        print(f"\n  📥 BIO Input ({len(sample_ner_bio_output)} tokens):")
        for t in sample_ner_bio_output:
            if t['label'] != 'O':
                print(f"    🏷️  word='{t['word']}' label='{t['label']}'")
        print(f"  ⬇️  Aggregation...")
        print(f"  📤 Output entities: {result}")
        print(f"  ✅ Expected:        {sample_ner_aggregated_entities}")
        assert result == sample_ner_aggregated_entities, (
            f"Expected {sample_ner_aggregated_entities}, got {result}"
        )

    def test_aggregate_output_is_list_of_strings(self, sample_ner_bio_output):
        """Output phải là List[str] — đúng schema NLUResult.entities."""
        result = aggregate_bio_entities(sample_ner_bio_output)
        assert isinstance(result, list)
        for entity in result:
            assert isinstance(entity, str), f"Entity phải là str, got {type(entity)}"

    def test_aggregate_underscore_removed(self):
        """Underscore trong word-segmented tokens phải được thay bằng space."""
        bio_output = [
            {"word": "viêm_phổi", "label": "B-SYMPTOM_AND_DISEASE"},
        ]
        result = aggregate_bio_entities(bio_output)
        print(f"\n  📥 Input:  word='viêm_phổi' (có underscore)")
        print(f"  📤 Output: {result}")
        print(f"  ✅ Underscore '_' → space ' '")
        assert result == ["viêm phổi"], f"Underscore phải được thay bằng space, got {result}"

    def test_aggregate_multiple_entities(self):
        """Nhiều entities riêng biệt phải tách đúng."""
        bio_output = [
            {"word": "đau", "label": "B-SYMPTOM_AND_DISEASE"},
            {"word": "đầu", "label": "I-SYMPTOM_AND_DISEASE"},
            {"word": "uống", "label": "O"},
            {"word": "paracetamol", "label": "B-MEDICINE"},
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == ["đau đầu", "paracetamol"]

    def test_aggregate_single_b_token(self):
        """Entity chỉ có 1 B-token (không có I-) vẫn được capture."""
        bio_output = [
            {"word": "sốt", "label": "B-SYMPTOM_AND_DISEASE"},
            {"word": "cao", "label": "O"},
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == ["sốt"]

    def test_aggregate_consecutive_b_tokens(self):
        """Hai B-tokens liên tiếp = 2 entities riêng biệt."""
        bio_output = [
            {"word": "sốt", "label": "B-SYMPTOM_AND_DISEASE"},
            {"word": "mổ", "label": "B-MEDICAL_PROCEDURE"},
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == ["sốt", "mổ"]

    def test_aggregate_entity_at_end(self):
        """Entity ở cuối chuỗi (không có O token sau) vẫn được flush."""
        bio_output = [
            {"word": "uống", "label": "O"},
            {"word": "aspirin", "label": "B-MEDICINE"},
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == ["aspirin"]


# ============================================================
# ⚡ EDGE CASE TESTS
# ============================================================

class TestMedicalNEREdgeCases:
    """Edge cases cho NER."""

    def test_empty_input_returns_empty_list(self):
        """Input rỗng → entities rỗng."""
        result = aggregate_bio_entities([])
        assert result == []

    def test_all_o_labels_returns_empty(self):
        """Tất cả O → không có entity nào."""
        bio_output = [
            {"word": "tôi", "label": "O"},
            {"word": "khỏe", "label": "O"},
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == []

    def test_all_entities_no_o(self):
        """Tất cả tokens đều là entity."""
        bio_output = [
            {"word": "đau", "label": "B-SYMPTOM_AND_DISEASE"},
            {"word": "dạ_dày", "label": "I-SYMPTOM_AND_DISEASE"},
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == ["đau dạ dày"]

    def test_orphan_i_token_ignored(self):
        """I-token đứng một mình (trước đó không có B-) bị bỏ qua."""
        bio_output = [
            {"word": "dạ_dày", "label": "I-SYMPTOM_AND_DISEASE"},
            {"word": "mệt", "label": "O"},
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == [], "Orphan I-token phải bị bỏ qua"

    def test_missing_word_key_handled(self):
        """Token thiếu key 'word' → không crash (dùng default '')."""
        bio_output = [
            {"label": "B-SYMPTOM_AND_DISEASE"},  # missing 'word'
            {"word": "đau", "label": "O"},
        ]
        result = aggregate_bio_entities(bio_output)
        # Không crash là đủ
        assert isinstance(result, list)

    def test_missing_label_key_handled(self):
        """Token thiếu key 'label' → mặc định 'O' (không crash)."""
        bio_output = [
            {"word": "đau"},  # missing 'label'
        ]
        result = aggregate_bio_entities(bio_output)
        assert result == [], "Missing label phải default là O → empty"
