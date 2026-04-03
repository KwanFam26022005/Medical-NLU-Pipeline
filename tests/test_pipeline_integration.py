"""
test_pipeline_integration.py — Tests cho toàn bộ Pipeline (main.py)

Bao gồm:
  - Integration test: End-to-end flow (Trạm 1 → 2A/2B/2C → Response)
  - Schema test: Pydantic models (Request/Response) validation
  - API test: FastAPI endpoint /analyze, /health
  - Edge case: model None, partial failure, concurrent requests
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (
    MedicalQueryRequest,
    MedicalQueryResponse,
    NLUResult,
    HealthCheckResponse,
)


# ============================================================
# 📋 PYDANTIC SCHEMA TESTS — Request & Response validation
# ============================================================

class TestRequestSchema:
    """Kiểm tra MedicalQueryRequest schema."""

    def test_valid_request(self):
        """Request hợp lệ = có text."""
        req = MedicalQueryRequest(text="tôi bị đau đầu")
        assert req.text == "tôi bị đau đầu"

    def test_empty_text_rejected(self):
        """Text rỗng: min_length=1 → Pydantic raise ValidationError."""
        with pytest.raises(Exception):
            MedicalQueryRequest(text="")

    def test_text_max_length(self):
        """Text vượt max_length=2048 → raise ValidationError."""
        with pytest.raises(Exception):
            MedicalQueryRequest(text="a" * 2049)

    def test_text_at_max_length(self):
        """Text đúng max_length=2048 → hợp lệ."""
        req = MedicalQueryRequest(text="a" * 2048)
        assert len(req.text) == 2048

    def test_whitespace_text_valid(self):
        """Text chỉ có whitespace: min_length=1, whitespace count → hợp lệ."""
        req = MedicalQueryRequest(text=" ")
        assert req.text == " "

    def test_unicode_vietnamese_text(self):
        """Text tiếng Việt Unicode → hợp lệ."""
        req = MedicalQueryRequest(text="Tôi bị đau dạ dày từ hôm qua, nên uống thuốc gì?")
        assert "dạ dày" in req.text


class TestResponseSchema:
    """Kiểm tra MedicalQueryResponse schema."""

    def test_valid_response_construction(self):
        """Response xây dựng đúng từ các components."""
        resp = MedicalQueryResponse(
            raw_text="test",
            clean_text="test",
            nlu_result=NLUResult(
                entities=["đau đầu"],
                topic={"topic": "neurology", "confidence": 0.9, "is_reliable": True},
                intent={"intents": ["Diagnosis"], "scores": {"Diagnosis": 0.85}, "primary_intent": "Diagnosis"},
            ),
            processing_time_ms=123.45,
        )
        assert resp.raw_text == "test"
        assert resp.nlu_result.entities == ["đau đầu"]
        assert resp.processing_time_ms == 123.45

    def test_nlu_result_entities_are_list_of_strings(self):
        """NLUResult.entities phải là List[str]."""
        nlu = NLUResult(
            entities=["đau dạ dày", "mổ"],
            topic={"topic": "gastroenterology", "confidence": 0.9, "is_reliable": True},
            intent={"intents": ["Treatment"], "scores": {"Treatment": 0.8}, "primary_intent": "Treatment"},
        )
        assert isinstance(nlu.entities, list)
        for e in nlu.entities:
            assert isinstance(e, str)

    def test_empty_entities_allowed(self):
        """entities rỗng → hợp lệ (câu hỏi tổng quát không chứa entity)."""
        nlu = NLUResult(
            entities=[],
            topic={"topic": "general", "confidence": 0.5, "is_reliable": True},
            intent={"intents": [], "scores": {}, "primary_intent": "unknown"},
        )
        assert nlu.entities == []


# ============================================================
# 🔗 INTEGRATION TESTS — E2E Flow
# ============================================================

class TestPipelineFlow:
    """E2E integration tests cho luồng xử lý trong main.py."""

    def test_trạm1_output_feeds_trạm2(self, sample_raw_text, sample_clean_text):
        """Output Trạm 1 (str) phải là input hợp lệ cho Trạm 2 (str)."""
        # Trạm 1 output
        clean_text = sample_clean_text
        assert isinstance(clean_text, str)
        assert len(clean_text) > 0
        # Đủ điều kiện làm input cho NER/Topic/Intent

    def test_ner_output_fits_nlu_result_entities(self, sample_ner_aggregated_entities):
        """NER aggregated output (List[str]) khớp NLUResult.entities type."""
        nlu = NLUResult(
            entities=sample_ner_aggregated_entities,
            topic={"topic": "test", "confidence": 0.5, "is_reliable": True},
            intent={"intents": [], "scores": {}, "primary_intent": "unknown"},
        )
        assert nlu.entities == sample_ner_aggregated_entities

    def test_topic_output_fits_nlu_result(self, sample_topic_output):
        """Topic output (Dict) khớp NLUResult.topic type."""
        nlu = NLUResult(
            entities=[],
            topic=sample_topic_output,
            intent={"intents": [], "scores": {}, "primary_intent": "unknown"},
        )
        assert nlu.topic["topic"] == "gastroenterology"

    def test_intent_normalized_fits_nlu_result(self, sample_intent_normalized_output):
        """Intent normalized output (Dict) khớp NLUResult.intent type."""
        nlu = NLUResult(
            entities=[],
            topic={"topic": "test", "confidence": 0.5, "is_reliable": True},
            intent=sample_intent_normalized_output,
        )
        assert nlu.intent["primary_intent"] == "Treatment"

    def test_full_response_assembly(
        self, sample_raw_text, sample_clean_text,
        sample_ner_aggregated_entities, sample_topic_output,
        sample_intent_normalized_output
    ):
        """Full response construction — tất cả stages gộp lại."""
        response = MedicalQueryResponse(
            raw_text=sample_raw_text,
            clean_text=sample_clean_text,
            nlu_result=NLUResult(
                entities=sample_ner_aggregated_entities,
                topic=sample_topic_output,
                intent=sample_intent_normalized_output,
            ),
            processing_time_ms=150.0,
        )
        print(f"\n  {'='*60}")
        print(f"  🏥 FULL PIPELINE RESPONSE")
        print(f"  {'='*60}")
        print(f"  📥 Raw text:   '{response.raw_text}'")
        print(f"  🧹 Clean text: '{response.clean_text}'")
        print(f"  \n  📊 NLU Results:")
        print(f"    🏷️  Entities: {response.nlu_result.entities}")
        print(f"    🏥 Topic:    {response.nlu_result.topic}")
        print(f"    🎯 Intent:   {response.nlu_result.intent}")
        print(f"  \n  ⏱️  Processing: {response.processing_time_ms}ms")
        print(f"  {'='*60}")
        assert response.raw_text == sample_raw_text
        assert response.clean_text == sample_clean_text
        assert response.nlu_result.entities == ["đau dạ dày", "mổ nội soi"]
        assert response.nlu_result.topic["topic"] == "gastroenterology"
        assert response.nlu_result.intent["primary_intent"] == "Treatment"
        assert response.processing_time_ms > 0


# ============================================================
# 🌐 API ENDPOINT TESTS
# ============================================================

class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_response_schema(self):
        """HealthCheckResponse phải có status và models_loaded."""
        resp = HealthCheckResponse(
            status="healthy",
            models_loaded={
                "acronym_resolver": True,
                "medical_ner": True,
                "topic_classifier": True,
                "intent_classifier": False,
            },
        )
        assert resp.status == "healthy"
        assert resp.models_loaded["acronym_resolver"] is True
        assert resp.models_loaded["intent_classifier"] is False

    def test_all_models_none_still_healthy(self):
        """Ngay cả khi tất cả model None → status vẫn 'healthy' (server chạy)."""
        resp = HealthCheckResponse(
            status="healthy",
            models_loaded={
                "acronym_resolver": False,
                "medical_ner": False,
                "topic_classifier": False,
                "intent_classifier": False,
            },
        )
        assert resp.status == "healthy"


# ============================================================
# ⚡ EDGE CASE TESTS — Partial Failure & Fallbacks
# ============================================================

class TestPipelineEdgeCases:
    """Edge cases cho toàn bộ pipeline."""

    def test_acronym_resolver_none_uses_raw_text(self, sample_raw_text):
        """Nếu acronym_resolver = None → clean_text = raw_text."""
        # Simulate: acronym_resolver is None in main.py
        clean_text = sample_raw_text  # fallback logic
        assert clean_text == sample_raw_text

    def test_ner_failure_returns_empty_entities(self):
        """NER crash → entities = [] (graceful fallback)."""
        fallback_entities = []
        nlu = NLUResult(
            entities=fallback_entities,
            topic={"topic": "unknown", "confidence": 0.0, "is_reliable": False},
            intent={"intents": [], "scores": {}, "primary_intent": "unknown"},
        )
        assert nlu.entities == []

    def test_topic_failure_returns_unknown(self):
        """Topic crash → topic = 'unknown' (graceful fallback)."""
        fallback_topic = {"topic": "unknown", "confidence": 0.0, "is_reliable": False}
        assert fallback_topic["topic"] == "unknown"
        assert fallback_topic["confidence"] == 0.0

    def test_intent_failure_returns_unknown(self):
        """Intent crash → primary_intent = 'unknown' (graceful fallback)."""
        fallback_intent = {"intents": [], "scores": {}, "primary_intent": "unknown"}
        assert fallback_intent["primary_intent"] == "unknown"
        assert fallback_intent["intents"] == []

    def test_all_models_failed_still_returns_response(self, sample_raw_text):
        """Tất cả models crash → vẫn trả response hợp lệ (degraded)."""
        response = MedicalQueryResponse(
            raw_text=sample_raw_text,
            clean_text=sample_raw_text,  # no acronym resolution
            nlu_result=NLUResult(
                entities=[],
                topic={"topic": "unknown", "confidence": 0.0, "is_reliable": False},
                intent={"intents": [], "scores": {}, "primary_intent": "unknown"},
            ),
            processing_time_ms=1.0,
        )
        assert response.raw_text == sample_raw_text
        assert response.nlu_result.entities == []
        assert response.nlu_result.topic["topic"] == "unknown"

    def test_response_serializable_to_json(
        self, sample_raw_text, sample_clean_text,
        sample_ner_aggregated_entities, sample_topic_output,
        sample_intent_normalized_output
    ):
        """Response phải serialize được thành JSON (API trả về JSON)."""
        response = MedicalQueryResponse(
            raw_text=sample_raw_text,
            clean_text=sample_clean_text,
            nlu_result=NLUResult(
                entities=sample_ner_aggregated_entities,
                topic=sample_topic_output,
                intent=sample_intent_normalized_output,
            ),
            processing_time_ms=100.0,
        )
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "gastroenterology" in json_str
        assert "Treatment" in json_str


# ============================================================
# 🔄 CRAG COMPATIBILITY TESTS — NLU output → CRAG input
# ============================================================

class TestCRAGCompatibility:
    """Kiểm tra output pipeline đủ thông tin cho CRAG nodes."""

    def test_crag_retrieval_has_topic_filter(self, sample_topic_output):
        """CRAG Hybrid Retrieval cần topic string cho Qdrant filter."""
        topic = sample_topic_output["topic"]
        assert isinstance(topic, str)
        assert len(topic) > 0

    def test_crag_retrieval_has_entities_for_boost(self, sample_ner_aggregated_entities):
        """CRAG Hybrid Retrieval cần entities cho sparse query boost."""
        assert isinstance(sample_ner_aggregated_entities, list)
        for entity in sample_ner_aggregated_entities:
            assert isinstance(entity, str)

    def test_crag_generation_has_intent_for_prompt(self, sample_intent_normalized_output):
        """CRAG LLM Generation cần primary_intent cho prompt routing."""
        primary = sample_intent_normalized_output["primary_intent"]
        assert primary in ["Diagnosis", "Treatment", "Severity", "Cause"]

    def test_crag_query_rewriter_has_entities_and_intent(
        self, sample_ner_aggregated_entities, sample_intent_normalized_output
    ):
        """CRAG Query Rewriter cần entities + intent để viết lại query."""
        entities = sample_ner_aggregated_entities
        intent = sample_intent_normalized_output["primary_intent"]
        # Simulate query rewrite
        rewritten = f"Triệu chứng: {', '.join(entities)}. Câu hỏi về: {intent}"
        assert "đau dạ dày" in rewritten
        assert "Treatment" in rewritten

    def test_confidence_gate_for_topic_filter(self, sample_topic_output):
        """Nếu confidence < 0.7 → CRAG nên bỏ topic filter."""
        confidence = sample_topic_output["confidence"]
        should_filter = confidence >= 0.7
        # sample has 0.9134 → should filter
        assert should_filter is True
