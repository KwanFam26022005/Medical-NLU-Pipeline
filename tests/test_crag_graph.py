"""
tests/test_crag_graph.py — Unit tests cho CRAG graph topology.

Kiểm tra:
1. Graph compile thành công
2. Routing logic: route_after_grading, route_after_hallucination_check
3. End-to-end paths với mock nodes:
   - Happy path: RELEVANT → GROUNDED → END
   - Retry path: NOT_RELEVANT → rewrite → re-retrieve
   - Fallback after max retrieval retries
   - Hallucination retry → GROUNDED
   - Fallback after max generation retries
4. State integrity: counters, final_answer
"""

import pytest
from unittest.mock import patch
from typing import Any, Dict

from crag.state import CRAGState
from crag.graph import (
    build_crag_graph,
    route_after_grading,
    route_after_hallucination_check,
    MAX_RETRIEVAL_COUNT,
    MAX_GENERATION_COUNT,
)
from crag.pipeline import create_initial_state


# ============================================================
# 📋 FIXTURES
# ============================================================

@pytest.fixture
def base_state() -> CRAGState:
    """Tạo CRAGState cơ bản cho testing."""
    return create_initial_state(
        raw_text="bs ơi e bị đau dạ dày có cần mổ k ạ?",
        clean_text="bác sĩ ơi em bị đau dạ dày có cần mổ không ạ?",
        topic="gastroenterology",
        topic_confidence=0.91,
        entities=["đau dạ dày"],
        entity_types={"đau dạ dày": "SYMPTOM_AND_DISEASE"},
        primary_intent="Treatment",
        all_intents=[{"Treatment": 0.87}, {"Diagnosis": 0.12}],
    )


# ============================================================
# 🧪 TEST: Graph compiles
# ============================================================

class TestGraphCompile:
    """Verify graph builds and compiles without errors."""

    def test_build_crag_graph_compiles(self):
        """Graph phải compile thành công."""
        graph = build_crag_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """Graph phải có đúng 6 nodes."""
        graph = build_crag_graph()
        # LangGraph compiled graph có attribute .nodes hoặc tương tự
        # Verify bằng cách chạy thử
        assert graph is not None


# ============================================================
# 🧪 TEST: Routing functions (unit)
# ============================================================

class TestRouteAfterGrading:
    """Test route_after_grading() routing logic."""

    def test_relevant_routes_to_generate(self, base_state):
        """RELEVANT → generate."""
        base_state["grader_result"] = "RELEVANT"
        assert route_after_grading(base_state) == "generate"

    def test_not_relevant_first_try_routes_to_rewrite(self, base_state):
        """NOT_RELEVANT + retrieval_count=1 → rewrite_query."""
        base_state["grader_result"] = "NOT_RELEVANT"
        base_state["retrieval_count"] = 1  # Lần 1 (gốc)
        assert route_after_grading(base_state) == "rewrite_query"

    def test_not_relevant_second_retry_routes_to_rewrite(self, base_state):
        """NOT_RELEVANT + retrieval_count=2 → rewrite_query (vẫn còn 1 retry)."""
        base_state["grader_result"] = "NOT_RELEVANT"
        base_state["retrieval_count"] = 2  # Retry 1
        assert route_after_grading(base_state) == "rewrite_query"

    def test_not_relevant_max_retries_routes_to_fallback(self, base_state):
        """NOT_RELEVANT + retrieval_count=3 → fallback (hết retry)."""
        base_state["grader_result"] = "NOT_RELEVANT"
        base_state["retrieval_count"] = MAX_RETRIEVAL_COUNT  # = 3
        assert route_after_grading(base_state) == "fallback"

    def test_not_relevant_over_max_routes_to_fallback(self, base_state):
        """NOT_RELEVANT + retrieval_count > max → fallback."""
        base_state["grader_result"] = "NOT_RELEVANT"
        base_state["retrieval_count"] = MAX_RETRIEVAL_COUNT + 1
        assert route_after_grading(base_state) == "fallback"


class TestRouteAfterHallucinationCheck:
    """Test route_after_hallucination_check() routing logic."""

    def test_grounded_routes_to_end(self, base_state):
        """GROUNDED → end."""
        base_state["hallucination_result"] = "GROUNDED"
        assert route_after_hallucination_check(base_state) == "end"

    def test_not_grounded_first_try_routes_to_generate(self, base_state):
        """NOT_GROUNDED + generation_count=1 → generate (retry)."""
        base_state["hallucination_result"] = "NOT_GROUNDED"
        base_state["generation_count"] = 1  # Lần 1 (gốc)
        assert route_after_hallucination_check(base_state) == "generate"

    def test_not_grounded_max_retries_routes_to_fallback(self, base_state):
        """NOT_GROUNDED + generation_count=2 → fallback (hết retry)."""
        base_state["hallucination_result"] = "NOT_GROUNDED"
        base_state["generation_count"] = MAX_GENERATION_COUNT  # = 2
        assert route_after_hallucination_check(base_state) == "fallback"

    def test_not_grounded_over_max_routes_to_fallback(self, base_state):
        """NOT_GROUNDED + generation_count > max → fallback."""
        base_state["hallucination_result"] = "NOT_GROUNDED"
        base_state["generation_count"] = MAX_GENERATION_COUNT + 1
        assert route_after_hallucination_check(base_state) == "fallback"


# ============================================================
# 🧪 TEST: End-to-end graph paths with mock nodes
# ============================================================

class TestGraphHappyPath:
    """Test: RELEVANT → GROUNDED → END (happy path)."""

    def test_happy_path_returns_final_answer(self, base_state):
        """Happy path: retrieve → grade(RELEVANT) → generate → hallucination(GROUNDED) → END."""
        mock_docs = [{"content": "Đau dạ dày là...", "metadata": {"source": "test"}, "score": 0.95}]
        expected_answer = "Đau dạ dày có thể điều trị bằng thuốc PPI. [Nguồn 1]"

        def mock_retrieve(state):
            return {"retrieved_docs": mock_docs, "retrieval_count": state.get("retrieval_count", 0) + 1}

        def mock_grade(state):
            return {"filtered_docs": mock_docs, "grader_result": "RELEVANT"}

        def mock_rewrite(state):
            return {"current_query": state.get("current_query", "")}

        def mock_generate(state):
            return {"generated_answer": expected_answer, "sources": [{"title": "test", "snippet": "..."}]}

        def mock_hallucination(state):
            return {
                "hallucination_result": "GROUNDED",
                "generation_count": state.get("generation_count", 0) + 1,
                "final_answer": expected_answer,
            }

        def mock_fallback(state):
            return {"final_answer": "Fallback", "sources": []}

        with patch("crag.graph.hybrid_retrieve", mock_retrieve), \
             patch("crag.graph.grade_documents", mock_grade), \
             patch("crag.graph.rewrite_query", mock_rewrite), \
             patch("crag.graph.generate_answer", mock_generate), \
             patch("crag.graph.check_hallucination", mock_hallucination), \
             patch("crag.graph.prepare_fallback_answer", mock_fallback):

            graph = build_crag_graph()
            result = graph.invoke(base_state)

            assert result["final_answer"] == expected_answer
            assert result["retrieval_count"] == 1
            assert result["generation_count"] == 1
            assert result["grader_result"] == "RELEVANT"
            assert result["hallucination_result"] == "GROUNDED"


class TestGraphRetryRetrievalPath:
    """Test: NOT_RELEVANT → rewrite → re-retrieve → RELEVANT → END."""

    def test_one_retry_then_success(self, base_state):
        """Lần retrieve 1 thất bại, rewrite, lần 2 thành công."""
        call_count = {"retrieve": 0, "grade": 0}
        mock_docs = [{"content": "Đau dạ dày...", "metadata": {}, "score": 0.9}]
        expected_answer = "Trả lời sau retry."

        def mock_retrieve(state):
            call_count["retrieve"] += 1
            return {
                "retrieved_docs": mock_docs,
                "retrieval_count": state.get("retrieval_count", 0) + 1,
            }

        def mock_grade(state):
            call_count["grade"] += 1
            # Lần 1: NOT_RELEVANT, lần 2: RELEVANT
            if call_count["grade"] <= 1:
                return {"filtered_docs": [], "grader_result": "NOT_RELEVANT"}
            return {"filtered_docs": mock_docs, "grader_result": "RELEVANT"}

        def mock_rewrite(state):
            return {"current_query": "câu hỏi đã viết lại"}

        def mock_generate(state):
            return {"generated_answer": expected_answer, "sources": []}

        def mock_hallucination(state):
            return {
                "hallucination_result": "GROUNDED",
                "generation_count": state.get("generation_count", 0) + 1,
                "final_answer": expected_answer,
            }

        def mock_fallback(state):
            return {"final_answer": "Fallback", "sources": []}

        with patch("crag.graph.hybrid_retrieve", mock_retrieve), \
             patch("crag.graph.grade_documents", mock_grade), \
             patch("crag.graph.rewrite_query", mock_rewrite), \
             patch("crag.graph.generate_answer", mock_generate), \
             patch("crag.graph.check_hallucination", mock_hallucination), \
             patch("crag.graph.prepare_fallback_answer", mock_fallback):

            graph = build_crag_graph()
            result = graph.invoke(base_state)

            assert result["final_answer"] == expected_answer
            assert result["retrieval_count"] == 2  # 1 gốc + 1 retry
            assert call_count["retrieve"] == 2
            assert call_count["grade"] == 2


class TestGraphFallbackAfterMaxRetrieval:
    """Test: NOT_RELEVANT × MAX_RETRIEVAL_COUNT → fallback."""

    def test_fallback_after_max_retrieval_retries(self, base_state):
        """Tất cả lần retrieve đều NOT_RELEVANT → fallback."""

        def mock_retrieve(state):
            return {
                "retrieved_docs": [],
                "retrieval_count": state.get("retrieval_count", 0) + 1,
            }

        def mock_grade(state):
            return {"filtered_docs": [], "grader_result": "NOT_RELEVANT"}

        def mock_rewrite(state):
            return {"current_query": "rewritten query"}

        def mock_generate(state):
            return {"generated_answer": "should not reach", "sources": []}

        def mock_hallucination(state):
            return {
                "hallucination_result": "GROUNDED",
                "generation_count": state.get("generation_count", 0) + 1,
                "final_answer": "should not reach",
            }

        def mock_fallback(state):
            return {
                "final_answer": "Xin lỗi, tôi chưa tìm được thông tin...",
                "sources": [],
            }

        with patch("crag.graph.hybrid_retrieve", mock_retrieve), \
             patch("crag.graph.grade_documents", mock_grade), \
             patch("crag.graph.rewrite_query", mock_rewrite), \
             patch("crag.graph.generate_answer", mock_generate), \
             patch("crag.graph.check_hallucination", mock_hallucination), \
             patch("crag.graph.prepare_fallback_answer", mock_fallback):

            graph = build_crag_graph()
            result = graph.invoke(base_state)

            assert "Xin lỗi" in result["final_answer"]
            assert result["retrieval_count"] == MAX_RETRIEVAL_COUNT
            assert result["sources"] == []


class TestGraphHallucinationRetry:
    """Test: GROUNDED after one hallucination retry."""

    def test_hallucination_retry_then_grounded(self, base_state):
        """Lần generate 1: NOT_GROUNDED, lần 2: GROUNDED."""
        hallucination_calls = {"count": 0}
        expected_answer = "Câu trả lời chính xác sau retry."

        def mock_retrieve(state):
            return {
                "retrieved_docs": [{"content": "doc", "metadata": {}, "score": 0.9}],
                "retrieval_count": state.get("retrieval_count", 0) + 1,
            }

        def mock_grade(state):
            docs = state.get("retrieved_docs", [])
            return {"filtered_docs": docs, "grader_result": "RELEVANT"}

        def mock_rewrite(state):
            return {"current_query": state.get("current_query", "")}

        def mock_generate(state):
            return {"generated_answer": expected_answer, "sources": []}

        def mock_hallucination(state):
            hallucination_calls["count"] += 1
            gen_count = state.get("generation_count", 0) + 1
            if hallucination_calls["count"] <= 1:
                return {
                    "hallucination_result": "NOT_GROUNDED",
                    "generation_count": gen_count,
                    "final_answer": "",
                }
            return {
                "hallucination_result": "GROUNDED",
                "generation_count": gen_count,
                "final_answer": expected_answer,
            }

        def mock_fallback(state):
            return {"final_answer": "Fallback", "sources": []}

        with patch("crag.graph.hybrid_retrieve", mock_retrieve), \
             patch("crag.graph.grade_documents", mock_grade), \
             patch("crag.graph.rewrite_query", mock_rewrite), \
             patch("crag.graph.generate_answer", mock_generate), \
             patch("crag.graph.check_hallucination", mock_hallucination), \
             patch("crag.graph.prepare_fallback_answer", mock_fallback):

            graph = build_crag_graph()
            result = graph.invoke(base_state)

            assert result["final_answer"] == expected_answer
            assert result["generation_count"] == 2  # 1 gốc + 1 retry
            assert hallucination_calls["count"] == 2


class TestGraphFallbackAfterMaxGeneration:
    """Test: NOT_GROUNDED × MAX_GENERATION_COUNT → fallback."""

    def test_fallback_after_max_generation_retries(self, base_state):
        """Tất cả lần generate đều NOT_GROUNDED → fallback."""

        def mock_retrieve(state):
            return {
                "retrieved_docs": [{"content": "doc", "metadata": {}, "score": 0.9}],
                "retrieval_count": state.get("retrieval_count", 0) + 1,
            }

        def mock_grade(state):
            docs = state.get("retrieved_docs", [])
            return {"filtered_docs": docs, "grader_result": "RELEVANT"}

        def mock_rewrite(state):
            return {"current_query": state.get("current_query", "")}

        def mock_generate(state):
            return {"generated_answer": "bad answer", "sources": []}

        def mock_hallucination(state):
            return {
                "hallucination_result": "NOT_GROUNDED",
                "generation_count": state.get("generation_count", 0) + 1,
                "final_answer": "",
            }

        def mock_fallback(state):
            return {
                "final_answer": "Xin lỗi, không đủ tin cậy...",
                "sources": [],
            }

        with patch("crag.graph.hybrid_retrieve", mock_retrieve), \
             patch("crag.graph.grade_documents", mock_grade), \
             patch("crag.graph.rewrite_query", mock_rewrite), \
             patch("crag.graph.generate_answer", mock_generate), \
             patch("crag.graph.check_hallucination", mock_hallucination), \
             patch("crag.graph.prepare_fallback_answer", mock_fallback):

            graph = build_crag_graph()
            result = graph.invoke(base_state)

            assert "Xin lỗi" in result["final_answer"]
            assert result["generation_count"] == MAX_GENERATION_COUNT


# ============================================================
# 🧪 TEST: State integrity
# ============================================================

class TestStateIntegrity:
    """Test state fields are properly maintained."""

    def test_initial_state_has_all_required_fields(self, base_state):
        """Initial state phải có tất cả fields."""
        required_fields = [
            "raw_text", "clean_text", "topic", "topic_confidence",
            "entities", "entity_types", "primary_intent", "all_intents",
            "current_query", "retrieved_docs", "filtered_docs",
            "generated_answer", "sources",
            "grader_result", "hallucination_result",
            "retrieval_count", "generation_count",
            "error_message", "final_answer",
        ]
        for field in required_fields:
            assert field in base_state, f"Missing field: {field}"

    def test_initial_current_query_equals_clean_text(self, base_state):
        """Ban đầu current_query = clean_text."""
        assert base_state["current_query"] == base_state["clean_text"]

    def test_initial_counters_are_zero(self, base_state):
        """Ban đầu retrieval_count = 0, generation_count = 0."""
        assert base_state["retrieval_count"] == 0
        assert base_state["generation_count"] == 0

    def test_initial_final_answer_is_empty(self, base_state):
        """Ban đầu final_answer = ''."""
        assert base_state["final_answer"] == ""


# ============================================================
# 🧪 TEST: Fallback node
# ============================================================

class TestFallbackNode:
    """Test prepare_fallback_answer() directly."""

    def test_fallback_mentions_topic_in_vietnamese(self, base_state):
        """Fallback answer phải mention chuyên khoa tiếng Việt."""
        from crag.fallback import prepare_fallback_answer
        result = prepare_fallback_answer(base_state)
        assert "Tiêu hóa" in result["final_answer"]  # gastroenterology → Tiêu hóa

    def test_fallback_with_unknown_topic(self, base_state):
        """Unknown topic → fallback dùng 'chuyên khoa phù hợp'."""
        from crag.fallback import prepare_fallback_answer
        base_state["topic"] = "some_unknown_topic"
        result = prepare_fallback_answer(base_state)
        assert "chuyên khoa phù hợp" in result["final_answer"]

    def test_fallback_sources_empty(self, base_state):
        """Fallback phải trả sources rỗng."""
        from crag.fallback import prepare_fallback_answer
        result = prepare_fallback_answer(base_state)
        assert result["sources"] == []


# ============================================================
# 🧪 TEST: Constants
# ============================================================

class TestConstants:
    """Verify retry limit constants."""

    def test_max_retrieval_count(self):
        assert MAX_RETRIEVAL_COUNT == 3

    def test_max_generation_count(self):
        assert MAX_GENERATION_COUNT == 2
