"""
crag/generator.py — LLM Answer Generation node.

Phase 1 (skeleton): chỉ định nghĩa signature và cập nhật state.
Logic thực tế (LLM generation có trích dẫn nguồn) sẽ implement ở Phase 2.
"""

from typing import Any, Dict

from crag.state import CRAGState


def generate_answer(state: CRAGState) -> Dict[str, Any]:
    """
    Generate: sinh câu trả lời dựa trên filtered_docs.

    Inject intent_focus suffix vào system prompt để LLM tập trung
    đúng loại thông tin (chẩn đoán, điều trị, mức độ nguy hiểm, nguyên nhân).

    Phase 1 skeleton: trả về state update dict.

    Returns:
        Dict chứa các key cần update trong CRAGState:
          - generated_answer: str — câu trả lời LLM
          - sources: List[Dict] — nguồn trích dẫn
    """
    # TODO: Phase 2 — implement actual LLM generation logic
    # 1. Build context từ filtered_docs
    # 2. Get intent_focus từ INTENT_PROMPTS[primary_intent]
    # 3. Invoke GENERATION_PROMPT
    # 4. Parse sources từ doc metadata

    return {
        "generated_answer": "",
        "sources": [],
    }
