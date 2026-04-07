"""
crag/hallucination.py — Hallucination Check node.

Phase 1 (skeleton): chỉ định nghĩa signature và cập nhật state.
Logic thực tế (LLM grounding check) sẽ implement ở Phase 2.
"""

from typing import Any, Dict

from crag.state import CRAGState


def check_hallucination(state: CRAGState) -> Dict[str, Any]:
    """
    Hallucination Check: kiểm tra câu trả lời có grounded trong tài liệu.

    So sánh generated_answer với filtered_docs. Nếu câu trả lời chứa
    thông tin không có trong tài liệu → NOT_GROUNDED.

    Phase 1 skeleton: trả về state update dict.

    Returns:
        Dict chứa các key cần update trong CRAGState:
          - hallucination_result: "GROUNDED" | "NOT_GROUNDED"
          - generation_count: int — tăng thêm 1
          - final_answer: str — set nếu GROUNDED
    """
    # TODO: Phase 2 — implement actual LLM hallucination check
    # 1. Build context từ filtered_docs
    # 2. Invoke HALLUCINATION_PROMPT
    # 3. Parse output: GROUNDED / NOT_GROUNDED

    generation_count = state.get("generation_count", 0) + 1
    generated_answer = state.get("generated_answer", "")

    # Skeleton: mặc định GROUNDED
    return {
        "hallucination_result": "GROUNDED",
        "generation_count": generation_count,
        "final_answer": generated_answer,
    }
