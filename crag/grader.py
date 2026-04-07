"""
crag/grader.py — Document Grader node (LLM-as-Judge).

Phase 1 (skeleton): chỉ định nghĩa signature và cập nhật state.
Logic thực tế (LLM grading từng doc) sẽ implement ở Phase 2.
"""

from typing import Any, Dict

from crag.state import CRAGState


def grade_documents(state: CRAGState) -> Dict[str, Any]:
    """
    Document Grader: LLM-as-Judge đánh giá relevance.

    Lọc từng document, giữ lại docs RELEVANT.
    Nếu không có doc nào relevant → grader_result = "NOT_RELEVANT".

    Phase 1 skeleton: trả về state update dict.

    Returns:
        Dict chứa các key cần update trong CRAGState:
          - filtered_docs: List[Dict] docs đã qua filter
          - grader_result: "RELEVANT" | "NOT_RELEVANT"
    """
    # TODO: Phase 2 — implement actual LLM grading logic
    # 1. Iterate qua retrieved_docs
    # 2. Gọi LLM với GRADER_PROMPT cho mỗi doc
    # 3. Parse output: RELEVANT / NOT_RELEVANT
    # 4. Giữ lại docs RELEVANT

    docs = state.get("retrieved_docs", [])

    # Skeleton: pass-through (giả định tất cả docs đều relevant)
    return {
        "filtered_docs": docs,
        "grader_result": "RELEVANT" if docs else "NOT_RELEVANT",
    }
