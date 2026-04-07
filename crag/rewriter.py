"""
crag/rewriter.py — Query Rewriter node.

Phase 1 (skeleton): chỉ định nghĩa signature và cập nhật state.
Logic thực tế (LLM rewrite dựa trên NLU metadata) sẽ implement ở Phase 2.
"""

from typing import Any, Dict

from crag.state import CRAGState


def rewrite_query(state: CRAGState) -> Dict[str, Any]:
    """
    Query Rewriter: viết lại query dựa trên entities + intent từ NLU.

    Dùng LLM để viết lại câu hỏi với thuật ngữ y khoa chính xác hơn,
    tập trung vào entities đã trích xuất và intent của người dùng.

    Phase 1 skeleton: trả về state update dict.

    Returns:
        Dict chứa key cần update trong CRAGState:
          - current_query: str — query đã viết lại
    """
    # TODO: Phase 2 — implement actual LLM rewriting logic
    # 1. Map intent → tiếng Việt (INTENT_VI_MAP)
    # 2. Invoke REWRITER_PROMPT với question, entities, intent, topic
    # 3. Parse output → new query string

    # Skeleton: giữ nguyên query (no-op)
    return {
        "current_query": state.get("current_query", state.get("clean_text", "")),
    }
