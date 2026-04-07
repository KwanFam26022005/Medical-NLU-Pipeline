"""
crag/graph.py — LangGraph state machine cho CRAG pipeline.

Xây dựng graph topology:
  retrieve → grade_documents → [generate | rewrite_query | fallback]
  generate → check_hallucination → [END | generate (retry) | fallback]
  rewrite_query → retrieve (loop back)
  prepare_fallback → END

Giới hạn vòng lặp:
  - retrieval_count max = 3 (1 gốc + 2 retry)
  - generation_count max = 2 (1 gốc + 1 retry)
"""

from langgraph.graph import END, StateGraph

from crag.fallback import prepare_fallback_answer
from crag.generator import generate_answer
from crag.grader import grade_documents
from crag.hallucination import check_hallucination
from crag.retriever import hybrid_retrieve
from crag.rewriter import rewrite_query
from crag.state import CRAGState


# ── Retry limits ───────────────────────────────────────────────
MAX_RETRIEVAL_COUNT = 3   # 1 lần gốc + 2 lần retry
MAX_GENERATION_COUNT = 2  # 1 lần gốc + 1 lần retry


# ── Routing functions ──────────────────────────────────────────

def route_after_grading(state: CRAGState) -> str:
    """
    Điều hướng sau Document Grader.

    - RELEVANT → "generate" (sinh câu trả lời)
    - NOT_RELEVANT + retrieval_count <= 2 → "rewrite_query" (viết lại query)
    - NOT_RELEVANT + retrieval_count > 2 → "fallback" (hết retry)
    """
    if state.get("grader_result") == "RELEVANT":
        return "generate"

    # NOT_RELEVANT: kiểm tra retry limit
    retrieval_count = state.get("retrieval_count", 1)
    if retrieval_count < MAX_RETRIEVAL_COUNT:
        return "rewrite_query"

    return "fallback"


def route_after_hallucination_check(state: CRAGState) -> str:
    """
    Điều hướng sau Hallucination Check.

    - GROUNDED → "end" (trả kết quả)
    - NOT_GROUNDED + generation_count < max → "generate" (generate lại)
    - NOT_GROUNDED + generation_count >= max → "fallback" (hết retry)
    """
    if state.get("hallucination_result") == "GROUNDED":
        return "end"

    # NOT_GROUNDED: kiểm tra retry limit
    generation_count = state.get("generation_count", 1)
    if generation_count < MAX_GENERATION_COUNT:
        return "generate"

    return "fallback"


# ── Graph builder ──────────────────────────────────────────────

def build_crag_graph() -> StateGraph:
    """
    Xây dựng và compile CRAG state machine.

    Returns:
        Compiled LangGraph StateGraph (runnable).

    Graph topology:
        Entry → retrieve → grade_documents
          → RELEVANT     → generate → check_hallucination
                            → GROUNDED     → END
                            → NOT_GROUNDED → generate (retry) | fallback
          → NOT_RELEVANT → rewrite_query → retrieve (loop) | fallback
    """
    workflow = StateGraph(CRAGState)

    # ── Đăng ký nodes ──
    workflow.add_node("retrieve", hybrid_retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("prepare_fallback", prepare_fallback_answer)

    # ── Entry point ──
    workflow.set_entry_point("retrieve")

    # ── Edges ──
    # retrieve → grade_documents (luôn luôn)
    workflow.add_edge("retrieve", "grade_documents")

    # grade_documents → conditional routing
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "prepare_fallback",
        },
    )

    # rewrite_query → retrieve (loop back)
    workflow.add_edge("rewrite_query", "retrieve")

    # generate → check_hallucination (luôn luôn)
    workflow.add_edge("generate", "check_hallucination")

    # check_hallucination → conditional routing
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "end": END,
            "generate": "generate",
            "fallback": "prepare_fallback",
        },
    )

    # prepare_fallback → END
    workflow.add_edge("prepare_fallback", END)

    return workflow.compile()
