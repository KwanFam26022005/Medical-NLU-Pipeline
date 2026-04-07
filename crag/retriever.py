"""
crag/retriever.py — Hybrid Retrieval node cho CRAG pipeline.

Phase 1 (skeleton): chỉ định nghĩa signature và cập nhật state.
Logic thực tế (Qdrant + bge-m3 + Reranker) sẽ implement ở Phase 2.
"""

from typing import Any, Dict

from crag.state import CRAGState


# ── Config constants ───────────────────────────────────────────
TOPIC_CONFIDENCE_THRESHOLD = 0.70  # Ngưỡng để áp dụng topic filter
TOP_K_RETRIEVAL = 50               # Số docs retrieve ban đầu
TOP_K_RERANK = 5                   # Số docs sau rerank


def hybrid_retrieve(state: CRAGState) -> Dict[str, Any]:
    """
    Hybrid Retrieval: Dense + Sparse + Topic Filter + Entity Boost + Reranker.

    Phase 1 skeleton: trả về state update dict (LangGraph convention).
    Logic thực tế sẽ gọi Qdrant + bge-m3 + CrossEncoder reranker.

    Returns:
        Dict chứa các key cần update trong CRAGState.
    """
    # TODO: Phase 2 — implement actual retrieval logic
    # 1. Entity-Boosted Query (entities → enrich sparse)
    # 2. Encode query (bge-m3 dense + sparse)
    # 3. Build Qdrant filter (topic_confidence >= threshold)
    # 4. Hybrid search (dense + sparse + RRF merge)
    # 5. Reranker (CrossEncoder bge-reranker-v2-m3)

    retrieval_count = state.get("retrieval_count", 0) + 1

    return {
        "retrieved_docs": [],      # Placeholder — sẽ được fill bởi actual retriever
        "retrieval_count": retrieval_count,
    }
