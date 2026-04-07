"""
crag/pipeline.py — Entry point cho CRAG pipeline.

Nhận NLU context từ main.py → khởi tạo CRAGState → chạy graph → trả kết quả.
"""

from typing import Any, Dict, List, Optional

from crag.graph import build_crag_graph
from crag.state import CRAGState


def create_initial_state(
    raw_text: str,
    clean_text: str,
    topic: str = "unknown",
    topic_confidence: float = 0.0,
    entities: Optional[List[str]] = None,
    entity_types: Optional[Dict[str, str]] = None,
    primary_intent: str = "unknown",
    all_intents: Optional[List[Dict[str, float]]] = None,
) -> CRAGState:
    """
    Tạo CRAGState ban đầu từ NLU output.

    Args:
        raw_text: Text gốc từ user.
        clean_text: Text đã giải viết tắt (Trạm 1).
        topic: Chuyên khoa y tế (Trạm 2B).
        topic_confidence: Độ tin cậy topic.
        entities: Danh sách entities (Trạm 2A).
        entity_types: Mapping entity → type.
        primary_intent: Intent chính (Trạm 2C).
        all_intents: Tất cả intents + scores.

    Returns:
        CRAGState đã khởi tạo, sẵn sàng cho graph.invoke().
    """
    return CRAGState(
        # Input (immutable)
        raw_text=raw_text,
        clean_text=clean_text,
        topic=topic,
        topic_confidence=topic_confidence,
        entities=entities or [],
        entity_types=entity_types or {},
        primary_intent=primary_intent,
        all_intents=all_intents or [],
        # Retrieval state
        current_query=clean_text,  # Ban đầu = clean_text
        retrieved_docs=[],
        filtered_docs=[],
        # Generation state
        generated_answer="",
        sources=[],
        # Control flow
        grader_result="",
        hallucination_result="",
        retrieval_count=0,
        generation_count=0,
        error_message=None,
        final_answer="",
    )


def run_crag_pipeline(
    raw_text: str,
    clean_text: str,
    topic: str = "unknown",
    topic_confidence: float = 0.0,
    entities: Optional[List[str]] = None,
    entity_types: Optional[Dict[str, str]] = None,
    primary_intent: str = "unknown",
    all_intents: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Chạy full CRAG pipeline.

    Args:
        Tham số giống create_initial_state().

    Returns:
        Dict chứa kết quả cuối cùng:
          - final_answer: str
          - sources: List[Dict]
          - retrieval_count: int
          - generation_count: int
          - error_message: Optional[str]
    """
    # Build graph (có thể cache sau nếu cần optimization)
    graph = build_crag_graph()

    # Tạo initial state
    initial_state = create_initial_state(
        raw_text=raw_text,
        clean_text=clean_text,
        topic=topic,
        topic_confidence=topic_confidence,
        entities=entities,
        entity_types=entity_types,
        primary_intent=primary_intent,
        all_intents=all_intents,
    )

    # Chạy graph
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        return {
            "final_answer": (
                "Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi của bạn. "
                "Vui lòng thử lại sau hoặc liên hệ bác sĩ để được tư vấn trực tiếp."
            ),
            "sources": [],
            "retrieval_count": initial_state.get("retrieval_count", 0),
            "generation_count": initial_state.get("generation_count", 0),
            "error_message": str(e),
        }

    return {
        "final_answer": final_state.get("final_answer", ""),
        "sources": final_state.get("sources", []),
        "retrieval_count": final_state.get("retrieval_count", 0),
        "generation_count": final_state.get("generation_count", 0),
        "error_message": final_state.get("error_message"),
    }
