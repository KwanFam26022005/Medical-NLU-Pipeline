"""
crag/state.py — CRAGState schema cho LangGraph pipeline.

Mọi node trong graph đọc/ghi state qua TypedDict này.
Chia làm 4 nhóm:
  - Input (immutable): dữ liệu từ NLU Pipeline (Trạm 1-2C)
  - Retrieval (mutable): trạng thái tìm kiếm
  - Generation: câu trả lời và nguồn trích dẫn
  - Control flow: biến điều khiển vòng lặp
"""

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class CRAGState(TypedDict):
    """State schema cho LangGraph CRAG pipeline."""

    # ── Input từ NLU Pipeline (immutable) ──────────────────────
    raw_text: str                          # Text gốc từ user
    clean_text: str                        # Text đã giải viết tắt (Trạm 1)
    topic: str                             # "gastroenterology" (Trạm 2B)
    topic_confidence: float                # 0.9134
    entities: List[str]                    # ["đau dạ dày", "trào ngược"] (Trạm 2A)
    entity_types: Dict[str, str]           # {"đau dạ dày": "SYMPTOM_AND_DISEASE"}
    primary_intent: str                    # "Treatment" (Trạm 2C)
    all_intents: List[Dict[str, float]]    # [{"Treatment": 0.87}, ...]

    # ── Retrieval state (mutable) ──────────────────────────────
    current_query: str                     # Query hiện tại (có thể đã rewrite)
    retrieved_docs: List[Dict[str, Any]]   # [{content, metadata, score}]
    filtered_docs: List[Dict[str, Any]]    # Docs đã qua Document Grader

    # ── Generation state ───────────────────────────────────────
    generated_answer: str                  # Câu trả lời LLM
    sources: List[Dict[str, str]]          # [{title, url/id, snippet}]

    # ── Control flow ───────────────────────────────────────────
    grader_result: str                     # "RELEVANT" | "NOT_RELEVANT"
    hallucination_result: str              # "GROUNDED" | "NOT_GROUNDED"
    retrieval_count: int                   # Số lần retrieve (max 3: 1 gốc + 2 retry)
    generation_count: int                  # Số lần generate (max 2: 1 gốc + 1 retry)
    error_message: Optional[str]           # Lỗi nếu có
    final_answer: str                      # Câu trả lời cuối cùng
