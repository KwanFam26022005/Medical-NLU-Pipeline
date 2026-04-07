"""
crag/fallback.py — Fallback Answer node.

Sinh câu trả lời an toàn khi pipeline CRAG thất bại
(không tìm được tài liệu phù hợp hoặc hallucination check thất bại).
"""

from typing import Any, Dict

from crag.state import CRAGState


# ── Topic → Tiếng Việt mapping ─────────────────────────────────
TOPIC_VI_MAP = {
    "cardiology": "Tim mạch",
    "gastroenterology": "Tiêu hóa",
    "neurology": "Thần kinh",
    "orthopedics": "Cơ xương khớp",
    "pediatrics": "Nhi khoa",
    "dermatology": "Da liễu",
    "obstetrics_gynecology": "Sản phụ khoa",
    "urology": "Tiết niệu",
    "endocrinology": "Nội tiết",
    "oncology": "Ung bướu",
    "ent": "Tai Mũi Họng",
    "ophthalmology": "Nhãn khoa",
    "dentistry": "Răng Hàm Mặt",
    "rheumatology": "Cơ Xương Khớp",
    "nutrition": "Dinh dưỡng",
    "internal_medicine": "Nội khoa",
    "reproductive_endocrinology": "Nội tiết sinh sản",
    "traditional_medicine": "Y học cổ truyền",
}


def prepare_fallback_answer(state: CRAGState) -> Dict[str, Any]:
    """
    Sinh câu trả lời an toàn khi pipeline CRAG không thể tìm được
    tài liệu phù hợp hoặc câu trả lời không đạt hallucination check.

    Returns:
        Dict chứa các key cần update trong CRAGState:
          - final_answer: str — câu trả lời fallback an toàn
          - sources: List — rỗng (không có nguồn trích dẫn)
    """
    topic = state.get("topic", "unknown")
    khoa = TOPIC_VI_MAP.get(topic, "chuyên khoa phù hợp")

    fallback_answer = (
        f"Xin lỗi, tôi chưa tìm được thông tin đủ tin cậy để trả lời câu hỏi của bạn. "
        f"Câu hỏi của bạn liên quan đến chuyên khoa **{khoa}**. "
        f"Để được tư vấn chính xác, vui lòng đặt lịch khám với bác sĩ chuyên khoa {khoa}. "
        f"Bạn có thể liên hệ các bệnh viện lớn như Vinmec, Tâm Anh, hoặc bệnh viện công tuyến trên."
    )

    return {
        "final_answer": fallback_answer,
        "sources": [],
    }
