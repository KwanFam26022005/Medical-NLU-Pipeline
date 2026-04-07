"""
crag/prompts.py — Tất cả prompt templates cho CRAG pipeline.

Bao gồm:
  - GRADER_PROMPT: đánh giá relevance của document
  - REWRITER_PROMPT: viết lại query khi document không relevant
  - GENERATION_PROMPT: sinh câu trả lời có trích dẫn
  - HALLUCINATION_PROMPT: kiểm tra grounding
  - INTENT_PROMPTS: suffix theo intent cho generation prompt
"""

from langchain_core.prompts import ChatPromptTemplate


# ── Intent → System Prompt Suffix ──────────────────────────────
INTENT_PROMPTS = {
    "Diagnosis": "Tập trung vào: triệu chứng gợi ý, xét nghiệm cần làm, chẩn đoán phân biệt.",
    "Treatment": "Tập trung vào: phương pháp điều trị, thuốc, phác đồ, lưu ý khi dùng thuốc.",
    "Severity":  "Tập trung vào: đánh giá mức độ nguy hiểm, dấu hiệu cần đi khám ngay.",
    "Cause":     "Tập trung vào: nguyên nhân gây bệnh, yếu tố nguy cơ, cơ chế bệnh sinh.",
}


# ── Document Grader ────────────────────────────────────────────
GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một chuyên gia đánh giá tài liệu y tế. Nhiệm vụ của bạn là xác định tài liệu truy xuất có THỰC SỰ LIÊN QUAN đến câu hỏi y tế của bệnh nhân hay không.

TIÊU CHÍ ĐÁNH GIÁ:
1. Tài liệu phải đề cập đến ít nhất MỘT triệu chứng, bệnh, hoặc thủ thuật y tế liên quan đến câu hỏi.
2. Tài liệu phải thuộc đúng chuyên khoa hoặc lĩnh vực y tế mà câu hỏi đang hỏi.
3. Tài liệu phải cung cấp thông tin có thể dùng để trả lời câu hỏi (dù chỉ một phần).

CHỈ trả lời bằng MỘT TỪ DUY NHẤT:
- RELEVANT — nếu tài liệu liên quan và hữu ích
- NOT_RELEVANT — nếu tài liệu không liên quan hoặc sai chuyên khoa"""),
    ("human", """Câu hỏi của bệnh nhân: {question}
Chuyên khoa dự đoán: {topic}
Thực thể y tế trong câu hỏi: {entities}

Tài liệu truy xuất:
---
{document}
---

Đánh giá (RELEVANT hoặc NOT_RELEVANT):""")
])


# ── Query Rewriter ─────────────────────────────────────────────
REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là chuyên gia viết lại câu hỏi y tế. Câu hỏi gốc không tìm được tài liệu phù hợp. 
Hãy viết lại câu hỏi sao cho:
1. Sử dụng thuật ngữ y khoa chính xác hơn
2. Tập trung vào các thực thể y tế đã được trích xuất
3. Phù hợp với ý định hỏi của bệnh nhân
4. Ngắn gọn, rõ ràng, dưới 100 từ

CHỈ trả về câu hỏi đã viết lại, KHÔNG giải thích."""),
    ("human", """Câu hỏi gốc: {question}
Thực thể y tế: {entities}
Ý định hỏi: {intent}
Chuyên khoa: {topic}

Câu hỏi viết lại:""")
])


# ── LLM Generation ─────────────────────────────────────────────
GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là Trợ lý Y tế AI, có vai trò tương đương một bác sĩ đa khoa tư vấn ban đầu.

QUY TẮC BẮT BUỘC:
1. CHỈ trả lời dựa trên tài liệu tham khảo được cung cấp bên dưới.
2. Mỗi khẳng định y tế PHẢI kèm trích dẫn nguồn: [Nguồn 1], [Nguồn 2], v.v.
3. Nếu tài liệu không đủ thông tin → nói rõ: "Thông tin hiện có chưa đủ để trả lời đầy đủ."
4. LUÔN kết thúc bằng lời khuyên đi khám bác sĩ chuyên khoa.
5. KHÔNG đưa ra chẩn đoán xác định. Chỉ cung cấp thông tin tham khảo.
6. Trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu cho bệnh nhân.
{intent_focus}

ĐỊNH DẠNG TRÍCH DẪN:
- Sử dụng [Nguồn X] ngay sau thông tin trích dẫn
- Liệt kê danh sách nguồn ở cuối câu trả lời"""),
    ("human", """TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI CỦA BỆNH NHÂN: {question}

CÂU TRẢ LỜI (có trích dẫn nguồn):""")
])


# ── Hallucination Check ────────────────────────────────────────
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là chuyên gia kiểm tra tính chính xác của câu trả lời y tế. 
Nhiệm vụ: Xác định câu trả lời có DỰA TRÊN (grounded) các tài liệu tham khảo hay không.

TIÊU CHÍ:
1. Mọi khẳng định y tế trong câu trả lời PHẢI có cơ sở từ tài liệu tham khảo.
2. Câu trả lời KHÔNG ĐƯỢC bịa thêm thông tin y tế mà tài liệu không đề cập.
3. Câu trả lời ĐƯỢC PHÉP diễn giải lại (paraphrase) miễn là đúng nghĩa gốc.
4. Các câu chỉ dẫn đi khám bác sĩ, lời khuyên chung (uống đủ nước, nghỉ ngơi) 
   ĐƯỢC PHÉP vì đây là tri thức y tế phổ quát.

CHỈ trả lời bằng MỘT TỪ DUY NHẤT:
- GROUNDED — câu trả lời dựa trên tài liệu
- NOT_GROUNDED — câu trả lời chứa thông tin bịa đặt"""),
    ("human", """Tài liệu tham khảo:
---
{context}
---

Câu trả lời cần kiểm tra:
---
{answer}
---

Đánh giá (GROUNDED hoặc NOT_GROUNDED):""")
])


# ── Intent mapping tiếng Việt (cho query rewriter) ─────────────
INTENT_VI_MAP = {
    "Diagnosis": "chẩn đoán bệnh",
    "Treatment": "phương pháp điều trị",
    "Severity": "mức độ nghiêm trọng",
    "Cause": "nguyên nhân gây bệnh",
}
