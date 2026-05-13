"""
postprocessor.py — Hậu xử lý output KEHN: logits → kết quả có nghĩa.

Xử lý:
- Topic: argmax + softmax → tên chuyên khoa tiếng Việt + confidence %
- Intent: argmax → ý định bệnh nhân tiếng Việt
- NER: CRF Viterbi decode → subword→word alignment → BIO decoding → entities
"""

import torch
import torch.nn.functional as F


# ── Label Mappings (đồng bộ config_joint.py) ──────────────────────────

# Topic: 17 chuyên khoa (ID → tên tiếng Việt)
TOPIC_VI = {
    0:  ("cardiology",                "Tim mạch"),
    1:  ("dentistry",                 "Nha khoa"),
    2:  ("dermatology",               "Da liễu"),
    3:  ("endocrinology",             "Nội tiết"),
    4:  ("ent",                       "Tai Mũi Họng"),
    5:  ("gastroenterology",          "Tiêu hóa"),
    6:  ("internal_medicine",         "Nội khoa"),
    7:  ("neurology",                 "Thần kinh"),
    8:  ("nutrition",                 "Dinh dưỡng"),
    9:  ("obstetrics_gynecology",     "Sản phụ khoa"),
    10: ("oncology",                  "Ung bướu"),
    11: ("ophthalmology",             "Nhãn khoa"),
    12: ("orthopedics",              "Chỉnh hình"),
    13: ("pediatrics",               "Nhi khoa"),
    14: ("reproductive_endocrinology", "Nội tiết sinh sản"),
    15: ("rheumatology",             "Cơ xương khớp"),
    16: ("urology",                  "Tiết niệu"),
}

# Intent: 4 nhãn (ID → tên tiếng Việt)
INTENT_VI = {
    0: ("method_diagnosis", "Hỏi phương pháp chẩn đoán/điều trị"),
    1: ("treatment",        "Hỏi điều trị"),
    2: ("severity",         "Hỏi mức độ nghiêm trọng"),
    3: ("cause",            "Hỏi nguyên nhân"),
}

# NER: 5 entity types, 11 BIO tags (đồng bộ config_joint.py)
NER_ID2TAG = {
    0: "O",
    1: "B-SYM", 2: "I-SYM",
    3: "B-PRO", 4: "I-PRO",
    5: "B-DRU", 6: "I-DRU",
    7: "B-DUR", 8: "I-DUR",
    9: "B-SEV", 10: "I-SEV",
}

NER_TYPE_VI = {
    "SYM": "Triệu chứng",
    "PRO": "Thủ thuật/Quy trình",
    "DRU": "Thuốc",
    "DUR": "Thời gian",
    "SEV": "Mức độ nghiêm trọng",
}

# Màu cho mỗi entity type (dùng trong Gradio HighlightedText)
NER_TYPE_COLOR = {
    "SYM": "#FF6B6B",   # Đỏ nhạt — Triệu chứng
    "PRO": "#4ECDC4",   # Xanh ngọc — Thủ thuật
    "DRU": "#45B7D1",   # Xanh dương — Thuốc
    "DUR": "#96CEB4",   # Xanh lá nhạt — Thời gian
    "SEV": "#FFEAA7",   # Vàng nhạt — Mức độ
}


class Postprocessor:
    """Hậu xử lý raw model output → kết quả có nghĩa tiếng Việt."""

    def process_topic(self, logits_topic: torch.Tensor) -> dict:
        """
        Topic classification: argmax + softmax → tên chuyên khoa + confidence.

        Args:
            logits_topic: [1, n_topic] — raw logits

        Returns:
            dict: {
                "label_en": str,
                "label_vi": str,
                "confidence": float,
                "all_probs": dict[str, float]  — tất cả chuyên khoa + xác suất
            }
        """
        probs = F.softmax(logits_topic, dim=-1)[0]  # [n_topic]
        top_idx = probs.argmax().item()
        label_en, label_vi = TOPIC_VI[top_idx]

        # Top-k probs cho hiển thị
        all_probs = {}
        for idx in range(probs.size(0)):
            _, vi_name = TOPIC_VI.get(idx, (f"unknown_{idx}", f"Unknown {idx}"))
            all_probs[vi_name] = round(probs[idx].item(), 4)

        return {
            "label_en": label_en,
            "label_vi": label_vi,
            "confidence": round(probs[top_idx].item(), 4),
            "all_probs": all_probs,
        }

    def process_intent(self, logits_intent: torch.Tensor) -> dict:
        """
        Intent detection: argmax → ý định bệnh nhân.

        Args:
            logits_intent: [1, n_intent] — sentence-level probs (đã mean pooled)

        Returns:
            dict: {"label_en": str, "label_vi": str, "confidence": float, "all_probs": dict}
        """
        probs = logits_intent[0]  # [n_intent] — đã là probs (softmax trong model)
        top_idx = probs.argmax().item()
        label_en, label_vi = INTENT_VI[top_idx]

        all_probs = {}
        for idx in range(probs.size(0)):
            _, vi_name = INTENT_VI.get(idx, (f"unknown_{idx}", f"Unknown {idx}"))
            all_probs[vi_name] = round(probs[idx].item(), 4)

        return {
            "label_en": label_en,
            "label_vi": label_vi,
            "confidence": round(probs[top_idx].item(), 4),
            "all_probs": all_probs,
        }

    def process_ner(
        self,
        ner_pred_ids: list,
        words: list,
        word_offsets: list,
    ) -> dict:
        """
        NER: CRF Viterbi predictions → subword→word alignment → BIO decode → entities.

        Args:
            ner_pred_ids : list[int] — CRF decoded tag IDs cho TẤT CẢ tokens
                           (bao gồm CLS, SEP, subwords)
            words        : list[str] — danh sách từ gốc (sau word segmentation)
            word_offsets : list[int] — vị trí token (0-indexed) của first subword
                           cho mỗi word

        Returns:
            dict: {
                "entities": list[dict] — [{entity_text, entity_type, type_vi, start, end}]
                "word_tags": list[tuple] — [(word, tag), ...] cho mỗi word
                "highlighted_text": list — format cho Gradio HighlightedText
            }
        """
        # ── Subword → Word alignment ──────────────────────────────────
        # Chỉ lấy predictions tại vị trí first subword (word_offsets)
        word_tags = []
        for i, offset in enumerate(word_offsets):
            if i < len(words) and offset < len(ner_pred_ids):
                tag_id = ner_pred_ids[offset]
                tag = NER_ID2TAG.get(tag_id, "O")
                word_tags.append((words[i], tag))
            elif i < len(words):
                word_tags.append((words[i], "O"))

        # ── BIO Decoding → Entity list ────────────────────────────────
        entities = []
        current_entity = None

        for idx, (word, tag) in enumerate(word_tags):
            if tag.startswith("B-"):
                # Kết thúc entity trước (nếu có)
                if current_entity is not None:
                    entities.append(current_entity)
                etype = tag[2:]
                current_entity = {
                    "entity_text": word,
                    "entity_type": etype,
                    "type_vi": NER_TYPE_VI.get(etype, etype),
                    "start": idx,
                    "end": idx,
                }
            elif tag.startswith("I-") and current_entity is not None:
                etype = tag[2:]
                if etype == current_entity["entity_type"]:
                    # Tiếp tục entity hiện tại
                    current_entity["entity_text"] += " " + word
                    current_entity["end"] = idx
                else:
                    # I-tag không khớp type → kết thúc entity cũ, bắt đầu mới
                    entities.append(current_entity)
                    current_entity = {
                        "entity_text": word,
                        "entity_type": etype,
                        "type_vi": NER_TYPE_VI.get(etype, etype),
                        "start": idx,
                        "end": idx,
                    }
            else:
                # O tag hoặc I- tag mà không có entity đang mở
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Flush entity cuối cùng
        if current_entity is not None:
            entities.append(current_entity)

        # ── Gradio HighlightedText format ─────────────────────────────
        # Format: list of (text, label_or_None)
        highlighted = []
        i = 0
        while i < len(word_tags):
            word, tag = word_tags[i]
            if tag == "O":
                highlighted.append((word + " ", None))
                i += 1
            else:
                # Tìm entity tương ứng
                etype = tag[2:] if "-" in tag else tag
                entity_words = [word]
                j = i + 1
                while j < len(word_tags):
                    next_word, next_tag = word_tags[j]
                    if next_tag.startswith("I-") and next_tag[2:] == etype:
                        entity_words.append(next_word)
                        j += 1
                    else:
                        break
                entity_text = " ".join(entity_words)
                type_vi = NER_TYPE_VI.get(etype, etype)
                highlighted.append((entity_text + " ", type_vi))
                i = j

        return {
            "entities": entities,
            "word_tags": word_tags,
            "highlighted_text": highlighted,
        }
