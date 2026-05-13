"""
preprocessor.py — Tiền xử lý văn bản tiếng Việt cho KEHN inference.

Pipeline: Raw text → Word Segmentation (underthesea) → ViHealthBERT Tokenize → Tensors
Trả kèm word_offsets để postprocessor align subword predictions về word-level.
"""

import torch
from underthesea import word_tokenize
from transformers import AutoTokenizer


class Preprocessor:
    """
    Tiền xử lý câu hỏi y tế tiếng Việt → tensors cho KEHN model.

    Tokenization strategy giống data_loader_joint.py:
    - Tokenize từng word riêng (sau word segmentation)
    - Track vị trí first subword của mỗi word (word_offsets)
    """

    def __init__(self, tokenizer_name: str, max_seq_len: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len

    def preprocess(self, text: str, device: torch.device = None) -> dict:
        """
        Xử lý raw text → dict tensors + metadata.

        Args:
            text: Câu hỏi y tế tiếng Việt (raw, chưa word segment)
            device: torch device (CPU/CUDA)

        Returns:
            dict:
                input_ids      : [1, L] long
                attention_mask : [1, L] long
                words          : list[str] — danh sách từ sau word segmentation
                word_offsets   : list[int] — vị trí token (0-indexed) của first
                                 subword cho mỗi word (dùng để align NER predictions)
        """
        if device is None:
            device = torch.device("cpu")

        # ── Bước 1: Word Segmentation ──────────────────────────────────
        words = word_tokenize(text)
        # Loại bỏ tokens rỗng (edge case)
        words = [w for w in words if w.strip()]

        # ── Bước 2: Tokenize word-by-word (giống data_loader_joint.py) ─
        input_ids = [self.tokenizer.cls_token_id]
        word_offsets = []  # Vị trí trong input_ids của first subword mỗi word

        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
            w_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)

            # Track vị trí first subword (trước khi thêm vào input_ids)
            word_offsets.append(len(input_ids))

            input_ids.extend(w_ids)

        # Thêm SEP token
        input_ids.append(self.tokenizer.sep_token_id)

        # ── Bước 3: Truncation ─────────────────────────────────────────
        # Xử lý câu dài hơn max_length
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len - 1] + [self.tokenizer.sep_token_id]
            # Cắt word_offsets tương ứng (chỉ giữ words có first subword trong range)
            word_offsets = [off for off in word_offsets if off < self.max_seq_len - 1]
            # Cắt words tương ứng
            words = words[: len(word_offsets)]

        attention_mask = [1] * len(input_ids)

        # ── Bước 4: Convert to tensors (batch_size=1) ─────────────────
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "words": words,
            "word_offsets": word_offsets,
        }
