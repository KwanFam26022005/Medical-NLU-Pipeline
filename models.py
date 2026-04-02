"""
models.py - Các class Inference cho Medical NLU Pipeline.
Bao gồm: AcronymResolver (Trạm 1), MedicalNER (2A), TopicClassifier (2B), IntentClassifier (2C).
Tất cả kế thừa từ BaseNLUModel để chuẩn hóa interface.
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from config import (
    ACRONYM_MODEL_DIR,
    ACRONYM_MODEL_NAME,
    INTENT_ID2LABEL,
    INTENT_LABELS,
    INTENT_MODEL_DIR,
    INTENT_MODEL_NAME,
    INTENT_NUM_LABELS,
    MAX_INPUT_LENGTH,
    NER_ID2LABEL,
    NER_LABELS,
    NER_MODEL_DIR,
    NER_MODEL_NAME,
    TOPIC_MODEL_DIR,
    TOPIC_MODEL_NAME,
)


# ============================================================
# 🧩 BASE CLASS - Interface chung cho tất cả NLU Models
# ============================================================

class BaseNLUModel(ABC):
    """
    Abstract base class cho các NLU model trong pipeline.
    Định nghĩa interface chuẩn: load model, predict, async predict.
    """

    def __init__(self, model_dir: Path, device: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights và tokenizer."""
        pass

    @abstractmethod
    def predict(self, text: str) -> Any:
        """Chạy inference synchronous."""
        pass

    async def async_predict(self, text: str) -> Any:
        """
        Chạy inference bất đồng bộ.
        Dùng run_in_executor để không block event loop khi chạy model inference.
        """
        # Fix #6: get_running_loop() thay vì get_event_loop() (deprecated Python 3.10+)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, text)

    def ensure_loaded(self) -> None:
        """Đảm bảo model đã được load trước khi predict."""
        if not self._is_loaded:
            self.load_model()
            self._is_loaded = True


# ============================================================
# 📍 TRẠM 1: ACRONYM CROSS-ENCODER (Pairwise Binary Scorer)
# ============================================================

class AcronymCrossEncoder(BaseNLUModel):
    """
    Cross-Encoder cho Acronym Disambiguation.

    Kiến trúc:
      Input: "[CLS] context_with_<e>acronym</e> [SEP] candidate_expansion [SEP]"
      Output: scalar logit → BCEWithLogitsLoss

    Tại inference: score tất cả candidates từ dictionary → argmax.
    """

    ENTITY_START = "<e>"
    ENTITY_END = "</e>"
    SPECIAL_TOKENS = ["<e>", "</e>"]

    def __init__(
        self,
        model_dir: Path = ACRONYM_MODEL_DIR,
        model_name: str = ACRONYM_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_dir, device)
        self.model_name = model_name
        self.acronym_dict: Dict[str, List[str]] = {}

    def load_model(self) -> None:
        """Load trained Cross-Encoder model + tokenizer + dictionary."""
        is_local = self.model_dir.exists()
        model_name_or_path = str(self.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=1
        )
        self.model.to(self.device)
        self.model.eval()

        # Load dictionary
        if is_local:
            dict_path = self.model_dir / "acronym_dict.json"
        else:
            try:
                from huggingface_hub import hf_hub_download
                dict_path = hf_hub_download(repo_id=model_name_or_path, filename="acronym_dict.json")
            except ImportError:
                print("⚠️ huggingface_hub not installed. Cannot download dictionary.")
                dict_path = None
            except Exception as e:
                print(f"⚠️ Failed to download acronym_dict.json: {e}")
                dict_path = None

        if dict_path:
            with open(dict_path, "r", encoding="utf-8") as f:
                self.acronym_dict = json.load(f)
            print(f"✅ AcronymCrossEncoder loaded: {len(self.acronym_dict)} acronyms")
        else:
            print(f"⚠️ No acronym_dict.json found for {model_name_or_path}")

        self._is_loaded = True

    @torch.no_grad()
    def score_candidates(
        self, marked_text: str, candidates: List[str]
    ) -> List[float]:
        """
        Score tất cả candidates cho một marked context.

        Args:
            marked_text: Context with <e>acronym</e> markers.
            candidates: List of expansion strings.

        Returns:
            List of logit scores (higher = more likely).
        """
        self.ensure_loaded()
        encodings = self.tokenizer(
            [marked_text] * len(candidates),
            candidates,
            max_length=128,
            padding=True,
            truncation="only_first",
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encodings).logits.squeeze(-1)
        return logits.cpu().tolist()

    def predict_from_raw(
        self, text: str, start_char_idx: int, length_acronym: int
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict expansion given raw text + acronym position.

        Returns:
            (best_expansion, confidence, ranked_list)
        """
        acronym = text[start_char_idx: start_char_idx + length_acronym]
        marked_text = (
            text[:start_char_idx]
            + self.ENTITY_START + acronym + self.ENTITY_END
            + text[start_char_idx + length_acronym:]
        )

        candidates = self.acronym_dict.get(acronym, [])
        if not candidates:
            return acronym, 0.0, [(acronym, 1.0)]
        if len(candidates) == 1:
            return candidates[0], 1.0, [(candidates[0], 1.0)]

        scores = self.score_candidates(marked_text, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        best_expansion = ranked[0][0]
        # Confidence = sigmoid(best) - sigmoid(second_best)
        import torch
        probs = [torch.sigmoid(torch.tensor(s)).item() for s in scores]
        probs_sorted = sorted(probs, reverse=True)
        confidence = probs_sorted[0] - probs_sorted[1] if len(probs_sorted) > 1 else 1.0

        return best_expansion, confidence, ranked

    def predict(self, text: str) -> str:
        """
        Auto-detect và giải tất cả từ viết tắt trong text.
        Returns: Clean text với các từ viết tắt đã được thay thế.
        """
        self.ensure_loaded()

        if not self.acronym_dict:
            return text

        # Detect acronyms using dictionary keys
        found = []
        for acronym in self.acronym_dict:
            pattern = re.compile(r'(?<!\w)' + re.escape(acronym) + r'(?!\w)')
            for match in pattern.finditer(text):
                found.append((acronym, match.start(), match.end()))

        if not found:
            return text

        # Sort by position descending so replacement doesn't shift indices
        found.sort(key=lambda x: x[1], reverse=True)

        clean_text = text
        for acronym, start, end in found:
            expansion, conf, _ = self.predict_from_raw(text, start, len(acronym))
            if conf > 0.1:  # minimal threshold
                clean_text = clean_text[:start] + expansion + clean_text[end:]

        return clean_text










# ============================================================
# 📍 TRẠM 2A: MEDICAL NER (Named Entity Recognition)
# ============================================================

import os
import torch
from transformers import AutoTokenizer
from custom_models import ViHealthBertCRF
from config import NER_MODEL_DIR, NER_MODEL_NAME, NER_ID2LABEL

class MedicalNER:
    def __init__(self, model_dir=NER_MODEL_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MedicalNER] Đang load tokenizer từ: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        print(f"[MedicalNER] Đang khởi tạo ViHealthBERT + CRF...")
        self.model = ViHealthBertCRF(model_name=NER_MODEL_NAME, num_labels=len(NER_ID2LABEL))
        
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy trọng số model tại {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.id2label = NER_ID2LABEL

    def predict(self, text: str):
        # Tách từ thô (giả định input đã được word-segment bằng underthesea)
        words = text.split()
        inputs = self.tokenizer(
            words, 
            is_split_into_words=True, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        )
        
        # Chuyển dữ liệu lên GPU (nếu có)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            # Vì không truyền labels, model sẽ tự động gọi self.crf.decode() và trả về danh sách dự đoán
            predictions = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy kết quả của câu đầu tiên trong batch
        pred_ids = predictions[0]
        word_ids = inputs.word_ids()
        
        # Lọc bỏ các sub-tokens (chỉ lấy nhãn của mảnh từ đầu tiên)
        result = []
        current_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != current_word_idx:
                label = self.id2label.get(pred_ids[idx], "O")
                result.append({"word": words[word_idx], "label": label})
                current_word_idx = word_idx
                
        return result


# ============================================================
# 📍 TRẠM 2B: TOPIC CLASSIFIER (PENDING DATA - SKELETON)
# ============================================================

class TopicClassifier(BaseNLUModel):
    """
    Phân loại Khoa y tế từ câu hỏi (vd: "gastroenterology", "cardiology").
    
    ⚠️ TRẠNG THÁI: CHỜ FINE-TUNE.
    Class này tạm load base model (dummy weights) để pipeline không bị gãy.
    Sẽ cập nhật khi dataset Topic sẵn sàng.
    """

    def __init__(
        self,
        model_dir: Path = TOPIC_MODEL_DIR,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_dir, device)
        self.id2label: Dict[int, str] = {}
        self._is_ready = False  # Flag: model đã fine-tune chưa

    def load_model(self) -> None:
        """Load model. Nếu chưa fine-tune, load base model với dummy head."""
        label_map_path = self.model_dir / "label_mapping.json"

        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            print(f"[TopicClassifier] Loading fine-tuned model từ {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)
            )
            self._is_ready = True
        else:
            # ⚠️ DUMMY MODE: Load base model, output sẽ không có ý nghĩa
            print(f"[TopicClassifier] ⚠️ Chưa có fine-tuned model.")
            print(f"  -> DUMMY MODE: Load base model. Output chỉ mang tính placeholder.")
            self.tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_NAME)
            # Tạm đặt num_labels = 20 (placeholder, sẽ thay đổi theo data thực tế)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                TOPIC_MODEL_NAME,
                num_labels=20,
                ignore_mismatched_sizes=True,
            )
            self._is_ready = False

        # Load label mapping nếu có
        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.id2label = {int(k): v for k, v in raw.items()}

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Phân loại topic/khoa.
        Returns: {"topic": str, "confidence": float, "is_reliable": bool}
        """
        self.ensure_loaded()

        inputs = self.tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        predicted_id = torch.argmax(probs).item()
        confidence = probs[predicted_id].item()

        topic_name = self.id2label.get(predicted_id, f"topic_{predicted_id}")

        return {
            "topic": topic_name,
            "confidence": round(confidence, 4),
            "is_reliable": self._is_ready,  # False nếu đang dùng dummy model
        }


# ============================================================
# 📍 TRẠM 2C: INTENT CLASSIFIER (Multi-label)
# ============================================================

class IntentClassifier(BaseNLUModel):
    """
    Phân loại ý định câu hỏi y tế: Diagnosis, Treatment, Severity, Cause.
    
    ⚠️ MULTI-LABEL: Một câu có thể có nhiều intent đồng thời.
    Sử dụng sigmoid threshold thay vì argmax.
    """

    def __init__(
        self,
        model_dir: Path = INTENT_MODEL_DIR,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_dir, device)
        self.threshold = threshold
        self.id2label = INTENT_ID2LABEL

    def load_model(self) -> None:
        """Load fine-tuned intent model hoặc base model."""
        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            print(f"[IntentClassifier] Loading fine-tuned model từ {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)
            )
        else:
            print(f"[IntentClassifier] ⚠️ Không tìm thấy fine-tuned model.")
            print(f"  -> Fallback: Load base model {INTENT_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                INTENT_MODEL_NAME,
                num_labels=INTENT_NUM_LABELS,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
            )

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Dự đoán intent(s) của câu hỏi y tế.
        
        Logic Multi-label:
        - Dùng sigmoid (không phải softmax) vì các intent KHÔNG loại trừ nhau.
        - Áp dụng threshold: intent nào có prob >= threshold thì được chọn.
        - VD: "Đau bụng có phải viêm ruột thừa không, cần mổ không?"
          -> Diagnosis (0.85) ✓, Treatment (0.72) ✓, Severity (0.30) ✗
        
        Returns: {"intents": [...], "scores": {...}, "primary_intent": str}
        """
        self.ensure_loaded()

        inputs = self.tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        # Sigmoid cho multi-label (KHÔNG DÙNG softmax)
        probs = torch.sigmoid(outputs.logits)[0]

        # Lọc intent vượt threshold
        predicted_intents: List[str] = []
        scores: Dict[str, float] = {}

        for idx, prob in enumerate(probs):
            label = self.id2label.get(idx, f"intent_{idx}")
            score = round(prob.item(), 4)
            scores[label] = score
            if score >= self.threshold:
                predicted_intents.append(label)

        # Intent chính = intent có score cao nhất
        primary_intent = max(scores, key=scores.get) if scores else "Unknown"

        # Nếu không có intent nào vượt threshold -> lấy intent cao nhất
        if not predicted_intents:
            predicted_intents = [primary_intent]

        return {
            "intents": predicted_intents,
            "scores": scores,
            "primary_intent": primary_intent,
        }
