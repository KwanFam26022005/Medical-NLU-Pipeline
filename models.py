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
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), num_labels=1
        )
        self.model.to(self.device)
        self.model.eval()

        # Load dictionary
        dict_path = self.model_dir / "acronym_dict.json"
        if dict_path.exists():
            with open(dict_path, "r", encoding="utf-8") as f:
                self.acronym_dict = json.load(f)
            print(f"✅ AcronymCrossEncoder loaded: {len(self.acronym_dict)} acronyms")
        else:
            print(f"⚠️ No acronym_dict.json found in {self.model_dir}")

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

class MedicalNER(BaseNLUModel):
    """
    Trích xuất thực thể y tế: Triệu chứng/Bệnh, Thủ thuật, Thuốc.
    
    ⚠️ RÀNG BUỘC: vihealthbert-base-word yêu cầu word-segmentation.
    Class này tích hợp bước word-segment trước khi tokenize.
    """

    def __init__(
        self,
        model_dir: Path = NER_MODEL_DIR,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_dir, device)
        self.id2label = NER_ID2LABEL
        self._segmenter = None

    def _init_word_segmenter(self) -> None:
        """
        Khởi tạo word segmenter cho tiếng Việt.
        Ưu tiên py_vncorenlp, fallback sang underthesea, cuối cùng dùng regex cơ bản.
        """
        try:
            import py_vncorenlp
            # py_vncorenlp cần download model VnCoreNLP
            self._segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"],
                save_dir=str(Path.home() / "vncorenlp"),
            )
            print("[MedicalNER] Word segmenter: py_vncorenlp ✓")
            return
        except Exception as e:
            print(f"[MedicalNER] py_vncorenlp không khả dụng: {e}")

        try:
            from underthesea import word_tokenize
            self._segmenter = word_tokenize
            print("[MedicalNER] Word segmenter: underthesea ✓")
            return
        except ImportError:
            print("[MedicalNER] underthesea không khả dụng.")

        # Fallback: regex đơn giản (không segment, giữ nguyên text)
        print("[MedicalNER] ⚠️ Fallback: Không có word segmenter. Sử dụng text gốc.")
        self._segmenter = None

    def _word_segment(self, text: str) -> str:
        """
        Word-segmentation cho tiếng Việt.
        Output: text với các từ ghép nối bằng '_' (vd: "đau_dạ_dày")
        """
        if self._segmenter is None:
            return text

        # py_vncorenlp trả về list of sentences
        if hasattr(self._segmenter, "word_segment"):
            segmented = self._segmenter.word_segment(text)
            if isinstance(segmented, list):
                return " ".join(segmented)
            return segmented

        # underthesea.word_tokenize trả về string
        if callable(self._segmenter):
            return self._segmenter(text, format="text")

        return text

    def load_model(self) -> None:
        """Load fine-tuned NER model hoặc base model."""
        self._init_word_segmenter()

        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            print(f"[MedicalNER] Loading fine-tuned model từ {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(self.model_dir)
            )
        else:
            print(f"[MedicalNER] ⚠️ Không tìm thấy fine-tuned model.")
            print(f"  -> Fallback: Load base model {NER_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
            self.model = AutoModelForTokenClassification.from_pretrained(
                NER_MODEL_NAME,
                num_labels=len(NER_LABELS),
                id2label=NER_ID2LABEL,
                ignore_mismatched_sizes=True,
            )

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    @torch.no_grad()
    def predict(self, text: str) -> List[str]:
        """
        Trích xuất entities từ text.
        
        Luồng xử lý:
        1. Word-segment text ("đau dạ dày" -> "đau_dạ_dày")
        2. Tokenize bằng WordPiece
        3. Model predict nhãn BIO cho mỗi token
        4. Gom các token B-xxx + I-xxx thành entity hoàn chỉnh
        5. Trả về list entity strings
        """
        self.ensure_loaded()

        # Bước 1: Word-segmentation (BẮT BUỘC cho vihealthbert-base-word)
        segmented_text = self._word_segment(text)

        # Bước 2: Tokenize
        inputs = self.tokenizer(
            segmented_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(self.device)

        offset_mapping = inputs.pop("offset_mapping")[0]

        # Bước 3: Predict
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]

        # Bước 4: Decode - gom BIO tags thành entities
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities: List[str] = []
        current_entity_tokens: List[str] = []
        current_entity_type: Optional[str] = None

        for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
            # Bỏ qua special tokens
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                continue

            label = self.id2label.get(pred_id.item(), "O")

            if label.startswith("B-"):
                # Lưu entity trước đó (nếu có)
                if current_entity_tokens:
                    entity_text = self._reconstruct_entity(current_entity_tokens)
                    entities.append(entity_text)

                # Bắt đầu entity mới
                current_entity_type = label[2:]
                current_entity_tokens = [token]

            elif label.startswith("I-") and current_entity_type:
                # Tiếp tục entity hiện tại
                current_entity_tokens.append(token)

            else:
                # O tag -> kết thúc entity hiện tại
                if current_entity_tokens:
                    entity_text = self._reconstruct_entity(current_entity_tokens)
                    entities.append(entity_text)
                    current_entity_tokens = []
                    current_entity_type = None

        # Entity cuối cùng
        if current_entity_tokens:
            entity_text = self._reconstruct_entity(current_entity_tokens)
            entities.append(entity_text)

        return entities

    @staticmethod
    def _reconstruct_entity(tokens: List[str]) -> str:
        """
        Ghép lại các sub-tokens thành entity text đọc được.
        - Loại bỏ prefix '##' của WordPiece sub-tokens
        - Thay '_' (word-segmentation) bằng space
        """
        result = ""
        for token in tokens:
            if token.startswith("##"):
                result += token[2:]
            else:
                if result:
                    result += " "
                result += token

        # Thay dấu '_' (word-segmentation marker) bằng space
        result = result.replace("_", " ")
        return result.strip()


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
