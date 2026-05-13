"""
model_loader.py — Load checkpoint KEHN và expose hàm predict().

Xử lý:
- Import KEHN class từ package gốc (qua sys.path)
- Instantiate model với đúng hyperparameters
- Load state_dict từ file .pt
- Auto device detection (CUDA/CPU)
- predict(text) → dict kết quả 3 tác vụ
"""

import sys
from pathlib import Path

import torch

# ── Setup sys.path để import KEHN package ─────────────────────────────
# demo/ nằm trong kehn_joint/ → cần thêm parent của kehn_joint vào path
_DEMO_DIR = Path(__file__).resolve().parent          # kehn_joint/demo/
_KEHN_DIR = _DEMO_DIR.parent                          # kehn_joint/
_PROJECT_ROOT = _KEHN_DIR.parent                      # Chatbot-Y tế/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kehn_joint.model.kehn_model import KEHN
from kehn_joint.config_joint import MODEL_CONFIG

from preprocessor import Preprocessor
from postprocessor import Postprocessor

# ── Default paths ─────────────────────────────────────────────────────
# TODO: điều chỉnh path file .pt tại đây
DEFAULT_CHECKPOINT = _KEHN_DIR / "model_outputs" / "best_model_vihealthbert.pt"
DEFAULT_BACKBONE = MODEL_CONFIG["vihealthbert"]  # "demdecuong/vihealthbert-base-word"


class KEHNPredictor:
    """
    Wrapper inference cho KEHN model.

    Usage:
        predictor = KEHNPredictor()
        result = predictor.predict("Tôi bị đau đầu và sốt cao")
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        backbone_name: str = None,
        device: str = None,
    ):
        """
        Khởi tạo predictor.

        Args:
            checkpoint_path: Path đến file .pt (default: model_outputs/best_model_vihealthbert.pt)
            backbone_name:   HuggingFace model ID cho backbone (default: vihealthbert)
            device:          "cuda" / "cpu" / None (auto-detect)
        """
        # ── Device detection ──────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Paths ─────────────────────────────────────────────────────
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.backbone_name = backbone_name or DEFAULT_BACKBONE

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint không tìm thấy: {self.checkpoint_path}\n"
                f"Hãy đặt file .pt vào: {DEFAULT_CHECKPOINT}"
            )

        # ── Build model ───────────────────────────────────────────────
        print(f"🤖 Đang khởi tạo KEHN model...")
        print(f"   Backbone: {self.backbone_name}")
        print(f"   Device: {self.device}")

        self.model = KEHN(
            backbone_name=self.backbone_name,
            n_topic=MODEL_CONFIG["n_topic"],
            n_intent=MODEL_CONFIG["n_intent"],
            n_ner_tag=MODEL_CONFIG["n_ner_tag"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            num_co_blocks=MODEL_CONFIG["num_co_interactive_blocks"],
            dropout=MODEL_CONFIG["hidden_dropout"],
            use_bilstm=MODEL_CONFIG["use_bilstm"],
            topic_class_weights=None,  # Không cần class weights cho inference
        )

        # ── Load checkpoint ───────────────────────────────────────────
        print(f"   📦 Loading checkpoint: {self.checkpoint_path.name}")
        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        # Loại bỏ key "topic_weights" nếu có (training-only buffer)
        if "topic_weights" in state_dict:
            del state_dict["topic_weights"]

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"   ✅ Model loaded thành công!")

        # ── Preprocessor & Postprocessor ──────────────────────────────
        self.preprocessor = Preprocessor(
            tokenizer_name=self.backbone_name,
            max_seq_len=128,
        )
        self.postprocessor = Postprocessor()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Inference pipeline: text → 3-task predictions.

        Args:
            text: Câu hỏi y tế tiếng Việt (raw, chưa word segment)

        Returns:
            dict: {
                "input_text": str,
                "segmented_words": list[str],
                "topic": {label_en, label_vi, confidence, all_probs},
                "intent": {label_en, label_vi, confidence, all_probs},
                "ner": {entities, word_tags, highlighted_text},
            }
        """
        # Bước 1: Tiền xử lý
        preprocessed = self.preprocessor.preprocess(text, device=self.device)

        input_ids = preprocessed["input_ids"]            # [1, L]
        attention_mask = preprocessed["attention_mask"]   # [1, L]
        words = preprocessed["words"]                     # list[str]
        word_offsets = preprocessed["word_offsets"]        # list[int]

        # Bước 2: Forward pass (public API, phase="full", không labels)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            phase="full",
        )

        # Bước 3: Hậu xử lý Topic & Intent
        topic_result = self.postprocessor.process_topic(output["logits_topic"])
        intent_result = self.postprocessor.process_intent(output["logits_intent"])

        # Bước 4: NER — CRF Viterbi decode + subword→word alignment
        ner_pred_ids = self.model.predict_ner(
            output["logits_ner"],
            attention_mask,
        )[0]  # [0] vì batch_size=1, trả về list of lists

        ner_result = self.postprocessor.process_ner(
            ner_pred_ids=ner_pred_ids,
            words=words,
            word_offsets=word_offsets,
        )

        return {
            "input_text": text,
            "segmented_words": words,
            "topic": topic_result,
            "intent": intent_result,
            "ner": ner_result,
        }
