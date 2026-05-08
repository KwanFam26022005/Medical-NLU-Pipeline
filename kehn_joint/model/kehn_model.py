"""
kehn_model.py — Knowledge-Enhanced Hierarchical Network (KEHN).

Kiến trúc 3 tầng kết hợp OneNet + Stack-Propagation + Co-Interactive Transformer:
  Tầng 1: PhoBERT/ViHealthBERT Encoder + BiLSTM
  Tầng 2: Feature Mining — Token-level Intent ↔ NER (Co-Interactive cross-attention)
  Tầng 3: Topic Decoder — Stack-Propagation (concat H_ctx + P(Intent) + P(NER))

CHANGES vs v1:
  [+TR]  CRF Transition Constraints: illegal BIO transitions set to −1e9 at init,
         re-clamped after each optimizer step via constrain_crf_transitions().
  [+CWL] Confidence-Weighted NER Loss: mỗi sample's CRF loss nhân với
         ner_confidence của sample đó (bảo vệ shared encoder khỏi noisy labels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchcrf import CRF

from .co_interactive import LabelAttention, CoInteractiveBlock
from ..config_joint import NER_TAGS, NER2ID, NER_ENTITY_TYPES


# ──────────────────────────────────────────────────────────────────────
# Transition constraint helpers
# ──────────────────────────────────────────────────────────────────────

def build_bio_constraint_masks(ner_tags: list, entity_types: list):
    """
    Xây dựng 2 boolean mask cho CRF transitions hợp lệ theo BIO scheme.

    Returns:
        illegal_start  : BoolTensor [n_tags]   — True = illegal start tag
        illegal_trans  : BoolTensor [n_tags, n_tags] — True = illegal [from→to]

    Luật BIO:
    (1) Sequence không được BẮT ĐẦU bằng I-X
    (2) O     → I-X  là illegal (phải dùng B-X)
    (3) B-X   → I-Y  là illegal nếu X ≠ Y
    (4) I-X   → I-Y  là illegal nếu X ≠ Y

    Ví dụ với entity_type SYM (B-SYM=1, I-SYM=2):
        illegal_start[2] = True                (I-SYM không thể bắt đầu)
        illegal_trans[0, 2] = True             (O → I-SYM)
        illegal_trans[3, 2] = True             (B-PRO → I-SYM)
        illegal_trans[4, 2] = True             (I-PRO → I-SYM)
        ...
    """
    n = len(ner_tags)
    tag2id = {t: i for i, t in enumerate(ner_tags)}

    # Identify I-* indices and their paired B-* index
    i_tag_pairs = {}   # i_idx → b_idx (may be None if B-X missing)
    for etype in entity_types:
        b_key = f"B-{etype}"
        i_key = f"I-{etype}"
        if i_key in tag2id:
            b_idx = tag2id.get(b_key)  # None if B-X missing
            i_tag_pairs[tag2id[i_key]] = b_idx

    # ── (1) Start constraint ──────────────────────────────────
    illegal_start = torch.zeros(n, dtype=torch.bool)
    for i_idx in i_tag_pairs:
        illegal_start[i_idx] = True

    # ── (2-4) Transition constraints ─────────────────────────
    illegal_trans = torch.zeros(n, n, dtype=torch.bool)

    for i_idx, b_idx in i_tag_pairs.items():
        # Tập "prev tags" hợp lệ cho I-X:
        #   - B-X  (khai đầu entity X)
        #   - I-X  (tiếp tục entity X)
        valid_prev = set()
        if b_idx is not None:
            valid_prev.add(b_idx)
        valid_prev.add(i_idx)  # I-X → I-X (continuation)

        # Mọi prev tag NGOÀI valid_prev → I-X đều illegal
        for from_idx in range(n):
            if from_idx not in valid_prev:
                illegal_trans[from_idx, i_idx] = True

    return illegal_start, illegal_trans


# ──────────────────────────────────────────────────────────────────────


class KEHN(nn.Module):
    """
    Knowledge-Enhanced Hierarchical Network.

    Args:
        backbone_name      : HuggingFace model ID (PhoBERT hoặc ViHealthBERT)
        n_topic            : Số lớp Topic classification (17)
        n_intent           : Số lớp Intent detection (4)
        n_ner_tag          : Số BIO tags cho NER (11, gồm I-SEV)
        hidden_dim         : Hidden dimension (768 cho base, 1024 cho large)
        num_co_blocks      : Số Co-Interactive blocks xếp chồng (2)
        dropout            : Dropout rate (0.1)
        use_bilstm         : Có dùng BiLSTM trên encoder output không
        topic_class_weights: Tensor trọng số cho topic CE loss (xử lý imbalance)
    """

    def __init__(
        self,
        backbone_name: str,
        n_topic: int = 17,
        n_intent: int = 4,
        n_ner_tag: int = 11,
        hidden_dim: int = 768,
        num_co_blocks: int = 2,
        dropout: float = 0.1,
        use_bilstm: bool = True,
        topic_class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.n_topic   = n_topic
        self.n_intent  = n_intent
        self.n_ner_tag = n_ner_tag
        self.hidden_dim = hidden_dim
        self.use_bilstm = use_bilstm

        # ── Tầng 1: Shared Encoder ──────────────────────────────────
        self.encoder = AutoModel.from_pretrained(backbone_name)
        if use_bilstm:
            self.bilstm = nn.LSTM(
                hidden_dim, hidden_dim // 2,
                bidirectional=True, batch_first=True, dropout=dropout,
            )
        self.enc_dropout = nn.Dropout(dropout)

        # ── Tầng 2: Feature Mining (Co-Interactive) ─────────────────
        self.intent_fc = nn.Linear(hidden_dim, n_intent)
        self.ner_fc    = nn.Linear(hidden_dim, n_ner_tag)

        self.label_attn = LabelAttention(self.intent_fc, self.ner_fc)
        self.co_blocks  = nn.ModuleList([
            CoInteractiveBlock(self.intent_fc, self.ner_fc, hidden_dim, dropout)
            for _ in range(num_co_blocks)
        ])

        # CRF for NER sequence decoding
        self.crf = CRF(n_ner_tag, batch_first=True)

        # ── [+TR] CRF Transition Constraints ────────────────────────
        # Xây dựng illegal transition mask từ NER_TAGS và ENTITY_TYPES
        # rồi initialize các transition này về -1e9.
        # Sau mỗi optimizer.step(), gọi constrain_crf_transitions()
        # để re-clamp về -1e9 (gradient không được phép "sửa" chúng).
        illegal_start, illegal_trans = build_bio_constraint_masks(
            NER_TAGS, NER_ENTITY_TYPES
        )
        # Register as non-trainable buffers (move to device cùng model)
        self.register_buffer("_crf_illegal_start", illegal_start)
        self.register_buffer("_crf_illegal_trans",  illegal_trans)
        # Apply initial constraints
        self._apply_crf_constraints()

        # ── Tầng 3: Topic Decoder (Stack-Propagation) ────────────────
        topic_input_dim = hidden_dim + n_intent + n_ner_tag
        self.topic_classifier = nn.Sequential(
            nn.Linear(topic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_topic),
        )

        # Loss functions
        if topic_class_weights is not None:
            self.register_buffer("topic_weights", topic_class_weights)
        else:
            self.topic_weights = None

    # ── CRF Constraint API ─────────────────────────────────────────────

    def _apply_crf_constraints(self):
        """
        Gán −1e9 vào các ô transition bất hợp lệ (không qua gradient).
        Gọi một lần khi init, rồi gọi constrain_crf_transitions()
        sau mỗi optimizer.step() trong vòng huấn luyện.
        """
        with torch.no_grad():
            self.crf.start_transitions.data.masked_fill_(
                self._crf_illegal_start, -1e9
            )
            self.crf.transitions.data.masked_fill_(
                self._crf_illegal_trans, -1e9
            )

    def constrain_crf_transitions(self):
        """
        [+TR] Public API — gọi sau MỖII optimizer.step() trong train loop.

        Vì CRF transitions là nn.Parameter, optimizer sẽ cập nhật cả
        các ô illegal. Hàm này re-clamp chúng về −1e9 sau mỗi bước,
        đảm bảo BIO constraints được duy trì xuyên suốt quá trình train.

        Usage trong train_joint.py:
            scaler.step(optimizer)
            scaler.update()
            model.constrain_crf_transitions()   # ← thêm dòng này
        """
        self._apply_crf_constraints()

    def get_illegal_transition_stats(self) -> dict:
        """
        Debug utility: kiểm tra số lượng và giá trị max của illegal transitions.
        Gọi trong evaluate() để đảm bảo constraints vẫn đang hoạt động.
        """
        with torch.no_grad():
            start_vals = self.crf.start_transitions[self._crf_illegal_start]
            trans_vals = self.crf.transitions[self._crf_illegal_trans]
            return {
                "n_illegal_start": self._crf_illegal_start.sum().item(),
                "n_illegal_trans": self._crf_illegal_trans.sum().item(),
                "max_illegal_start_val": start_vals.max().item() if start_vals.numel() else float("-inf"),
                "max_illegal_trans_val": trans_vals.max().item() if trans_vals.numel() else float("-inf"),
            }

    # ── Forward Tầng 1 ─────────────────────────────────────────────────

    def _encode(self, input_ids, attention_mask):
        """Tầng 1: Encode input qua PhoBERT + optional BiLSTM."""
        H = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        if self.use_bilstm:
            H, _ = self.bilstm(H)
        H = self.enc_dropout(H)
        return H  # [B, L, hidden_dim]

    # ── Forward Tầng 2 ─────────────────────────────────────────────────

    def _feature_mining(self, H, mask):
        """
        Tầng 2: Co-Interactive Feature Mining.
        Intent ↔ NER tương tác bidirectional qua Label Attention + CoInteractive Blocks.
        """
        H_I, H_N = self.label_attn(H, H, mask)

        for i, block in enumerate(self.co_blocks):
            if i == 0:
                H_I, H_N = block(H_I + H, H_N + H, mask)
            else:
                H_I_new, H_N_new = self.label_attn(H_I, H_N, mask)
                H_I, H_N = block(H_I + H_I_new, H_N + H_N_new, mask)

        # Token-level Intent logits
        logits_intent_token = self.intent_fc(H_I + H)  # [B, L, n_intent]

        # Sentence-level Intent (masked mean voting)
        mask_expanded = mask.unsqueeze(-1).float()
        intent_probs_masked = F.softmax(logits_intent_token, dim=-1) * mask_expanded
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1)
        intent_probs_sentence = intent_probs_masked.sum(dim=1) / mask_sum  # [B, n_intent]

        # NER logits & pooled probs
        logits_ner = self.ner_fc(H_N + H)  # [B, L, n_ner_tag]
        ner_probs  = F.softmax(logits_ner, dim=-1)
        ner_probs_masked = ner_probs.masked_fill(mask_expanded == 0, float('-inf'))
        ner_probs_pooled = ner_probs_masked.max(dim=1)[0]  # [B, n_ner_tag]

        return logits_intent_token, intent_probs_sentence, logits_ner, ner_probs_pooled

    # ── Forward Tầng 3 ─────────────────────────────────────────────────

    def _topic_decode(self, H, intent_probs, ner_probs, mask, use_stack_prop=True):
        """Tầng 3: Topic Classification với Stack-Propagation."""
        mask_expanded = mask.unsqueeze(-1).float()
        H_masked = H * mask_expanded
        H_pool = F.max_pool1d(
            H_masked.transpose(1, 2), kernel_size=H.size(1)
        ).squeeze(2)  # [B, hidden_dim]

        if use_stack_prop:
            topic_input = torch.cat([
                H_pool,
                intent_probs.detach(),
                ner_probs.detach(),
            ], dim=-1)
        else:
            zeros_intent = torch.zeros_like(intent_probs)
            zeros_ner    = torch.zeros_like(ner_probs)
            topic_input  = torch.cat([H_pool, zeros_intent, zeros_ner], dim=-1)

        return self.topic_classifier(topic_input)  # [B, n_topic]

    # ── Main Forward ───────────────────────────────────────────────────

    def forward(
        self,
        input_ids,
        attention_mask,
        topic_labels=None,
        intent_labels=None,
        ner_labels=None,
        phase="full",
        token_intent_ids=None,
        ner_confidence=None,
    ):
        """
        Forward pass qua 3 tầng.

        Args:
            input_ids        : [B, L]
            attention_mask   : [B, L]
            topic_labels     : [B] long — Topic class indices
            intent_labels    : [B] long — Intent class indices
            ner_labels       : [B, L] long — NER BIO tag indices (-100 = ignored)
            phase            : "topic_only" | "mining_only" | "joint_no_prop" | "full"
            token_intent_ids : [B, L] long — Token-level intent tags (-100 = ignored)
            ner_confidence   : [B] float — Per-sample NER confidence (0-1)
                               NEW for Confidence-Weighted Loss [+CWL]

        Returns:
            dict with logits_topic, logits_intent, logits_ner, loss (if labels given)
        """
        H = self._encode(input_ids, attention_mask)

        logits_intent_token, intent_probs, logits_ner, ner_probs = \
            self._feature_mining(H, attention_mask)

        use_stack_prop = (phase == "full")
        logits_topic   = self._topic_decode(H, intent_probs, ner_probs,
                                            attention_mask, use_stack_prop)

        output = {
            "logits_topic":        logits_topic,
            "logits_intent":       intent_probs,       # sentence-level
            "logits_intent_token": logits_intent_token,
            "logits_ner":          logits_ner,
        }

        if topic_labels is not None or intent_labels is not None or ner_labels is not None:
            loss = self._compute_loss(
                logits_topic, logits_intent_token, logits_ner,
                topic_labels, intent_labels, ner_labels,
                attention_mask, phase,
                token_intent_ids=token_intent_ids,
                ner_confidence=ner_confidence,
            )
            output["loss"] = loss

        return output

    # ── Loss Computation ───────────────────────────────────────────────

    def _compute_loss(
        self,
        logits_topic, logits_intent_token, logits_ner,
        topic_labels, intent_labels, ner_labels,
        mask, phase,
        token_intent_ids=None,
        ner_confidence=None,
    ):
        """
        Compute joint loss theo curriculum phase.

        [+CWL] Confidence-Weighted NER Loss:
            Loss_NER = mean(ner_confidence × NLL_CRF_per_sample)

        Lý do dùng confidence weighting:
        - Dataset `hospital` có nhiều pseudo-labels (avg_ner_confidence=0.89-0.95)
        - Samples với confidence thấp hơn (~0.8) có khả năng nhãn sai cao hơn
        - Weighting giảm gradient từ noisy samples → bảo vệ shared encoder
        - Kết quả: encoder học pattern thực sự thay vì overfit noise
        """
        from .._get_loss_weights import get_loss_weights
        weights = get_loss_weights(phase)

        total_loss = torch.tensor(0.0, device=logits_topic.device)

        # ── Topic Loss (Weighted CrossEntropy) ──────────────────
        if weights["topic"] > 0 and topic_labels is not None:
            loss_fn = (
                nn.CrossEntropyLoss(weight=self.topic_weights)
                if self.topic_weights is not None
                else nn.CrossEntropyLoss()
            )
            loss_topic = loss_fn(logits_topic, topic_labels)
            total_loss = total_loss + weights["topic"] * loss_topic

        # ── Intent Loss (Token-level CE) ─────────────────────────
        if weights["intent"] > 0 and intent_labels is not None:
            B, L, C = logits_intent_token.shape
            intent_expanded = (
                token_intent_ids
                if token_intent_ids is not None
                else intent_labels.unsqueeze(1).expand(B, L)
            )
            logits_flat = logits_intent_token.reshape(-1, C)
            labels_flat = intent_expanded.reshape(-1)
            mask_flat   = mask.reshape(-1).bool()

            if mask_flat.any():
                loss_intent = F.cross_entropy(
                    logits_flat[mask_flat], labels_flat[mask_flat],
                    ignore_index=-100
                )
                total_loss = total_loss + weights["intent"] * loss_intent

        # ── NER Loss [+CWL] (CRF NLL, Confidence-Weighted) ──────
        if weights["ner"] > 0 and ner_labels is not None:
            ner_labels_crf = ner_labels.clone()
            ner_labels_crf[ner_labels_crf == -100] = 0

            # ✅ FIX: Dùng first-subword mask thay vì attention_mask.
            # ner_labels != -100 chỉ True tại: vị trí first sub-token của mỗi word.
            # CLS, SEP, sub-tokens đều bị loại → không còn O→I-X illegal transitions.
            first_token_mask = (ner_labels != -100) & mask.bool()  # ← THAY ĐỔI

            loss_ner_sum = -self.crf(
                logits_ner.float(),       # ← cast fp32 để tránh fp16 numerical issues trong CRF
                ner_labels_crf,
                mask=first_token_mask,    # ← THAY ĐỔI (cũ là mask.bool())
                reduction='none'
            )

            # Normalize theo số word-level tokens (không phải sub-tokens)
            n_valid = first_token_mask.float().sum(dim=1).clamp(min=1)  # ← cập nhật
            loss_ner_per_sample = loss_ner_sum / n_valid

            # [+CWL] Nhân với ner_confidence của từng sample
            if ner_confidence is not None:
                # ner_confidence: [B] float trong [0,1]
                # Clamp để tránh grad vanish khi confidence quá thấp
                conf = ner_confidence.to(logits_ner.device).clamp(min=0.05)
                loss_ner = (conf * loss_ner_per_sample).mean()
            else:
                # Fallback: unweighted mean (backward compatible)
                loss_ner = loss_ner_per_sample.mean()

            total_loss = total_loss + weights["ner"] * loss_ner

        return total_loss

    # ── Inference ──────────────────────────────────────────────────────

    def predict_ner(self, logits_ner, mask):
        """
        Decode NER predictions bằng CRF Viterbi.
        CRF transition constraints đã được áp dụng → output sẽ không
        chứa các sequence bất hợp lệ như I-SYM sau O.
        """
        return self.crf.decode(logits_ner, mask=mask.bool())