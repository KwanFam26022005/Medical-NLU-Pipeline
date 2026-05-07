"""
kehn_model.py — Knowledge-Enhanced Hierarchical Network (KEHN).

Kiến trúc 3 tầng kết hợp OneNet + Stack-Propagation + Co-Interactive Transformer:
  Tầng 1: PhoBERT/ViHealthBERT Encoder + BiLSTM
  Tầng 2: Feature Mining — Token-level Intent ↔ NER (Co-Interactive cross-attention)
  Tầng 3: Topic Decoder — Stack-Propagation (concat H_ctx + P(Intent) + P(NER))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchcrf import CRF

from .co_interactive import LabelAttention, CoInteractiveBlock


class KEHN(nn.Module):
    """
    Knowledge-Enhanced Hierarchical Network.
    
    Args:
        backbone_name: HuggingFace model ID (PhoBERT hoặc ViHealthBERT)
        n_topic: Số lớp Topic classification (18)
        n_intent: Số lớp Intent detection (4)
        n_ner_tag: Số BIO tags cho NER (7)
        hidden_dim: Hidden dimension (768 cho base models)
        num_co_blocks: Số Co-Interactive blocks xếp chồng (2)
        dropout: Dropout rate (0.1)
        use_bilstm: Có dùng BiLSTM trên encoder output không
        topic_class_weights: Tensor trọng số cho topic CE loss (xử lý imbalance)
    """

    def __init__(
        self,
        backbone_name: str,
        n_topic: int = 17,
        n_intent: int = 4,
        n_ner_tag: int = 7,
        hidden_dim: int = 768,
        num_co_blocks: int = 2,
        dropout: float = 0.1,
        use_bilstm: bool = True,
        topic_class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.n_topic = n_topic
        self.n_intent = n_intent
        self.n_ner_tag = n_ner_tag
        self.hidden_dim = hidden_dim
        self.use_bilstm = use_bilstm

        # ── Tầng 1: Shared Encoder ──────────────────────────
        self.encoder = AutoModel.from_pretrained(backbone_name)
        if use_bilstm:
            self.bilstm = nn.LSTM(
                hidden_dim, hidden_dim // 2,
                bidirectional=True, batch_first=True, dropout=dropout,
            )
        self.enc_dropout = nn.Dropout(dropout)

        # ── Tầng 2: Feature Mining (Co-Interactive) ─────────
        # Task-specific classification heads (label embeddings come from weights)
        self.intent_fc = nn.Linear(hidden_dim, n_intent)
        self.ner_fc = nn.Linear(hidden_dim, n_ner_tag)

        # Co-Interactive modules
        self.label_attn = LabelAttention(self.intent_fc, self.ner_fc)
        self.co_blocks = nn.ModuleList([
            CoInteractiveBlock(self.intent_fc, self.ner_fc, hidden_dim, dropout)
            for _ in range(num_co_blocks)
        ])

        # CRF for NER sequence decoding
        self.crf = CRF(n_ner_tag, batch_first=True)

        # ── Tầng 3: Topic Decoder (Stack-Propagation) ───────
        # Input: H_pool (hidden_dim) + intent_probs (n_intent) + ner_probs (n_ner_tag)
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

    def _encode(self, input_ids, attention_mask):
        """Tầng 1: Encode input qua PhoBERT + optional BiLSTM."""
        H = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        if self.use_bilstm:
            H, _ = self.bilstm(H)
        H = self.enc_dropout(H)
        return H  # [B, L, hidden_dim]

    def _feature_mining(self, H, mask):
        """
        Tầng 2: Co-Interactive Feature Mining.
        
        Intent ↔ NER tương tác bidirectional qua Label Attention + CoInteractive Blocks.
        Token-level Intent prediction → sentence voting.
        """
        # Initial label attention: tạo task-specific representations
        H_I, H_N = self.label_attn(H, H, mask)

        # Stacked Co-Interactive blocks
        for i, block in enumerate(self.co_blocks):
            if i == 0:
                H_I, H_N = block(H_I + H, H_N + H, mask)
            else:
                # Re-apply label attention trước block tiếp (theo DCA-Net)
                H_I_new, H_N_new = self.label_attn(H_I, H_N, mask)
                H_I, H_N = block(H_I + H_I_new, H_N + H_N_new, mask)

        # Token-level Intent logits
        logits_intent_token = self.intent_fc(H_I + H)  # [B, L, n_intent]

        # Sentence-level Intent bằng mean voting (Stack-Propagation)
        # Mask padding tokens trước khi voting
        mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
        intent_probs_masked = F.softmax(logits_intent_token, dim=-1) * mask_expanded
        intent_probs_sum = intent_probs_masked.sum(dim=1)  # [B, n_intent]
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        intent_probs_sentence = intent_probs_sum / mask_sum  # [B, n_intent]

        # NER logits
        logits_ner = self.ner_fc(H_N + H)  # [B, L, n_ner_tag]

        # NER probs (pooled for Stack-Propagation)
        ner_probs = F.softmax(logits_ner, dim=-1)
        ner_probs_masked = ner_probs.masked_fill(mask_expanded == 0, float('-inf'))
        ner_probs_pooled = ner_probs_masked.max(dim=1)[0]  # [B, n_ner_tag]

        return logits_intent_token, intent_probs_sentence, logits_ner, ner_probs_pooled

    def _topic_decode(self, H, intent_probs, ner_probs, mask, use_stack_prop=True):
        """
        Tầng 3: Topic Classification với Stack-Propagation.
        
        Concat context vector + Intent probs + NER probs → Topic prediction.
        """
        # Max-pool encoder output → context vector
        # Mask padding trước pooling
        mask_expanded = mask.unsqueeze(-1).float()
        H_masked = H * mask_expanded
        H_pool = F.max_pool1d(
            H_masked.transpose(1, 2), kernel_size=H.size(1)
        ).squeeze(2)  # [B, hidden_dim]

        # Stack-Propagation: concat features từ tầng dưới
        if use_stack_prop:
            # Detach intent/ner probs để tránh gradient conflict
            topic_input = torch.cat([
                H_pool,
                intent_probs.detach(),
                ner_probs.detach(),
            ], dim=-1)  # [B, hidden_dim + n_intent + n_ner_tag]
        else:
            # Phase 1-3: chưa dùng Stack-Propagation
            zeros_intent = torch.zeros_like(intent_probs)
            zeros_ner = torch.zeros_like(ner_probs)
            topic_input = torch.cat([H_pool, zeros_intent, zeros_ner], dim=-1)

        logits_topic = self.topic_classifier(topic_input)  # [B, n_topic]
        return logits_topic

    def forward(
        self,
        input_ids,
        attention_mask,
        topic_labels=None,
        intent_labels=None,
        ner_labels=None,
        phase="full",
        token_intent_ids=None,
    ):
        """
        Forward pass qua 3 tầng.
        
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            topic_labels: [B] (long) — Topic class indices
            intent_labels: [B] (long) — Intent class indices
            ner_labels: [B, L] (long) — NER BIO tag indices (-100 for ignored)
            phase: "topic_only" | "mining_only" | "joint_no_prop" | "full"
            token_intent_ids: [B, L] (long) — Token-level intent tags (-100 for ignored)
        
        Returns:
            dict with logits_topic, logits_intent, logits_ner, loss (if labels provided)
        """
        # ── Tầng 1: Encode ──
        H = self._encode(input_ids, attention_mask)

        # ── Tầng 2: Feature Mining ──
        logits_intent_token, intent_probs, logits_ner, ner_probs = \
            self._feature_mining(H, attention_mask)

        # ── Tầng 3: Topic Decode ──
        use_stack_prop = (phase == "full")
        logits_topic = self._topic_decode(H, intent_probs, ner_probs, attention_mask, use_stack_prop)

        # ── Compute Loss ──
        output = {
            "logits_topic": logits_topic,
            "logits_intent": intent_probs,  # sentence-level
            "logits_intent_token": logits_intent_token,
            "logits_ner": logits_ner,
        }

        if topic_labels is not None or intent_labels is not None or ner_labels is not None:
            loss = self._compute_loss(
                logits_topic, logits_intent_token, logits_ner,
                topic_labels, intent_labels, ner_labels,
                attention_mask, phase,
                token_intent_ids=token_intent_ids,
            )
            output["loss"] = loss

        return output

    def _compute_loss(
        self, logits_topic, logits_intent_token, logits_ner,
        topic_labels, intent_labels, ner_labels, mask, phase,
        token_intent_ids=None,
    ):
        """Compute joint loss theo curriculum phase."""
        from .._get_loss_weights import get_loss_weights
        weights = get_loss_weights(phase)

        total_loss = torch.tensor(0.0, device=logits_topic.device)

        # Topic loss (Weighted CrossEntropy)
        if weights["topic"] > 0 and topic_labels is not None:
            if self.topic_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=self.topic_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss_topic = loss_fn(logits_topic, topic_labels)
            total_loss = total_loss + weights["topic"] * loss_topic

        # Intent loss (Token-level CE)
        if weights["intent"] > 0 and intent_labels is not None:
            B, L, C = logits_intent_token.shape
            if token_intent_ids is not None:
                intent_expanded = token_intent_ids
            else:
                # Broadcast sentence-level intent label to all tokens
                intent_expanded = intent_labels.unsqueeze(1).expand(B, L)  # [B, L]
                
            # Flatten và mask padding
            logits_flat = logits_intent_token.reshape(-1, C)  # [B*L, C]
            labels_flat = intent_expanded.reshape(-1)  # [B*L]
            mask_flat = mask.reshape(-1).bool()

            if mask_flat.any():
                loss_intent = F.cross_entropy(
                    logits_flat[mask_flat], labels_flat[mask_flat], ignore_index=-100
                )
                total_loss = total_loss + weights["intent"] * loss_intent

        # NER loss (CRF negative log-likelihood)
        if weights["ner"] > 0 and ner_labels is not None:
            # CRF cần tag ids ≥ 0. Replace -100 (ignored) → 0, mask sẽ lo
            ner_labels_crf = ner_labels.clone()
            ner_labels_crf[ner_labels_crf == -100] = 0
            crf_mask = mask.bool()
            loss_ner = -self.crf(logits_ner, ner_labels_crf, mask=crf_mask, reduction="mean")
            total_loss = total_loss + weights["ner"] * loss_ner

        return total_loss

    def predict_ner(self, logits_ner, mask):
        """Decode NER predictions bằng CRF Viterbi."""
        return self.crf.decode(logits_ner, mask=mask.bool())
