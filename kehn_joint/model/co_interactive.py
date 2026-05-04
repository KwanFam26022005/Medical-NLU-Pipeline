"""
co_interactive.py — Port kiến trúc Co-Interactive Transformer từ DCA-Net (Qin et al., 2021).
Bao gồm: LabelAttention, CoInteractiveSelfAttention, CoInteractiveBlock, CoInteractiveFFN.

Nguồn gốc: https://github.com/kangbrilliant/DCA-Net/blob/master/model/joint_model_trans.py
Thay đổi: hidden_dim 300 → 768 (PhoBERT); clean up code style; thêm docstrings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm kiểu TF (epsilon bên trong sqrt)."""

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class SelfOutput(nn.Module):
    """Residual + LayerNorm sau attention."""

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class CoInteractiveSelfAttention(nn.Module):
    """
    Cross-attention hai chiều giữa 2 task representations.
    
    Thay vì self-attention thông thường (Q=K=V cùng nguồn),
    module này cho Intent query Slot keys/values và ngược lại:
      - Intent representation attends to Slot → intent nhận thông tin từ slot
      - Slot representation attends to Intent → slot nhận thông tin từ intent
    
    Multi-head attention với 8 heads mặc định.
    """

    def __init__(self, input_size, hidden_size, out_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size
        self.out_size = out_size

        # Intent → queries Slot
        self.query_a = nn.Linear(input_size, self.all_head_size)
        self.key_a = nn.Linear(input_size, self.all_head_size)
        self.value_a = nn.Linear(input_size, out_size)

        # Slot → queries Intent
        self.query_b = nn.Linear(input_size, self.all_head_size)
        self.key_b = nn.Linear(input_size, self.all_head_size)
        self.value_b = nn.Linear(input_size, out_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, repr_a, repr_b, mask):
        """
        Args:
            repr_a: Task A representation (e.g., Intent) [B, L, d]
            repr_b: Task B representation (e.g., NER)    [B, L, d]
            mask:   Attention mask [B, L]
        Returns:
            context_b: B enriched by A [B, L, out_size]
            context_a: A enriched by B [B, L, out_size]
        """
        # Attention mask → [B, 1, 1, L]
        extended_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - extended_mask) * -10000.0

        # A queries B (Intent attends to NER)
        Q_a = self.transpose_for_scores(self.query_a(repr_a))
        K_b = self.transpose_for_scores(self.key_a(repr_b))
        V_b = self.transpose_for_scores(self.value_a(repr_b))

        scores_a = torch.matmul(Q_a, K_b.transpose(-1, -2)) / math.sqrt(self.head_size)
        scores_a = scores_a + attention_mask
        probs_a = self.dropout(F.softmax(scores_a, dim=-1))
        context_a = torch.matmul(probs_a, V_b)

        # B queries A (NER attends to Intent)
        Q_b = self.transpose_for_scores(self.query_b(repr_b))
        K_a = self.transpose_for_scores(self.key_b(repr_a))
        V_a = self.transpose_for_scores(self.value_b(repr_a))

        scores_b = torch.matmul(Q_b, K_a.transpose(-1, -2)) / math.sqrt(self.head_size)
        scores_b = scores_b + attention_mask
        probs_b = self.dropout(F.softmax(scores_b, dim=-1))
        context_b = torch.matmul(probs_b, V_a)

        # Reshape back
        context_a = context_a.permute(0, 2, 1, 3).contiguous()
        context_b = context_b.permute(0, 2, 1, 3).contiguous()
        context_a = context_a.view(context_a.size()[:-2] + (self.out_size,))
        context_b = context_b.view(context_b.size()[:-2] + (self.out_size,))

        return context_b, context_a


class CoInteractiveFFN(nn.Module):
    """
    Extended Feed-Forward Network cho 2 streams (từ DCA-Net: Intermediate_I_S).
    
    Concat 2 stream representations + context trái/phải (sliding window),
    rồi project qua FFN để fuse thông tin.
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        # Input: concat(A, B) + left + right = 6 * hidden_size
        self.dense_in = nn.Linear(hidden_size * 6, hidden_size)
        self.act = nn.ReLU()
        self.dense_out = nn.Linear(hidden_size, hidden_size)
        self.norm_a = LayerNorm(hidden_size)
        self.norm_b = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, repr_a, repr_b):
        """Fuse 2 streams với local context (left/right sliding window)."""
        combined = torch.cat([repr_a, repr_b], dim=2)  # [B, L, 2*d]
        B, L, D = combined.size()

        # Sliding window: shift left & right
        pad = torch.zeros(B, 1, D, device=combined.device)
        h_left = torch.cat([pad, combined[:, :L - 1, :]], dim=1)
        h_right = torch.cat([combined[:, 1:, :], pad], dim=1)
        combined = torch.cat([combined, h_left, h_right], dim=2)  # [B, L, 6*d]

        hidden = self.dense_in(combined)
        hidden = self.act(hidden)
        hidden = self.dense_out(hidden)
        hidden = self.dropout(hidden)

        out_a = self.norm_a(hidden + repr_a)
        out_b = self.norm_b(hidden + repr_b)
        return out_a, out_b


class LabelAttention(nn.Module):
    """
    Label Attention Layer (từ DCA-Net).
    
    Project hidden states qua label embeddings (weights của classification heads)
    để tạo explicit task-aware representations.
    
    Công thức:
        score = H @ W_label.T          → [B, L, n_labels]
        probs = softmax(score)          → [B, L, n_labels]
        H_task = probs @ W_label        → [B, L, d]
    """

    def __init__(self, fc_a, fc_b):
        """
        Args:
            fc_a: nn.Linear head cho task A (e.g., intent_fc)
            fc_b: nn.Linear head cho task B (e.g., ner_fc)
        """
        super().__init__()
        self.W_a = fc_a.weight  # [n_labels_a, hidden_dim]
        self.W_b = fc_b.weight  # [n_labels_b, hidden_dim]

    def forward(self, input_a, input_b, mask):
        """
        Args:
            input_a, input_b: [B, L, d] — hidden states cho 2 tasks
            mask: [B, L] — attention mask (unused here, kept for API consistency)
        Returns:
            repr_a, repr_b: [B, L, d] — label-attention enriched representations
        """
        # Task A label attention
        score_a = torch.matmul(input_a, self.W_a.t())  # [B, L, n_a]
        probs_a = F.softmax(score_a, dim=-1)
        repr_a = torch.matmul(probs_a, self.W_a)  # [B, L, d]

        # Task B label attention
        score_b = torch.matmul(input_b, self.W_b.t())  # [B, L, n_b]
        probs_b = F.softmax(score_b, dim=-1)
        repr_b = torch.matmul(probs_b, self.W_b)  # [B, L, d]

        return repr_a, repr_b


class CoInteractiveBlock(nn.Module):
    """
    Một block Co-Interactive hoàn chỉnh (từ DCA-Net: I_S_Block).
    
    Pipeline: CrossAttention → SelfOutput (residual) → CoInteractiveFFN (fuse)
    """

    def __init__(self, fc_a, fc_b, hidden_size, dropout=0.1):
        super().__init__()
        self.cross_attn = CoInteractiveSelfAttention(
            input_size=hidden_size,
            hidden_size=2 * hidden_size,
            out_size=hidden_size,
            dropout=dropout,
        )
        self.out_a = SelfOutput(hidden_size, dropout)
        self.out_b = SelfOutput(hidden_size, dropout)
        self.ffn = CoInteractiveFFN(hidden_size, dropout)

    def forward(self, repr_a, repr_b, mask):
        """
        Args:
            repr_a: Task A representation [B, L, d]
            repr_b: Task B representation [B, L, d]
            mask: [B, L]
        Returns:
            new_a, new_b: Updated representations [B, L, d]
        """
        # Cross-attention: A ↔ B tương tác 2 chiều
        ctx_b, ctx_a = self.cross_attn(repr_a, repr_b, mask)

        # Residual connection + LayerNorm
        new_a = self.out_a(ctx_a, repr_a)
        new_b = self.out_b(ctx_b, repr_b)

        # Extended FFN fuse
        new_a, new_b = self.ffn(new_a, new_b)

        return new_a, new_b
