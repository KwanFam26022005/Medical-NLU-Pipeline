# 🤖 AGENT PROMPT — Rebuild Acronym Disambiguation Pipeline (SOTA-Grade)

---

## 🎯 NHIỆM VỤ TỔNG QUAN

Bạn là một Senior NLP Engineer. Nhiệm vụ của bạn là **viết lại hoàn toàn** module Acronym
Disambiguation (Trạm 1) trong Medical NLU Pipeline cho tiếng Việt y tế, theo kiến trúc
SOTA hiện tại. Các module còn lại (NER, Topic, Intent) **không được chỉnh sửa**.

Toàn bộ code phải production-ready: có type hints, docstring, logging, error handling, và
unit test cơ bản cho mỗi layer.

---

## 📂 CONTEXT HỆ THỐNG HIỆN TẠI

### Codebase structure (hiện tại — KHÔNG thay đổi các file không liên quan)
```
Medical-NLU-Pipeline/
├── config.py           ← Cấu hình trung tâm (CẦN cập nhật phần ACRONYM)
├── models.py           ← 4 model class (CẦN viết lại AcronymResolver)
├── data_loader.py      ← 4 data loader (CẦN viết lại AcronymDataLoader)
├── train_acronym.py    ← Training script (VIẾT LẠI HOÀN TOÀN)
├── train_ner.py        ← KHÔNG chỉnh sửa
├── train_topic.py      ← KHÔNG chỉnh sửa
├── train_intent.py     ← KHÔNG chỉnh sửa
├── main.py             ← FastAPI server (CẦN cập nhật inference call)
├── build_topic_dataset.py ← KHÔNG chỉnh sửa
└── data/
    ├── acrDrAid/
    │   ├── train.csv       (columns: context, acronym, expansion)
    │   ├── dev.csv
    │   └── test.csv
    └── acronym_dict.json   ({"CT": ["cắt lớp vi tính", "công thức máu", ...], ...})
```

### Pretrained backbone ưu tiên (theo thứ tự)
```
1. manhtt-079/vipubmed-deberta          ← Ưu tiên cao nhất (DeBERTaV3, domain y tế)
2. demdecuong/vihealthbert-base-syllable ← Fallback (đang dùng)
3. xlm-roberta-base                     ← Fallback đa ngôn ngữ
```

### Dataset đặc điểm quan trọng
- 4,000 train / 523 dev / 1,130 test samples
- 135 unique acronyms, 424 unique expansions
- Avg candidates per acronym: 3.16
- **19.26% test acronyms KHÔNG có trong train** → model phải generalize
- Avg context length: ~30 syllables (radiology report domain)

---

## ❌ VẤN ĐỀ CỦA CODE HIỆN TẠI (PHẢI FIX)

```python
# HIỆN TẠI — SAI VỀ FORMULATION
# train_acronym.py đang dùng:
# CrossEntropyLoss over 424 classes → 4000/424 ≈ 9.4 samples/class
# Focal Loss + class weights chỉ là band-aid, không fix root problem
# Model KHÔNG generalize được với unseen acronyms (19.26% test set)

model = AutoModelForSequenceClassification.from_pretrained(
    "demdecuong/vihealthbert-base-syllable",
    num_labels=424   # ← SAI: sparse classes, không generalize
)
```

---

## ✅ KIẾN TRÚC MỚI CẦN XÂY DỰNG — 4 TẦNG

```
┌─────────────────────────────────────────────────────────────┐
│  TẦNG 1: Data Layer — Multi-Choice Negative Sampling        │
│  (AcronymDataset + AcronymDataLoader)                       │
├─────────────────────────────────────────────────────────────┤
│  TẦNG 2: Model Layer — Pairwise Binary Scorer               │
│  (AcronymResolver với Siamese Encoding + Acronym Position)  │
├─────────────────────────────────────────────────────────────┤
│  TẦNG 3: Training Layer — Contrastive + Self-Supervised     │
│  (AcronymTrainer với Hard Negative Mining + Pseudo-Label)   │
├─────────────────────────────────────────────────────────────┤
│  TẦNG 4: Inference Layer — Candidate Ranking + Fallback     │
│  (AcronymPredictor với confidence threshold + fallback)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 ĐẶC TẢ CHI TIẾT TỪNG TẦNG

---

### TẦNG 1 — Data Layer (`data_loader.py` → class `AcronymDataset` + `AcronymDataLoader`)

**Formulation thay đổi:**
```
CŨ:  Input → [CLS] context [SEP]  →  softmax(424 classes)
MỚI: Input → [CLS] context [SEP] candidate_expansion [SEP]  →  scalar score
     Với mỗi acronym: score tất cả candidates → argmax → expansion được chọn
```

**Yêu cầu class `AcronymDataset(Dataset)`:**

```python
# Khởi tạo
def __init__(
    self,
    data_path: str,              # path đến train/dev/test.csv
    acronym_dict: dict,          # {"CT": ["cắt lớp vi tính", ...], ...}
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    mode: str = "train",         # "train" | "eval"
    n_hard_negatives: int = 2,   # số hard negatives thêm vào khi train
    augment_substitution: bool = True  # ContextAD substitution trick
)
```

**Logic `__getitem__` phải:**
1. Load 1 row: `(context, acronym, correct_expansion)`
2. Lấy `candidates = acronym_dict[acronym]` — tất cả expansions của acronym đó
3. **Nếu mode="train"**: thêm `n_hard_negatives` hard negatives từ expansions của
   acronym **khác** (chọn những expansion có cùng chữ cái đầu với acronym — harder)
4. **Nếu `augment_substitution=True`**: tạo thêm `substituted_context` bằng cách
   thay `acronym` trong câu bằng `correct_expansion`, dùng cả hai version để train
5. Shuffle candidates, ghi nhớ index của `correct_expansion`
6. Encode từng `(context, candidate)` pair qua tokenizer với format:
   ```
   [CLS] context [SEP] candidate [SEP]
   ```
7. Nếu backbone là DeBERTa: dùng thêm **acronym position mask** — binary tensor
   đánh dấu vị trí acronym trong context để model biết token nào cần disambiguate

**Output của `__getitem__`:**
```python
{
    "input_ids":       torch.Tensor,  # shape [N_candidates, max_length]
    "attention_mask":  torch.Tensor,  # shape [N_candidates, max_length]
    "token_type_ids":  torch.Tensor,  # shape [N_candidates, max_length] (nếu có)
    "acronym_mask":    torch.Tensor,  # shape [N_candidates, max_length] — position của acronym
    "label":           int,           # index của correct expansion trong candidates
    "n_candidates":    int,           # để collate_fn biết cách pad
    "acronym":         str,           # để debug
    "candidates":      List[str]      # để inference
}
```

**Yêu cầu `collate_fn`:**
- Xử lý batch với số candidates khác nhau giữa các samples
- Pad/stack thành tensor thống nhất `[batch_size, max_N_candidates, seq_len]`
- Tạo `candidate_mask` để ignore padded candidates trong loss computation

**Yêu cầu `AcronymDataLoader`:**
- Wrap `AcronymDataset` với `DataLoader`
- Cung cấp method `build_acronym_dict(data_dir)` để auto-build dict từ CSV
- Cung cấp method `get_class_frequencies()` để tính class distribution

---

### TẦNG 2 — Model Layer (`models.py` → class `AcronymResolver`)

**Yêu cầu class `AcronymResolver(nn.Module)`:**

```python
def __init__(
    self,
    model_name: str = "manhtt-079/vipubmed-deberta",
    dropout: float = 0.1,
    use_acronym_position: bool = True,  # dùng position embedding của acronym
    pooling: str = "cls"                # "cls" | "mean" | "cls+mean"
)
```

**Sub-components cần implement:**

**2a. Encoder:**
```python
self.encoder = AutoModel.from_pretrained(model_name)
# hidden_size tự động lấy từ config
```

**2b. Acronym Position Attention (nếu `use_acronym_position=True`):**
```python
# Học attention weight tập trung vào vị trí acronym trong context
# Input: hidden_states [batch, seq, hidden], acronym_mask [batch, seq]
# Output: acronym_aware_repr [batch, hidden]
self.acronym_attention = nn.Sequential(
    nn.Linear(hidden_size, 1),
    nn.Softmax(dim=1)
)
# Trong forward: weighted sum của hidden_states bởi acronym_mask + attention
```

**2c. Pooling Layer:**
```python
# "cls": lấy hidden_state[:, 0, :]
# "mean": mean pooling có mask
# "cls+mean": concat cả hai → Linear để project về hidden_size
```

**2d. Scorer Head:**
```python
self.scorer = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(hidden_size, hidden_size // 2),
    nn.GELU(),
    nn.LayerNorm(hidden_size // 2),
    nn.Dropout(dropout),
    nn.Linear(hidden_size // 2, 1)   # scalar score
)
```

**Forward logic:**
```python
def forward(
    self,
    input_ids:      torch.Tensor,   # [batch, n_cands, seq_len]
    attention_mask: torch.Tensor,   # [batch, n_cands, seq_len]
    token_type_ids: Optional[torch.Tensor],
    acronym_mask:   Optional[torch.Tensor],  # [batch, n_cands, seq_len]
    candidate_mask: torch.Tensor,   # [batch, n_cands] — 1=valid, 0=padded
    labels:         Optional[torch.Tensor],  # [batch]
) -> dict:
    # 1. Reshape: [batch * n_cands, seq_len] để encode song song
    # 2. Encode qua BERT/DeBERTa
    # 3. Apply acronym position attention nếu bật
    # 4. Apply pooling strategy
    # 5. Score mỗi pair → [batch * n_cands, 1]
    # 6. Reshape → [batch, n_cands]
    # 7. Mask padded candidates: logits[~candidate_mask] = -inf
    # 8. Nếu có labels: tính loss (xem phần loss bên dưới)
    # 9. Return {"loss": loss, "logits": logits, "probs": softmax(logits)}
```

**Loss function — 2 thành phần:**
```python
# Loss 1: Cross-Entropy trên multi-choice logits (chính)
loss_ce = F.cross_entropy(logits, labels)

# Loss 2: Margin Ranking Loss (phụ — tăng separation)
# Với mỗi (positive, negative) pair:
# margin_loss = max(0, margin - score_pos + score_neg)
# margin = 0.5 (hyperparameter)
loss_margin = compute_margin_loss(logits, labels, margin=0.5)

# Tổng loss
loss = loss_ce + 0.3 * loss_margin
```

**Inference method:**
```python
@torch.no_grad()
def predict(
    self,
    context: str,
    acronym: str,
    candidates: List[str],
    tokenizer: PreTrainedTokenizer,
    device: str = "cpu"
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Returns:
        best_expansion: str
        confidence: float (max_prob - second_max_prob)
        ranked_list: [(expansion, prob), ...] sorted by prob desc
    """
```

---

### TẦNG 3 — Training Layer (`train_acronym.py` — VIẾT LẠI HOÀN TOÀN)

**Yêu cầu class `AcronymTrainer`:**

```python
def __init__(
    self,
    model: AcronymResolver,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig    # dataclass với các hyperparams
)
```

**Dataclass `TrainingConfig`:**
```python
@dataclass
class TrainingConfig:
    # Optimizer
    encoder_lr: float = 2e-5      # LR cho BERT layers
    head_lr: float = 1e-4         # LR cho scorer head (5x encoder)
    weight_decay: float = 0.01
    
    # Scheduler
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Training
    epochs: int = 10
    batch_size: int = 8           # nhỏ vì mỗi sample có nhiều pairs
    gradient_accumulation: int = 4  # effective batch = 32
    
    # Self-supervised
    use_self_supervised: bool = True
    self_sup_start_epoch: int = 2   # bắt đầu từ epoch 2
    pseudo_label_threshold: float = 0.85  # confidence để accept pseudo-label
    
    # Hard negative mining
    use_hard_negative_mining: bool = True
    hard_negative_refresh_steps: int = 100  # refresh mỗi N steps
    
    # Loss
    margin: float = 0.5
    margin_loss_weight: float = 0.3
    
    # Saving
    output_dir: str = "saved_models/acronym"
    save_best_metric: str = "accuracy"
```

**Training loop phải implement đủ 3 strategies:**

**Strategy A — Hard Negative Mining:**
```python
def mine_hard_negatives(self, step: int):
    """
    Mỗi `hard_negative_refresh_steps` steps:
    1. Forward pass toàn bộ train set với model hiện tại (no_grad)
    2. Với mỗi sample: tìm negative candidates có score cao nhất (confusing cases)
    3. Update dataset.hard_negatives dict
    4. Những confusing negatives này được ưu tiên sample trong batches tiếp theo
    """
```

**Strategy B — Self-Supervised với Pseudo-Labels:**
```python
def generate_pseudo_labels(self, unlabeled_loader: DataLoader):
    """
    Sau mỗi epoch (bắt đầu từ self_sup_start_epoch):
    1. Forward pass trên unlabeled data (raw radiology reports từ corpus)
    2. Nếu confidence (max_prob - 2nd_max_prob) > threshold → accept as pseudo-label
    3. Thêm pseudo-labeled samples vào train set cho epoch tiếp theo
    4. Log số lượng và quality của pseudo-labels
    """
```

**Strategy C — Span-Noise Augmentation (kế thừa từ ViMQ paper):**
```python
def apply_span_noise(self, context: str, acronym: str, p: float = 0.1):
    """
    Với probability p: shift start/end index của acronym span ±1 token
    → Model học robust với annotation noise
    """
```

**Metrics cần compute và log mỗi epoch:**
```python
metrics = {
    "accuracy":          float,   # % correct predictions
    "accuracy_seen":     float,   # accuracy chỉ trên acronyms đã thấy trong train
    "accuracy_unseen":   float,   # accuracy trên acronyms chưa thấy (quan trọng!)
    "mrr":               float,   # Mean Reciprocal Rank
    "confidence_mean":   float,   # avg (max_prob - 2nd_max_prob)
    "pseudo_label_count": int,    # số pseudo-labels được thêm vào
}
```

**Checkpoint và early stopping:**
```python
# Save best model theo accuracy_unseen (metric quan trọng nhất)
# Early stopping patience = 3 epochs
# Save training curve dưới dạng JSON để plot sau
```

---

### TẦNG 4 — Inference Layer (cập nhật `models.py` + `main.py`)

**Yêu cầu class `AcronymPredictor` (wrapper around `AcronymResolver`):**

```python
class AcronymPredictor:
    """
    Production inference wrapper với:
    - Candidate lookup từ acronym_dict
    - Confidence threshold + fallback strategy
    - Batch inference support
    - Acronym detection trong câu (nếu không được chỉ định)
    """
    
    def __init__(
        self,
        model: AcronymResolver,
        tokenizer: PreTrainedTokenizer,
        acronym_dict: dict,           # {"CT": [...], "MRI": [...], ...}
        confidence_threshold: float = 0.6,  # dưới ngưỡng này → fallback
        device: str = "cpu"
    )
    
    def resolve(
        self,
        context: str,
        acronym: str
    ) -> AcronymResult:
        """
        1. Lookup candidates = acronym_dict.get(acronym, [])
        2. Nếu candidates rỗng → return fallback (acronym giữ nguyên)
        3. Nếu chỉ 1 candidate → return trực tiếp (không cần model)
        4. Encode và score tất cả candidates
        5. Nếu confidence < threshold → log warning, return best_guess với flag
        6. Return AcronymResult
        """
    
    def resolve_sentence(self, sentence: str) -> str:
        """
        Auto-detect ALL acronyms trong câu (dùng regex + dict lookup),
        resolve từng cái, replace vào câu gốc.
        Return câu đã expand đầy đủ.
        """
    
    def resolve_batch(self, items: List[Tuple[str, str]]) -> List[AcronymResult]:
        """Batch inference cho throughput tốt hơn."""
```

**Dataclass `AcronymResult`:**
```python
@dataclass
class AcronymResult:
    acronym:        str
    expansion:      str
    confidence:     float   # max_prob - 2nd_max_prob
    is_certain:     bool    # confidence > threshold
    ranked_list:    List[Tuple[str, float]]  # [(expansion, prob), ...]
    fallback_used:  bool    # True nếu không có trong dict
```

**Cập nhật `main.py`:**
```python
# Thay thế call cũ:
# acronym_result = acronym_resolver.predict(...)

# Bằng call mới:
# predictor = AcronymPredictor(model, tokenizer, acronym_dict)
# resolved_sentence = predictor.resolve_sentence(raw_query)
# Sau đó pass resolved_sentence vào các module tiếp theo (NER, Topic, Intent)

# Thêm vào JSON response:
{
    "acronyms_detected": [
        {
            "original": "CT",
            "expansion": "cắt lớp vi tính",
            "confidence": 0.94,
            "is_certain": true
        }
    ],
    "resolved_text": "câu sau khi expand acronyms",
    "ner": {...},
    "topic": {...},
    "intent": {...}
}
```

---

## 🏗️ CẤU TRÚC FILE MỚI CẦN TẠO

```
Medical-NLU-Pipeline/
│
├── data_loader.py          ← Cập nhật AcronymDataset + AcronymDataLoader
├── models.py               ← Cập nhật AcronymResolver + thêm AcronymPredictor
├── train_acronym.py        ← VIẾT LẠI: AcronymTrainer + TrainingConfig
├── config.py               ← Cập nhật phần ACRONYM config
├── main.py                 ← Cập nhật inference call + response schema
│
└── scripts/
    ├── build_acronym_dict.py    ← [NEW] Script tạo acronym_dict.json từ CSV
    ├── evaluate_acronym.py      ← [NEW] Evaluation script với breakdown seen/unseen
    └── generate_silver_data.py  ← [NEW] Auto-extract acronym pairs từ raw corpus
```

---

## 📌 CONSTRAINTS & GROUND RULES

### Constraints bắt buộc
1. **KHÔNG thay đổi** `train_ner.py`, `train_topic.py`, `train_intent.py`, `build_topic_dataset.py`
2. **KHÔNG thay đổi** API contract của `main.py` — endpoint `/analyze_medical_query` giữ nguyên input schema
3. **KHÔNG xóa** class `MedicalNER`, `TopicClassifier`, `IntentClassifier` khỏi `models.py`
4. Tất cả code phải tương thích với `Python 3.9+`, `PyTorch 2.0+`, `transformers 4.35+`
5. Phải chạy được trên cả CPU (inference) và GPU (training)

### Code quality
- Mỗi class/function phải có **Google-style docstring**
- **Type hints** đầy đủ trên mọi function signature
- **Logging** thay vì print statements (`import logging`)
- **Error handling** rõ ràng với custom exceptions khi cần
- Mỗi file phải có block `if __name__ == "__main__"` với quick sanity test

### Testing
Sau khi viết xong, cung cấp **unit tests** cho:
```python
# test_acronym.py
def test_data_loader_output_shape()      # kiểm tra tensor shapes
def test_model_forward_pass()            # kiểm tra forward không crash
def test_collate_fn_variable_candidates() # kiểm tra collate với N candidates khác nhau
def test_predictor_fallback()            # kiểm tra fallback khi acronym không có trong dict
def test_predictor_resolve_sentence()    # kiểm tra auto-detect trong câu
```

---

## 📊 EXPECTED PERFORMANCE TARGETS

Sau khi implement đầy đủ 4 tầng:

| Metric | Baseline (hiện tại) | Target sau rebuild |
|--------|--------------------|--------------------|
| Overall Accuracy | ~72–75% | ≥ 87% |
| Accuracy (seen acronyms) | ~78% | ≥ 92% |
| Accuracy (unseen acronyms) | ~45% | ≥ 72% |
| MRR | ~0.78 | ≥ 0.91 |
| Inference latency (CPU, 1 sample) | ~50ms | ≤ 80ms |

---

## 🔁 THỨ TỰ THỰC HIỆN

Implement theo thứ tự sau để có thể test từng tầng độc lập:

```
Bước 1: scripts/build_acronym_dict.py   → tạo acronym_dict.json
Bước 2: data_loader.py (Tầng 1)         → verify với test_data_loader_output_shape()
Bước 3: models.py (Tầng 2)             → verify với test_model_forward_pass()
Bước 4: train_acronym.py (Tầng 3)      → chạy 1 epoch để verify không crash
Bước 5: Inference + main.py (Tầng 4)   → verify với test_predictor_*()
Bước 6: scripts/evaluate_acronym.py    → chạy full evaluation, in breakdown seen/unseen
Bước 7: scripts/generate_silver_data.py → optional, nếu cần thêm data
```

---

## ✅ DEFINITION OF DONE

Bạn HOÀN THÀNH nhiệm vụ khi và chỉ khi:

- [ ] Tất cả 5 unit tests pass
- [ ] `train_acronym.py` chạy được end-to-end 1 epoch không có error
- [ ] `evaluate_acronym.py` in ra breakdown accuracy seen/unseen/overall
- [ ] `main.py` vẫn respond đúng với request format cũ
- [ ] Không có thay đổi nào trong `train_ner.py`, `train_topic.py`, `train_intent.py`
- [ ] Mỗi file mới/sửa đổi đều có docstring + type hints + logging

---

*Prompt version: 1.0 | Task: Acronym Disambiguation SOTA Rebuild | Domain: Vietnamese Medical NLP*
