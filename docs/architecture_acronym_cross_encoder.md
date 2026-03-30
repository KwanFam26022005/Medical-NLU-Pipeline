# 📐 Kiến trúc Chi tiết: Cross-Encoder cho Acronym Disambiguation (WSD)

> **Module:** Trạm 1 — Acronym Word Sense Disambiguation  
> **Backbone:** `demdecuong/vihealthbert-base-syllable` (RoBERTa-base, 135M params)  
> **Dataset:** acrDrAid (LREC 2022) — Vietnamese Medical Acronyms  
> **Kết quả:** Overall Accuracy **91.77%** | Unseen Accuracy **84.94%** | MRR **0.9533**

---

## 1. Tổng quan Kiến trúc (Architecture Overview)

### 1.1 Tại sao Cross-Encoder vượt trội hơn Multi-class Classification?

Bài toán **Acronym Disambiguation** (hay Word Sense Disambiguation — WSD) yêu cầu chọn đúng nghĩa mở rộng cho một từ viết tắt trong ngữ cảnh cụ thể. Ví dụ: `"kt"` có thể là *"kích thước"*, *"kỹ thuật"*, hoặc *"kiểm tra"* tùy vào câu.

**Kiến trúc cũ — Multi-class Classification (280 labels):**

```
Input:  "Giải nghĩa từ viết tắt kt trong câu: Kết quả XQ phổi cho thấy kt 3cm"
         ↓
      Tokenizer → BERT Encoder → [CLS] vector → Linear(768, 280) → Softmax
         ↓
Output: class_id = 42  →  id2expansion[42] = "kích thước"
```

**Vấn đề chết người:** Mỗi nghĩa mở rộng được gán một `class_id` cố định. Nếu tại inference mô hình gặp một từ viết tắt **chưa từng xuất hiện trong tập huấn luyện** (ví dụ: `"pta"` — *phẫu thuật appendix*), thì **không tồn tại class nào** trong 280 classes để mô hình chọn. Kết quả: mô hình **buộc phải đoán bừa** vào một class ngẫu nhiên → sai 100%.

**Kiến trúc mới — Cross-Encoder (Pairwise Binary Scorer):**

```
Input:  ("Kết quả XQ phổi cho thấy <e>kt</e> 3cm", "kích thước")  →  score = 0.92
        ("Kết quả XQ phổi cho thấy <e>kt</e> 3cm", "kỹ thuật")    →  score = 0.15
        ("Kết quả XQ phổi cho thấy <e>kt</e> 3cm", "kiểm tra")    →  score = 0.08
         ↓
Output: argmax → "kích thước" ✅
```

> **Điểm then chốt:** Cross-Encoder không cần biết trước tập label cố định. Nó **mã hóa trực tiếp chuỗi ký tự** của nghĩa mở rộng ứng viên (candidate expansion) thành vector ngữ nghĩa rồi so sánh với ngữ cảnh. Nhờ đó, chỉ cần thêm một dòng vào `dictionary.json` là mô hình có thể xử lý từ viết tắt hoàn toàn mới mà **không cần train lại** (Zero-shot Generalization).

### 1.2 Sơ đồ Luồng Dữ liệu Tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA FLOW PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① Raw JSON (acrDrAid)                                         │
│     data.json: {text, start_char_idx, length_acronym, expansion}│
│     dictionary.json: {acronym → [candidate_1, candidate_2, ...]}│
│          │                                                      │
│          ▼                                                      │
│  ② Entity Marking                                              │
│     "Kết quả XQ phổi" → "Kết quả <e>XQ</e> phổi"             │
│          │                                                      │
│          ▼                                                      │
│  ③ Pair Generation (Hard Negative Mining)                      │
│     Positive: (marked_ctx, "X-quang")      → label = 1.0       │
│     Negative: (marked_ctx, "xét quang")    → label = 0.0       │
│          │                                                      │
│          ▼                                                      │
│  ④ Tokenizer (sentence-pair encoding)                          │
│     [CLS] marked_context [SEP] candidate [SEP]                 │
│     → input_ids: [0, 451, 23, <e>, 89, </e>, ..., 2, 367, 2]  │
│     → attention_mask: [1, 1, 1, ..., 0, 0, 0]                 │
│          │                                                      │
│          ▼                                                      │
│  ⑤ BERT Encoder (12 Transformer Layers)                        │
│     → Hidden states: [batch, seq_len, 768]                     │
│     → [CLS] pooler output: [batch, 768]                        │
│          │                                                      │
│          ▼                                                      │
│  ⑥ Classification Head                                         │
│     Linear(768, 1) → scalar logit                              │
│          │                                                      │
│          ├── Training: BCEWithLogitsLoss(logit, label)          │
│          └── Inference: argmax trên N logits → best expansion   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Tầng 1: Tiền xử lý Dữ liệu (Data Preprocessing Layer)

### 2.1 Nguyên liệu thô

**File `data/{train,dev,test}/data.json`:**

Mỗi sample là một JSON object chứa:

| Field | Type | Ý nghĩa |
|---|---|---|
| `text` | `string` | Câu gốc chứa từ viết tắt |
| `start_char_idx` | `int` | Vị trí ký tự bắt đầu của từ viết tắt trong `text` |
| `length_acronym` | `int` | Chiều dài (tính bằng ký tự) của từ viết tắt |
| `expansion` | `string` | Nghĩa mở rộng **đúng** (ground truth) |

Ví dụ:
```json
{
  "text": "Kết quả XQ phổi cho thấy tổn thương dạng kính mờ",
  "start_char_idx": 8,
  "length_acronym": 2,
  "expansion": "X-quang"
}
```

**File `data/dictionary.json`:**

Mapping mỗi từ viết tắt → tất cả các nghĩa mở rộng hợp lệ:
```json
{
  "XQ": ["X-quang", "xét quang"],
  "kt": ["kích thước", "kỹ thuật", "kiểm tra"],
  "bn": ["bệnh nhân", "bình nguyên"],
  ...
}
```

> **Thống kê dataset:** 135 acronyms, 424 total expansions, trung bình ~3.1 nghĩa/từ viết tắt. Phân bố: Train 4000, Dev 523, Test 1130 samples.

### 2.2 Kỹ thuật Entity Marking (Đánh dấu Thực thể)

**Mục đích:** Cho mô hình Transformer biết **chính xác vị trí** của từ viết tắt cần giải nghĩa bên trong câu. Nếu không đánh dấu, mô hình sẽ mơ hồ không biết phải focus vào đoạn nào.

**Cơ chế:** Chèn hai **special tokens** `<e>` (entity start) và `</e>` (entity end) bao quanh từ viết tắt bằng phép string slicing:

```python
marked_text = text[:start] + "<e>" + text[start:start+length] + "</e>" + text[start+length:]
```

**Ví dụ cụ thể:**

| Bước | Giá trị |
|---|---|
| Text gốc | `"Kết quả XQ phổi cho thấy tổn thương dạng kính mờ"` |
| `start_char_idx` | `8` |
| `length_acronym` | `2` |
| Acronym trích xuất | `text[8:10]` → `"XQ"` |
| **Text sau Entity Marking** | `"Kết quả <e>XQ</e> phổi cho thấy tổn thương dạng kính mờ"` |

**Tại sao phải thêm `<e>`, `</e>` vào Vocabulary của Tokenizer?**

Khi ta gọi `tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})`:

1. Tokenizer sẽ **không** tách `<e>` thành các subword `<`, `e`, `>` mà giữ nguyên nó như **một token duy nhất** có embedding riêng.
2. Embedding mới này được **khởi tạo từ phân phối Gaussian** dựa trên mean/cov của embedding cũ, sau đó được **fine-tune** trong quá trình training.
3. Sau khi train, mô hình học được rằng: *"token nằm giữa `<e>` và `</e>` chính là đối tượng cần disambiguate"* — tương tự kỹ thuật trong Relation Extraction (RE).

### 2.3 Kỹ thuật Hard Negative Mining

Mỗi sample trong dataset cung cấp **một cặp đúng** (positive pair). Nhưng để model học phân biệt đúng/sai, ta cần **cặp sai** (negative pairs). Đây chính là lúc `dictionary.json` phát huy tác dụng:

**Quy trình sinh cặp cho 1 sample:**

Giả sử sample có `acronym = "kt"`, `correct_expansion = "kích thước"`, và dictionary cho biết `kt → ["kích thước", "kỹ thuật", "kiểm tra"]`:

| # | Marked Context | Candidate | Label |
|---|---|---|---|
| 1 | `"...cho thấy <e>kt</e> 3cm..."` | `"kích thước"` | **1.0** ✅ |
| 2 | `"...cho thấy <e>kt</e> 3cm..."` | `"kỹ thuật"` | **0.0** ❌ |
| 3 | `"...cho thấy <e>kt</e> 3cm..."` | `"kiểm tra"` | **0.0** ❌ |

> **Tại sao gọi là "Hard" Negatives?** Vì các ứng viên sai (`"kỹ thuật"`, `"kiểm tra"`) đều là **nghĩa hợp lệ** của cùng một từ viết tắt `"kt"` — chúng cùng là nghĩa y tế có thật, chỉ khác ngữ cảnh. Điều này buộc mô hình phải **thực sự hiểu ngữ nghĩa** của câu xung quanh, không thể chỉ dựa vào surface-level overlap.

### 2.4 Định dạng Đầu vào cho Tokenizer (Sentence-Pair Encoding)

Tokenizer của HuggingFace khi nhận 2 chuỗi sẽ tự động nối chúng theo format:

```
[CLS]  Kết quả <e> XQ </e> phổi cho thấy tổn thương  [SEP]  X-quang  [SEP]
  ↑                                                      ↑              ↑
  Token đặc biệt                                    Ngăn cách        Kết thúc
  bắt đầu câu                                   sentence A-B         câu B
```

Sau khi encode:
```
input_ids:      [0, 4511, 2309, <e_id>, 8901, </e_id>, 7620, ..., 2, 36702, 2]
attention_mask: [1,    1,    1,      1,     1,       1,    1, ..., 1,     1, 1, 0, 0, ...]
token_type_ids: [0,    0,    0,      0,     0,       0,    0, ..., 0,     1, 1, 0, 0, ...]
                 ├──── Sentence A (context) ──────────────┤  ├── Sentence B ──┤  ├ PAD ┤
```

> **Lưu ý kỹ thuật:** Chúng ta sử dụng `truncation="only_first"` thay vì `truncation=True` (mặc định `longest_first`). Lý do: nghĩa mở rộng (sentence B) thường rất ngắn (1-3 từ), nên khi cần cắt bớt, ta **chỉ cắt phần context dài** (sentence A) để đảm bảo candidate expansion luôn được giữ nguyên vẹn.

### 2.5 Collate Function — Xử lý Batch có Kích thước Bất đồng nhất

Mỗi từ viết tắt có số lượng nghĩa mở rộng ứng viên **khác nhau** (2-7 candidates). Vì vậy, mỗi sample sinh ra **số lượng cặp khác nhau**, DataLoader không thể stack trực tiếp.

**Giải pháp: `acronym_train_collate_fn`** — Flatten tất cả các cặp từ mọi sample trong batch thành một tensor phẳng:

```
Sample 0: "kt"  → 3 candidates → 3 pairs
Sample 1: "bn"  → 2 candidates → 2 pairs  
Sample 2: "XQ"  → 2 candidates → 2 pairs
Sample 3: "HA"  → 5 candidates → 5 pairs
─────────────────────────────────────────
Collated batch: 12 pairs total

input_ids:      Tensor[12, 128]
attention_mask: Tensor[12, 128]
labels:         Tensor[12]  →  [1, 0, 0,  1, 0,  1, 0,  1, 0, 0, 0, 0]
```

---

## 3. Tầng 2: Kiến trúc Mô hình (Model Architecture & Computation)

### 3.1 Backbone: ViHealthBERT-Syllable

**`demdecuong/vihealthbert-base-syllable`** là một mô hình RoBERTa-base đã được **pre-train trên 3GB+ dữ liệu y tế tiếng Việt** (bài báo, hồ sơ bệnh án, diễn đàn y khoa). Điểm đặc biệt: tokenizer làm việc ở mức **âm tiết** (syllable-level), phù hợp với đặc tính ngôn ngữ Việt.

| Thông số | Giá trị |
|---|---|
| Kiến trúc | RoBERTa-base |
| Hidden size ($d$) | 768 |
| Số Transformer layers | 12 |
| Số Attention heads | 12 |
| Vocabulary size | 64,001 (+2 special tokens = 64,003) |
| Max position | 514 |
| Total parameters | ~135M |

### 3.2 Forward Pass — Luồng Tính toán Chi tiết

#### Bước 1: Embedding Layer

Token IDs được chuyển thành vector liên tục:

$$\mathbf{E} = \text{TokenEmb}(\text{input\_ids}) + \text{PosEmb}(\text{position\_ids})$$

Kết quả: Tensor $\mathbf{E} \in \mathbb{R}^{B \times L \times 768}$ (B = batch size, L = sequence length).

#### Bước 2: 12 Transformer Encoder Layers

Mỗi layer thực hiện 2 phép biến đổi chính:

**Multi-Head Self-Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

trong đó $Q = \mathbf{H}W^Q$, $K = \mathbf{H}W^K$, $V = \mathbf{H}W^V$, và $d_k = 768 / 12 = 64$.

> **Ý nghĩa trong bài toán WSD:** Nhờ Self-Attention, token `[CLS]` có thể "nhìn thấy" đồng thời cả ngữ cảnh xung quanh từ viết tắt (sentence A) **VÀ** chuỗi ký tự của nghĩa mở rộng (sentence B). Attention weights giữa `<e>kt</e>` và `"kích thước"` sẽ cao nếu chúng tương thích ngữ nghĩa.

**Feed-Forward Network (FFN):**

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}W_1 + b_1)W_2 + b_2$$

Qua 12 layers, output cuối cùng: $\mathbf{H}^{(12)} \in \mathbb{R}^{B \times L \times 768}$.

#### Bước 3: Trích xuất [CLS] Representation

Chúng ta **chỉ lấy vector ẩn tại vị trí đầu tiên** (token `[CLS]`):

$$\mathbf{h}_{\text{CLS}} = \mathbf{H}^{(12)}[:, 0, :] \in \mathbb{R}^{B \times 768}$$

> **Tại sao lại là `[CLS]`?** Trong kiến trúc BERT/RoBERTa, token `[CLS]` được thiết kế đặc biệt: nó **không gắn với bất kỳ từ nào** trong câu, nên Self-Attention buộc nó phải tổng hợp thông tin từ **toàn bộ sequence** (cả context lẫn candidate). Sau pre-training, `[CLS]` trở thành một **compressed representation** của toàn bộ cặp câu — lý tưởng cho classification/scoring.

#### Bước 4: Classification Head (Scoring Layer)

Một lớp **Linear đơn giản** chiếu vector 768-chiều xuống còn 1 scalar:

$$\text{logit} = \mathbf{W} \cdot \mathbf{h}_{\text{CLS}} + b$$

trong đó $\mathbf{W} \in \mathbb{R}^{1 \times 768}$ và $b \in \mathbb{R}^1$.

**Kết quả:** Mỗi cặp (context, candidate) cho ra **một con số duy nhất** (logit). Logit càng cao → mô hình càng tự tin rằng candidate đó là nghĩa đúng.

### 3.3 Hàm Mất mát: BCEWithLogitsLoss

#### Tại sao KHÔNG dùng CrossEntropyLoss?

| Tiêu chí | `CrossEntropyLoss` | `BCEWithLogitsLoss` |
|---|---|---|
| Input | Logits cho **tất cả K classes** cùng lúc | Logit cho **từng cặp** riêng lẻ |
| Giả định | Các class mutually exclusive, tập label cố định | Mỗi cặp là bài toán nhị phân độc lập |
| Số output neurons | K (= 280 trong kiến trúc cũ) | **1** |
| Thêm class mới | ❌ Phải thay đổi head + train lại | ✅ Chỉ cần thêm vào dictionary |

`BCEWithLogitsLoss` kết hợp **Sigmoid + Binary Cross Entropy** trong một bước (numerically stable):

#### Công thức toán học

Cho logit $z_i$ và label $y_i \in \{0, 1\}$:

$$\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(\sigma(z_i)) + (1 - y_i) \cdot \log(1 - \sigma(z_i)) \right]$$

**Phân tích trực giác:**

- Khi $y_i = 1$ (cặp đúng): Loss giảm khi $\sigma(z_i) \to 1$ (logit dương lớn).
- Khi $y_i = 0$ (cặp sai): Loss giảm khi $\sigma(z_i) \to 0$ (logit âm lớn).

> **Numerically Stable:** `BCEWithLogitsLoss` **KHÔNG** tính sigmoid trước rồi mới tính log (dễ bị underflow/overflow). Thay vào đó, nó dùng công thức biến đổi tương đương:
> $$\mathcal{L} = \max(z_i, 0) - z_i \cdot y_i + \log(1 + e^{-|z_i|})$$
> Điều này đảm bảo **không bao giờ xảy ra NaN/Inf** — nguyên nhân chính khiến kiến trúc cũ training bị crash.

### 3.4 Chiến lược Training nâng cao

| Kỹ thuật | Giá trị | Mục đích |
|---|---|---|
| **Differential LR** | Encoder: 2e-5, Head: 1e-4 | Encoder đã pre-train → learn rate thấp để giữ kiến thức; Head mới → learn rate cao để hội tụ nhanh |
| **Gradient Accumulation** | 4 steps (effective batch = 32) | Mô phỏng batch size lớn trên GPU nhỏ (Colab T4: 15GB VRAM) |
| **Cosine Annealing** | Warmup 10% → decay to 0 | Tránh divergence ở đầu training; giảm mượt cuối training |
| **FP16 Mixed Precision** | Enabled | Giảm 50% VRAM, tăng tốc 1.5-2x trên GPU Ampere/Turing |
| **Early Stopping** | Patience = 5 epochs | Dừng sớm nếu dev accuracy không cải thiện → tránh overfitting |
| **Gradient Clipping** | Max norm = 1.0 | Ngăn exploding gradients |

---

## 4. Tầng 3: Chiến lược Suy luận (Inference Strategy)

### 4.1 Luồng Inference End-to-End

Khi hệ thống nhận một câu text mới từ người dùng, quy trình diễn ra như sau:

```
Input: "Bệnh nhân có kt khối u 3cm trên phim XQ ngực"
                    ↓
   ┌────────────────────────────────────────┐
   │ Bước 1: Acronym Detection              │
   │ Quét regex cho mỗi key trong dictionary │
   │ → Tìm thấy: "kt" (pos 15), "XQ" (pos 34) │
   └────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────┐
   │ Bước 2: Entity Marking (cho từng acronym)│
   │ "...có <e>kt</e> khối u..."             │
   │ "...phim <e>XQ</e> ngực"                │
   └────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────┐
   │ Bước 3: Dictionary Lookup               │
   │ kt → ["kích thước", "kỹ thuật", "kiểm tra"]│
   │ XQ → ["X-quang", "xét quang"]           │
   └────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────┐
   │ Bước 4: Batch Inference (xem §4.2)     │
   │ Score tất cả candidates → argmax        │
   │ kt → "kích thước" (score: 0.92)         │
   │ XQ → "X-quang" (score: 0.88)            │
   └────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────┐
   │ Bước 5: Text Replacement                │
   │ Replace từ cuối lên đầu (giữ index)     │
   └────────────────────────────────────────┘
                    ↓
Output: "Bệnh nhân có kích thước khối u 3cm trên phim X-quang ngực"
```

### 4.2 Batch Inference — Tối ưu Latency

Khi cần score $N$ ứng viên cho một từ viết tắt, cách **naive** là chạy vòng lặp `for`:

```python
# ❌ CHẬM — N lần forward pass
for candidate in candidates:
    score = model(marked_text, candidate)  # 1 forward pass mỗi lần
```

**Giải pháp tối ưu:** Duplicate context thành $N$ bản sao, ghép với $N$ candidates, tạo thành **một batch duy nhất** kích thước $[N, L]$:

```python
# ✅ NHANH — 1 lần forward pass duy nhất
encodings = tokenizer(
    [marked_text] * N,     # N bản sao giống nhau của context
    candidates,            # N candidates khác nhau
    max_length=128,
    padding=True,
    truncation="only_first",
    return_tensors="pt",
)
# input_ids shape: [N, 128]
# Chỉ CẦN 1 forward pass:
logits = model(**encodings).logits.squeeze(-1)  # shape: [N]
```

**Phân tích hiệu suất:**

| Phương pháp | Số forward passes | Latency (T4 GPU) |
|---|---|---|
| Vòng lặp for | $N$ | ~$N \times 15$ ms |
| Batch inference | **1** | ~$20$ ms (bất kể $N$) |

Với $N = 5$ candidates trung bình, batch inference nhanh hơn **~3.75 lần**.

### 4.3 Chọn Đáp án Cuối cùng & Confidence Score

Sau khi có vector logits $\mathbf{z} = [z_1, z_2, ..., z_N]$:

**Đáp án:**
$$\hat{y} = \text{candidates}[\arg\max(\mathbf{z})]$$

**Confidence Score (Margin-based):**

Thay vì dùng probability tuyệt đối (có thể misleading), ta dùng **margin** giữa top-1 và top-2:

$$\text{confidence} = \sigma(z_{\text{top-1}}) - \sigma(z_{\text{top-2}})$$

- Confidence **cao** (> 0.3): Mô hình rất chắc chắn, chênh lệch lớn giữa lựa chọn tốt nhất và thứ 2.
- Confidence **thấp** (< 0.1): Mô hình phân vân, nên fallback giữ nguyên từ viết tắt thay vì giải sai.

> **Threshold hiện tại:** `confidence > 0.1` — ngưỡng thấp có chủ đích vì bài toán y tế ưu tiên **recall** (phát hiện + giải nghĩa nhiều nhất có thể) hơn **precision**.

---

## 5. Kết quả Đánh giá & Phân tích

### 5.1 Benchmark trên Test Set (1130 samples)

| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| **Overall Accuracy** | **91.77%** | 1037/1130 dự đoán chính xác |
| **MRR** | **0.9533** | Đáp án đúng gần như luôn ở Top-1 hoặc Top-2 |
| **Seen Accuracy** | **94.38%** | 818 samples thuộc 109 acronyms đã thấy khi train |
| **Unseen Accuracy** | **84.94%** | 312 samples thuộc 26 acronyms **chưa từng thấy** khi train |

### 5.2 Phân tích Zero-shot Generalization

Con số **84.94% Unseen Accuracy** chứng minh Cross-Encoder thực sự hiểu ngữ nghĩa:

- Mô hình **chưa bao giờ** thấy 26 acronyms này trong training.
- Nó chỉ dựa vào (1) ngữ cảnh xung quanh để hiểu ý nghĩa, và (2) semantic encoding của candidate expansion string.
- Đối chiếu: kiến trúc Multi-class cũ sẽ đạt **0%** trên unseen vì không có class nào tương ứng.

---

## 6. Deployment & Tích hợp

### 6.1 Model Registry

Model đã được publish lên **Hugging Face Hub**:
- **Repo ID:** `KwanFam26022005/model1-acronym-wsd`
- **Files:** `model.safetensors`, `tokenizer.json`, `config.json`, `acronym_dict.json`

### 6.2 Tích hợp vào FastAPI Pipeline

```
config.py:  ACRONYM_MODEL_DIR = "KwanFam26022005/model1-acronym-wsd"
                ↓
models.py:  AcronymCrossEncoder.load_model()
                → AutoTokenizer.from_pretrained(repo_id)
                → AutoModelForSequenceClassification.from_pretrained(repo_id)
                → hf_hub_download(repo_id, "acronym_dict.json")
                ↓
main.py:    acronym_resolver.predict(user_text)
                → detect acronyms → score candidates → return clean text
```

> **Lưu ý:** Lần chạy đầu tiên sẽ download model (~540MB) về cache local (~/.cache/huggingface/). Các lần sau sẽ load từ cache, latency chỉ còn ~2 giây khởi tạo.
