# 🏥 KEHN Demo — Hệ thống Hiểu Ngôn ngữ Y tế Đa nhiệm

Demo inference cho mô hình **KEHN** (Knowledge-Enhanced Hierarchical Network) — phân tích câu hỏi y tế tiếng Việt qua 3 tác vụ đồng thời:

| Tác vụ | Mô tả | Output |
|--------|-------|--------|
| **Topic** | Phân loại chuyên khoa | 17 chuyên khoa + confidence % |
| **Intent** | Nhận diện ý định | 4 loại: Chẩn đoán / Điều trị / Mức độ / Nguyên nhân |
| **NER** | Trích xuất thực thể | 5 loại: Triệu chứng / Thuốc / Thủ thuật / Thời gian / Mức độ |

## 📦 Cài đặt

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Đặt file checkpoint

Đảm bảo file `.pt` nằm đúng vị trí:

```
kehn_joint/
├── model_outputs/
│   └── best_model_vihealthbert.pt    ← checkpoint đã train
├── demo/
│   ├── app.py
│   ├── model_loader.py
│   ├── preprocessor.py
│   ├── postprocessor.py
│   ├── requirements.txt
│   └── README.md
├── model/
│   ├── kehn_model.py
│   └── co_interactive.py
└── config_joint.py
```

> **Lưu ý:** Nếu file `.pt` ở vị trí khác, sửa biến `DEFAULT_CHECKPOINT` trong `model_loader.py`.

### 3. Backbone ViHealthBERT

Lần chạy đầu tiên, model sẽ tự tải backbone ViHealthBERT từ HuggingFace (`demdecuong/vihealthbert-base-word`). Cần kết nối internet.

## 🚀 Chạy Demo

```bash
# Từ thư mục kehn_joint/
cd demo
python app.py
```

Mở trình duyệt tại: **http://localhost:7860**

## 💡 Ví dụ

Nhập câu hỏi:
```
Tôi bị đau đầu, sốt cao và ho nhiều ngày
```

Kết quả:
- **Chuyên khoa**: Nội khoa (confidence ~85%)
- **Ý định**: Hỏi phương pháp chẩn đoán/điều trị
- **Thực thể NER**: đau đầu (Triệu chứng), sốt cao (Triệu chứng), ho (Triệu chứng), nhiều ngày (Thời gian)

## 🔧 Cấu hình

| Tham số | File | Mặc định |
|---------|------|----------|
| Checkpoint path | `model_loader.py` | `model_outputs/best_model_vihealthbert.pt` |
| Backbone | `model_loader.py` | `demdecuong/vihealthbert-base-word` |
| Max sequence length | `preprocessor.py` | 128 |
| Server port | `app.py` | 7860 |

## 📐 Kiến trúc

```
Tầng 1: ViHealthBERT → BiLSTM → H ∈ ℝ^(L×768)
Tầng 2: Label Attention → Cross-Attention (Intent ↔ NER) → Token-level decoders
Tầng 3: MaxPool(H) ⊕ MeanPool(P_intent) ⊕ MaxPool(P_ner) → Linear → Softmax → Topic
```
