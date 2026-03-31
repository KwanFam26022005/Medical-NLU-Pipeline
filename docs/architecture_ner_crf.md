# 📐 kiến trúc chi tiết: Medical NER — trích xuất thực thể y tế (trạm 2A)

> **module:** trạm 2A — medical named entity recognition (token classification)  
> **backbone:** `demdecuong/vihealthbert-base-word` (roberta-base, word-level)  
> **tầng bổ trợ (SOTA):** CRF (conditional random fields)  
> **không gian nhãn:** BIO format (SYMPTOM_AND_DISEASE, MEDICAL_PROCEDURE, MEDICINE)  
> **mục tiêu:** giải quyết triệt để vấn đề span-noise và luật chuyển nhãn trong văn bản y tế.

---

## 1. tổng quan kiến trúc (architecture overview)

### 1.1 mục tiêu của trạm 2A
trạm 2A nhận đầu vào là các câu hỏi thô đã được word-segmented (tách từ bằng dấu `_`). nhiệm vụ là gán nhãn cho từng token để nhặt ra các thực thể y tế quan trọng phục vụ cho các trạm phía sau (như RAG hoặc tìm kiếm bác sĩ).

khác với các bài toán classification thông thường, NER là **sequence labeling**. độ chính xác không chỉ nằm ở việc đoán đúng từ đó là gì, mà còn phải đúng ranh giới (B- hay I-) để tạo thành một thực thể hoàn chỉnh.

### 1.2 sơ đồ luồng dữ liệu (data flow)
văn bản thô -> word segmentation (`_`) -> vihealthbert tokenizer -> align labels -> BERT encoder -> Linear layer -> **CRF layer** -> Predicted sequence.

---

## 2. tầng dữ liệu: Label Alignment & Span-Noise handling

### 2.1 xử lý ranh giới token (label alignment)
vì backbone xài word-level nhưng tokenizer vẫn có thể cắt một từ phức thành các sub-tokens (vd: "đau_đầu" -> "đau_@@", "đầu"). 
chiến lược của trạm 2A là:
- chỉ giữ nhãn cho sub-token đầu tiên.
- các sub-token sau được gán nhãn `IGN` hoặc nhãn `I-` tương ứng để tránh việc mô hình bị rối khi tính loss trên các mẩu từ vụn.

### 2.2 chiến thuật trị Span-Noise từ dataset ViMQ
như đã phân tích EDA, data ViMQ bị lỗi lệch index thực thể khá nhiều. 
**triết lý:** thay vì cố gắng fix tay từng mẫu data (vốn rất tốn kém), trạm 2A dùng tầng **CRF** để làm "bộ lọc ngữ pháp". CRF sẽ học xác suất chuyển nhãn, vd nó sẽ biết xác suất $P(I \text{-MED} | B \text{-MED})$ là rất cao, còn $P(I \text{-MED} | B \text{-SYMP})$ gần như bằng 0. điều này giúp tự động "nắn" lại các lỗi gán nhãn sai ranh giới của con người.

---

## 3. kiến trúc mô hình (model architecture)

### 3.1 tầng Backbone & Dropout
xài `vihealthbert-base-word` vì nó được pre-train trên các cặp từ y tế tiếng việt. output của lớp encoder cuối cùng sẽ là một tensor $\mathbf{H} \in \mathbb{R}^{L \times d}$ (với $L$ là độ dài chuỗi, $d=768$).

### 3.2 tầng CRF (Conditional Random Fields) - điểm chạm SOTA
thay vì dùng hàm Softmax độc lập cho từng token (vốn coi các từ là rời rạc), trạm 2A dùng CRF để tính xác suất cho **toàn bộ chuỗi nhãn** $\mathbf{y}$:

$$
P(\mathbf{y} | \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp \left( \sum_{i=1}^{L} \mathbf{E}_{i, y_i} + \sum_{i=1}^{L-1} \mathbf{T}_{y_i, y_{i+1}} \right)
$$

trong đó:
- $\mathbf{E}$: ma trận emission (điểm số từ BERT ném ra).
- $\mathbf{T}$: ma trận transition (điểm số chuyển nhãn mà CRF học được).
- $Z(\mathbf{x})$: hàm chuẩn hóa (partition function).

**lợi ích:** mô hình sẽ không bao giờ nhả ra kết quả vô lý kiểu một nhãn `I-` đứng một mình mà không có `B-` đi trước.

---

## 4. đánh giá & metric
không dùng accuracy vì nhãn `O` (outside) chiếm đa số (>90%). trạm 2A tập trung vào:
- **Strict F1-score (seqeval):** chỉ tính là đúng nếu đúng cả loại thực thể và ranh giới (exact match).
- **Partial match:** đánh giá khả năng bắt trúng một phần thực thể trong môi trường nhiễu.