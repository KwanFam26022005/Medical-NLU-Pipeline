# 📐 kiến trúc chi tiết: Medical Intent — phân loại đa ý định y tế (trạm 2C)

> **module:** trạm 2C — medical intent classification (multi-label, multi-class)
> **backbone:** `demdecuong/vihealthbert-base-syllable` (roberta-base, ~135M tham số)
> **chiến lược loss:** BCEWithLogitsLoss + Adaptive Pos-Weight
> **không gian nhãn:** 4 intents (Diagnosis, Treatment, Severity, Cause)

---

## 1. tổng quan kiến trúc (architecture overview)

### 1.1 mục tiêu của trạm 2C
khác với trạm 2B (Topic) là đơn nhãn loại trừ lẫn nhau, trạm 2C giải bài toán **đa ý định**. một câu hỏi của bệnh nhân thường phức tạp và chứa nhiều mong muốn cùng lúc. 
output là một vector xác suất cho từng nhãn, và một mẫu có thể có nhiều nhãn cùng đạt ngưỡng (threshold).

### 1.2 sơ đồ luồng dữ liệu
Raw text -> Syllable tokenizer (no segmenting) -> ViHealthBERT-Syllable -> Pooler layer (`[CLS]`) -> Linear head -> Sigmoid activation -> Multi-label prediction.

---

## 2. tầng dữ liệu: xử lý Multi-label & Imbalance

### 2.1 định dạng Multi-hot Encoding
data được chuyển từ dạng text sang vector nhị phân $\mathbf{y} \in \{0, 1\}^K$ (với $K=4$). 
vd: câu "bệnh này chữa sao và có chết ko?" sẽ có nhãn `[0, 1, 1, 0]` tương ứng với Treatment và Severity.

### 2.2 chiến thuật Positive Weighting (trị imbalanced)
trong data y tế, nhãn `Diagnosis` luôn áp đảo, trong khi `Cause` hay `Severity` rất ít. nếu dùng loss thông thường, model sẽ có xu hướng đoán toàn `0` cho các nhãn hiếm để giảm loss.
trạm 2C tính toán `pos_weight` cho từng class $c$:

$$
\operatorname{posweight}_c = \frac{N - N_c}{N_c}
$$

hệ số này sẽ nhân vào loss của các mẫu positive (nhãn 1). lớp càng hiếm thì `pos_weight` càng lớn, buộc mô hình phải "trân trọng" những lần bắt trúng nhãn thiểu số.

---

## 3. kiến trúc mô hình (model architecture)

### 3.1 backbone: ViHealthBERT-Syllable
lý do chọn bản syllable: câu hỏi intent thường mang tính hội thoại tự nhiên, việc xài syllable giúp model nhạy bén với các trợ từ, thán từ và cấu trúc câu hỏi tiếng việt mà không bị phụ thuộc vào chất lượng của tool tách từ bên ngoài.

### 3.2 hàm mất mát: Weighted Binary Cross-Entropy
sử dụng `BCEWithLogitsLoss` để tính loss độc lập cho từng nhãn trên cùng một head classification:

$$
\mathcal{L} = -\frac{1}{K} \sum_{c=1}^{K} w_c [y_c \cdot \log \sigma(z_c) + (1 - y_c) \cdot \log (1 - \sigma(z_c))]
$$

với $z_c$ là logit đầu ra của lớp thứ $c$. cơ chế này cho phép các nhãn không cạnh tranh xác suất với nhau (khác với Softmax của bài Topic).

---

## 4. đánh giá (evaluation)
- **Micro-F1 & Macro-F1:** cân bằng giữa hiệu năng tổng thể và hiệu năng trên từng nhãn đơn lẻ.
- **Hamming Loss:** đo tỷ lệ các nhãn bị dự đoán sai trên tổng số nhãn.
- **Per-class Precision/Recall:** đặc biệt theo dõi Recall của nhãn `Severity` vì trong y tế, bỏ sót ý định hỏi về mức độ nguy hiểm là rủi ro lớn.