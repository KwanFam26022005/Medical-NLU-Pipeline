# Kiến trúc KEHN

### 3.1. Tầng 1: Mã hóa Ngữ cảnh Dùng chung (Shared Context Encoder)

Lớp mã hóa ngữ cảnh đóng vai trò là bộ trích xuất đặc trưng nền tảng (**shared encoder**) cung cấp biểu diễn không gian chung cho cả ba bài toán: Phân loại Chủ đề (Topic), Phát hiện Ý định (Intent) và Nhận diện Thực thể y tế (NER). Nhằm tối ưu hóa khả năng nắm bắt ngữ nghĩa chuyên ngành và giải quyết sự phụ thuộc xa (**long-term dependencies**) trong các câu hỏi y tế phức tạp, kiến trúc đề xuất kết hợp **mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Model)** với **mạng bộ nhớ ngắn-dài hai chiều (BiLSTM)**.

Quá trình tính toán tại Tầng 1 diễn ra tuần tự qua các bước:

---

### A. Ánh xạ không gian vector (Word Embedding)

Giả sử câu hỏi đầu vào là một chuỗi gồm L từ sau phân tách (Word Segmentation):

$$
X={x1,x2,…,xL}X = \{x_1, x_2, \dots, x_L\}
$$

$$
X={x1,x2,…,xL}
$$

Thay vì dùng nhúng từ tĩnh (như GloVe), mô hình sử dụng **PhoBERT** (hoặc **ViHealthBERT**) làm backbone. Mỗi token $x_i$ được đưa qua hàm ánh xạ  $\phi_{emb}$

$$
e_i = \phi_{emb}(x_i)
$$

Trong đó $e_i \in \mathbb{R}^d$ là vector nhúng ngữ nghĩa, với d=768d  tương ứng hidden dimension của kiến trúc base.

---

### B. Mã hóa chuỗi thời gian (Sequential Encoding với BiLSTM)

Chuỗi vector $E = \{e_1, e_2, \dots, e_L\}$ được đưa qua lớp **BiLSTM** để nắm bắt ngữ cảnh cấu trúc câu theo cả hai chiều. Tại mỗi bước thời gian i:

$$
\overrightarrow{h_i} = \text{LSTM}_{\text{forward}}(e_i,\, \overrightarrow{h_{i-1}})
$$

$$
\overleftarrow{h_i} = \text{LSTM}_{\text{backward}}(e_i,\, \overleftarrow{h_{i+1}})
$$

---

### C. Tổng hợp biểu diễn ẩn (Contextual Representation)

Đặc trưng ngữ cảnh cuối cùng cho mỗi từ $x_i$  là phép ghép nối (concatenation, $\oplus$) của trạng thái ẩn tiến và lùi:

$$
h_i = \overrightarrow{h_i} \oplus \overleftarrow{h_i}
$$

---

**Đầu ra của Tầng 1** là ma trận ẩn dùng chung:

$$
H = (h_1, h_2, \dots, h_L) \in \mathbb{R}^{L \times d}
$$

Ma trận ***H*** mang toàn bộ thông tin ngữ cảnh của câu và đóng vai trò đầu vào trực tiếp (dưới dạng **Query, Key, Value**) cho **Khối Tương tác đa nhiệm (Co-Interactive Transformer)** ở Tầng 2.

### 3.2. Tầng 2: Khai phá Đặc trưng Đa nhiệm (Co-Interactive Feature Mining Layer)

Nếu Tầng 1 cung cấp biểu diễn ngữ cảnh tĩnh dùng chung, thì Tầng 2 đóng vai trò là **lõi tương tác đa nhiệm** nhằm mô hình hóa mối quan hệ mật thiết giữa Thực thể y tế (NER) và Ý định người bệnh (Intent). Thay vì dùng Self-Attention thông thường hay truyền thông tin một chiều, kiến trúc KEHN áp dụng **mô-đun Co-Interactive Transformer** kết hợp kỹ thuật **Giải mã Ý định Cấp độ từ (Token-level Intent Detection)**.

Quá trình tính toán tại Tầng 2 gồm 3 bước toán học trọng tâm:

---

### A. Cơ chế Chú ý Nhãn (Label Attention)

Để tách biệt hai luồng đặc trưng từ ma trận ngữ cảnh chung  $H \in \mathbb{R}^{L \times d}$ (đầu ra Tầng 1), mô hình dùng các ma trận nhúng nhãn $W_I(Intent)$ và $W_S (NER)$ để truy xuất thông tin từ $H$, tạo ra các **biểu diễn chuyên biệt** mang ngữ nghĩa của từng tác vụ:

**Đặc trưng Ý định ban đầu:**

$$
A_I = \text{softmax}(H W_I)
$$

$$
H_{\text{intent}} = H + A_I W_I
$$

**Đặc trưng Thực thể ban đầu:**

$$
A_S = \text{softmax}(H W_S)
$$

$$
H_{\text{ner}} = H + A_S W_S
$$

---

### B. Tương tác Chéo Hai chiều (Co-Interactive Attention Layer)

Từ $H_{\text{intent}}$ và $H_{\text{ner}}$, mô hình ánh xạ sang các ma trận **Query (Q)**, **Key (K)**, **Value (V)** qua các phép biến đổi tuyến tính, sau đó thực hiện **Cross-Attention** để hai tác vụ trao đổi tri thức cho nhau.

**① Thực thể nhận thức Ý định (Intent-aware Slot):**

Nhánh **NER** dùng $Q_S$ để truy xuất từ $K_I$, $V_I$ của nhánh **Intent**:

$$
C_S = \text{softmax}\!\left(\frac{Q_S K_I^T}{\sqrt{d_k}}\right) V_I
$$

$$
H_S' = \text{LayerNorm}(H_{\text{ner}} + C_S)
$$

**② Ý định nhận thức Thực thể (Slot-aware Intent):**

Nhánh Intent dùng $Q_I$  để truy xuất từ $K_S, V_S$ của nhánh NER:

$$
C_I = \text{softmax}\!\left(\frac{Q_I K_S^T}{\sqrt{d_k}}\right) V_S
$$

$$
H_I' = \text{LayerNorm}(H_{\text{intent}} + C_I)
$$

Sau bước này, hai luồng đặc trưng đi qua **Feed-Forward Network (FFN)** để hợp nhất thông tin ẩn, tạo ra biểu diễn cập nhật cuối cùng $\hat{H}_I$ và $\hat{H}_S.$

---

### C. Giải mã Cấp độ từ (Token-level Decoders)

Khác với các mô hình truyền thống dự đoán một ý định duy nhất cho cả câu, KEHN áp dụng lý thuyết từ **Stack-Propagation**, chuyển bài toán Ý định thành bài toán **gán nhãn chuỗi (sequence labeling)**. Tại mỗi bước thời gian $i$, phân phối xác suất của từng tác vụ được tính độc lập:

$$
y_i^I = \text{softmax}(W_{\text{dec}\_I} \cdot \hat{h}_i^I)
$$

$$
y_i^S = \text{softmax}(W_{\text{dec}\_S} \cdot \hat{h}_i^S)
$$

---

**Đầu ra của Tầng 2** là hai ma trận phân phối xác suất $P(\text{Intent})$ và $P(\text{NER})$ trên toàn bộ $L$ từ, sẵn sàng để **lan truyền xếp chồng (stack-propagation)** lên Tầng 3.

### 3.3. Tầng 3: Giải mã Chuyên khoa bằng Lan truyền Xếp chồng (Stack-Propagation Topic Decoder)

Nếu Tầng 1 và Tầng 2 đóng vai trò trích xuất và tương tác các đặc trưng phụ trợ (**auxiliary tasks**), thì Tầng 3 chính là nơi giải quyết **bài toán trọng tâm (primary task)**: Phân loại câu hỏi của bệnh nhân vào các chuyên khoa y tế tương ứng.

Thay vì chỉ dùng đặc trưng văn bản thô, Tầng 3 áp dụng **cơ chế Lan truyền Xếp chồng (Stack-Propagation)** — trực tiếp sử dụng phân phối xác suất đầu ra của Intent và NER từ Tầng 2 như một dạng **"tri thức y khoa" (explicit knowledge)** để dẫn hướng cho bộ phân loại Chủ đề, giúp mô hình không phải tự học lại đặc trưng bệnh học từ đầu.

Quá trình toán học tại Tầng 3 gồm 4 bước:

---

### A. Gộp đặc trưng không gian và thời gian (Pooling)

Do độ dài câu hỏi $L$  khác nhau, các ma trận đầu ra từ các tầng dưới cần được nén thành vector kích thước cố định đại diện cho toàn câu (**sentence-level representation**).

**Đặc trưng Ngữ cảnh** $h_{\text{pool}}$  — Áp dụng **Max-pooling** theo chiều thời gian lên $H \in \mathbb{R}^{L \times d}$  (Tầng 1), giữ lại tín hiệu ngữ nghĩa mạnh nhất:

$$
h_{\text{pool}} = \text{MaxPooling}(H)
$$

**Đặc trưng Ý định**  $p_{\text{intent}}$ — Áp dụng **Mean-pooling** lên $P(\text{Intent}) \in \mathbb{R}^{L \times |I_{\text{label}}|}$ (Tầng 2), tổng hợp ý định toàn câu dựa trên sự đồng thuận của tất cả các từ:

$$
p_{\text{intent}} = \text{MeanPooling}(P(\text{Intent}))
$$

**Đặc trưng Thực thể** $p_{\text{ner}}$ — Áp dụng **Max-pooling** lên $P(\text{NER}) \in \mathbb{R}^{L \times |S_{\text{label}}|}$ , đóng vai trò như bộ **trigger**: chỉ cần một từ trong câu mang xác suất cao là Bệnh lý hoặc Thuốc, vector đại diện sẽ ghi nhận sự tồn tại của thực thể đó:

$$
p_{\text{ner}} = \text{MaxPooling}(P(\text{NER}))
$$

---

### B. Lan truyền Xếp chồng (Stack-Propagation / Concatenation)

Ba vector đặc trưng được ghép nối tạo thành **siêu vector tổng hợp** $V_{\text{topic}}$ , mang đồng thời thông tin ngữ cảnh tĩnh lẫn tri thức đa nhiệm:

$$
V_{\text{topic}} = h_{\text{pool}} \oplus p_{\text{intent}} \oplus p_{\text{ner}}
$$

---

### C. Giải mã Chuyên khoa (Topic Decoding)

$V_{\text{topic}}$ được đưa qua **Linear layer** và **Softmax** để tính phân phối xác suất $\hat{y}_{\text{topic}}$ cho 18 chuyên khoa y tế:

$$
\hat{y}_{\text{topic}} = \text{Softmax}(W_{\text{topic}} \cdot V_{\text{topic}} + b_{\text{topic}})
$$

---

### D. Tối ưu hóa với Mất cân bằng lớp (Weighted Loss Optimization)

Dữ liệu y tế thực tế đối mặt với **mất cân bằng lớp cực đoan (extreme class imbalance)**. Tầng 3 sử dụng **Weighted Cross-Entropy Loss** $\mathcal{L}_{\text{topic}}^{\text{weighted CE}}$, trong đó các chuyên khoa hiếm (như **Y học cổ truyền**) được gán trọng số phạt cao hơn nhiều lần so với các chuyên khoa phổ biến (như **Nội khoa**), ép mô hình chú ý đồng đều đến mọi lớp:

$$
\mathcal{L}_{\text{topic}}^{\text{weighted CE}} = -\sum_{c=1}^{C} w_c \cdot y_c \cdot \log(\hat{y}_c)
$$

Trong đó $w_c$ là trọng số của lớp $c$, tỉ lệ nghịch với tần suất xuất hiện của chuyên khoa đó trong tập huấn luyện.