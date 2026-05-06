# Kiến trúc OneNet cho KEHNKién

**1. Ứng dụng về Triết lý: Chuyển đổi từ Pipeline sang Joint Learning**

- **Vấn đề của hệ thống cũ:** Trong bài báo gốc, OneNet chỉ ra rằng các hệ thống hiểu ngôn ngữ nói (SLU) truyền thống thường hoạt động theo dạng đường ống (Pipeline): Dự đoán Domain (Lĩnh vực) trước, sau đó mới đến Intent (Ý định) và Slot (Thực thể). Cách làm này gây ra "hiệu ứng lan truyền lỗi" (error propagation) – nếu dự đoán Domain sai, toàn bộ kết quả Intent và Slot phía sau sẽ hỏng.
- **Cách KEHN kế thừa:** Kiến trúc KEHN áp dụng triệt để triết lý **Unified Neural Network (Mạng nơ-ron hợp nhất)** của OneNet. Thay vì tách rời, KEHN đưa cả 3 bài toán vào giải quyết đồng thời trên cùng một kiến trúc. Trong đó, định nghĩa của OneNet được ánh xạ sang bài toán Y tế của bạn như sau:
    - *Domain Prediction* → **Topic Classification** (Phân loại 18 chuyên khoa Y tế).
    - *Intent Prediction* → **Intent Detection** (Phát hiện 4 ý định của bệnh nhân).
    - *Slot Tagging* → **Medical NER** (Bóc tách 7 loại Thực thể y tế).

**2. Ứng dụng vào Tầng 1 (Đáy tháp): Shared Context Encoder**

- **Thiết kế gốc của OneNet:** Lớp dưới cùng của OneNet sử dụng một mạng BiLSTM dùng chung (Shared Bidirectional LSTM) để đọc chuỗi văn bản và tạo ra các biểu diễn ngữ cảnh cho từng từ.
- **Sự cải tiến trong KEHN:** Tầng 1 của KEHN kế thừa chính xác cấu trúc "Shared Encoder" này. Toàn bộ văn bản đầu vào đều phải đi qua một bộ mã hóa chung trước khi rẽ nhánh đi làm các nhiệm vụ khác nhau.
    - *Điểm nâng cấp:* Thay vì sử dụng nhúng từ/ký tự (Word/Char Embeddings) đơn giản như OneNet, KEHN nâng cấp bộ trích xuất đặc trưng bằng mô hình ngôn ngữ tiền huấn luyện **PhoBERT** hoặc **ViHealthBERT**, sau đó mới đẩy qua lớp BiLSTM để tăng cường khả năng bắt sự phụ thuộc xa (long-term dependencies) trong câu hỏi y tế phức tạp.

**3. Ứng dụng vào Tầng 3 (Đỉnh tháp): Khối phân loại Domain / Topic**

- **Thiết kế gốc của OneNet:** Để dự đoán Domain, OneNet sử dụng một mạng truyền thẳng (Feedforward network) nhận đầu vào là sự tổng hợp (sum) của các trạng thái ẩn từ lớp BiLSTM dùng chung.
- **Sự cải tiến trong KEHN:** Tầng 3 (Topic Decoder) của KEHN chính là phiên bản nâng cấp của khối Domain Classifier trong OneNet. Mặc dù vẫn giữ nguyên mục tiêu là bài toán phân loại tổng thể cho cả câu (sentence classification), KEHN không chỉ dùng đặc trưng từ Tầng 1 như OneNet. Nó áp dụng cơ chế *Stack-Propagation* để hút thêm phân phối xác suất Ý định và Thực thể từ Tầng 2 lên, ghép nối lại thành siêu vector *Vtopic* trước khi đưa qua lớp Linear và Softmax để chốt kết quả chuyên khoa.

**4. Ứng dụng vào Chiến lược Huấn luyện: Curriculum Learning (Học theo giáo trình)**

Đây là di sản quan trọng nhất của OneNet được apply vào KEHN. Việc ép một mô hình học cùng lúc 3 tác vụ khó ngay từ đầu thường khiến hàm Loss bị nhiễu và khó hội tụ.

- **Thiết kế gốc của OneNet:** OneNet đề xuất huấn luyện mô hình theo từng giai đoạn (Curriculum learning): Đầu tiên tối ưu hóa bộ phân loại Domain, sau đó tối ưu Intent, tiếp đến là huấn luyện chung (Joint) Domain + Intent, và cuối cùng mới tối ưu toàn bộ cả 3 hàm loss. Bài báo OneNet chứng minh rằng nếu không có chiến lược này, hiệu suất dự đoán Intent có thể giảm 1.7%.
- **Cách apply vào KEHN:** KEHN áp dụng chiến lược học theo giáo trình này và điều chỉnh lại thành **4 Phase (Giai đoạn)** tinh vi hơn:
    - **Phase 1 (Topic Only - Epoch 1-3):** Khóa (Freeze) Tầng 2, chỉ ép Tầng 1 học cách phân loại Topic giống hệt cách OneNet khởi động.
    - **Phase 2 (Mining Only - Epoch 4-6):** Khóa Tầng 3, ép Tầng 2 tập trung học bóc tách NER và Intent cho sắc bén.
    - **Phase 3 (Joint without Propagation - Epoch 7-10):** Cho phép 3 tác vụ học chung trên Tầng 1 (giống với Phase 3 của OneNet), nhưng tạm thời tắt luồng Stack-Propagation.
    - **Phase 4 (Full Stack-Propagation - Epoch 11-30):** Bật toàn bộ luồng truyền thông tin. Tối ưu hóa hàm Joint Loss tổng hợp (*Ljoint*), trong đó Topic (tác vụ chính) chiếm trọng số cao nhất (0.5), Intent (0.3) và NER (0.2).