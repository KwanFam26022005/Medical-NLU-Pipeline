# 🎤 Lời Thuyết Trình — Hệ thống Trợ lý Y tế Thông minh Tiếng Việt
**Báo cáo Tiến độ Giữa kỳ | ĐH Tôn Đức Thắng | Tháng 4, 2026**

---

## PHẦN MỞ ĐẦU

### Slide 1 — Trang tiêu đề

> Kính thưa thầy và các bạn,
>
> Hôm nay nhóm em xin trình bày đề tài **"Nghiên cứu và Xây dựng Hệ thống Trợ lý Y tế Thông minh Tiếng Việt"** — một hệ thống kết hợp kiến trúc Phân rã Đa tác vụ về NLU với cơ chế Truy xuất có Kiểm chứng, hay còn gọi là CRAG.
>
> Đây là báo cáo tiến độ giữa kỳ. Trong buổi hôm nay, nhóm em sẽ tập trung vào **Giai đoạn 1** — toàn bộ phần xử lý ngôn ngữ tự nhiên đầu vào gồm 4 trạm NLU mà nhóm đã xây dựng và fine-tune thành công.

---

### Slide 2 — Phân công thành viên

> Nhóm em gồm 2 thành viên. Bạn **Phạm Hồng Đăng Khoa** phụ trách thiết kế kiến trúc tổng thể, xây dựng Trạm 1 về giải nghĩa từ viết tắt, Trạm 2B về phân loại chuyên khoa, đồng thời phụ trách phần báo cáo và slide. Bạn **Nguyễn Hoàng Khang** phụ trách phát triển Trạm 2A về NER, Trạm 2C về phân loại ý định, và tích hợp luồng dữ liệu giữa các trạm. Cả hai thành viên đều đóng góp 100%.

---

### Slide 3 & 4 — Mục lục / Nội dung trình bày

> Bài báo cáo hôm nay gồm 5 phần chính. Sau phần giới thiệu bài toán chung, chúng em sẽ lần lượt trình bày từng trạm: Trạm 1 giải nghĩa từ viết tắt y tế, Trạm 2A nhận dạng thực thể y tế, Trạm 2B phân loại chuyên khoa, và Trạm 2C phân loại đa ý định.
>
> Mỗi trạm chúng em sẽ trình bày theo cấu trúc nhất quán: bắt đầu từ **dữ liệu**, đi qua **tiền xử lý**, rồi đến **kiến trúc mô hình**, và kết thúc bằng **kết quả đánh giá**.

---

## PHẦN 1 — GIỚI THIỆU BÀI TOÁN

### Slide 5 — Đặt vấn đề

> Bài toán mà nhóm em hướng đến xuất phát từ một thực tế rất đặc thù của y tế trực tuyến Việt Nam. Khi bệnh nhân đặt câu hỏi, họ thường viết theo kiểu hội thoại rất tự nhiên, đầy từ lóng, viết tắt và sai cấu trúc. Ví dụ điển hình là câu: *"bs ơi em hay đau dd, đi siêu âm ổ bg thì bth..."* — một câu mà người ngoài ngành rất khó hiểu ngay được "dd" là dạ dày, "bg" là bụng gan, "bth" là bình thường.
>
> Điều này tạo ra 4 rào cản kỹ thuật cực kỳ nghiêm trọng. Thứ nhất, **khuyết từ vựng** — các mô hình ngôn ngữ lớn (LLM) tổng quát không thể tự nội suy được nghĩa của các viết tắt chuyên ngành y. Thứ hai, **lệch bối cảnh** — nếu đưa thẳng câu hỏi thô vào hệ thống RAG, nó sẽ tìm sai tài liệu vì từ khóa nhiễu. Thứ ba, **nguy cơ ảo giác** — LLM có thể sinh ra phác đồ điều trị sai lệch nếu không có thông tin ngữ cảnh đủ chuẩn. Thứ tư, **tính phi tuyến** — ý định của người bệnh thường phân mảnh trong cùng một câu hỏi.
>
> Từ đây, nhóm em đặt ra kết luận: hệ thống **không thể phó mặc** độ chính xác cho đầu vào thô. Bắt buộc phải thiết lập một **màng lọc ngôn ngữ tự nhiên** — tức là NLU Pipeline — trước khi đưa thông tin vào RAG và LLM.

---

### Slide 6 — Sơ đồ Kiến trúc Tổng thể

> Đây là kiến trúc tổng thể của hệ thống mà nhóm em đang xây dựng, gồm 4 tầng.
>
> **Tầng 1** là Interface — người dùng tương tác qua giao diện Streamlit và API FastAPI. **Tầng 2** là trái tim của hệ thống — Medical NLU Pipeline với 4 trạm xử lý song song. **Tầng 3** là hệ thống RAG với Vector Database và Semantic Retriever. **Tầng 4** là bộ sinh câu trả lời sử dụng LLM như Gemini Pro hoặc Llama 3, kết hợp với bộ lọc an toàn y tế.
>
> Trọng tâm của bản báo cáo giữa kỳ hôm nay là **Tầng 2** — toàn bộ NLU Pipeline mà nhóm em đã hoàn thành.

---

### Slide 7 — Đề xuất Giải pháp & Tiến độ

> Về giải pháp, nhóm em chọn kiến trúc phân tầng — tức là tách biệt rõ ràng từng bước xử lý. Điều này giúp từng module có thể được tối ưu độc lập mà không ảnh hưởng đến nhau.
>
> Về tiến độ: giai đoạn 1 đã **hoàn thành**. Nhóm đã xây dựng xong bộ dữ liệu từ các nguồn uy tín, giải quyết vấn đề mất cân bằng nhãn, và fine-tune thành công cả 4 trạm với kết quả đánh giá tốt. Giai đoạn 2 sau giữa kỳ sẽ tập trung vào xây dựng hệ thống Corrective RAG và triển khai API song song.

---

### Slide 8 — Minh họa Đầu ra: Cơ chế Hội tụ Đa tác vụ

> Trước khi đi vào từng trạm, nhóm em muốn cho thầy thấy **kết quả cuối cùng** của toàn bộ pipeline trông như thế nào.
>
> Lấy ví dụ câu hỏi thô: *"bs ơi em hay đau dd, có nguy hiểm không?"*. Câu này đi qua 4 trạm song song. Trạm 1 giải nghĩa: "bs" thành "bác sĩ", "dd" thành "dạ dày". Trạm 2A trích xuất thực thể: "đau dạ dày" được gán nhãn Symptom với vị trí chính xác. Trạm 2B xác định chuyên khoa: Gastroenterology với độ tin cậy 0.98. Trạm 2C phân tích ý định: Diagnosis 0.85, Severity 0.92.
>
> Tất cả kết quả này được hợp nhất vào một khối **Unified Context JSON** — đây là "kim chỉ nam" định hướng cho Query Rewriter và Vector Router ở tầng RAG phía sau. Thay vì gửi một câu hỏi nhiễu loạn, hệ thống gửi đi một cấu trúc ngữ nghĩa hoàn chỉnh và chuẩn xác.

---

## PHẦN 2 — TRẠM 1: GIẢI NGHĨA TỪ VIẾT TẮT Y TẾ (WSD)

### Slide 9 — Kiến trúc Tập dữ liệu acrDrAid

> Bây giờ chúng ta đi vào Trạm 1. Nhiệm vụ của trạm này là **Word Sense Disambiguation** — tức là khi gặp một từ viết tắt y tế, hệ thống phải xác định chính xác nghĩa nào phù hợp với ngữ cảnh cụ thể đó.
>
> Dữ liệu nhóm sử dụng là bộ **acrDrAid**, được xây dựng từ hai nguồn kết hợp: một là *Query Context* — các câu hỏi thực tế chứa từ viết tắt; hai là *Knowledge Base* — từ điển y khoa ánh xạ từng viết tắt với các nghĩa mở rộng tương ứng.
>
> Cơ chế sinh dữ liệu hoạt động như sau: với mỗi câu hỏi chứa từ viết tắt, hệ thống tra từ điển để lấy tất cả nghĩa có thể, rồi sinh ra các cặp câu. Cặp nào ghép đúng nghĩa sẽ được gán nhãn 1 (Positive), cặp ghép sai nghĩa được gán nhãn 0 (Negative).
>
> Về thống kê: bộ dữ liệu có **135 từ viết tắt**, **424 nghĩa mở rộng**, độ đa nghĩa trung bình là **3.14 nghĩa/từ**. Tập huấn luyện gồm 4.000 mẫu, kiểm định 523 mẫu, thử nghiệm 1.130 mẫu.

---

### Slide 10 — EDA 1: Sự Đa nghĩa & Bế tắc của Multi-class

> Nhìn vào biểu đồ EDA, chúng ta thấy ngay mức độ phức tạp của bài toán. Từ viết tắt **"tt"** dẫn đầu với tới **17 nghĩa hoàn toàn khác biệt**. Các từ như "bt", "nt", "tp" mỗi cái có 8 nghĩa. Đây là bức tranh điển hình của sự đa nghĩa cực đoan trong y khoa Tiếng Việt.
>
> Điều này dẫn đến **tử huyệt của kiến trúc Multi-class Classification cổ điển**: nếu dùng mô hình phân loại thông thường, ta buộc mô hình phải chọn 1 trong 280 nhãn cố định. Vấn đề lớn là nếu gặp một từ viết tắt **chưa xuất hiện trong tập huấn luyện**, mô hình không có class nào để chọn, dẫn đến đoán bừa và sai hoàn toàn.
>
> Đây chính là lý do nhóm em chuyển hướng sang kiến trúc **Cross-Encoder**. Thay vì "nhớ nhãn", Cross-Encoder học cách **đánh giá độ tương đồng ngữ nghĩa** giữa ngữ cảnh và từng nghĩa mở rộng, cho phép xử lý được cả từ hoàn toàn mới mà mô hình chưa từng gặp.

---

### Slide 11 — EDA 2: Độ dài & Sự bùng nổ Dữ liệu

> Về phân bố độ dài: sau khi chèn thẻ đánh dấu thực thể vào ngữ cảnh, chiều dài trung vị của cặp câu chỉ khoảng **47 từ**. Điều này có nghĩa là ngưỡng `max_length=128` hoàn toàn đủ để bao trọn toàn bộ thông tin mà không bao giờ cắt đứt phần nghĩa ứng viên ở đuôi câu.
>
> Điểm quan trọng thứ hai là kỹ thuật **Hard Negative Mining**: từ 4.000 mẫu gốc, nhóm em sinh ra các cặp sai nhưng "hợp lý" — tức là cùng từ viết tắt nhưng ghép với nghĩa sai. Kết quả là bộ dữ liệu **bùng nổ thành 12.656 cặp**, tỷ lệ 1:2.2. Kỹ thuật này ép buộc mô hình phải học sâu ngữ nghĩa thay vì đoán theo tần suất, đồng thời chống overfitting rất hiệu quả.

---

### Slide 12 — Tiền xử lý 1: Tổng quan Luồng Dữ liệu

> Pipeline tiền xử lý của Trạm 1 gồm **4 bước tuần tự**.
>
> **Bước 1 — Data Extraction**: đọc file JSON, trích xuất văn bản, vị trí bắt đầu và độ dài của từ viết tắt trong câu. **Bước 2 — Entity Marking**: chèn cặp thẻ `<e>` và `</e>` bao quanh từ viết tắt để "đánh dấu" vị trí cần chú ý cho mô hình. **Bước 3 — Pair Generation**: ghép câu ngữ cảnh đã đánh dấu với từng nghĩa ứng viên, gán nhãn 1 hoặc 0 tương ứng. **Bước 4 — Encode & Collate**: tokenize theo chuẩn Cross-Encoder `[CLS]...[SEP]`, flatten batch và dynamic pad.
>
> Triết lý cốt lõi ở đây là chuyển hóa bài toán từ việc "nhìn 1 câu, đoán 1 nhãn" sang "nhìn 2 chuỗi, đo độ tương đồng ngữ nghĩa".

---

### Slide 13 — Tiền xử lý 2: Cơ chế Đánh dấu Thực thể

> Slide này giải thích tại sao cặp thẻ `<e>` và `</e>` lại quan trọng đến vậy.
>
> Có 3 lý do kỹ thuật. Thứ nhất, **bảo vệ tính toàn vẹn**: hai thẻ này bắt buộc phải được đăng ký là Special Tokens trong từ điển của Tokenizer. Nếu bỏ qua bước này, Tokenizer sẽ cắt chúng thành các ký tự lẻ `<`, `e`, `>` và hoàn toàn phá hủy cấu trúc định vị.
>
> Thứ hai, **khởi tạo không gian nhúng**: vector embedding của 2 thẻ này được khởi tạo ngẫu nhiên và học được ý nghĩa thông qua quá trình fine-tuning cùng với backbone RoBERTa.
>
> Thứ ba, và quan trọng nhất, **điều hướng Self-Attention**: giống kỹ thuật Relation Extraction, cặp thẻ này hoạt động như biển báo vị trí. Các Attention Heads học cách dồn trọng số cực đại vào span nằm giữa `<e>` và `</e>` khi tổng hợp ngữ cảnh tại token `[CLS]`. Nói cách khác, mô hình được "chỉ điểm" chính xác vào vị trí cần giải nghĩa.

---

### Slide 14 — Tiền xử lý 3: Mã hóa Cặp câu

> Sau khi đánh dấu thực thể, bước tiếp theo là ghép cặp câu theo chuẩn Cross-Encoder. Ngữ cảnh đã đánh dấu là **Sentence A**, nghĩa ứng viên là **Sentence B**, hai phần được phân tách bởi token `[SEP]`.
>
> Chiến lược truncation cũng được thiết kế có chủ đích: nếu tổng độ dài vượt `max_length`, chỉ cắt bớt phần **Sentence A** (ngữ cảnh), còn phần **Sentence B** (nghĩa ứng viên) luôn được bảo toàn 100% thông tin. Lý do là Candidate String thường rất ngắn và chứa thông tin định nghĩa cốt lõi, không thể mất đi.
>
> Đầu ra của bước này là 3 tensor chuẩn: `input_ids`, `token_type_ids` để phân biệt Sentence A và B, và `attention_mask`.

---

### Slide 15 — Tiền xử lý 4: Flattening & Dynamic Padding

> Có một thách thức đặc thù của Cross-Encoder: mỗi câu hỏi có thể có số lượng nghĩa ứng viên khác nhau. Câu hỏi 1 có 2 nghĩa, câu hỏi 2 có 3 nghĩa — đây là dữ liệu **không đồng nhất kích thước**, DataLoader thông thường không thể xử lý được.
>
> Giải pháp là kỹ thuật **Flatten & Dynamic Pad**: phá vỡ ranh giới giữa các sample, gom tất cả cặp câu thành một danh sách 1D độc lập. Sau đó padding chỉ kéo dài đến chuỗi dài nhất trong batch hiện tại (L_max), thay vì padding đến global `max_length`.
>
> Lợi ích rất lớn: vì độ phức tạp của Self-Attention là O(L²), việc giảm L_max từ 128 xuống mức thực tế của batch giúp tiết kiệm bộ nhớ GPU đáng kể. Đây là một tối ưu kỹ thuật quan trọng khi training với tài nguyên hạn chế.

---

**💡 TÓM TẮT RÚT GỌN TIỀN XỬ LÝ - TRẠM 1:**
* **Entity Marking:** Chèn thẻ `<e>` và `</e>` để định vị chính xác từ viết tắt, giúp điều hướng Attention Heads của mô hình.
* **Data Structure:** Ghép cặp ngữ cảnh và từng nghĩa ứng viên (Cross-Encoder) thay vì mô hình phân loại nhãn thông thường.
* **Flattening & Dynamic Padding:** Gom cặp câu thành danh sách 1D và padding theo chiều dài tối đa của từng batch (L_max) để tối ưu hóa bộ nhớ GPU.

---

## PHẦN 3 — TRẠM 2A: NHẬN DẠNG THỰC THỂ Y TẾ (NER)

### Slide 16 — Token Classification & Định dạng CoNLL

> Bước vào Trạm 2A. Đây là trạm nhận dạng thực thể y tế — tức là từ câu hỏi đầu vào, hệ thống phải xác định chính xác đâu là triệu chứng, đâu là tên thuốc, đâu là thủ thuật y tế.
>
> Điểm khác biệt quan trọng cần nhấn mạnh ngay: khác với các trạm phân loại khác hoạt động ở cấp độ câu, NER là bài toán **Token Classification** — mô hình phải gán nhãn cho từng token riêng lẻ trong chuỗi.
>
> Định dạng dữ liệu là **CoNLL với chiến lược BIO**. Quy tắc rất rõ ràng: nhãn **B- (Begin)** đánh dấu token đầu tiên của một thực thể, ví dụ "đau" trong "đau thắt ngực". Nhãn **I- (Inside)** đánh dấu các token tiếp theo trong cùng một cụm thực thể, ví dụ "thắt" và "ngực". Nhãn **O (Outside)** gán cho mọi token không thuộc thực thể y tế nào.
>
> Bộ nhãn của Trạm 2A gồm 3 loại thực thể: **SYMPTOM_AND_DISEASE** cho bệnh lý và triệu chứng, **MEDICAL_PROCEDURE** cho thủ thuật, và **MEDICINE** cho tên thuốc.

---

### Slide 17 — EDA: Phân bố Độ dài & Nhãn

> Phân tích dữ liệu của Trạm 2A cho thấy hai đặc trưng quan trọng cần đối phó.
>
> Nhìn vào biểu đồ bên trái về độ dài chuỗi: các câu hỏi sau khi tách thành đơn vị CoNLL thường rất ngắn, trung vị chỉ khoảng **20-30 tokens**. Điều này là cơ sở kỹ thuật để chốt tham số `max_length=128` — ngưỡng này bao trọn an toàn mà không lãng phí bộ nhớ GPU.
>
> Biểu đồ bên phải cho thấy vấn đề nghiêm trọng hơn: hiện tượng **O-dominance** — nhãn O luôn chiếm hơn 80% tổng số tokens. Đây là mất cân bằng mang tính cấu trúc, không thể loại bỏ vì nhãn O mang thông tin nền cần thiết. Nếu không có biện pháp kiến trúc đặc biệt, mô hình sẽ học theo lối tắt: luôn đoán O cho mọi token để giảm Loss, và hoàn toàn bỏ sót các thực thể y tế quan trọng.
>
> Đây chính là lý do tại sao kiến trúc CRF được đưa vào — chúng em sẽ giải thích ở phần sau.

---

### Slide 18 — Tiền xử lý: Cơ chế Subword Alignment

> Đây là slide về một vấn đề kỹ thuật tinh tế mà nhóm đã phải giải quyết trong quá trình xây dựng Trạm 2A.
>
> ViHealthBERT sử dụng thuật toán BPE (Byte Pair Encoding), có thể cắt vụn một từ phức tạp thành nhiều sub-tokens. Ví dụ từ "Paracetamol" bị cắt thành "para", "##ceta", "##mol". Vấn đề: nhãn gốc chỉ có 1 — đó là `B-Drug` — nhưng giờ ta có 3 sub-tokens. Gán nhãn thế nào?
>
> Nếu gán `B-Drug` cho cả 3 sub-token, mô hình sẽ bối rối vì vi phạm quy tắc BIO — mỗi thực thể chỉ có 1 Begin. Do đó, nhóm áp dụng **chiến lược First-token Rule**: chỉ giữ nhãn gốc cho **sub-token đầu tiên** của mỗi từ, còn các sub-token hậu tố được gán nhãn đặc biệt **-100**.
>
> Tại sao -100? Đây là index mặc định mà hàm `CrossEntropyLoss` trong PyTorch sẽ **hoàn toàn bỏ qua** khi tính toán Gradient. Điều này ngăn chặn việc phạt oan mô hình vì sự mơ hồ không thể tránh được của quá trình tokenize.

---

**💡 TÓM TẮT RÚT GỌN TIỀN XỬ LÝ - TRẠM 2A:**
* **Subword Alignment:** Áp dụng First-token Rule để chỉ gán nhãn BIO gốc cho sub-token đầu tiên (do BPE có thể cắt vụn từ).
* **Label Masking (-100):** Các sub-token hậu tố được gán nhãn `-100` để ép hàm Loss tự động bỏ qua, tránh phạt oan mô hình và bảo toàn quy tắc BIO.

---

## PHẦN 4 — TRẠM 2B: PHÂN LOẠI CHUYÊN KHOA (TOPIC)

### Slide 19 — Hệ sinh thái Dữ liệu Chuyên khoa

> Trạm 2B có nhiệm vụ xác định **câu hỏi này thuộc chuyên khoa y tế nào** — Tim mạch, Thần kinh, Tiêu hóa, hay 15 chuyên khoa khác.
>
> Điểm đặc biệt của bộ dữ liệu là tính **đa nguồn**: 70.5% dữ liệu đến từ Vinmec — nguồn bệnh án lâm sàng chuyên sâu; 23.8% từ Bệnh viện Tâm Anh — các ca bệnh phức tạp; 5.7% từ AloBacsi — văn phong hỏi đáp bình dân. Sự pha trộn này cực kỳ có giá trị vì nó giúp mô hình tiếp xúc với cả hai đầu spectrum: văn phong chuyên môn cao và văn phong hội thoại thường ngày.
>
> Tổng quy mô tập Train là **10.216 mẫu**, ban đầu có 39 nhãn thô, sau tiền xử lý được quy hoạch còn **18 nhãn lõi**. Loại bài toán là **Single-label** — mỗi câu hỏi chỉ thuộc về đúng một chuyên khoa.

---

### Slide 20 — EDA 1: Đặc trưng Nguồn & Phân bố Độ dài

> Biểu đồ bên trái xác nhận tỷ trọng đóng góp từ 3 nguồn như đã đề cập. Điểm đáng chú ý là Vinmec đóng vai trò xương sống còn AloBacsi bổ sung văn phong Q&A bình dân — sự đa dạng này là chìa khóa chống overfitting.
>
> Biểu đồ bên phải về phân bố độ dài văn bản cho thấy phân phối **lệch phải điển hình**: chiều dài trung vị là **84 từ**, đuôi phải kéo dài đến vài trăm từ. Từ phân tích này, nhóm chốt tham số `max_length=256` — ngưỡng này bao trọn hơn 95% dữ liệu lâm sàng mà không lãng phí VRAM cho padding dư thừa.

---

### Slide 21 — EDA 2: Phân mảnh & Mất cân bằng Cực đoan

> Đây là slide phân tích vấn đề nghiêm trọng nhất của Trạm 2B.
>
> Biểu đồ bên trái cho thấy 39 nhãn gốc có **chồng lấn ngữ nghĩa nghiêm trọng**. Ví dụ điển hình: "neurosurgery" và "neurology" — cả hai đều liên quan đến thần kinh, nhưng một cái là phẫu thuật, một cái là nội thần kinh. Đối với bệnh nhân mô tả triệu chứng chung chung, ranh giới này gần như vô hình. Điều này phá vỡ decision boundary của mô hình và bắt buộc phải áp dụng bước Canonical Mapping.
>
> Biểu đồ bên phải — đường cong Lorenz — cho thấy mức độ mất cân bằng thuộc loại **cực đoan**: chỉ khoảng 20% chuyên khoa chiếm hơn 80% tổng mẫu, tỷ lệ mất cân bằng chạm ngưỡng **159.2x** giữa chuyên khoa nhiều nhất và ít nhất. Con số này bác bỏ hoàn toàn khả năng dùng hàm mất mát Standard Cross-Entropy thông thường.

---

### Slide 22 — Tiền xử lý 1: Pipeline Overview (4 bước)

> Quy trình tiền xử lý của Trạm 2B gồm 4 bước tuần tự, được thiết kế như **4 màng lọc chồng nhau**.
>
> **Bước 1 — Domain Cleaning**: khử nhiễu HTML, boilerplate, và đặc biệt là loại bỏ dấu gạch dưới word-segmentation `_` vì Trạm 2B dùng Syllable Tokenizer không cần tách từ. **Bước 2 — Canonical Mapping**: gộp 39 nhãn thô về 18 nhãn lõi, loại bỏ các nhãn phi lâm sàng như radiology, laboratory. **Bước 3 — Double-Filter Augment**: dùng Self-Training trên kho FAQ để tăng cường dữ liệu cho nhãn thiểu số, với 2 bộ lọc nghiêm ngặt. **Bước 4 — Weighted Collate**: tính trọng số lớp và thực hiện Dynamic Padding.

---

### Slide 23 — Tiền xử lý 2: Làm sạch văn bản (7-Step Filter)

> Slide này cho thấy cụ thể quá trình làm sạch văn bản qua **7 bước lọc** với regex.
>
> Nhìn vào ví dụ: từ một đoạn văn bản thô lộn xộn với HTML tags, URL Vinmec, số điện thoại và lời chào cảm ơn — sau 7 bước lọc, chỉ còn lại đúng phần nội dung lâm sàng cốt lõi: *"Dạo này tôi hay bị đau bụng vùng thượng vị, đau âm ỉ liên tục. Đặc biệt là sau khi ăn đồ cay."*
>
> Triết lý của bước này là **tối ưu hóa mật độ thông tin**: 100% các token đưa vào mô hình phải là triệu chứng lâm sàng có giá trị, không còn một token nhiễu nào. Mỗi token trong `max_length` đều quý giá — không được lãng phí cho lời chào hay link quảng cáo.

---

### Slide 24 — Tiền xử lý 2A: Đồng nhất Nhãn — Canonical Mapping

> Đây là bước giải quyết vấn đề chồng lấn ngữ nghĩa. Ý tưởng là **gộp các chuyên khoa hẹp có chung đặc tính lâm sàng** vào cùng một nhãn lõi.
>
> Ví dụ minh họa: "hepatology" (gan mật) và "gastroenterology" (tiêu hóa) — cả hai đều được gộp vào nhãn **Gastroenterology** vì bệnh nhân thường không thể phân biệt được hai khoa này qua triệu chứng. Tương tự, "neonatology" (sơ sinh) và "pediatrics" (nhi khoa) được gộp thành **Pediatrics**.
>
> Mục tiêu là tạo ra ranh giới quyết định (Decision Boundary) **đủ sắc nét** để mô hình có thể học hiệu quả. Gộp nhãn không phải là mất thông tin — đó là thiết kế taxonomy thông minh phù hợp với khả năng phân biệt từ câu hỏi của bệnh nhân.

---

### Slide 25 — Tiền xử lý 2B: Khử nhiễu & Kết quả Phân bố

> Song song với Canonical Mapping, nhóm áp dụng **Exclusion Principle**: loại bỏ hoàn toàn các nhãn như "radiology", "laboratory", "research".
>
> Lý do: các nhãn này thiên về cận lâm sàng thay vì bệnh lý thực thể. Một câu hỏi như "xét nghiệm máu nên làm ở đâu?" không liên quan đến chuyên khoa bệnh lý — đây là câu hỏi về dịch vụ, không phải về bệnh. Giữ lại chúng sẽ ép mô hình học nhiễu và làm giảm chất lượng tổng thể.
>
> Kết quả sau khi quy hoạch: dù đã chuẩn hóa về 18 nhãn, mất cân bằng vẫn còn nghiêm trọng — tỷ lệ 159.2x giữa Sản khoa (2548 mẫu) và Dinh dưỡng (16 mẫu). Đây là tiền đề bắt buộc cho bước Self-Training ở phần tiếp theo.

---

### Slide 26 — Tiền xử lý 3: Màng lọc Kép (Double-Filter Pseudo-Labeling)

> Đây là giải pháp sáng tạo nhất của Trạm 2B để giải quyết mất cân bằng cực đoan mà **không dùng Oversampling thông thường**.
>
> Cơ chế: nhóm có một kho FAQ y tế lớn với khoảng 10.000 mẫu chưa có nhãn. Dùng mô hình Teacher đã train sơ bộ để dự đoán nhãn cho kho này, rồi áp dụng **hai bộ lọc chồng nhau** trước khi đưa vào tập Train.
>
> **Filter 1** yêu cầu độ tin cậy Softmax **>= 0.95** — chỉ lấy những mẫu mà mô hình cực kỳ chắc chắn, tránh nhãn giả nhiễu do domain shift. **Filter 2** chỉ cho phép các mẫu thuộc **nhóm thiểu số** đi qua — kiên quyết chặn Sản và Nhi vì nếu augment thêm nhãn đa số, mất cân bằng sẽ bị khuếch đại thêm.
>
> Sự kết hợp hai bộ lọc này là **học bán giám sát có kiểm soát**: an toàn về chất lượng nhãn, và mục tiêu về phân bố.

---

**💡 TÓM TẮT RÚT GỌN TIỀN XỬ LÝ - TRẠM 2B:**
* **Làm sạch (7-Step Filter):** Loại bỏ hoàn toàn nhiễu bằng regex (HTML, URL, chào hỏi) để tối ưu mật độ thông tin lâm sàng.
* **Canonical Mapping:** Gộp 39 nhãn gốc có sự chồng lấn về 18 nhãn lõi sắc nét, loại bỏ các nhãn cận lâm sàng/dịch vụ.
* **Màng lọc kép (Double-Filter):** Bổ sung dữ liệu cho nhóm thiểu số thông qua Self-Training có kiểm soát với hai điều kiện khắt khe (chỉ lấy nhãn thiểu số và độ tin cậy >= 0.95).

---

## PHẦN 5 — TRẠM 2C: PHÂN LOẠI ĐA Ý ĐỊNH Y TẾ (INTENT)

### Slide 27 — Bản chất của Đa ý định (Multi-label)

> Trạm 2C là trạm phức tạp nhất về mặt bài toán. Nhiệm vụ là xác định **ý định của bệnh nhân** — nhưng không phải chỉ một ý định, mà có thể là nhiều ý định cùng lúc.
>
> Hệ thống có 4 nhãn ý định: **Diagnosis** — hỏi bệnh là bệnh gì; **Treatment** — hỏi cách điều trị; **Severity** — hỏi mức độ nguy hiểm; **Cause** — hỏi nguyên nhân.
>
> Nhìn vào ví dụ: câu *"Bác sĩ ơi, em hay đau thắt ngực lan ra tay trái, bệnh này có nguy hiểm không và chữa thế nào ạ?"* — câu này đồng thời chứa ý định hỏi về **Treatment** và **Severity**. Nếu dùng Softmax như Trạm 2B, mô hình chỉ được chọn 1 trong 4, và chắc chắn sẽ bỏ sót một phần quan trọng.
>
> Do đó, kiến trúc **bắt buộc phải dùng Sigmoid độc lập** cho từng nhãn, thay vì Softmax chia sẻ xác suất. Mô hình không còn chọn "1 trong N", mà trả lời **N câu hỏi Có/Không song song và độc lập**.

---

### Slide 28 — EDA: Mất cân bằng Cực đoan

> Biểu đồ EDA của Trạm 2C phơi bày một thực tế rất khắc nghiệt. Nhãn **Diagnosis** xuất hiện 3.471 lần trong tập Train, trong khi **Cause** chỉ có 492 lần — chênh lệch gấp khoảng **7 lần**.
>
> Trong ngữ cảnh y tế, đây không chỉ là vấn đề kỹ thuật mà còn là vấn đề an toàn. Nếu mô hình bỏ sót nhãn **Severity** — tức là câu hỏi "có nguy hiểm không" — hệ thống sẽ không biết rằng người dùng đang lo lắng về tính mạng, và prompt sinh ra sẽ thiếu đi phần trả lời quan trọng nhất.
>
> Nếu dùng hàm Loss thông thường, Gradient sẽ bị thống trị bởi nhãn Diagnosis. Mô hình sẽ hình thành thói quen **lười biếng**: luôn dự đoán 0 cho các nhãn hiếm để tối ưu Loss tổng — một kết quả hoàn toàn vô dụng trong thực tế.

---

### Slide 29 — Tiền xử lý: Cơ chế Multi-hot Encoding

> Bước tiền xử lý đầu tiên của Trạm 2C là chuyển đổi nhãn văn bản sang **vector nhị phân multi-hot**.
>
> Ví dụ câu *"Bệnh này chữa sao và có chết không?"* — phân tích ngữ nghĩa xác định câu này chứa ý định **Treatment** (chữa thế nào) và **Severity** (có chết không). Kết quả là vector `[0, 1, 1, 0]` tương ứng với 4 chiều `[Diagnosis, Treatment, Severity, Cause]`.
>
> Thiết kế này có ý nghĩa toán học quan trọng: hàm Loss có thể tính toán **song song** sự sai lệch trên cả 4 chiều cùng một lúc, mỗi chiều là một bài toán binary classification hoàn toàn độc lập.

---

### Slide 30 — Tiền xử lý: Chuẩn hóa Trọng số Động (pos_weight)

> Đây là giải pháp kỹ thuật cốt lõi để trị vấn đề mất cân bằng của Trạm 2C.
>
> Công thức tính `pos_weight` cho mỗi nhãn c rất trực quan: bằng số mẫu Negative chia cho số mẫu Positive. Nhãn càng hiếm, hệ số phạt càng lớn. Lấy ví dụ nhãn Cause: với tổng N=4000 và N_c=492, `pos_weight` tính ra khoảng **7.13**. Nghĩa là mỗi khi mô hình bỏ sót một câu hỏi về nguyên nhân bệnh, hình phạt (Gradient) sẽ **lớn gấp 7 lần** so với khi bỏ sót nhãn Diagnosis phổ biến.
>
> Điều đặc biệt là hệ số này được **tính toán tự động trong Data Loader**, không phải hard-code. Mỗi lần bộ dữ liệu thay đổi, trọng số tự cập nhật tương ứng.
>
> Kết hợp với hàm mất mát **Asymmetric Loss (ASL)** — vừa down-weight các nhãn Negative dễ đoán, vừa up-weight các nhãn Positive khó đoán — đây là kiến trúc SOTA cho bài toán multi-label imbalanced. Kết quả chứng minh điều này: F1 của nhãn Cause đạt **0.9941**, Macro-F1 tổng thể đạt **0.9688**.
>
> Đây cũng là điểm kết thúc phần trình bày về dữ liệu và tiền xử lý. Nhóm em xin mời thầy đặt câu hỏi hoặc tiếp tục sang phần kiến trúc mô hình.

---

**💡 TÓM TẮT RÚT GỌN TIỀN XỬ LÝ - TRẠM 2C:**
* **Multi-hot Encoding:** Chuyển đổi nhãn thành vector nhị phân (ví dụ: `[0, 1, 1, 0]`) để xử lý bài toán Multi-label thành các phép phân loại nhị phân song song.
* **Trọng số động (pos_weight):** Tự động tính toán hệ số phạt cho nhãn hiếm ngay trong Data Loader, kết hợp cùng Asymmetric Loss (ASL) để giải quyết triệt để vấn đề mất cân bằng cực đoan.

---

## LỜI KẾT — TỔNG KẾT & HỎI ĐÁP

> Kính thưa thầy và các bạn, nhóm em xin tổng kết lại những gì đã trình bày hôm nay.
>
> Chúng em đã hoàn thành **toàn bộ Giai đoạn 1** của hệ thống: từ thiết kế pipeline dữ liệu, giải quyết mất cân bằng đa tầng trên cả 4 trạm, đến fine-tune thành công từng mô hình.
>
> Kết quả nổi bật: Trạm 1 WSD xử lý được cả từ viết tắt chưa từng thấy nhờ Cross-Encoder; Trạm 2A NER đạt Strict F1 ~80% với bộ lọc CRF; Trạm 2B Topic đạt Macro-F1 >90% sau Self-Training; và Trạm 2C Intent đạt Macro-F1 0.9688 với F1 nhãn Cause đặc biệt là 0.9941 — con số chứng minh ASL hoạt động xuất sắc.
>
> Giai đoạn tiếp theo nhóm sẽ xây dựng hệ thống Corrective RAG và tích hợp toàn bộ pipeline end-to-end.
>
> Nhóm em xin cảm ơn thầy đã lắng nghe và rất mong nhận được nhận xét, góp ý từ phía thầy.

---

*📝 Ghi chú sử dụng: Mỗi đoạn in nghiêng trong dấu `>` là lời thuyết trình đọc trực tiếp. Các tiêu đề chỉ để định vị slide tương ứng, không đọc ra.*