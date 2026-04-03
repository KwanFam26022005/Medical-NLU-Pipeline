# 📐 Kiến trúc Chi tiết: RAG Pipeline — Retrieval-Augmented Generation cho Chatbot Y tế (Trạm 3)

> **Module:** Trạm 3 — RAG-based Answer Generation  
> **Vai trò:** Nhận metadata NLU (topic, entities, intent) + clean_text → Truy xuất tri thức → Sinh câu trả lời có trích dẫn  
> **Upstream:** Trạm 1 (Acronym WSD) + Trạm 2A/2B/2C (NER, Topic, Intent)  
> **Downstream:** Response API → Frontend (Chatbot Widget / Mobile App)

---

## 1. Hồ sơ Dữ liệu & Phân tích Khám phá (EDA) — Corpus Tri thức

Trước khi lựa chọn kiến trúc RAG, cần hiểu rõ bản chất corpus tri thức mà hệ thống sẽ phục vụ.

### 1.1 Nguồn Tri thức Hiện có (Knowledge Sources)

Hệ thống thừa hưởng một kho dữ liệu y tế tiếng Việt đa dạng, đã được thu thập và chuẩn hóa trong các giai đoạn trước:

| Nguồn | Định dạng | Quy mô ước tính | Đặc trưng nội dung |
|---|---|---|---|
| `data/FAQ_summarization/` | `.txt.src` / `.txt.tgt.tagged` | ~10,000+ cặp Q/A | Cặp Hỏi-Đáp y tế đa chuyên khoa, đã phân tách source/target |
| `data/train_ml.csv` (Vinmec) | CSV | ~4,600 rows | Câu hỏi bệnh nhân + nhãn topic (xương sống dữ liệu, >60%) |
| `data/alobacsi_processed.csv` | CSV | ~1,500 rows | Q&A bình dân, văn phong hội thoại tự nhiên |
| `data/ml_training_data_tamanh.csv` | CSV | ~3,000 rows | Ca bệnh chuyên sâu từ BV Tâm Anh |

### 1.2 Thách thức Đặc thù của Domain Y tế Việt Nam

- **Ngôn ngữ hỗn hợp:** Bệnh nhân viết tắt tùy tiện (`kt`, `bn`, `XQ`), xen lẫn thuật ngữ Latinh (`PET/CT`, `MRI`) với tiếng Việt bình dân — Trạm 1 (WSD) đã giải quyết lớp này.
- **Semantic overlap giữa các khoa:** "Đau bụng" có thể thuộc Tiêu hóa, Sản phụ khoa, hay Tiết niệu — Trạm 2B (Topic) cung cấp tín hiệu phân loại, nhưng RAG cần thêm bước lọc ngữ cảnh.
- **High-stakes domain:** Câu trả lời sai có thể gây hậu quả nghiêm trọng → buộc phải có cơ chế **self-correction** và **citation** (trích dẫn nguồn).
- **Long-tail distribution:** Các khoa hiếm (Dinh dưỡng, Nhãn khoa, Y học cổ truyền) có rất ít FAQ → retrieval phải hoạt động tốt ngay cả khi corpus mỏng.

> **💡 Insight:** Không một kiến trúc RAG nào có thể hoạt động tốt trên domain y tế nếu thiếu **guardrails** (rào chắn an toàn). Đây là yếu tố quyết định mọi lựa chọn kiến trúc phía dưới.

---

## 2. Phần 1 — Lựa chọn Kiến trúc RAG Tối ưu

Ba kiến trúc dưới đây được chọn lọc dựa trên: (1) mức độ phù hợp với pipeline NLU đã có, (2) khả năng đảm bảo an toàn y tế, (3) tính khả thi triển khai trên Colab/self-hosted.

---

### 2.1 Kiến trúc A: Advanced RAG (Hybrid Search + Reranker)

> **Paper tham chiếu:** Gao et al., *"Retrieval-Augmented Generation for Large Language Models: A Survey"*, arXiv:2312.10997, 2023.

#### Mô tả Kiến trúc

Advanced RAG cải tiến Naive RAG bằng cách thêm **pre-retrieval optimization** (tối ưu trước khi tìm) và **post-retrieval refinement** (tinh chỉnh sau khi tìm):

```
User Query
    │
    ▼
┌──────────────────┐
│ Pre-Retrieval    │  ← Query Rewriting, HyDE, NLU metadata injection
│ Optimization     │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Hybrid Retrieval │  ← Dense (bge-m3) + Sparse (BM25) 
│ (Stage 1)        │     + Metadata Filtering (topic)
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Cross-Encoder    │  ← ViRanker / bge-reranker-v2-m3
│ Reranker (Stage 2)│    Top-50 → Top-5
└───────┬──────────┘
        ▼
┌──────────────────┐
│ LLM Generation   │  ← Grounded answer + Citations
│ + Prompt Template│
└──────────────────┘
```

#### Lý do Phù hợp với Đề án

| Yếu tố | Phân tích |
|---|---|
| **NLU-Guided Retrieval** | Output NLU (topic, entities, intent) trở thành **pre-retrieval filter** — thu hẹp không gian tìm kiếm 5-10x, giảm noise triệt để |
| **Hybrid Search** | Thuật ngữ y tế (tên thuốc, mã ICD) cần BM25 (exact match); triệu chứng mô tả tự nhiên cần Dense (semantic match) → phải dùng cả hai |
| **Reranker** | Cross-encoder chấm điểm từng cặp (query, passage) — tương tự cách Cross-Encoder Trạm 1 chấm cặp (context, expansion) → kiến trúc thống nhất |

#### Ưu điểm
- ✅ Dễ triển khai nhất trong 3 kiến trúc (pipeline tuần tự, không có vòng lặp)
- ✅ Tận dụng trực tiếp metadata NLU đã có → không cần thêm component mới
- ✅ Reranker trang bị sẵn cho tiếng Việt (ViRanker, bge-reranker-v2-m3)
- ✅ Latency thấp, phù hợp real-time chatbot

#### Hạn chế
- ❌ Không có cơ chế **self-correction** — nếu retrieval thất bại, LLM vẫn cố generate → hallucination
- ❌ Không kiểm tra output → nguy hiểm trong y tế

#### Mức độ Phức tạp: ⭐⭐ (Trung bình)

---

### 2.2 Kiến trúc B: Corrective RAG (CRAG) — ⭐ **KHUYẾN NGHỊ #1**

> **Paper tham chiếu:** Yan et al., *"Corrective Retrieval Augmented Generation"*, arXiv:2401.15884, ICML 2024.  
> **Bổ trợ:** Asai et al., *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"*, arXiv:2310.11511, NeurIPS 2023.

#### Mô tả Kiến trúc

CRAG bổ sung **3 van an toàn** (safety valves) vào Advanced RAG, tạo thành một **state machine có vòng lặp phản hồi**:

```
User Query → NLU Pipeline → {topic, entities, intent, clean_text}
    │
    ▼
┌──────────────────────┐
│ ① Hybrid Retrieval   │  Dense + Sparse + Topic Filter
│    + Reranker         │  Top-50 → Top-5
└───────┬──────────────┘
        ▼
┌──────────────────────┐
│ ② Document Grader    │  LLM-as-Judge: "Tài liệu này có liên quan
│    (Van an toàn #1)  │   đến câu hỏi y tế không?"
└───────┬──────────────┘
        │
   ┌────┴────┐
   │         │
Relevant   Not Relevant
   │         │
   ▼         ▼
Generate   ┌──────────────────┐
   │       │ ③ Query Rewriter  │  Viết lại truy vấn dựa trên
   │       │    (Van an toàn #2)│  entities + intent + feedback
   │       └───────┬───────────┘
   │               │
   │               ▼
   │         Re-Retrieve (quay lại ①)
   │         (tối đa 2 lần retry)
   ▼
┌──────────────────────┐
│ ④ LLM Generation     │  System prompt (role: bác sĩ AI)
│    + Citations        │  + Retrieved context + Clean query
└───────┬──────────────┘
        ▼
┌──────────────────────┐
│ ⑤ Hallucination Check│  "Câu trả lời có dựa trên tài liệu
│    (Van an toàn #3)  │   đã truy xuất không?"
└───────┬──────────────┘
        │
   ┌────┴────┐
   │         │
Grounded   Not Grounded
   │         │
   ▼         ▼
Return     Re-Generate (quay lại ④)
Answer     (tối đa 1 lần retry)
+ Sources
```

#### Lý do Phù hợp với Đề án

| Yếu tố | Phân tích |
|---|---|
| **High-stakes domain** | Y tế = sai 1 câu có thể nguy hiểm → 3 van an toàn là **bắt buộc**, không phải tùy chọn |
| **NLU metadata as guardrail** | `topic = cardiology` → Document Grader kiểm tra cả relevance LẪN topic consistency |
| **Query rewriting thông minh** | Khi retrieval thất bại, hệ thống dùng `entities` (đau dạ dày, trào ngược) để viết lại query có chủ đích, thay vì "đoán mò" |
| **Plug-and-play** | CRAG **không yêu cầu train model mới** — chỉ cần thêm logic nodes vào pipeline. Dùng LangGraph orchestration |
| **Self-RAG hybrid** | Kết hợp nguyên lý Self-RAG (reflection token) vào bước Hallucination Check: LLM tự đánh giá output |

#### Ưu điểm
- ✅ **An toàn nhất** cho domain y tế — 3 lớp kiểm tra chồng chéo
- ✅ Plug-and-play: xây trên nền Advanced RAG, không cần train thêm model
- ✅ Khai thác tối đa NLU metadata (topic filter + entity-boosted rewrite + intent-aware prompt)
- ✅ Triển khai bằng LangGraph (state machine, dễ debug, dễ mở rộng)
- ✅ Có thể bắt đầu với Advanced RAG rồi **nâng cấp dần** lên CRAG

#### Hạn chế
- ❌ Latency cao hơn Advanced RAG (~1.5-2x) do có vòng lặp retry
- ❌ Document Grader và Hallucination Check cần LLM calls bổ sung → tăng chi phí nếu dùng API
- ❌ Cần thiết kế prompt cẩn thận cho từng "judge" node

#### Mức độ Phức tạp: ⭐⭐⭐ (Trung bình — Cao)

---

### 2.3 Kiến trúc C: RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

> **Paper tham chiếu:** Sarthi et al., *"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"*, arXiv:2401.18059, ICLR 2024.

#### Mô tả Kiến trúc

RAPTOR xây dựng một **cây tri thức phân tầng** (hierarchical tree) từ corpus, cho phép retrieve ở nhiều mức trừu tượng:

```
                    ┌─────────────────────┐
          Level 3:  │  Tóm tắt Chuyên khoa │  "Tim mạch: tổng quan các bệnh lý..."
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
         ┌─────────┐    ┌──────────┐     ┌──────────┐
Level 2: │ Cluster: │    │ Cluster:  │     │ Cluster:  │
         │ Tăng HA  │    │ Nhịp nhanh│     │ Suy tim   │
         └────┬────┘    └────┬─────┘     └────┬─────┘
              │              │                 │
         ┌────┼────┐   ┌────┼────┐       ┌────┼────┐
         ▼    ▼    ▼   ▼    ▼    ▼       ▼    ▼    ▼
Level 1: [FAQ1][FAQ2]  [FAQ3][FAQ4]      [FAQ5][FAQ6]
         (chunks gốc)  (chunks gốc)      (chunks gốc)
```

**Quy trình xây cây:**
1. Chunk corpus thành các đoạn nhỏ (Level 1)
2. Cluster các chunks theo semantic similarity (GMM / k-means trên embeddings)
3. Tóm tắt từng cluster bằng LLM → tạo node Level 2
4. Lặp lại: cluster các summary → tóm tắt → Level 3, ...
5. Tại inference: retrieve từ **mọi level** của cây → trả về cả chi tiết (Level 1) lẫn bối cảnh rộng (Level 2-3)

#### Lý do Phù hợp với Đề án

| Yếu tố | Phân tích |
|---|---|
| **Multi-scale questions** | Bệnh nhân hỏi cả câu cụ thể ("liều thuốc X bao nhiêu?") lẫn câu tổng quát ("bệnh tim mạch nguy hiểm không?") → RAPTOR phục vụ cả hai |
| **18 chuyên khoa = 18 subtrees** | Canonical topics (output Trạm 2B) map tự nhiên thành các nhánh của cây tri thức |
| **FAQ có cấu trúc** | Các cặp Q/A y tế có tính phân cụm cao (theo khoa, theo bệnh) → clustering hiệu quả |

#### Ưu điểm
- ✅ Trả lời được cả câu hỏi chi tiết lẫn tổng quát
- ✅ Cải thiện **20% absolute accuracy** trên QuALITY benchmark so với flat retrieval (theo paper gốc)
- ✅ Cây tri thức xây 1 lần, query nhiều lần → amortized cost

#### Hạn chế
- ❌ **Chi phí xây cây rất cao:** Cần gọi LLM tóm tắt cho MỌI cluster ở MỌI level → hàng trăm đến hàng nghìn LLM calls cho corpus ~10K FAQ
- ❌ Không có cơ chế self-correction (cần kết hợp với CRAG)
- ❌ Khó cập nhật incremental: thêm FAQ mới → phải rebuild nhánh cây
- ❌ Chưa có paper nào validate RAPTOR trên domain y tế tiếng Việt

#### Mức độ Phức tạp: ⭐⭐⭐⭐ (Cao)

---

### 2.4 Bảng So sánh Tổng hợp — 3 Kiến trúc RAG

| Tiêu chí | **Advanced RAG** | **CRAG** ⭐ | **RAPTOR** |
|---|---|---|---|
| **An toàn y tế** | ⚠️ Trung bình — không self-correct | ✅ **Cao nhất** — 3 van an toàn | ⚠️ Trung bình — cần ghép CRAG |
| **Tận dụng NLU metadata** | ✅ Tốt (pre-filter) | ✅ **Xuất sắc** (filter + rewrite + prompt) | ⭐ Tốt (subtree routing) |
| **Latency** | ✅ **Thấp nhất** (~200-500ms) | ⚠️ Trung bình (~500-1500ms, có retry) | ⚠️ Thấp khi query, cao khi build |
| **Chi phí triển khai** | ✅ Thấp | ⭐ Trung bình | ❌ Cao (xây cây tốn LLM tokens) |
| **Chống Hallucination** | ❌ Không có | ✅ **Document Grader + Hallucination Check** | ❌ Không có |
| **Incremental update** | ✅ Dễ — thêm doc, re-embed | ✅ Dễ — giống Advanced RAG | ❌ Khó — rebuild subtree |
| **Paper (verifiable)** | Gao et al., arXiv:2312.10997 | Yan et al., arXiv:2401.15884 | Sarthi et al., arXiv:2401.18059 |
| **Mức phức tạp** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Khuyến nghị** | Phase 1 (MVP) | **Phase 2 (Production)** ⭐ | Phase 3 (nếu cần multi-scale) |

> **🏆 Kết luận Phần 1:** Kiến trúc **CRAG (Corrective RAG)** được khuyến nghị mạnh nhất vì:
> 1. **An toàn nhất** cho domain y tế (3 lớp kiểm tra)
> 2. **Tận dụng triệt để** NLU metadata đã có (topic filter, entity rewrite, intent prompt)
> 3. **Plug-and-play**: xây trên nền Advanced RAG, không cần train model mới
> 4. **Lộ trình tự nhiên**: Bắt đầu MVP bằng Advanced RAG → nâng cấp lên CRAG bằng cách thêm 3 nodes

---

## 3. Lựa chọn Thành phần Cốt lõi (Component Selection)

### 3.1 Embedding Model

| Model | Loại | Tiếng Việt | Y tế | Hybrid Native | Max Tokens | Paper / Source |
|---|---|---|---|---|---|---|
| **`BAAI/bge-m3`** ⭐ | Dense + Sparse + ColBERT | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ Có | 8,192 | Chen et al., arXiv:2402.03216 |
| `multilingual-e5-large` | Dense only | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | 512 | Wang et al., arXiv:2402.05672 |
| `AITeamVN/Vietnamese_Embedding` | Dense only | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | 8,000 | HuggingFace Hub |

> **Khuyến nghị:** `BAAI/bge-m3` — model duy nhất trả về dense + sparse + multi-vector trong **1 forward pass**, loại bỏ nhu cầu chạy BM25 engine riêng.

### 3.2 Reranker

| Model | Ngôn ngữ | Kiểu | Paper / Source |
|---|---|---|---|
| **ViRanker** ⭐ | 🇻🇳 Vietnamese-first | Cross-Encoder | Nguyen et al., 2024 (ResearchGate) |
| `BAAI/bge-reranker-v2-m3` | Multilingual | Cross-Encoder | BAAI, HuggingFace Hub |

### 3.3 Vector Database

| DB | Hybrid Search | Metadata Filter | Production | Ghi chú |
|---|---|---|---|---|
| **Qdrant** ⭐ | ✅ Native | ✅ Payload filter | ✅ | Nhẹ, Docker 1 container, free cloud tier |
| Milvus | ✅ Native | ✅ | ✅ | Scale lớn (>1M docs), phức tạp hơn |
| ChromaDB | ❌ | Limited | ⚠️ Dev-only | Chỉ phù hợp prototype |

### 3.4 LLM Generator

| Model | Kịch bản | Tiếng Việt | Y tế | VRAM | Paper / Source |
|---|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** ⭐ | Self-hosted (Colab T4) | ⭐⭐⭐⭐ | ⭐⭐⭐ | 16GB (4-bit) | Qwen Team, arXiv:2407.10671 |
| **Gemini 2.0 Flash** ⭐ | API (free tier) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | N/A | Google DeepMind |
| Vistral-7B-Chat | Self-hosted | ⭐⭐⭐⭐ | ⭐⭐⭐ | 16GB | VinAI |

---

## 4. Sức mạnh Độc quyền: NLU-Guided Retrieval

Đây là tính năng **không hệ thống RAG y tế thông thường nào có được** — vì không ai có sẵn 4 trạm NLU production-grade trước khi retrieve.

### 4.1 Topic-Filtered Search (Thu hẹp không gian tìm kiếm)

```python
# Trạm 2B output: topic = "cardiology", confidence = 0.94
# Thay vì search toàn bộ 10,000 FAQ → chỉ search trong ~500 FAQ cardiology
results = qdrant.search(
    query_vector=bge_m3.encode(clean_query),
    query_filter=Filter(
        must=[FieldCondition(key="topic", match=MatchValue(value="cardiology"))]
    ),
    limit=50,
)
# → Precision@5 tăng ~40%, recall@50 giữ nguyên
```

### 4.2 Entity-Boosted Hybrid Search

```python
# Trạm 2A output: entities = ["đau dạ dày", "trào ngược axit"]
# Boost sparse component bằng medical entities
enriched_query = f"{clean_query} đau dạ dày trào ngược axit"
# → BM25 component nhận thêm tín hiệu keyword mạnh
```

### 4.3 Intent-Aware Prompt Engineering

```python
# Trạm 2C output: intent = "Treatment"
INTENT_PROMPTS = {
    "Diagnosis": "Tập trung vào triệu chứng, xét nghiệm cần làm, và chẩn đoán phân biệt.",
    "Treatment": "Tập trung vào phương pháp điều trị, thuốc, phác đồ, và lưu ý khi dùng.",
    "Severity":  "Tập trung đánh giá mức độ nguy hiểm, khi nào cần đi khám ngay.",
    "Cause":     "Tập trung vào nguyên nhân gây bệnh, yếu tố nguy cơ, và cơ chế bệnh sinh.",
}
system_suffix = INTENT_PROMPTS.get(primary_intent, "")
```

### 4.4 Sơ đồ Tích hợp NLU → RAG

```
              NLU Pipeline Output
              ┌─────────────────────────────────────────┐
              │ clean_text: "Tôi bị đau dạ dày..."      │
              │ topic: "gastroenterology" (conf: 0.91)  │
              │ entities: ["đau dạ dày", "trào ngược"]  │
              │ intent: "Treatment"                     │
              └────────┬─────────┬──────────┬───────────┘
                       │         │          │
            ┌──────────┘         │          └──────────┐
            ▼                    ▼                     ▼
   Topic Filter           Entity Boost          Intent Prompt
   (Qdrant payload)    (Enriched sparse)     (System message)
            │                    │                     │
            └──────────┬─────────┘                     │
                       ▼                               │
              Hybrid Retrieval                         │
              + Reranker                               │
                       │                               │
                       ▼                               │
              ┌────────────────┐                       │
              │ Document Grader│ (CRAG Van #1)         │
              └───────┬────────┘                       │
                      ▼                                │
              ┌────────────────┐                       │
              │ LLM Generation │ ◄─────────────────────┘
              │ + Citations    │
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Hallucination  │ (CRAG Van #3)
              │ Check          │
              └───────┬────────┘
                      ▼
              Final Answer + Sources
```

---

## 5. Phần 2 — Dataset Phù hợp cho Từng Kiến trúc RAG

### 5.1 Datasets cho Advanced RAG & CRAG

Cả hai kiến trúc này cùng chia sẻ nhu cầu: **corpus FAQ y tế tiếng Việt** để indexing, và **benchmark QA** để đánh giá chất lượng retrieval + generation.

#### A. Corpus Tri thức (Knowledge Base — dùng để Index)

| # | Dataset | Nguồn | Quy mô | Task | Lý do phù hợp |
|---|---|---|---|---|---|
| 1 | **ViHealthQA** | [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA) (HuggingFace) | 10,015 cặp Q/A | Extractive / Abstractive QA | Corpus Q/A y tế lớn nhất tiếng Việt, đã dùng trong nhiều nghiên cứu (SPBERTQA). Mỗi cặp = 1 chunk lý tưởng cho RAG |
| 2 | **Vietnamese-Medical-QA** | [hungnm/vietnamese-medical-qa](https://huggingface.co/datasets/hungnm/vietnamese-medical-qa) (HuggingFace) | 9,335 cặp Q/A | Open-domain Medical QA | Thu thập từ **eDoctor + Vinmec** — trùng domain với hệ thống, đảm bảo phân phối ngôn ngữ nhất quán |
| 3 | **Vietnamese-Healthcare** | [urnus11/Vietnamese-Healthcare](https://huggingface.co/datasets/urnus11/Vietnamese-Healthcare) (HuggingFace) | 100K+ entries | Medical articles + QA | Bao gồm bài viết y tế + subtitle Vinmec — phù hợp làm **corpus mở rộng** khi cần coverage rộng hơn |

#### B. Benchmark Đánh giá (Evaluation — dùng để test pipeline)

| # | Dataset | Nguồn | Quy mô | Task | Lý do phù hợp |
|---|---|---|---|---|---|
| 1 | **VM14K** | [bmd1905/VM14K](https://huggingface.co/datasets/bmd1905/VM14K) (HuggingFace, arXiv 2025) | 14,000 MCQ | Medical MCQ (USMLE-style) | Benchmark y tế Việt Nam **lớn nhất và mới nhất**, 34 chuyên khoa, 4 mức độ khó — dùng để đánh giá khả năng reasoning của toàn pipeline |
| 2 | **ViMedAQA** | ACL Anthology, 2024 | ~5,000 Q/A | Abstractive Medical QA | Chuyên biệt cho **abstractive QA** y tế Việt — đánh giá khả năng sinh câu trả lời tự nhiên (generation quality) |
| 3 | **VMHQA** | [EAI Conference, 2025](https://huggingface.co/datasets/) | 10,000 MCQ | Mental Health QA + RAG eval | Đặc biệt có **RAG evaluation protocol** tích hợp — dùng để benchmark retrieval quality |

### 5.2 Datasets Bổ sung cho RAPTOR

RAPTOR yêu cầu **corpus dài và có cấu trúc phân cấp** (articles > sections > paragraphs) để xây cây tri thức hiệu quả:

| # | Dataset | Nguồn | Quy mô | Task | Lý do phù hợp |
|---|---|---|---|---|---|
| 1 | **Vietnamese-Healthcare** | [urnus11/Vietnamese-Healthcare](https://huggingface.co/datasets/urnus11/Vietnamese-Healthcare) | 100K+ | Long-form articles | Bài viết y tế dài, có cấu trúc (tiêu đề → mục → nội dung) — lý tưởng cho hierarchical chunking |
| 2 | **MedQA** (English, tham chiếu) | [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa), arXiv:2009.13081 | 12,723 MCQ | Multi-hop Medical QA | Benchmark quốc tế để so sánh RAPTOR vs flat retrieval — paper gốc RAPTOR dùng USMLE questions tương tự |
| 3 | **FAQ_summarization** (nội bộ) | `data/FAQ_summarization/` | ~10K+ cặp | Summarization + QA | Đã có sẵn trong repo — source/target pairs có thể dùng làm cả chunks (source) lẫn summaries (target) cho RAPTOR tree |

### 5.3 Bảng Tổng hợp Dataset — Mapping với Kiến trúc

| Dataset | Advanced RAG | CRAG | RAPTOR | Vai trò chính |
|---|---|---|---|---|
| **ViHealthQA** | ✅ Knowledge Base | ✅ Knowledge Base | ⭐ Level-1 chunks | Corpus Q/A lõi |
| **Vietnamese-Medical-QA** | ✅ Knowledge Base | ✅ Knowledge Base | ⭐ Level-1 chunks | Corpus bổ sung (Vinmec/eDoctor) |
| **Vietnamese-Healthcare** | ⭐ Mở rộng | ⭐ Mở rộng | ✅ **Tree building** | Bài viết dài → hierarchical |
| **VM14K** | ✅ Eval benchmark | ✅ Eval benchmark | ✅ Eval benchmark | Đánh giá reasoning |
| **ViMedAQA** | ✅ Generation eval | ✅ Generation eval | — | Đánh giá abstractive QA |
| **VMHQA** | ✅ RAG eval | ✅ **RAG eval** ⭐ | — | Benchmark RAG protocol |
| **FAQ_summarization** (nội bộ) | ✅ Knowledge Base | ✅ Knowledge Base | ✅ Chunks + Summaries | Đã có sẵn trong repo |

---

## 6. Lộ trình Triển khai Đề xuất (Implementation Roadmap)

```
Phase 1: Advanced RAG (MVP)          Phase 2: CRAG (Production)         Phase 3: RAPTOR (Optional)
┌─────────────────────────┐         ┌─────────────────────────┐        ┌─────────────────────────┐
│ • Chunk FAQ → bge-m3    │         │ • + Document Grader     │        │ • Hierarchical tree     │
│ • Qdrant index          │         │ • + Query Rewriter      │        │ • Multi-scale retrieval │
│ • Topic filter          │         │ • + Hallucination Check │        │ • LLM summarization     │
│ • Basic generation      │         │ • + LangGraph state     │        │ • Rebuild strategy      │
│ • ~1-2 tuần             │───────▶ │ • ~1 tuần               │──────▶ │ • ~2 tuần               │
│                         │         │                         │        │                         │
│ Eval: VM14K accuracy    │         │ Eval: + Faithfulness    │        │ Eval: Multi-hop QA      │
│       ViMedAQA          │         │       + VMHQA RAG       │        │       Complex reasoning │
└─────────────────────────┘         └─────────────────────────┘        └─────────────────────────┘
```

---

## 7. Tham khảo Chính (Verifiable References)

| # | Tài liệu | Identifier | Năm |
|---|---|---|---|
| 1 | Gao et al., *"RAG for LLMs: A Survey"* | arXiv:2312.10997 | 2023 |
| 2 | Yan et al., *"Corrective Retrieval Augmented Generation (CRAG)"* | arXiv:2401.15884 | 2024 (ICML) |
| 3 | Asai et al., *"Self-RAG: Learning to Retrieve, Generate, and Critique"* | arXiv:2310.11511 | 2023 (NeurIPS) |
| 4 | Sarthi et al., *"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"* | arXiv:2401.18059 | 2024 (ICLR) |
| 5 | Chen et al., *"BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity"* | arXiv:2402.03216 | 2024 |
| 6 | Xiong et al., *"Benchmarking RAG on Medical QA (MedRAG/MIRAGE)"* | arXiv:2402.13178 | 2024 |
| 7 | Qwen Team, *"Qwen2.5 Technical Report"* | arXiv:2407.10671 | 2024 |
| 8 | VM14K, *"Vietnamese Medical MCQ Benchmark"* | HuggingFace: bmd1905/VM14K | 2025 |
| 9 | ViHealthQA | HuggingFace: tarudesu/ViHealthQA | 2023 |
| 10 | ViRanker, *"Cross-Encoder Reranker for Vietnamese"* | ResearchGate (Nguyen et al.) | 2024 |
