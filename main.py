"""
main.py - FastAPI endpoint cho Medical NLU Pipeline.
Trạm 3: Tích hợp toàn bộ pipeline - Tiền xử lý + NLU đa luồng.

Luồng xử lý:
  1. Nhận request text thô
  2. Trạm 1: Giải viết tắt (Acronym WSD) -> Clean text
  3. Trạm 2: Chạy song song bằng asyncio.gather:
     - 2A: Medical NER (trích xuất entities)
     - 2B: Topic Classification (phân loại khoa)
     - 2C: Intent Classification (phân loại ý định)
  4. Gom kết quả -> Trả JSON response

Cách chạy:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import (
    ACRONYM_MODEL_DIR,
    API_HOST,
    API_PORT,
    DATA_DIR,
    INTENT_MODEL_DIR,
    NER_MODEL_DIR,
    TOPIC_MODEL_DIR,
)
from models import AcronymCrossEncoder, IntentClassifier, MedicalNER, TopicClassifier


# ============================================================
# 📦 REQUEST / RESPONSE SCHEMAS
# ============================================================

class MedicalQueryRequest(BaseModel):
    """Schema cho request đầu vào."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Câu hỏi y tế cần phân tích",
        examples=["bs ơi e bị đau dạ dày có cần mổ k ạ?"],
    )


class NLUResult(BaseModel):
    """Kết quả phân tích NLU."""
    entities: List[str] = Field(
        default_factory=list,
        description="Danh sách thực thể y tế (triệu chứng, bệnh, thuốc)",
    )
    topic: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phân loại khoa y tế",
    )
    intent: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phân loại ý định câu hỏi",
    )


class MedicalQueryResponse(BaseModel):
    """Schema cho response trả về."""
    raw_text: str = Field(description="Text gốc từ user")
    clean_text: str = Field(description="Text đã giải viết tắt")
    nlu_result: NLUResult = Field(description="Kết quả NLU")
    processing_time_ms: float = Field(description="Thời gian xử lý (ms)")


class HealthCheckResponse(BaseModel):
    """Schema cho health check."""
    status: str
    models_loaded: Dict[str, bool]


# ============================================================
# 🚀 APPLICATION LIFECYCLE
# ============================================================

# Global model instances (khởi tạo 1 lần khi startup)
acronym_resolver: Optional[AcronymCrossEncoder] = None
medical_ner: Optional[MedicalNER] = None
topic_classifier: Optional[TopicClassifier] = None
intent_classifier: Optional[IntentClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager: Load tất cả models khi server khởi động.
    Dùng asynccontextmanager thay vì on_event (deprecated).
    """
    global acronym_resolver, medical_ner, topic_classifier, intent_classifier

    print("=" * 60)
    print("🏥 Medical NLU Pipeline - Khởi tạo models...")
    print("=" * 60)

    # Load từng model (xử lý lỗi từng model để pipeline không bị gãy)
    try:
        acronym_resolver = AcronymCrossEncoder(
            model_dir=ACRONYM_MODEL_DIR,
        )
        acronym_resolver.load_model()
    except Exception as e:
        print(f"⚠️ Không thể load AcronymResolver: {e}")
        acronym_resolver = None

    try:
        medical_ner = MedicalNER(model_dir=NER_MODEL_DIR)
        medical_ner.load_model()
    except Exception as e:
        print(f"⚠️ Không thể load MedicalNER: {e}")
        medical_ner = None

    try:
        topic_classifier = TopicClassifier(model_dir=TOPIC_MODEL_DIR)
        topic_classifier.load_model()
    except Exception as e:
        print(f"⚠️ Không thể load TopicClassifier: {e}")
        topic_classifier = None

    try:
        intent_classifier = IntentClassifier(model_dir=INTENT_MODEL_DIR)
        intent_classifier.load_model()
    except Exception as e:
        print(f"⚠️ Không thể load IntentClassifier: {e}")
        intent_classifier = None

    print("\n✅ Pipeline sẵn sàng!")
    print("=" * 60)

    yield  # Server chạy ở đây

    # Cleanup khi shutdown
    print("\n🛑 Shutting down Medical NLU Pipeline...")


# ============================================================
# 🌐 FASTAPI APP
# ============================================================

app = FastAPI(
    title="Medical NLU Pipeline API",
    description=(
        "Hệ thống Trợ lý Y tế Thông minh - Phân tích câu hỏi y tế tiếng Việt.\n\n"
        "**Pipeline:** Giải viết tắt → NER + Topic + Intent (song song)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check() -> HealthCheckResponse:
    """Kiểm tra trạng thái hệ thống và các models."""
    return HealthCheckResponse(
        status="healthy",
        models_loaded={
            "acronym_resolver": acronym_resolver is not None,
            "medical_ner": medical_ner is not None,
            "topic_classifier": topic_classifier is not None,
            "intent_classifier": intent_classifier is not None,
        },
    )


@app.post(
    "/analyze_medical_query",
    response_model=MedicalQueryResponse,
    tags=["NLU Pipeline"],
    summary="Phân tích câu hỏi y tế",
    description=(
        "Nhận câu hỏi y tế thô → Giải viết tắt → "
        "Trích xuất entities + Phân loại Khoa + Phân loại Ý định"
    ),
)
async def analyze_medical_query(
    request: MedicalQueryRequest,
) -> MedicalQueryResponse:
    """
    Endpoint chính của Medical NLU Pipeline.
    
    Luồng xử lý:
    1. TRẠM 1: Acronym Resolution (sequential - cần clean text trước)
    2. TRẠM 2: NER + Topic + Intent chạy SONG SONG bằng asyncio.gather
    """
    start_time = time.perf_counter()
    raw_text = request.text.strip()

    # ──────────────────────────────────────────────
    # TRẠM 1: Giải viết tắt (Acronym WSD)
    # Chạy TRƯỚC vì clean text là input cho Trạm 2
    # ──────────────────────────────────────────────
    if acronym_resolver is not None:
        try:
            clean_text = await acronym_resolver.async_predict(raw_text)
        except Exception as e:
            print(f"⚠️ AcronymResolver error: {e}")
            clean_text = raw_text
    else:
        clean_text = raw_text

    # ──────────────────────────────────────────────
    # TRẠM 2: Chạy 3 nhánh NLU ĐỒNG THỜI
    # asyncio.gather chạy 2A, 2B, 2C song song
    # ──────────────────────────────────────────────
    async def run_ner() -> List[str]:
        """2A: Medical NER — trả về danh sách entity strings đã aggregate từ BIO."""
        if medical_ner is not None:
            try:
                raw_ner = await medical_ner.async_predict(clean_text)
                # Aggregate BIO tokens thành entity strings
                entities = []
                current_entity = []
                for token in raw_ner:
                    label = token.get("label", "O")
                    word = token.get("word", "").replace("_", " ")
                    if label.startswith("B-"):
                        if current_entity:
                            entities.append(" ".join(current_entity))
                        current_entity = [word]
                    elif label.startswith("I-") and current_entity:
                        current_entity.append(word)
                    else:
                        if current_entity:
                            entities.append(" ".join(current_entity))
                            current_entity = []
                if current_entity:
                    entities.append(" ".join(current_entity))
                return entities
            except Exception as e:
                print(f"⚠️ MedicalNER error: {e}")
        return []

    async def run_topic() -> Dict[str, Any]:
        """2B: Topic Classification."""
        if topic_classifier is not None:
            try:
                return await topic_classifier.async_predict(clean_text)
            except Exception as e:
                print(f"⚠️ TopicClassifier error: {e}")
        return {"topic": "unknown", "confidence": 0.0, "is_reliable": False}

    async def run_intent() -> Dict[str, Any]:
        """2C: Intent Classification — normalize List[Dict] thành Dict chuẩn."""
        if intent_classifier is not None:
            try:
                raw_intents = await intent_classifier.async_predict(clean_text)
                # Normalize: List[Dict] → Dict với primary_intent
                if raw_intents:
                    sorted_intents = sorted(raw_intents, key=lambda x: x["score"], reverse=True)
                    primary = sorted_intents[0]["intent"]
                    scores = {item["intent"]: round(item["score"], 4) for item in sorted_intents}
                    return {
                        "intents": [item["intent"] for item in sorted_intents],
                        "scores": scores,
                        "primary_intent": primary,
                    }
            except Exception as e:
                print(f"⚠️ IntentClassifier error: {e}")
        return {"intents": [], "scores": {}, "primary_intent": "unknown"}

    # 🔥 Chạy song song 3 nhánh NLU
    entities, topic_result, intent_result = await asyncio.gather(
        run_ner(),
        run_topic(),
        run_intent(),
    )

    # ──────────────────────────────────────────────
    # GOM KẾT QUẢ
    # ──────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return MedicalQueryResponse(
        raw_text=raw_text,
        clean_text=clean_text,
        nlu_result=NLUResult(
            entities=entities,
            topic=topic_result,
            intent=intent_result,
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )


# ============================================================
# 🏃 RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
