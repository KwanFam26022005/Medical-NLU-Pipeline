"""
app.py — Giao diện demo KEHN (Gradio).

Chạy: python app.py
Mở trình duyệt tại: http://localhost:7860
"""

import sys
from pathlib import Path

# ── Setup imports ─────────────────────────────────────────────────────
_DEMO_DIR = Path(__file__).resolve().parent
_KEHN_DIR = _DEMO_DIR.parent
_PROJECT_ROOT = _KEHN_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import gradio as gr
from model_loader import KEHNPredictor
from postprocessor import NER_TYPE_VI, NER_TYPE_COLOR

# ── Khởi tạo model (một lần duy nhất) ────────────────────────────────
print("=" * 60)
print("🏥 KEHN Medical NLU Demo")
print("=" * 60)
predictor = KEHNPredictor()
print()


# ── Hàm inference cho Gradio ─────────────────────────────────────────

def run_inference(text: str):
    """
    Chạy inference pipeline và format output cho Gradio.

    Returns:
        topic_output : dict — cho gr.Label (chuyên khoa + confidence)
        intent_output: dict — cho gr.Label (ý định)
        ner_highlight: list — cho gr.HighlightedText (entities highlighted)
        ner_table    : list — cho gr.Dataframe (bảng entities)
        words_display: str — câu sau word segmentation
    """
    if not text or not text.strip():
        return (
            {"Vui lòng nhập câu hỏi": 1.0},
            {"Vui lòng nhập câu hỏi": 1.0},
            [("Vui lòng nhập câu hỏi y tế...", None)],
            [],
            "",
        )

    # Chạy inference
    result = predictor.predict(text.strip())

    # ── Topic output (gr.Label) ───────────────────────────────────
    topic = result["topic"]
    topic_output = topic["all_probs"]

    # ── Intent output (gr.Label) ──────────────────────────────────
    intent = result["intent"]
    intent_output = intent["all_probs"]

    # ── NER highlighted text ──────────────────────────────────────
    ner = result["ner"]
    ner_highlight = ner["highlighted_text"]
    if not ner_highlight:
        ner_highlight = [(" ".join(result["segmented_words"]), None)]

    # ── NER table ─────────────────────────────────────────────────
    ner_table = []
    for entity in ner["entities"]:
        ner_table.append([
            entity["entity_text"],
            entity["type_vi"],
            entity["entity_type"],
        ])

    # ── Segmented words display ───────────────────────────────────
    words_display = " | ".join(result["segmented_words"])

    return topic_output, intent_output, ner_highlight, ner_table, words_display


# ── Xây dựng giao diện Gradio ────────────────────────────────────────

# CSS tùy chỉnh
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
}
.main-title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
"""

# Entity type color mapping cho HighlightedText
color_map = {}
for etype, vi_name in NER_TYPE_VI.items():
    color_map[vi_name] = NER_TYPE_COLOR.get(etype, "#CCCCCC")


with gr.Blocks(
    title="KEHN — Hệ thống Hiểu Ngôn ngữ Y tế",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
) as demo:

    # ── Header ────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🏥 KEHN — Hệ thống Hiểu Ngôn ngữ Y tế Đa nhiệm
        <p class="subtitle">
            Knowledge-Enhanced Hierarchical Network · 3 tác vụ: Chuyên khoa · Ý định · Thực thể y tế<br>
            Backbone: ViHealthBERT · Co-Interactive Transformer · CRF
        </p>
        """,
    )

    with gr.Row():
        # ── Input column ──────────────────────────────────────────
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📝 Nhập câu hỏi y tế tiếng Việt",
                placeholder="Ví dụ: Tôi bị đau đầu, sốt cao và ho nhiều ngày...",
                lines=3,
                max_lines=5,
            )

            with gr.Row():
                submit_btn = gr.Button(
                    "🔍 Phân tích",
                    variant="primary",
                    scale=2,
                )
                clear_btn = gr.ClearButton(
                    value="🗑️ Xóa",
                    scale=1,
                )

            words_display = gr.Textbox(
                label="📐 Tách từ (Word Segmentation)",
                interactive=False,
                lines=1,
            )

    gr.Markdown("---")

    # ── Output panels ─────────────────────────────────────────────
    with gr.Row():
        # Panel 1: Topic
        with gr.Column(scale=1):
            gr.Markdown("### 🏷️ Chuyên khoa dự đoán")
            topic_output = gr.Label(
                label="Chuyên khoa",
                num_top_classes=5,
            )

        # Panel 2: Intent
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Ý định bệnh nhân")
            intent_output = gr.Label(
                label="Ý định",
                num_top_classes=4,
            )

    gr.Markdown("---")

    # Panel 3: NER
    gr.Markdown("### 🔬 Thực thể y tế (NER)")

    with gr.Row():
        with gr.Column(scale=3):
            ner_highlight = gr.HighlightedText(
                label="Văn bản với thực thể được đánh dấu",
                color_map=color_map,
                show_legend=True,
                combine_adjacent=False,
            )

        with gr.Column(scale=2):
            ner_table = gr.Dataframe(
                headers=["Thực thể", "Loại (Tiếng Việt)", "Mã"],
                label="Bảng thực thể",
                wrap=True,
            )

    # ── Chú giải màu NER ──────────────────────────────────────────
    legend_items = " · ".join([
        f'<span style="background:{color};padding:2px 8px;border-radius:4px;color:#333;font-size:0.85rem">{vi_name}</span>'
        for etype, vi_name in NER_TYPE_VI.items()
        for color in [NER_TYPE_COLOR.get(etype, "#CCC")]
    ])
    gr.Markdown(f"**Chú giải:** {legend_items}")

    # ── Examples ──────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 📋 Câu mẫu")

    gr.Examples(
        examples=[
            ["Tôi bị đau đầu, sốt cao và ho nhiều ngày"],
            ["Con tôi 3 tuổi bị nổi mẩn đỏ khắp người, có cần đưa đi khám da liễu không?"],
            ["Tôi đang uống thuốc Metformin để điều trị tiểu đường, có tác dụng phụ gì không?"],
        ],
        inputs=[input_text],
        label="Bấm vào câu mẫu để thử nghiệm:",
    )

    # ── Event handlers ────────────────────────────────────────────
    outputs = [topic_output, intent_output, ner_highlight, ner_table, words_display]

    submit_btn.click(
        fn=run_inference,
        inputs=[input_text],
        outputs=outputs,
    )

    input_text.submit(
        fn=run_inference,
        inputs=[input_text],
        outputs=outputs,
    )

    clear_btn.add([input_text, words_display, topic_output, intent_output, ner_highlight, ner_table])

    # ── Footer ────────────────────────────────────────────────────
    gr.Markdown(
        """
        ---
        <div style="text-align:center; color:#95a5a6; font-size:0.85rem;">
            KEHN Medical NLU · ViHealthBERT + Co-Interactive Transformer + CRF · 3-Task Joint Learning
        </div>
        """,
    )


# ── Launch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
