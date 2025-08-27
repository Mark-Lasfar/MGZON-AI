import os
import logging
import gradio as gr
from openai import OpenAI
from pydoc import html
from utils import LATEX_DELIMS

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://router.huggingface.co/v1")
MODEL_NAME = "openai/gpt-oss-120b:cerebras"  # استخدام مزود Cerebras لـ gpt-oss-120b
client = OpenAI(api_key=HF_TOKEN, base_url=API_ENDPOINT)

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# وظيفة التنسيق النهائي (من الـ Space الأصلي)
def format_final(analysis_text: str, visible_text: str) -> str:
    """Render final message with collapsible analysis + normal Markdown answer."""
    reasoning_safe = html.escape((analysis_text or "").strip())
    response = (visible_text or "").strip()
    return (
        "<details><summary><strong>🤔 Analysis</strong></summary>\n"
        "<pre style='white-space:pre-wrap;'>"
        f"{reasoning_safe}"
        "</pre>\n</details>\n\n"
        "**💬 Response:**\n\n"
        f"{response}"
    )

# وظيفة التوليد مع محاكاة streaming
def generate(message, history, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens):
    if not message.strip():
        yield "Please enter a prompt."
        return

    # Flatten gradio history
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history:
        if isinstance(h, dict):
            msgs.append(h)
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            u, a = h
            if u: msgs.append({"role": "user", "content": u})
            if a: msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": message})

    # إعداد الأدوات إذا تم تفعيل البحث على الويب
    tools = [{"type": "function", "function": {
        "name": "web_search_preview",
        "description": "Simulate web search for additional context",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    }}] if enable_browsing else None
    tool_choice = "auto" if enable_browsing else None

    in_analysis = False
    in_visible = False
    raw_analysis = ""
    raw_visible = ""
    raw_started = False
    last_flush_len = 0

    def make_raw_preview() -> str:
        return (
            "```text\n"
            "Analysis (live):\n"
            f"{raw_analysis}\n\n"
            "Response (draft):\n"
            f"{raw_visible}\n"
            "```"
        )

    try:
        # طلب التوليد عبر Inference API مع streaming
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stream=True,
            extra_body={"reasoning": {"effort": reasoning_effort}},
            tools=tools,
            tool_choice=tool_choice
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if content == "<|channel|>analysis<|message|>":
                    in_analysis, in_visible = True, False
                    if not raw_started:
                        raw_started = True
                        yield make_raw_preview()
                    continue
                if content == "<|channel|>final<|message|>":
                    in_analysis, in_visible = False, True
                    if not raw_started:
                        raw_started = True
                        yield make_raw_preview()
                    continue

                if in_analysis:
                    raw_analysis += content
                elif in_visible:
                    raw_visible += content
                else:
                    raw_visible += content

                total_len = len(raw_analysis) + len(raw_visible)
                if total_len - last_flush_len >= 120 or "\n" in content:
                    last_flush_len = total_len
                    yield make_raw_preview()

        final_markdown = format_final(raw_analysis, raw_visible)
        if final_markdown.count("$") % 2:
            final_markdown += "$"
        yield final_markdown

    except Exception as e:
        logger.exception("Stream failed")
        yield f"❌ Error: {e}"

# إعداد واجهة Gradio
chatbot_ui = gr.ChatInterface(
    fn=generate,
    type="messages",
    chatbot=gr.Chatbot(
        label="GPT-OSS-120B Chatbot",
        type="messages",
        height=600,
        latex_delimiters=LATEX_DELIMS,
    ),
    additional_inputs_accordion=gr.Accordion("⚙️ Settings", open=True),
    additional_inputs=[
        gr.Textbox(label="System prompt", value="You are a helpful assistant.", lines=2),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7),
        gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium"),
        gr.Checkbox(label="Enable web browsing (simulated)", value=False),
        gr.Slider(label="Max New Tokens", minimum=50, maximum=1024, step=50, value=200),
    ],
    stop_btn=True,
    examples=[
        ["Explain the difference between supervised and unsupervised learning."],
        ["Summarize the plot of Inception in two sentences."],
        ["Show me the LaTeX for the quadratic formula."],
        ["What are advantages of AMD Instinct MI300X GPU?"],
        ["Derive the gradient of softmax cross-entropy loss."],
        ["Explain why ∂/∂x xⁿ = n·xⁿ⁻¹ holds."],
    ],
    title="GPT-OSS-120B Chatbot on Hugging Face",
    description="This Space demonstrates the OpenAI GPT-OSS-120B model running via Hugging Face Inference API. Includes analysis for chain of thought insights. Licensed under Apache 2.0. ***DISCLAIMER:*** Analysis may contain internal thoughts not suitable for final response.",
)

# دمج FastAPI مع Gradio
from fastapi import FastAPI
from gradio import mount_gradio_app

app = FastAPI(title="GPT-OSS-120B API")
app = mount_gradio_app(app, chatbot_ui, path="/")

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    chatbot_ui.queue(max_size=QUEUE_SIZE, concurrency_count=CONCURRENCY_LIMIT).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False
    )