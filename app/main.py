import os
import json
import logging
import gradio as gr
from openai import OpenAI
from pydoc import html
from typing import List, Generator, Optional

# تعريف LATEX_DELIMS
LATEX_DELIMS = [
    {"left": "$$",  "right": "$$",  "display": True},
    {"left": "$",   "right": "$",   "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحقق من الملفات في /app/ (للتصحيح)
logger.info("Files in /app/: %s", os.listdir("/app"))

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:cerebras")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")
client = OpenAI(api_key=HF_TOKEN, base_url=API_ENDPOINT)

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# دالة request_generation (دمج من gateway.py)
def request_generation(
    api_key: str,
    api_base: str,
    message: str,
    system_prompt: str,
    model_name: str,
    chat_history: Optional[List[dict]] = None,
    temperature: float = 0.3,
    max_new_tokens: int = 1024,
    reasoning_effort: str = "off",
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[str] = None,
) -> Generator[str, None, None]:
    """Streams Responses API events. Emits:
      - "analysis" sentinel once, then raw reasoning deltas
      - "assistantfinal" sentinel once, then visible output deltas
    If no visible deltas, emits a tool-call fallback message."""
    client = OpenAI(api_key=api_key, base_url=api_base)

    # تنظيف الـ messages من metadata
    input_messages: List[dict] = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for msg in chat_history:
            clean_msg = {"role": msg.get("role"), "content": msg.get("content")}
            if clean_msg["content"]:
                input_messages.append(clean_msg)
    input_messages.append({"role": "user", "content": message})

    # إعداد tools و tool_choice
    tools = tools if tools else []
    tool_choice = tool_choice if tool_choice in ["auto", "none", "any", "required"] else "none"

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=input_messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
        )

        reasoning_started = False
        reasoning_closed = False
        saw_visible_output = False
        last_tool_name = None
        last_tool_args = None
        buffer = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if content == "<|channel|>analysis<|message|>":
                    if not reasoning_started:
                        yield "analysis"
                        reasoning_started = True
                    continue
                if content == "<|channel|>final<|message|>":
                    if reasoning_started and not reasoning_closed:
                        yield "assistantfinal"
                        reasoning_closed = True
                    continue

                saw_visible_output = True
                buffer += content

                if "\n" in buffer or len(buffer) > 150:
                    yield buffer
                    buffer = ""
                continue

            if chunk.choices[0].delta.tool_calls:
                tool_call = chunk.choices[0].delta.tool_calls[0]
                name = getattr(tool_call, "function", {}).get("name", None)
                args = getattr(tool_call, "function", {}).get("arguments", None)
                if name:
                    last_tool_name = name
                if args:
                    last_tool_args = args
                continue

            if chunk.choices[0].finish_reason in ("stop", "tool_calls", "error"):
                if buffer:
                    yield buffer
                    buffer = ""

                if reasoning_started and not reasoning_closed:
                    yield "assistantfinal"
                    reasoning_closed = True

                if not saw_visible_output:
                    msg = "I attempted to call a tool, but tools aren't executed in this environment, so no final answer was produced."
                    if last_tool_name:
                        try:
                            args_text = json.dumps(last_tool_args, ensure_ascii=False, default=str)
                        except Exception:
                            args_text = str(last_tool_args)
                        msg += f"\n\n• Tool requested: **{last_tool_name}**\n• Arguments: `{args_text}`"
                    yield msg

                if chunk.choices[0].finish_reason == "error":
                    yield f"Error: Unknown error"
                break

        if buffer:
            yield buffer

    except Exception as e:
        logger.exception("[Gateway] Streaming failed")
        yield f"Error: {e}"

# وظيفة التنسيق النهائي
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

    # Flatten gradio history وتنظيف metadata
    chat_history = []
    for h in history:
        if isinstance(h, dict):
            clean_msg = {"role": h.get("role"), "content": h.get("content")}
            if clean_msg["content"]:
                chat_history.append(clean_msg)
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            u, a = h
            if u: chat_history.append({"role": "user", "content": u})
            if a: chat_history.append({"role": "assistant", "content": a})

    # إعداد الأدوات إذا تم تفعيل البحث على الويب
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search_preview",
                "description": "Simulate web search for additional context",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
        }
    ] if enable_browsing else []
    tool_choice = "auto" if enable_browsing else "none"

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
        # استدعاء request_generation
        stream = request_generation(
            api_key=HF_TOKEN,
            api_base=API_ENDPOINT,
            message=message,
            system_prompt=system_prompt,
            model_name=MODEL_NAME,
            chat_history=chat_history,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )

        for chunk in stream:
            if chunk == "analysis":
                in_analysis, in_visible = True, False
                if not raw_started:
                    raw_started = True
                    yield make_raw_preview()
                continue
            if chunk == "assistantfinal":
                in_analysis, in_visible = False, True
                if not raw_started:
                    raw_started = True
                    yield make_raw_preview()
                continue

            if in_analysis:
                raw_analysis += chunk
            elif in_visible:
                raw_visible += chunk
            else:
                raw_visible += chunk

            total_len = len(raw_analysis) + len(raw_visible)
            if total_len - last_flush_len >= 120 or "\n" in chunk:
                last_flush_len = total_len
                yield make_raw_preview()

        final_markdown = format_final(raw_analysis, raw_visible)
        if final_markdown.count("$") % 2:
            final_markdown += "$"
        yield final_markdown

    except Exception as e:
        logger.exception("Stream failed")
        yield f"❌ Error: {e}"

# إعداد CSS
css = """
.gradio-container { max-width: 800px; margin: auto; }
.chatbot { border: 1px solid #ccc; border-radius: 10px; }
.input-textbox { font-size: 16px; }
"""

# إعداد واجهة Gradio
chatbot_ui = gr.ChatInterface(
    fn=generate,
    type="messages",
    chatbot=gr.Chatbot(
        label="MGZon-120B Chatbot",
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
    stop_btn="Stop",
    examples=[
        ["Explain the difference between supervised and unsupervised learning."],
        ["Summarize the plot of Inception in two sentences."],
        ["Show me the LaTeX for the quadratic formula."],
        ["What are advantages of AMD Instinct MI300X GPU?"],
        ["Derive the gradient of softmax cross-entropy loss."],
        ["Explain why ∂/∂x xⁿ = n·xⁿ⁻¹ holds."],
    ],
    title="MGZon-120B Chatbot on Hugging Face",
    description="This Space demonstrates the  GPT-MGZon-120B model running via Hugging Face Inference API. Includes analysis for chain of thought insights. Licensed under Apache 2.0. ***DISCLAIMER:*** Analysis may contain internal thoughts not suitable for final response.",
    theme="gradio/soft",
    css=css,
)

# دمج FastAPI مع Gradio
from fastapi import FastAPI
from gradio import mount_gradio_app

app = FastAPI(title="MGZon-120B API")
app = mount_gradio_app(app, chatbot_ui, path="/")

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    chatbot_ui.queue(max_size=QUEUE_SIZE, concurrency_count=CONCURRENCY_LIMIT).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=False
    )