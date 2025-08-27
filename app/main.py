import os
import logging
import gradio as gr
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import LATEX_DELIMS
from pydoc import html  # لـ html.escape في التنسيق

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحميل النموذج (من التدميج: استخدام FLAN-T5 بدلاً من gpt-oss-120b)
MODEL_NAME = "MGZON/mgzon-flan-t5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")
    logger.info(f"Model {MODEL_NAME} loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# إعداد FastAPI (للـ API backend)
app = FastAPI(title="MGZON FLAN-T5 API (Inspired by GPT-OSS-120B)")

class RequestText(BaseModel):
    text: str
    max_length: int = 200

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/generate/")
async def generate_api(req: RequestText):
    try:
        prompt = f"You are a helpful assistant.\n\nUser: {req.text}"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=req.max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"API generation error: {e}")
        return {"error": str(e)}

# وظيفة التنسيق النهائي (من الكود الأصلي للتدميج)
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

# وظيفة التوليد الرئيسية (دمج: محاكاة streaming مع FLAN-T5)
def generate(message, history, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens):
    if not message.strip():
        yield "Please enter a prompt."
        return

    # Flatten gradio history (من الكود الأصلي)
    msgs = []
    for h in history:
        if isinstance(h, dict):
            msgs.append(h)
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            u, a = h
            if u: msgs.append({"role": "user", "content": u})
            if a: msgs.append({"role": "assistant", "content": a})

    # بناء الـ prompt الكامل
    full_prompt = f"{system_prompt}\n\n"
    for msg in msgs:
        role = msg["role"]
        content = msg["content"]
        full_prompt += f"{role.capitalize()}: {content}\n"
    full_prompt += f"User: {message}\nAssistant:"

    # محاكاة الـ tools (web browsing بسيط)
    if enable_browsing:
        full_prompt += "\n[Web Search Preview: Simulating search results for enhanced response.]"

    in_analysis = False
    raw_analysis = ""
    raw_visible = ""

    # توليد الرد (غير streaming حقيقي، لكن yield للمحاكاة)
    try:
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_new_tokens + inputs["input_ids"].shape[1],
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_visible = full_response.replace(full_prompt, "").strip()  # استخراج الرد فقط

        # محاكاة الـ analysis بناءً على reasoning_effort (تدميج من الـ Space الأصلي)
        if reasoning_effort == "low":
            raw_analysis = "Quick reasoning: Direct response generation."
        elif reasoning_effort == "medium":
            raw_analysis = f"Medium reasoning: Analyzed input '{message[:50]}...' and generated context-aware response."
        else:  # high
            raw_analysis = f"High reasoning: Step-by-step chain of thought - Tokenized input, considered history, applied {system_prompt}, simulated browsing if enabled, and optimized for {temperature} temperature."
        if enable_browsing:
            raw_analysis += "\nBrowsing simulation: Incorporated potential web insights into response."

        # Yield الـ preview البسيط (محاكاة live)
        yield f"```text\nAnalysis (live):\n{raw_analysis}\n\nResponse (draft):\n{raw_visible[:100]}...\n```"

        # الرد النهائي
        final_markdown = format_final(raw_analysis, raw_visible)
        if final_markdown.count("$") % 2:
            final_markdown += "$"  # إصلاح LaTeX
        yield final_markdown

    except Exception as e:
        logger.exception("Generation failed")
        yield f"❌ Error: {e}"

# إعداد واجهة Gradio (تدميج كامل مع الـ Space الأصلي)
chatbot_ui = gr.ChatInterface(
    fn=generate,
    type="messages",
    chatbot=gr.Chatbot(
        label="MGZON FLAN-T5 Chatbot (Inspired by GPT-OSS-120B)",
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
        ["What are advantages of AMD Instinct MI300X GPU?"],  # إشارة إلى AMD من الـ Space الأصلي
        ["Derive the gradient of softmax cross-entropy loss."],
        ["Explain why ∂/∂x xⁿ = n·xⁿ⁻¹ holds."],
    ],
    title="MGZON FLAN-T5 Chatbot on Hugging Face (Inspired by AMD GPT-OSS-120B)",
    description="This Space integrates features from the AMD GPT-OSS-120B chatbot, using MGZON FLAN-T5 model. Includes analysis for chain of thought insights. ***DISCLAIMER:*** Analysis may contain internal thoughts not suitable for final response. Built with Apache 2.0 License inspiration.",
)

# دمج FastAPI مع Gradio (لجعل الواجهة تظهر على /)
from gradio import mount_gradio_app
app = mount_gradio_app(app, chatbot_ui, path="/")

# إعدادات الـ queue (من الأصلي)
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    chatbot_ui.queue(max_size=QUEUE_SIZE, concurrency_count=CONCURRENCY_LIMIT).launch(server_name="0.0.0.0", server_port=7860, share=False)
    # uvicorn.run(app, host="0.0.0.0", port=7860)  # استخدم هذا إذا أردت FastAPI فقط، لكن Gradio أفضل للواجهة