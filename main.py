import os
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import gradio as gr
from api.endpoints import router as api_router
from utils.generation import generate, format_final, LATEX_DELIMS
from utils.web_search import web_search

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحقق من الملفات في /app/
logger.info("Files in /app/: %s", os.listdir("/app"))

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# إعداد CSS
css = """
.gradio-container { max-width: 800px; margin: auto; padding: 20px; }
.logo { width: 100px; height: 100px; margin: 20px auto; display: block; }
.chatbot { border: 1px solid #ccc; border-radius: 10px; }
.input-textbox { font-size: 16px; }
h1 { text-align: center; color: #fff; font-size: 2rem; margin-bottom: 20px; }
"""

# إعداد واجهة Gradio بـ Blocks
with gr.Blocks(css=css, theme="gradio/soft") as chatbot_ui:
    gr.HTML("""
        <img src="https://raw.githubusercontent.com/Mark-Lasfar/MGZon/main/public/icons/mg.svg" alt="MGZon Logo" class="logo">
        <h1>MGZon Chatbot</h1>
        <p style="text-align: center; color: #ccc;">A versatile chatbot powered by GPT-OSS-20B and a fine-tuned model for MGZon queries. Licensed under Apache 2.0.</p>
    """)
    chatbot = gr.Chatbot(
        label="MGZon Chatbot",
        type="messages",
        height=600,
        latex_delimiters=LATEX_DELIMS,
    )
    with gr.Row():
        message = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=2)
        submit = gr.Button("Send")
        stop_btn = gr.Button("Stop")
    with gr.Accordion("⚙️ Settings", open=False):
        system_prompt = gr.Textbox(
            label="System prompt",
            value="You are a helpful assistant capable of code generation, analysis, review, and more.",
            lines=2
        )
        temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.9)
        reasoning_effort = gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium")
        enable_browsing = gr.Checkbox(label="Enable DeepSearch (web browsing)", value=True)
        max_new_tokens = gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=4096)
    gr.Examples(
        examples=[
            ["Explain the difference between supervised and unsupervised learning."],
            ["Generate a React component for a login form."],
            ["Review this Python code: print('Hello World')"],
            ["Analyze the performance of a Django REST API."],
            ["Tell me about MGZon products and services."],
            ["Create a Flask route for user authentication."],
            ["What are the latest trends in AI?"],
            ["Provide guidelines for publishing a technical blog post."],
            ["Who is the founder of MGZon?"],
        ],
        inputs=[message]
    )

    def handle_submit(msg, history, sys_prompt, temp, effort, browsing, max_tokens):
        for response in generate(msg, history, sys_prompt, temp, effort, browsing, max_tokens):
            yield response

    submit.click(
        fn=handle_submit,
        inputs=[message, chatbot, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens],
        outputs=[chatbot]
    )
    stop_btn.click(fn=None, cancels=[submit])  # زر Stop لإيقاف الـ generation

# إعداد FastAPI
app = FastAPI(title="MGZon Chatbot API")

# ربط Gradio مع FastAPI
app = gr.mount_gradio_app(app, chatbot_ui, path="/gradio")

# ربط الملفات الثابتة والقوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# تضمين API endpoints
app.include_router(api_router)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))