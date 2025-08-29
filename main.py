import os
import io
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import gradio as gr

from api.endpoints import router as api_router
from utils.generation import generate, LATEX_DELIMS

# ================= إعداد اللوج =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Files in /app/: %s", os.listdir("/app"))

# ================= مفاتيح HuggingFace =================
HF_TOKEN = os.getenv("HF_TOKEN")
BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

# ================= إعداد Queue =================
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# ================= CSS =================
css = """
#input-row {display:flex; gap:6px; align-items:center;}
#msg-box {flex:1;}
"""

# ================= دالة المعالجة =================
def process_input(message, history, audio_input=None, file_input=None):
    input_type = "text"
    audio_data, image_data = None, None

    if audio_input:  # 🎤 صوت
        input_type = "audio"
        with open(audio_input, "rb") as f:
            audio_data = f.read()
        message = "Transcribe this audio"

    elif file_input:  # 📎 صورة / ملف
        if file_input.endswith(('.png', '.jpg', '.jpeg')):
            input_type = "image"
            with open(file_input, "rb") as f:
                image_data = f.read()
            message = f"Analyze image: {file_input}"
        else:
            input_type = "file"
            message = f"Analyze file: {file_input}"

    response_text, audio_response = "", None
    for chunk in generate(
        message=message,
        history=history,
        input_type=input_type,
        audio_data=audio_data,
        image_data=image_data
    ):
        if isinstance(chunk, bytes):  # 🔊 صوت
            audio_response = io.BytesIO(chunk)
            audio_response.name = "reply.wav"
        else:  # 📝 نص
            response_text += chunk

        yield response_text, audio_response

# ================= واجهة Gradio =================
with gr.Blocks(css=css, theme="gradio/soft") as chatbot_ui:
    chatbot = gr.Chatbot(label="MGZon Chatbot", height=700, latex_delimiters=LATEX_DELIMS)
    state = gr.State([])

    with gr.Row(elem_id="input-row"):
        msg = gr.Textbox(placeholder="Type your message...", elem_id="msg-box")
        mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤", elem_classes="audio-input")
        file = gr.File(file_types=["image", ".pdf", ".txt"], label="📎", elem_classes="upload-button")
        send_btn = gr.Button("Send")

    voice_reply = gr.Audio(label="🔊 Voice Reply", type="filepath", autoplay=True)

    def user_submit(message, history):
        history = history + [(message, None)]
        return "", history

    msg.submit(user_submit, [msg, state], [msg, state]).then(
        process_input, [msg, state, mic, file], [chatbot, voice_reply]
    )
    send_btn.click(user_submit, [msg, state], [msg, state]).then(
        process_input, [msg, state, mic, file], [chatbot, voice_reply]
    )

# ================= FastAPI =================
app = FastAPI(title="MGZon Chatbot API")

# ربط Gradio داخل FastAPI
app = gr.mount_gradio_app(app, chatbot_ui, path="/gradio")

# ملفات ثابتة + قوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Middleware 404
class NotFoundMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            if response.status_code == 404:
                logger.warning(f"404 Not Found: {request.url}")
                return templates.TemplateResponse("404.html", {"request": request}, status_code=404)
            return response
        except Exception as e:
            logger.exception(f"Error processing request {request.url}: {e}")
            return templates.TemplateResponse("500.html", {"request": request, "error": str(e)}, status_code=500)

app.add_middleware(NotFoundMiddleware)

# Root
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Docs
@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

# Swagger
@app.get("/swagger", response_class=HTMLResponse)
async def swagger_ui():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MGZon API Documentation")

# Redirect
@app.get("/launch-chatbot", response_class=RedirectResponse)
async def launch_chatbot():
    return RedirectResponse(url="/gradio", status_code=302)

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))

