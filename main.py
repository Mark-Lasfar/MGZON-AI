import os
import logging
import io
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

# ================= CSS مخصص =================
css = """
.gradio-container { max-width: 1200px; margin: auto; }
.chatbot { border: 1px solid #ccc; border-radius: 10px; padding: 15px; background-color: #f9f9f9; }
.input-textbox { font-size: 16px; padding: 10px; }
.upload-button::before {
    content: '📷';
    margin-right: 8px;
    font-size: 22px;
}
.audio-input::before {
    content: '🎤';
    margin-right: 8px;
    font-size: 22px;
}
.audio-output::before {
    content: '🔊';
    margin-right: 8px;
    font-size: 22px;
}
"""

# ================= دالة المعالجة =================
def process_input(message, audio_input=None, file_input=None, history=None,
                  system_prompt=None, temperature=0.7, reasoning_effort="medium",
                  enable_browsing=True, max_new_tokens=128000):
    input_type = "text"
    audio_data = None
    image_data = None

    # 🎤 إدخال صوتي من المايك
    if audio_input:
        input_type = "audio"
        with open(audio_input, "rb") as f:
            audio_data = f.read()
        message = "Transcribe this audio"

    # 📂 صورة أو ملف
    elif file_input:
        input_type = "file"
        if file_input.endswith(('.png', '.jpg', '.jpeg')):
            input_type = "image"
            with open(file_input, "rb") as f:
                image_data = f.read()
            message = f"Analyze image: {file_input}"
        else:
            message = f"Analyze file: {file_input}"

    response_text = ""
    audio_response = None

    # 🚀 استدعاء generate (يرجع نص + صوت)
    for chunk in generate(
        message=message,
        history=history,
        system_prompt=system_prompt,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        enable_browsing=enable_browsing,
        max_new_tokens=max_new_tokens,
        input_type=input_type,
        audio_data=audio_data,
        image_data=image_data
    ):
        if isinstance(chunk, bytes):  # 🔊 الصوت
            audio_response = io.BytesIO(chunk)
            audio_response.name = "reply.wav"
        else:  # 📝 النص
            response_text += chunk

        # نرجع (النص + الصوت) لحظي
        yield response_text, audio_response

# ================= واجهة Gradio =================
chatbot_ui = gr.ChatInterface(
    fn=process_input,
    chatbot=gr.Chatbot(
        label="MGZon Chatbot",
        height=800,
        latex_delimiters=LATEX_DELIMS,
    ),
    additional_inputs_accordion=gr.Accordion("⚙️ Settings", open=True),
    additional_inputs=[
        gr.Textbox(
            label="System Prompt",
            value="You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, image, and file inputs. For audio, transcribe using Whisper. For text-to-speech, use Parler-TTS. For images and files, analyze content appropriately. Continue generating content until the query is fully addressed, leveraging the full capacity of the model.",
            lines=4
        ),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7),
        gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium"),
        gr.Checkbox(label="Enable DeepSearch (web browsing)", value=True),
        gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=128000),
        gr.Audio(label="🎤 Voice Input", sources=["microphone"], type="filepath", elem_classes="audio-input"),
        gr.File(label="Upload Image/File", file_types=["image", ".pdf", ".txt"], elem_classes="upload-button"),
    ],
    additional_outputs=[gr.Audio(label="🔊 Voice Reply", type="filepath", elem_classes="audio-output", autoplay=True)],
    stop_btn="Stop",
    examples=[
        ["Explain the difference between supervised and unsupervised learning in detail with examples."],
        ["Generate a complete React component for a login form with form validation and error handling."],
        ["Describe this image: https://example.com/image.jpg"],
        ["Transcribe this audio: [upload audio file]."],
        ["Convert this text to speech: Hello, welcome to MGZon!"],
        ["Analyze this file: [upload PDF or text file]."],
    ],
    title="MGZon Chatbot",
    description="A versatile chatbot powered by DeepSeek, CLIP, Whisper, and Parler-TTS for text, image, audio, and file queries. Supports long responses, voice input/output, file uploads with custom icons, and backup token switching. Licensed under Apache 2.0.",
    theme="gradio/soft",
    css=css,
)

# ================= FastAPI =================
app = FastAPI(title="MGZon Chatbot API")

# Mount Gradio داخل FastAPI
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

