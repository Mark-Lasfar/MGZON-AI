import os
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
import io

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحقق من الملفات في /app/
logger.info("Files in /app/: %s", os.listdir("/app"))

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# إعداد CSS
css = """
.gradio-container { max-width: 1200px; margin: auto; font-family: Arial, sans-serif; }
.chatbot { border: 1px solid #ccc; border-radius: 12px; padding: 20px; background-color: #f5f5f5; }
.input-textbox { font-size: 16px; padding: 12px; border-radius: 8px; }
.upload-button, .capture-button, .record-button {
    background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 8px; font-size: 16px; cursor: pointer;
}
.upload-button:hover, .capture-button:hover, .record-button:hover { background-color: #45a049; }
.upload-button::before { content: '📷 '; font-size: 20px; }
.capture-button::before { content: '🎥 '; font-size: 20px; }
.record-button::before { content: '🎤 '; font-size: 20px; }
.audio-output::before { content: '🔊 '; font-size: 20px; }
.loading::after {
    content: ''; display: inline-block; width: 18px; height: 18px; border: 3px solid #333;
    border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-left: 10px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.output-container {
    margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 10px; background-color: #fff;
}
.audio-output-container {
    display: flex; align-items: center; gap: 12px; margin-top: 15px;
}
"""

# دالة لمعالجة الإدخال
def process_input(message, audio_input=None, image_input=None, history=None, system_prompt=None, temperature=0.7, reasoning_effort="medium", enable_browsing=True, max_new_tokens=128000):
    input_type = "text"
    audio_data = None
    image_data = None
    if audio_input:
        input_type = "audio"
        audio_data = audio_input
        message = "Transcribe this audio"
    elif image_input:
        input_type = "image"
        image_data = image_input
        message = "Analyze this image"
    
    response_text = ""
    audio_response = None
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
        if isinstance(chunk, bytes):
            audio_response = io.BytesIO(chunk)
            audio_response.name = "response.wav"
        else:
            response_text += chunk
        yield response_text, audio_response

# دالة لتفعيل تسجيل الصوت
def start_recording():
    return gr.update(visible=True)

# دالة لتفعيل التقاط الصورة
def start_image_capture():
    return gr.update(visible=True)

# إعداد واجهة Gradio
chatbot_ui = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(label="Message", placeholder="Type your message or use buttons below...", elem_classes="input-textbox"),
        gr.Audio(label="Record Audio", sources=["microphone"], type="numpy", streaming=True, visible=False, elem_classes="record-button"),
        gr.Image(label="Capture/Upload Image", sources=["webcam", "upload"], type="numpy", visible=False, elem_classes="capture-button"),
        gr.State(value=[]),  # History
        gr.Textbox(
            label="System Prompt",
            value="You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, image inputs. For audio, transcribe using Whisper. For text-to-speech, use Parler-TTS. For images, analyze using CLIP. Respond with voice output when requested. Continue until the query is fully addressed.",
            lines=4
        ),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7),
        gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium"),
        gr.Checkbox(label="Enable DeepSearch", value=True),
        gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=128000),
    ],
    outputs=[
        gr.Markdown(label="Response", elem_classes="output-container"),
        gr.Audio(label="Voice Output", type="filepath", elem_classes="audio-output", autoplay=True)
    ],
    additional_inputs=[
        gr.Button("Record Audio", elem_classes="record-button", onclick=start_recording),
        gr.Button("Capture/Upload Image", elem_classes="capture-button", onclick=start_image_capture),
    ],
    examples=[
        ["Explain the history of AI in detail."],
        ["Generate a React component for a login form."],
        ["Transcribe this audio: [record audio]."],
        ["Convert this text to speech: Hello, welcome to MGZon!"],
        ["Analyze this image: [capture/upload image]."],
    ],
    title="MGZon Chatbot",
    description="A versatile chatbot powered by advanced AI models. Supports text, audio, and image inputs with voice responses. Licensed under Apache 2.0.",
    theme="gradio/soft",
    css=css,
)

# إعداد FastAPI
app = FastAPI(title="MGZon Chatbot API")
app.include_router(api_router)

# ربط Gradio مع FastAPI
app = gr.mount_gradio_app(app, chatbot_ui, path="/gradio")

# ربط الملفات الثابتة والقوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Middleware لمعالجة 404
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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

@app.get("/swagger", response_class=HTMLResponse)
async def swagger_ui():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MGZon API Documentation")

@app.get("/launch-chatbot", response_class=RedirectResponse)
async def launch_chatbot():
    return RedirectResponse(url="/gradio", status_code=302)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
