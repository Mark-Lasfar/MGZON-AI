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

# إعداد CSS محسّن
css = """
.gradio-container { max-width: 1200px; margin: auto; font-family: Arial, sans-serif; }
.chatbot { border: 1px solid #ccc; border-radius: 12px; padding: 20px; background-color: #f5f5f5; }
.input-textbox { font-size: 16px; padding: 12px; border-radius: 8px; }
.upload-button, .audio-button, .camera-button { 
    background-color: #007bff; color: white; padding: 10px 20px; border-radius: 8px; 
    display: inline-flex; align-items: center; gap: 8px; font-size: 16px; 
}
.upload-button::before { content: '📷'; font-size: 20px; }
.audio-button::before { content: '🎤'; font-size: 20px; }
.camera-button::before { content: '📸'; font-size: 20px; }
.audio-output-container { 
    display: flex; align-items: center; gap: 12px; margin-top: 15px; 
    background-color: #e9ecef; padding: 10px; border-radius: 8px; 
}
.audio-output-container::before { content: '🔊'; font-size: 20px; }
.loading::after {
    content: ''; display: inline-block; width: 18px; height: 18px; 
    border: 3px solid #007bff; border-top-color: transparent; 
    border-radius: 50%; animation: spin 1s linear infinite; margin-left: 10px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.output-container { 
    margin-top: 20px; padding: 15px; border: 1px solid #ddd; 
    border-radius: 10px; background-color: white; 
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
        message = f"Analyze image: {message or 'describe this image'}"
    
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

# إعداد واجهة Gradio
chatbot_ui = gr.ChatInterface(
    fn=process_input,
    chatbot=gr.Chatbot(
        label="MGZon Chatbot",
        height=800,
        latex_delimiters=LATEX_DELIMS,
        elem_classes="chatbot",
    ),
    additional_inputs_accordion=gr.Accordion("⚙️ Settings", open=True),
    additional_inputs=[
        gr.Textbox(
            label="System Prompt",
            value="You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, image inputs. Transcribe audio using Whisper, convert text to speech using Parler-TTS, and analyze images using CLIP. Respond with text or audio based on input type. Continue until the query is fully addressed.",
            lines=4
        ),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7),
        gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium"),
        gr.Checkbox(label="Enable DeepSearch", value=True),
        gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=128000),
        gr.Audio(label="Record Audio", source="microphone", type="numpy", elem_classes="audio-button"),
        gr.Image(label="Capture Image", source="webcam", type="numpy", elem_classes="camera-button"),
        gr.File(label="Upload Image/File", file_types=["image", ".pdf", ".txt"], elem_classes="upload-button"),
    ],
    additional_outputs=[gr.Audio(label="Voice Output", type="filepath", elem_classes="audio-output-container", autoplay=True)],
    stop_btn="Stop",
    examples=[
        ["Explain the history of AI in detail."],
        ["Generate a React login component with validation."],
        ["Describe this image: [capture or upload image]"],
        ["Transcribe this audio: [record audio]"],
        ["Convert to speech: Hello, welcome to MGZon!"],
    ],
    title="MGZon Chatbot",
    description="A versatile chatbot powered by Hugging Face models for text, image, and audio queries. Supports real-time audio recording, webcam image capture, and web search. Licensed under Apache 2.0.",
    theme="gradio/soft",
    css=css,
)

# إعداد FastAPI
app = FastAPI(title="MGZon Chatbot API")

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

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Docs endpoint
@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

# Swagger UI endpoint
@app.get("/swagger", response_class=HTMLResponse)
async def swagger_ui():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MGZon API Documentation")

# Redirect لـ /gradio
@app.get("/launch-chatbot", response_class=RedirectResponse)
async def launch_chatbot():
    return RedirectResponse(url="/gradio", status_code=302)

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
