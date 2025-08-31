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
.chatbot { 
    border: 1px solid #ccc; 
    border-radius: 15px; 
    padding: 20px; 
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
}
.input-textbox { 
    font-size: 18px; 
    padding: 12px; 
    border-radius: 8px; 
    border: 1px solid #aaa; 
}
.upload-button, .audio-input-button, .audio-record-button { 
    background: #4CAF50; 
    color: white; 
    border-radius: 8px; 
    padding: 10px 20px; 
    font-size: 16px; 
    cursor: pointer; 
}
.upload-button:hover, .audio-input-button:hover, .audio-record-button:hover { 
    background: #45a049; 
}
.upload-button::before { 
    content: '📷 '; 
    font-size: 20px; 
}
.audio-input-button::before { 
    content: '🎤 '; 
    font-size: 20px; 
}
.audio-record-button::before { 
    content: '🔊 '; 
    font-size: 20px; 
}
.loading::after {
    content: '';
    display: inline-block;
    width: 18px;
    height: 18px;
    border: 3px solid #333;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-left: 10px;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.output-container {
    margin-top: 25px;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 10px;
    background: #fff;
}
.audio-output-container {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: 15px;
}
.output-format-radio { 
    margin-top: 10px; 
}
"""

# دالة لمعالجة الإدخال
def process_input(message, audio_input=None, image_input=None, history=None, system_prompt=None, temperature=0.7, reasoning_effort="medium", enable_browsing=True, max_new_tokens=128000, output_format="text"):
    input_type = "text"
    audio_data = None
    image_data = None
    if audio_input:
        input_type = "audio"
        with open(audio_input, "rb") as f:
            audio_data = f.read()
        message = "Transcribe this audio"
    elif image_input:
        input_type = "image"
        with open(image_input, "rb") as f:
            image_data = f.read()
        message = f"Analyze this image"
    
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
        image_data=image_data,
        output_format=output_format
    ):
        if isinstance(chunk, bytes):
            audio_response = io.BytesIO(chunk)
            audio_response.name = "response.wav"
        else:
            response_text += chunk
        yield response_text, audio_response

# دالة لمعالجة زر إرسال الصوت
def submit_audio(audio_input, output_format):
    if not audio_input:
        return "Please upload or record an audio file.", None
    return process_input(message="", audio_input=audio_input, output_format=output_format)

# دالة لمعالجة زر إرسال الصورة
def submit_image(image_input, output_format):
    if not image_input:
        return "Please upload an image.", None
    return process_input(message="", image_input=image_input, output_format=output_format)

# إعداد واجهة Gradio
with gr.Blocks(css=css, theme="gradio/soft") as chatbot_ui:
    gr.Markdown(
        """
        # MGZon Chatbot 🤖
        A versatile chatbot powered by DeepSeek, GPT-OSS, CLIP, Whisper, and Parler-TTS. Supports text, audio, and image inputs with text or voice outputs. Upload files, record audio, or type your query and choose your output format!
        """
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500, latex_delimiters=LATEX_DELIMS)
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Settings", open=True):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, image, and file inputs. For audio, transcribe using Whisper. For text-to-speech, use Parler-TTS. For images, analyze content appropriately. Respond in the requested output format (text or audio).",
                    lines=4
                )
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7)
                reasoning_effort = gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium")
                enable_browsing = gr.Checkbox(label="Enable DeepSearch (web browsing)", value=True)
                max_new_tokens = gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=128000)
                output_format = gr.Radio(
                    label="Output Format",
                    choices=["text", "audio"],
                    value="text",
                    elem_classes="output-format-radio"
                )
    with gr.Row():
        message = gr.Textbox(label="Type your message", placeholder="Enter your query or describe your request...", lines=2, elem_classes="input-textbox")
        submit_btn = gr.Button("Send", variant="primary")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Record or Upload Audio", type="filepath", elem_classes="audio-input")
            audio_submit_btn = gr.Button("Send Audio", elem_classes="audio-input-button")
        with gr.Column(scale=1):
            image_input = gr.File(label="Upload Image", file_types=["image"], elem_classes="upload-button")
            image_submit_btn = gr.Button("Send Image", elem_classes="upload-button")
    output_text = gr.Textbox(label="Response", lines=10, elem_classes="output-container")
    output_audio = gr.Audio(label="Voice Output", type="filepath", elem_classes="audio-output-container", autoplay=True)

    # ربط الأزرار
    submit_btn.click(
        fn=process_input,
        inputs=[message, audio_input, image_input, chatbot, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, output_format],
        outputs=[output_text, output_audio]
    )
    audio_submit_btn.click(
        fn=submit_audio,
        inputs=[audio_input, output_format],
        outputs=[output_text, output_audio]
    )
    image_submit_btn.click(
        fn=submit_image,
        inputs=[image_input, output_format],
        outputs=[output_text, output_audio]
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
