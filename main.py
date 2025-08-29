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
BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")  # إضافة التوكن الاحتياطي
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# إعداد CSS
css = """
.gradio-container { max-width: 1200px; margin: auto; }


.chatbot {
    background: #fafafa;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 10px;
}

.header h1 {
    font-size: 32px;
    color: #333;
}
.header p {
    color: #555;
    margin-top: -10px;
}

.gr-accordion {
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}



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
.loading::after {
    content: '';
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid #333;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-left: 8px;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.output-container {
    margin-top: 20px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 8px;
}
.audio-output-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
}
"""

# دالة لمعالجة الإدخال (نص، صوت، صور، ملفات)
def process_input(message, audio_input=None, file_input=None, history=None, system_prompt=None, temperature=0.7, reasoning_effort="medium", enable_browsing=True, max_new_tokens=128000):
    input_type = "text"
    audio_data = None
    image_data = None
    if audio_input:
        input_type = "audio"
        with open(audio_input, "rb") as f:
            audio_data = f.read()
        message = "Transcribe this audio"
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
with gr.Blocks(css=css, theme="gradio/soft") as chatbot_ui:
    with gr.Column(elem_classes="header"):
        gr.Markdown("<h1 style='text-align:center'>🤖 MGZon Chatbot</h1>")
        gr.Markdown(
            "<p style='text-align:center; font-size:16px'>"
            "A versatile assistant powered by DeepSeek, CLIP, Whisper, and Parler-TTS for text, image, audio, and file queries."
            "</p>"
        )

    with gr.Row():
        chatbot = gr.Chatbot(
            label="MGZon Chatbot",
            height=750,
            latex_delimiters=LATEX_DELIMS,
            elem_classes="chatbot",
            type="messages"
        )

    gr.Markdown("---")  # فاصل بسيط

    with gr.Accordion("⚙️ Settings", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                system_prompt = gr.Textbox(
                    label="System Prompt",
		    value="""You are an expert assistant providing detailed, comprehensive, and well-structured responses. 
		Support text, audio, image, and file inputs. 
		For audio, transcribe using Whisper. For text-to-speech, use Parler-TTS. 
		For images and files, analyze content appropriately. 
		Continue generating content until the query is fully addressed, leveraging the full capacity of the model.""",
                    lines=6,
                    elem_classes="input-textbox"
                )
                reasoning_effort = gr.Radio(
                    label="Reasoning Effort",
                    choices=["low", "medium", "high"],
                    value="medium"
                )
                enable_browsing = gr.Checkbox(
                    label="Enable DeepSearch (web browsing)",
                    value=True
                )
            with gr.Column(scale=1):
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.7
                )
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=50,
                    maximum=128000,
                    step=50,
                    value=128000
                )
                audio_input = gr.Audio(
                    label="Voice Input",
                    type="filepath",
                    elem_classes="audio-input"
                )
                file_input = gr.File(
                    label="Upload Image/File",
                    file_types=["image", ".pdf", ".txt"],
                    elem_classes="upload-button"
                )

    audio_output = gr.Audio(
        label="Voice Output",
        type="filepath",
        elem_classes="audio-output",
        autoplay=True
    )

    gr.ChatInterface(
        fn=process_input,
        chatbot=chatbot,
        additional_inputs=[system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, audio_input, file_input],
        additional_outputs=[audio_output],
        stop_btn="Stop",
        examples=[
            ["Explain the difference between supervised and unsupervised learning in detail with examples."],
            ["Generate a complete React component for a login form with form validation and error handling."],
            ["Describe this image: https://example.com/image.jpg"],
            ["Transcribe this audio: [upload audio file]."],
            ["Convert this text to speech: Hello, welcome to MGZon!"],
            ["Analyze this file: [upload PDF or text file]."],
        ],
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
