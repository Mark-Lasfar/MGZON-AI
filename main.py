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
.gradio-container { 
    max-width: 1000px; 
    margin: auto; 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
    background: #f0f2f5;
}
.chatbot { 
    border: none; 
    border-radius: 20px; 
    padding: 15px; 
    background: #fff; 
    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
    height: 600px; 
    overflow-y: auto;
}
.input-container { 
    display: flex; 
    align-items: center; 
    gap: 8px; 
    border: 1px solid #ddd; 
    border-radius: 25px; 
    padding: 8px; 
    background: #fff; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
    position: sticky; 
    bottom: 10px; 
    margin: 10px;
}
.input-textbox { 
    flex-grow: 1; 
    border: none; 
    outline: none; 
    font-size: 16px; 
    padding: 10px 15px; 
    border-radius: 20px; 
    background: transparent;
}
.input-icon { 
    background: none; 
    border: none; 
    cursor: pointer; 
    font-size: 22px; 
    padding: 8px; 
    color: #555; 
    transition: color 0.2s;
}
.input-icon:hover { 
    color: #0084ff; 
}
.submit-btn { 
    background: #0084ff; 
    color: white; 
    border-radius: 50%; 
    width: 36px; 
    height: 36px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    font-size: 18px; 
    cursor: pointer; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.submit-btn:hover { 
    background: #0066cc; 
}
.output-container { 
    margin: 15px 0; 
    padding: 15px; 
    border-radius: 10px; 
    background: #f9f9f9; 
    border: 1px solid #e0e0e0;
}
.settings-accordion { 
    background: #fff; 
    border-radius: 10px; 
    padding: 15px; 
    box-shadow: 0 1px 5px rgba(0,0,0,0.1); 
    margin-bottom: 10px;
}
.audio-output-container { 
    display: flex; 
    align-items: center; 
    gap: 10px; 
    margin-top: 10px; 
    background: #fff; 
    padding: 10px; 
    border-radius: 10px;
}
.gr-button { 
    transition: background-color 0.2s, transform 0.1s; 
}
.gr-button:hover { 
    transform: scale(1.05); 
}
"""
# دالة لمعالجة الإدخال
def process_input(message, audio_input=None, image_input=None, history=None, system_prompt=None, temperature=0.7, reasoning_effort="medium", enable_browsing=True, max_new_tokens=128000, output_format="text"):
    input_type = "text"
    audio_data = None
    image_data = None
    if audio_input:
        input_type = "audio"
        try:
            with open(audio_input, "rb") as f:
                audio_data = f.read()
            message = "Transcribe this audio"
        except Exception as e:
            logger.error(f"Failed to read audio file: {e}")
            return f"Error: Failed to read audio file: {e}", None
    elif image_input:
        input_type = "image"
        try:
            with open(image_input, "rb") as f:
                image_data = f.read()
            message = "Analyze this image"
        except Exception as e:
            logger.error(f"Failed to read image file: {e}")
            return f"Error: Failed to read image file: {e}", None
    
    response_text = ""
    audio_response = None
    try:
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
            yield response_text or "Processing...", audio_response
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        yield f"Error: Generation failed: {e}", None

# دالة لمعالجة زر إرسال الصوت
def submit_audio(audio_input, output_format):
    if not audio_input:
        return "Please upload or record an audio file.", None
    response_text = ""
    audio_response = None
    try:
        for text, audio in process_input(message="", audio_input=audio_input, output_format=output_format):
            response_text = text or "No text response generated."
            audio_response = audio
        return response_text, audio_response
    except Exception as e:
        logger.error(f"Audio submission failed: {e}")
        return f"Error: Audio processing failed: {e}", None

# دالة لمعالجة زر إرسال الصورة
def submit_image(image_input, output_format):
    if not image_input:
        return "Please upload an image.", None
    response_text = ""
    audio_response = None
    try:
        for text, audio in process_input(message="", image_input=image_input, output_format=output_format):
            response_text = text or "No text response generated."
            audio_response = audio
        return response_text, audio_response
    except Exception as e:
        logger.error(f"Image submission failed: {e}")
        return f"Error: Image processing failed: {e}", None

# إعداد واجهة Gradio
# إعداد واجهة Gradio
with gr.Blocks(css=css, theme="gradio/soft") as chatbot_ui:
    gr.Markdown(
        """
        # MGZon Chatbot 🤖
        A versatile chatbot powered by DeepSeek, GPT-OSS, CLIP, Whisper, and Parler-TTS. Type your query, upload images/files, or record audio in one sleek input form!
        """
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=600, latex_delimiters=LATEX_DELIMS, elem_classes="chatbot")
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Settings", open=False, elem_classes="settings-accordion"):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, image, and file inputs. For audio, transcribe using Whisper. For text-to-speech, use Parler-TTS. For images, analyze content appropriately. Respond in the requested output format (text or audio).",
                    lines=4
                )
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7)
                reasoning_effort = gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium")
                enable_browsing = gr.Checkbox(label="Enable DeepSearch (web browsing)", value=True)
                max_new_tokens = gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=128000)
                output_format = gr.Radio(label="Output Format", choices=["text", "audio"], value="text")
    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="input-container"):
                message = gr.Textbox(
                    placeholder="Type your message, or use icons to upload files/audio...",
                    lines=1,
                    elem_classes="input-textbox",
                    show_label=False
                )
                file_input = gr.File(
                    file_types=["image", ".mp3", ".wav"],
                    show_label=False,
                    elem_classes="input-icon",
                    visible=False
                )
                audio_input = gr.Audio(
                    type="filepath",
                    show_label=False,
                    elem_classes="input-icon",
                    visible=False
                )
                file_btn = gr.Button("📎", elem_classes="input-icon")
                audio_btn = gr.Button("🎤", elem_classes="input-icon")
                submit_btn = gr.Button("➡️", elem_classes="submit-btn")
    output_text = gr.Textbox(label="Response", lines=10, elem_classes="output-container")
    output_audio = gr.Audio(label="Voice Output", type="filepath", elem_classes="audio-output-container", autoplay=True)

    # ربط الأحداث
    file_btn.click(
        fn=lambda: gr.update(visible=True),
        outputs=file_input
    )
    audio_btn.click(
        fn=lambda: gr.update(visible=True),
        outputs=audio_input
    )
    submit_btn.click(
        fn=process_input,
        inputs=[message, audio_input, file_input, chatbot, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, output_format],
        outputs=[output_text, output_audio, chatbot, message],
        _js="() => { return ['', null, null, []]; }"  # تنظيف المدخلات
    )
    message.submit(
        fn=process_input,
        inputs=[message, audio_input, file_input, chatbot, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, output_format],
        outputs=[output_text, output_audio, chatbot, message],
        _js="() => { return ['', null, null, []]; }"  # تنظيف المدخلات
    )
    file_input.change(
        fn=submit_image,
        inputs=[file_input, output_format],
        outputs=[output_text, output_audio, chatbot, message],
        _js="() => { return ['', null, null, []]; }"  # تنظيف المدخلات
    )
    audio_input.change(
        fn=submit_audio,
        inputs=[audio_input, output_format],
        outputs=[output_text, output_audio, chatbot, message],
        _js="() => { return ['', null, null, []]; }"  # تنظيف المدخلات
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
