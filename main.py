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
    max-width: 800px;
    margin: 0 auto;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa;
    padding: 20px;
}
.chatbot {
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    height: 70vh;
    overflow-y: auto;
    padding: 20px;
}
.chatbot .message {
    margin-bottom: 15px;
    padding: 10px 15px;
    border-radius: 8px;
}
.chatbot .user {
    background: #25D366;
    color: white;
    margin-left: 20%;
    border-radius: 8px 8px 0 8px;
}
.chatbot .assistant {
    background: #f1f0f0;
    margin-right: 20%;
    border-radius: 8px 8px 8px 0;
}
.input-container {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 50px;
    padding: 10px;
    margin: 20px 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.input-textbox {
    flex-grow: 1;
    border: none;
    outline: none;
    font-size: 16px;
    padding: 12px 15px;
    background: transparent;
}
.input-icon, .submit-btn {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 20px;
    padding: 10px;
    color: #333;
    transition: color 0.2s;
}
.input-icon:hover, .submit-btn:hover {
    color: #25D366;
}
.submit-btn {
    background: #25D366;
    color: white;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.output-container {
    background: #f9f9f9;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
.audio-output-container {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #ffffff;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}
.settings-accordion {
    background: #ffffff;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.upload-preview {
    max-width: 200px;
    max-height: 200px;
    border-radius: 8px;
    margin: 10px 0;
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
            return f"Error: Failed to read audio file: {e}", None, [], ""
    elif image_input:
        input_type = "image"
        try:
            with open(image_input, "rb") as f:
                image_data = f.read()
            message = "Analyze this image"
        except Exception as e:
            logger.error(f"Failed to read image file: {e}")
            return f"Error: Failed to read image file: {e}", None, [], ""
    
    response_text = ""
    audio_response = None
    chatbot_history = history or []
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
            chatbot_history.append({"role": "assistant", "content": response_text})
            yield response_text or "Processing...", audio_response, chatbot_history, ""
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error: Generation failed: {e}", None, [], ""

# دالة لتنظيف المدخلات بعد الإرسال
def clear_inputs(response_text, audio_response, chatbot, message):
    return response_text, audio_response, [], ""

# دالة لمعاينة الملفات
def preview_file(file_input, audio_input):
    if file_input:
        return gr.update(value=f"<img src='{file_input}' class='upload-preview'>", visible=True), gr.update(visible=False)
    if audio_input:
        return gr.update(visible=False), gr.update(value=f"<audio controls src='{audio_input}'></audio>", visible=True)
    return gr.update(visible=False), gr.update(visible=False)

# دالة لمعالجة زر إرسال الصوت
def submit_audio(audio_input, output_format):
    if not audio_input:
        return "Please upload or record an audio file.", None, [], ""
    response_text = ""
    audio_response = None
    chatbot_history = []
    try:
        for text, audio in process_input(message="", audio_input=audio_input, output_format=output_format):
            response_text = text or "No text response generated."
            audio_response = audio
            chatbot_history.append({"role": "assistant", "content": response_text})
        return response_text, audio_response, chatbot_history, ""
    except Exception as e:
        logger.error(f"Audio submission failed: {e}")
        return f"Error: Audio processing failed: {e}", None, [], ""

# دالة لمعالجة زر إرسال الصورة
def submit_image(image_input, output_format):
    if not image_input:
        return "Please upload an image.", None, [], ""
    response_text = ""
    audio_response = None
    chatbot_history = []
    try:
        for text, audio in process_input(message="", image_input=image_input, output_format=output_format):
            response_text = text or "No text response generated."
            audio_response = audio
            chatbot_history.append({"role": "assistant", "content": response_text})
        return response_text, audio_response, chatbot_history, ""
    except Exception as e:
        logger.error(f"Image submission failed: {e}")
        return f"Error: Image processing failed: {e}", None, [], ""

# إعداد واجهة Gradio
with gr.Blocks(css=css, theme="gradio/soft") as chatbot_ui:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <img src="https://raw.githubusercontent.com/Mark-Lasfar/MGZon/main/public/icons/mg.svg" alt="MGZon Logo" style="width: 100px; margin-bottom: 10px;">
            <h1>MGZon Chatbot</h1>
            <p>A versatile chatbot powered by DeepSeek, GPT-OSS, CLIP, Whisper, and TTS. Type your query, upload images/files, or record audio in one sleek input form!</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=600,
                latex_delimiters=LATEX_DELIMS,
                elem_classes="chatbot",
                type="messages"
            )
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Settings", open=False, elem_classes="settings-accordion"):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, image, and file inputs. For audio, transcribe using Whisper. For text-to-speech, use TTS. For images, analyze content appropriately. Respond in the requested output format (text or audio).",
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
                file_preview = gr.HTML(label="File Preview", visible=False)
                audio_preview = gr.HTML(label="Audio Preview", visible=False)
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
    file_input.change(
        fn=preview_file,
        inputs=[file_input, audio_input],
        outputs=[file_preview, audio_preview]
    )
    audio_input.change(
        fn=preview_file,
        inputs=[file_input, audio_input],
        outputs=[file_preview, audio_preview]
    )
    submit_btn.click(
        fn=process_input,
        inputs=[message, audio_input, file_input, chatbot, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, output_format],
        outputs=[output_text, output_audio, chatbot, message],
    ).then(
        fn=clear_inputs,
        inputs=[output_text, output_audio, chatbot, message],
        outputs=[output_text, output_audio, chatbot, message]
    ).then(
        fn=lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[file_preview, audio_preview]
    )
    message.submit(
        fn=process_input,
        inputs=[message, audio_input, file_input, chatbot, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, output_format],
        outputs=[output_text, output_audio, chatbot, message],
    ).then(
        fn=clear_inputs,
        inputs=[output_text, output_audio, chatbot, message],
        outputs=[output_text, output_audio, chatbot, message]
    ).then(
        fn=lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[file_preview, audio_preview]
    )
    file_input.change(
        fn=submit_image,
        inputs=[file_input, output_format],
        outputs=[output_text, output_audio, chatbot, message],
    ).then(
        fn=clear_inputs,
        inputs=[output_text, output_audio, chatbot, message],
        outputs=[output_text, output_audio, chatbot, message]
    ).then(
        fn=lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[file_preview, audio_preview]
    )
    audio_input.change(
        fn=submit_audio,
        inputs=[audio_input, output_format],
        outputs=[output_text, output_audio, chatbot, message],
    ).then(
        fn=clear_inputs,
        inputs=[output_text, output_audio, chatbot, message],
        outputs=[output_text, output_audio, chatbot, message]
    ).then(
        fn=lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[file_preview, audio_preview]
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
