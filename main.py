import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
import gradio as gr
from api.endpoints import router as api_router
from utils.generation import generate, LATEX_DELIMS
from utils.web_search import web_search
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from jose import JWTError, jwt

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحقق من الملفات في /app/
logger.info("Files in /app/: %s", os.listdir("/app"))

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set.")
    raise ValueError("HF_TOKEN is required for Inference API.")

# إعدادات JWT لـ OAuth 2.0
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")  # غيّرها في الإنتاج
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# إعداد OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/oauth/token")

# إعداد FastAPI
app = FastAPI(
    title="MGZon Chatbot API",
    description="API for MGZon Chatbot with support for code generation, analysis, and MGZon-specific queries.",
    version="1.0.0",
)

# ربط الملفات الثابتة والقوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# تضمين API endpoints
app.include_router(api_router)

# إعداد CSS لـ Gradio
css = """
.gradio-container { max-width: 900px; margin: auto; }
.chatbot { border: 2px solid #ff6f61; border-radius: 15px; background: rgba(255, 255, 255, 0.05); }
.input-textbox { font-size: 18px; border-radius: 10px; }
.submit-btn { background: linear-gradient(45deg, #ff6f61, #e55a50); }
"""

# إعداد واجهة Gradio
chatbot_ui = gr.ChatInterface(
    fn=generate,
    type="messages",
    chatbot=gr.Chatbot(
        label="MGZon Chatbot",
        type="messages",
        height=700,
        latex_delimiters=LATEX_DELIMS,
        avatar_images=(
            "https://raw.githubusercontent.com/Mark-Lasfar/MGZon/main/public/icons/mg.svg",
            "https://raw.githubusercontent.com/Mark-Lasfar/MGZon/main/public/icons/mg.svg",
        ),
    ),
    additional_inputs_accordion=gr.Accordion("⚙️ Settings", open=True),
    additional_inputs=[
        gr.Textbox(
            label="System prompt",
            value="You are a helpful assistant capable of code generation, analysis, review, and more. Support frameworks like React, Django, Flask, Rails, Laravel, and others.",
            lines=2,
        ),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7),
        gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium"),
        gr.Checkbox(label="Enable DeepSearch (web browsing)", value=True),
        gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=4096),
    ],
    stop_btn="Stop",
    examples=[
        ["Explain the difference between supervised and unsupervised learning."],
        ["Generate a React component for a login form."],
        ["Review this Python code: print('Hello World')"],
        ["Analyze the performance of a Django REST API."],
        ["Tell me about MGZon products and services."],
        ["Create a Flask route for user authentication."],
        ["Generate a Laravel controller for user management."],
        ["What are the latest trends in AI?"],
        ["Provide guidelines for publishing a technical blog post."],
        ["Who is the founder of MGZon?"],
    ],
    title="MGZon Chatbot",
    description=(
        "A versatile chatbot powered by GPT-OSS-20B and MGZON/Veltrix. Supports code generation, analysis, review, web search, and MGZon-specific queries. "
        "Built by <a href='https://mark-elasfar.web.app/' target='_blank'>Mark Al-Asfar</a>. "
        "Licensed under Apache 2.0. <strong>DISCLAIMER:</strong> Analysis may contain internal thoughts not suitable for final response."
    ),
    theme="gradio/soft",
    css=css,
)

# ربط Gradio مع FastAPI
app = gr.mount_gradio_app(app, chatbot_ui, path="/gradio")

# Middleware لمعالجة 404
class NotFoundMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            if response.status_code == 404:
                return templates.TemplateResponse("404.html", {"request": request}, status_code=404)
            return response
        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

app.add_middleware(NotFoundMiddleware)

# OAuth 2.0 token endpoint (مثال أولي)
@app.post("/oauth/token")
async def token():
    """إنشاء رمز OAuth 2.0 (مثال أولي)."""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = jwt.encode(
        {"sub": "mgzon_user", "exp": datetime.utcnow() + access_token_expires},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """فحص حالة النموذج والخدمة."""
    try:
        # اختبار اتصال بسيط بنموذج افتراضي
        client = OpenAI(api_key=HF_TOKEN, base_url=FALLBACK_API_ENDPOINT, timeout=10.0)
        client.chat.completions.create(
            model=TERTIARY_MODEL_NAME,
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=10,
        )
        return JSONResponse({"status": "healthy", "model": TERTIARY_MODEL_NAME})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Docs endpoint
@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

# Redirect لـ /gradio
@app.get("/launch-chatbot", response_class=RedirectResponse)
async def launch_chatbot():
    try:
        return RedirectResponse(url="/gradio")
    except Exception as e:
        logger.error(f"Failed to redirect to /gradio: {e}")
        raise HTTPException(status_code=500, detail="Failed to redirect to chatbot")

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
