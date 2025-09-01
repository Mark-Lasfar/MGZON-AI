import os
import logging
from fastapi import FastAPI, Request, Depends, HTTPException, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router
from api.auth import fastapi_users, auth_backend, current_active_user, google_oauth_client, github_oauth_client
from api.database import get_db, engine, Base
from api.models import User, UserRead, UserCreate
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List
import uvicorn
import markdown2

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

# إعداد MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    logger.error("MONGODB_URI is not set in environment variables.")
    raise ValueError("MONGODB_URI is required for MongoDB.")

client = AsyncIOMotorClient(MONGO_URI)
db = client["hager"]

# إعداد Jinja2 مع دعم Markdown
templates = Jinja2Templates(directory="templates")
templates.env.filters['markdown'] = lambda text: markdown2.markdown(text)

# موديل للمقالات
class BlogPost(BaseModel):
    id: str
    title: str
    content: str
    author: str
    date: str
    created_at: str

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# إعداد FastAPI
app = FastAPI(title="MGZon Chatbot API")

# إضافة SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("JWT_SECRET"))

# إنشاء الجداول تلقائيًا
Base.metadata.create_all(bind=engine)

# ربط الملفات الثابتة
app.mount("/static", StaticFiles(directory="static"), name="static")

# إضافة CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mgzon-mgzon-app.hf.space"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إضافة auth routers
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserCreate),
    prefix="/users",
    tags=["users"],
)
app.include_router(
    fastapi_users.get_oauth_router(google_oauth_client, auth_backend, os.getenv("JWT_SECRET"), redirect_url="https://mgzon-mgzon-app.hf.space/auth/google/callback"),
    prefix="/auth/google",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_oauth_router(github_oauth_client, auth_backend, os.getenv("JWT_SECRET"), redirect_url="https://mgzon-mgzon-app.hf.space/auth/github/callback"),
    prefix="/auth/github",
    tags=["auth"],
)

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
async def root(request: Request, user: User = Depends(fastapi_users.current_user(optional=True))):
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

# Chat endpoint
@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request, user: User = Depends(fastapi_users.current_user(optional=True))):
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

# About endpoint
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, user: User = Depends(fastapi_users.current_user(optional=True))):
    return templates.TemplateResponse("about.html", {"request": request, "user": user})

# Blog endpoint (قائمة المقالات)
@app.get("/blog", response_class=HTMLResponse)
async def blog(request: Request, skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    posts = await db.blog_posts.find().skip(skip).limit(limit).to_list(limit)
    return templates.TemplateResponse("blog.html", {"request": request, "posts": posts})

# Blog post endpoint (عرض مقالة كاملة)
@app.get("/blog/{post_id}", response_class=HTMLResponse)
async def blog_post(request: Request, post_id: str):
    post = await db.blog_posts.find_one({"id": post_id})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return templates.TemplateResponse("blog_post.html", {"request": request, "post": post})

# Docs endpoint
@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

# Swagger UI endpoint
@app.get("/swagger", response_class=HTMLResponse)
async def swagger_ui():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MGZon API Documentation")

# Sitemap endpoint (ديناميكي)
@app.get("/sitemap.xml", response_class=PlainTextResponse)
async def sitemap():
    posts = await db.blog_posts.find().to_list(100)
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/</loc>\n'
    xml += '    <lastmod>2025-09-01</lastmod>\n'
    xml += '    <changefreq>daily</changefreq>\n'
    xml += '    <priority>1.0</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/chat</loc>\n'
    xml += '    <lastmod>2025-09-01</lastmod>\n'
    xml += '    <changefreq>daily</changefreq>\n'
    xml += '    <priority>0.8</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/about</loc>\n'
    xml += '    <lastmod>2025-09-01</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.7</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/login</loc>\n'
    xml += '    <lastmod>2025-09-01</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.8</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/register</loc>\n'
    xml += '    <lastmod>2025-09-01</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.8</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/docs</loc>\n'
    xml += '    <lastmod>2025-09-01</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.9</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/blog</loc>\n'
    xml += '    <lastmod>2025-09-01</lastmod>\n'
    xml += '    <changefreq>daily</changefreq>\n'
    xml += '    <priority>0.9</priority>\n'
    xml += '  </url>\n'
    for post in posts:
        xml += '  <url>\n'
        xml += f'    <loc>https://mgzon-mgzon-app.hf.space/blog/{post["id"]}</loc>\n'
        xml += f'    <lastmod>{post["date"]}</lastmod>\n'
        xml += '    <changefreq>weekly</changefreq>\n'
        xml += '    <priority>0.9</priority>\n'
        xml += '  </url>\n'
    xml += '</urlset>'
    return xml

# Redirect لـ /gradio
@app.get("/gradio", response_class=RedirectResponse)
async def launch_chatbot():
    return RedirectResponse(url="/chat", status_code=302)

# ربط API endpoints
app.include_router(api_router)

# تشغيل الخادم
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
