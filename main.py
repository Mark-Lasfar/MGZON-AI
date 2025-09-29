# main.py
# SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
# SPDX-License-Identifier: Apache-2.0

import os
import logging
from fastapi import FastAPI, Request, Depends, HTTPException, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router
from api.auth import fastapi_users, auth_backend, current_active_user, get_auth_router
from api.database import User, Conversation, get_user_db, init_db
from api.models import UserRead, UserCreate, UserUpdate
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import uvicorn
import markdown2
from pathlib import Path
from hashlib import md5
from datetime import datetime
from httpx_oauth.exceptions import GetIdEmailError
import re
import anyio

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("motor").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info("Starting application...")
logger.debug("Files in current directory: %s", os.listdir(os.getcwd()))

# Check environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")
if not BACKUP_HF_TOKEN:
    logger.warning("BACKUP_HF_TOKEN is not set. Fallback to secondary model will not work if primary token fails.")

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    logger.error("MONGODB_URI is not set in environment variables.")
    raise ValueError("MONGODB_URI is required for MongoDB.")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    logger.error("JWT_SECRET is not set or too short.")
    raise ValueError("JWT_SECRET is required (at least 32 characters).")

ROUTER_API_URL = os.getenv("ROUTER_API_URL", "https://router.huggingface.co")
logger.debug(f"ROUTER_API_URL set to: {ROUTER_API_URL}")

# MongoDB setup
client = AsyncIOMotorClient(MONGO_URI)
mongo_db = client["hager"]
session_message_counts = mongo_db["session_message_counts"]

# Create MongoDB index
async def setup_mongo_index():
    try:
        await session_message_counts.create_index("session_id", unique=True)
        logger.info("MongoDB index created successfully for session_id")
    except Exception as e:
        logger.error(f"Failed to create MongoDB index: {e}")

# Jinja2 setup
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")
templates.env.filters['markdown'] = lambda text: markdown2.markdown(text)

# Pydantic model for blog posts
class BlogPost(BaseModel):
    id: str
    title: str
    content: str
    author: str
    date: str
    created_at: str

# Application settings
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))
logger.debug(f"Application settings: QUEUE_SIZE={QUEUE_SIZE}, CONCURRENCY_LIMIT={CONCURRENCY_LIMIT}")

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing MongoDB...")
    await init_db()
    await setup_mongo_index()
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="MGZon Chatbot API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

# Add SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=JWT_SECRET)
logger.debug("SessionMiddleware added with JWT_SECRET")

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
logger.debug("Static files mounted at /static")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mgzon-mgzon-app.hf.space",
        "http://localhost:7860",
        "http://localhost:8000",
        "http://localhost",
        "https://localhost",
        "capacitor://localhost",
        "file://",
        "https://hager-zon.vercel.app",
        "https://mgzon-mgzon-app.hf.space/auth/google/callback",
        "https://mgzon-mgzon-app.hf.space/auth/github/callback",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.debug("CORS middleware configured with allowed origins")

# Include routers
app.include_router(api_router)
get_auth_router(app)
logger.debug("API and auth routers included")

# Add logout endpoint
@app.post("/logout")
async def logout(request: Request):
    logger.info("User logout requested")
    session_data = request.session.copy()
    request.session.clear()
    logger.debug(f"Cleared session data: {session_data}")
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("access_token")
    response.delete_cookie("session")
    logger.debug("Session and access_token cookies deleted")
    return response

# Debug routes endpoint
@app.get("/debug/routes", response_class=PlainTextResponse)
async def debug_routes():
    logger.debug("Fetching debug routes")
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", [])
        path = getattr(route, "path", "Unknown")
        routes.append(f"{methods} {path}")
    return "\n".join(sorted(routes))

# Custom middleware for 404 and 500 errors
class NotFoundMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            if response.status_code == 404:
                logger.warning(f"404 Not Found: {request.url}")
                return templates.TemplateResponse("404.html", {"request": request}, status_code=404)
            return response
        except Exception as e:
            logger.exception(f"Error processing request {request.url}: {str(e)}")
            if isinstance(e, anyio.EndOfStream):
                logger.error("EndOfStream error detected - likely async context issue")
                return templates.TemplateResponse(
                    "500.html",
                    {"request": request, "error": "Async context error"},
                    status_code=500
                )
            return templates.TemplateResponse(
                "500.html",
                {"request": request, "error": str(e)},
                status_code=500
            )

app.add_middleware(NotFoundMiddleware)
logger.debug("NotFoundMiddleware added")

# OAuth error handler
@app.exception_handler(GetIdEmailError)
async def handle_oauth_error(request: Request, exc: GetIdEmailError):
    logger.error(f"OAuth error: {exc}")
    error_message = "Failed to authenticate with OAuth. Please try again or contact support."
    return RedirectResponse(
        url=f"/login?error={error_message}",
        status_code=302
    )

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user: User = Depends(current_active_user)):
    logger.debug(f"Root endpoint accessed by user: {user.email if user else 'Anonymous'}")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "is_authenticated": user is not None
    })

# Google verification
@app.get("/google97468ef1f6b6e804.html", response_class=PlainTextResponse)
async def google_verification():
    logger.debug("Google verification endpoint accessed")
    return "google-site-verification: google97468ef1f6b6e804.html"

# Login page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User = Depends(current_active_user)):
    if user:
        logger.debug(f"User {user.email} already logged in, redirecting to /chat")
        return RedirectResponse(url="/chat", status_code=302)
    logger.debug("Login page accessed")
    return templates.TemplateResponse("login.html", {"request": request})

# Register page
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: User = Depends(current_active_user)):
    if user:
        logger.debug(f"User {user.email} already logged in, redirecting to /chat")
        return RedirectResponse(url="/chat", status_code=302)
    logger.debug("Register page accessed")
    return templates.TemplateResponse("register.html", {"request": request})

# Chat page
@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request, user: User = Depends(current_active_user)):
    logger.debug(f"Chat page accessed by user: {user.email if user else 'Anonymous'}")
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

# Specific conversation page
@app.get("/chat/{conversation_id}", response_class=HTMLResponse)
async def chat_conversation(
    request: Request,
    conversation_id: str,
    user: User = Depends(current_active_user),
):
    if not user:
        logger.debug("Anonymous user attempted to access conversation page, redirecting to /login")
        return RedirectResponse(url="/login", status_code=302)
    
    conversation = await mongo_db.conversation.find_one({
        "conversation_id": conversation_id,
        "user_id": user.id
    })
    if not conversation:
        logger.warning(f"Conversation {conversation_id} not found for user {user.email}")
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    logger.debug(f"Conversation page accessed: {conversation_id} by user: {user.email}")
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": user,
            "conversation_id": conversation["conversation_id"],
            "conversation_title": conversation["title"] or "Untitled Conversation"
        }
    )

# About page
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, user: User = Depends(current_active_user)):
    logger.debug(f"About page accessed by user: {user.email if user else 'Anonymous'}")
    return templates.TemplateResponse("about.html", {"request": request, "user": user})

# Serve static files
@app.get("/static/{path:path}")
async def serve_static(path: str):
    clean_path = re.sub(r'\?.*', '', path)
    file_path = Path("static") / clean_path
    if not file_path.exists():
        logger.warning(f"Static file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    cache_duration = 31536000 if not clean_path.endswith(('.js', '.css')) else 3600
    with open(file_path, "rb") as f:
        file_hash = md5(f.read()).hexdigest()
    headers = {
        "Cache-Control": f"public, max-age={cache_duration}",
        "ETag": file_hash,
        "Last-Modified": datetime.utcfromtimestamp(file_path.stat().st_mtime).strftime('%a, %d %b %Y %H:%M:%S GMT')
    }
    logger.debug(f"Serving static file: {file_path}")
    return FileResponse(file_path, headers=headers)

# Blog page
@app.get("/blog", response_class=HTMLResponse)
async def blog(request: Request, skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    logger.debug(f"Blog page accessed with skip={skip}, limit={limit}")
    posts = await mongo_db.blog_posts.find().skip(skip).limit(limit).to_list(limit)
    return templates.TemplateResponse("blog.html", {"request": request, "posts": posts})

# Individual blog post
@app.get("/blog/{post_id}", response_class=HTMLResponse)
async def blog_post(request: Request, post_id: str):
    logger.debug(f"Blog post accessed: {post_id}")
    post = await mongo_db.blog_posts.find_one({"id": post_id})
    if not post:
        logger.warning(f"Blog post not found: {post_id}")
        raise HTTPException(status_code=404, detail="Post not found")
    return templates.TemplateResponse("blog_post.html", {"request": request, "post": post})

# Docs page
@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    logger.debug("Docs page accessed")
    return templates.TemplateResponse("docs.html", {"request": request})

# Swagger UI
@app.get("/swagger", response_class=HTMLResponse)
async def swagger_ui():
    logger.debug("Swagger UI accessed")
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MGZon API Documentation")

# Sitemap
@app.get("/sitemap.xml", response_class=PlainTextResponse)
async def sitemap():
    logger.debug("Sitemap accessed")
    posts = await mongo_db.blog_posts.find().to_list(100)
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/</loc>\n'
    xml += f'    <lastmod>{current_date}</lastmod>\n'
    xml += '    <changefreq>daily</changefreq>\n'
    xml += '    <priority>1.0</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/chat</loc>\n'
    xml += f'    <lastmod>{current_date}</lastmod>\n'
    xml += '    <changefreq>daily</changefreq>\n'
    xml += '    <priority>0.8</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/about</loc>\n'
    xml += f'    <lastmod>{current_date}</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.7</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/login</loc>\n'
    xml += f'    <lastmod>{current_date}</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.8</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/register</loc>\n'
    xml += f'    <lastmod>{current_date}</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.8</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/docs</loc>\n'
    xml += f'    <lastmod>{current_date}</lastmod>\n'
    xml += '    <changefreq>weekly</changefreq>\n'
    xml += '    <priority>0.9</priority>\n'
    xml += '  </url>\n'
    xml += '  <url>\n'
    xml += '    <loc>https://mgzon-mgzon-app.hf.space/blog</loc>\n'
    xml += f'    <lastmod>{current_date}</lastmod>\n'
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

# Redirect /gradio to /chat
@app.get("/gradio", response_class=RedirectResponse)
async def launch_chatbot():
    logger.debug("Redirecting /gradio to /chat")
    return RedirectResponse(url="/chat", status_code=302)

# Health check endpoint
@app.get("/health", response_class=PlainTextResponse)
async def health_check():
    logger.debug("Health check endpoint accessed")
    return "OK"

if __name__ == "__main__":
    logger.info(f"Starting uvicorn server on port {os.getenv('PORT', 7860)}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
