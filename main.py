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
from api.auth import fastapi_users, auth_backend, current_active_user, google_oauth_client, github_oauth_client
from api.database import get_db, engine, Base
from api.models import User, UserRead, UserCreate, Conversation
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List
import uvicorn
import markdown2
from sqlalchemy.orm import Session
from pathlib import Path
from hashlib import md5
from datetime import datetime
import re
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.exceptions import GetIdEmailError

# Setup logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Files in current dir: %s", os.listdir(os.getcwd()))

# Check environment variables for required configurations
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    logger.error("MONGODB_URI is not set in environment variables.")
    raise ValueError("MONGODB_URI is required for MongoDB.")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    logger.error("JWT_SECRET is not set or too short.")
    raise ValueError("JWT_SECRET is required (at least 32 characters).")

# MongoDB setup for blog posts and session message counts
client = AsyncIOMotorClient(MONGO_URI)
mongo_db = client["hager"]
session_message_counts = mongo_db["session_message_counts"]

# Create MongoDB index for session_id to ensure uniqueness
async def setup_mongo_index():
    await session_message_counts.create_index("session_id", unique=True)

# Jinja2 setup with Markdown filter for rendering Markdown content
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

# Application settings from environment variables
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# Initialize FastAPI app
app = FastAPI(title="MGZon Chatbot API")

# Add SessionMiddleware for handling non-logged-in user sessions
app.add_middleware(SessionMiddleware, secret_key=JWT_SECRET)

# Create SQLAlchemy database tables
Base.metadata.create_all(bind=engine)

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS setup to allow requests from specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Kept as wildcard for multiple projects as per request
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for authentication, user management, and API endpoints
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
    fastapi_users.get_oauth_router(google_oauth_client, auth_backend, JWT_SECRET, redirect_url="https://mgzon-mgzon-app.hf.space/auth/google/callback"),
    prefix="/auth/google",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_oauth_router(github_oauth_client, auth_backend, JWT_SECRET, redirect_url="https://mgzon-mgzon-app.hf.space/auth/github/callback"),
    prefix="/auth/github",
    tags=["auth"],
)
app.include_router(api_router)

# Custom middleware for handling 404 and 500 errors
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

# Exception handler for OAuth errors
@app.exception_handler(GetIdEmailError)
async def handle_oauth_error(request: Request, exc: GetIdEmailError):
    logger.error(f"OAuth error: {exc}")
    return RedirectResponse(url="/login?error=oauth_failed", status_code=302)

# Custom Google OAuth callback to redirect to /chat
@app.get("/auth/google/callback")
async def google_oauth_callback(request: Request, user=Depends(fastapi_users.get_oauth_callback(auth_backend))):
    if user:
        return RedirectResponse(url="/chat", status_code=302)
    else:
        return RedirectResponse(url="/login?error=oauth_failed", status_code=302)

# Manual OAuth authorize endpoints (to ensure they work even if router fails)
@app.get("/auth/google/authorize")
async def google_authorize():
    redirect_uri = "https://mgzon-mgzon-app.hf.space/auth/google/callback"
    authorization_url = await google_oauth_client.get_authorization_url(
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"],
    )
    return RedirectResponse(authorization_url)

@app.get("/auth/github/authorize")
async def github_authorize():
    redirect_uri = "https://mgzon-mgzon-app.hf.space/auth/github/callback"
    authorization_url = await github_oauth_client.get_authorization_url(
        redirect_uri=redirect_uri,
        scope=["user", "user:email"],
    )
    return RedirectResponse(authorization_url)

# Root endpoint for homepage
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user: User = Depends(current_active_user)):
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

# Google verification endpoint
@app.get("/google97468ef1f6b6e804.html", response_class=PlainTextResponse)
async def google_verification():
    return "google-site-verification: google97468ef1f6b6e804.html"

# Login page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User = Depends(current_active_user)):
    if user:
        return RedirectResponse(url="/chat", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})

# Register page
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: User = Depends(current_active_user)):
    if user:
        return RedirectResponse(url="/chat", status_code=302)
    return templates.TemplateResponse("register.html", {"request": request})

# Chat page
@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request, user: User = Depends(current_active_user)):
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

# Specific conversation page
@app.get("/chat/{conversation_id}", response_class=HTMLResponse)
async def chat_conversation(
    request: Request,
    conversation_id: str,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": user,
            "conversation_id": conversation.conversation_id,
            "conversation_title": conversation.title or "Untitled Conversation"
        }
    )

# About page
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, user: User = Depends(current_active_user)):
    return templates.TemplateResponse("about.html", {"request": request, "user": user})

# Serve static files with caching and ETag support
@app.get("/static/{path:path}")
async def serve_static(path: str):
    # Remove query parameters (e.g., ?v=1.0) for versioning
    clean_path = re.sub(r'\?.*', '', path)
    file_path = Path("static") / clean_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # Set cache duration: 1 year for images, 1 hour for JS/CSS
    cache_duration = 31536000  # 1 year
    if clean_path.endswith(('.js', '.css')):
        cache_duration = 3600  # 1 hour
    # Generate ETag and Last-Modified headers
    with open(file_path, "rb") as f:
        file_hash = md5(f.read()).hexdigest()
    headers = {
        "Cache-Control": f"public, max-age={cache_duration}",
        "ETag": file_hash,
        "Last-Modified": datetime.utcfromtimestamp(file_path.stat().st_mtime).strftime('%a, %d %b %Y %H:%M:%S GMT')
    }
    return FileResponse(file_path, headers=headers)

# Blog page with pagination
@app.get("/blog", response_class=HTMLResponse)
async def blog(request: Request, skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    posts = await mongo_db.blog_posts.find().skip(skip).limit(limit).to_list(limit)
    return templates.TemplateResponse("blog.html", {"request": request, "posts": posts})

# Individual blog post
@app.get("/blog/{post_id}", response_class=HTMLResponse)
async def blog_post(request: Request, post_id: str):
    post = await mongo_db.blog_posts.find_one({"id": post_id})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return templates.TemplateResponse("blog_post.html", {"request": request, "post": post})

# Docs page
@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

# Swagger UI for API documentation
@app.get("/swagger", response_class=HTMLResponse)
async def swagger_ui():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MGZon API Documentation")

# Sitemap with dynamic dates
@app.get("/sitemap.xml", response_class=PlainTextResponse)
async def sitemap():
    posts = await mongo_db.blog_posts.find().to_list(100)
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    # Main pages with dynamic lastmod
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
    # Blog posts from MongoDB
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
    return RedirectResponse(url="/chat", status_code=302)

# Startup event to initialize MongoDB index
@app.on_event("startup")
async def startup_event():
    await setup_mongo_index()

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
