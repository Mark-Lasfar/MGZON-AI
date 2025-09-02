# api/auth.py
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, JWTStrategy, AuthenticationBackend
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from api.database import SessionLocal
from api.models import User, OAuthAccount
import os

# إعداد Cookie Transport
cookie_transport = CookieTransport(cookie_max_age=3600)

# إعداد JWT Strategy
SECRET = os.getenv("JWT_SECRET", "your_jwt_secret_here")  # ضعه في Space Secrets

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# إعداد عملاء OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

google_oauth_client = GoogleOAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
github_oauth_client = GitHubOAuth2(GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET)

# إعداد FastAPIUsers
fastapi_users = FastAPIUsers[User, int](
    lambda: SQLAlchemyUserDatabase(User, SessionLocal(), oauth_account_table=OAuthAccount),
    [auth_backend],
)

# Dependency للحصول على المستخدم الحالي (اختياري)
current_active_user = fastapi_users.current_user(active=True, optional=True)  # ← تغيير هنا
