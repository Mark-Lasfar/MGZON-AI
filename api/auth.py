from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, JWTStrategy, AuthenticationBackend
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from api.database import SessionLocal
from api.models import User, OAuthAccount
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

cookie_transport = CookieTransport(cookie_max_age=3600)

SECRET = os.getenv("JWT_SECRET")
if not SECRET or len(SECRET) < 32:
    logger.error("JWT_SECRET is not set or too short.")
    raise ValueError("JWT_SECRET is required (at least 32 characters).")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# OAuth credentials
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Log OAuth credentials status
logger.info("GOOGLE_CLIENT_ID is set: %s", bool(GOOGLE_CLIENT_ID))
logger.info("GOOGLE_CLIENT_SECRET is set: %s", bool(GOOGLE_CLIENT_SECRET))
logger.info("GITHUB_CLIENT_ID is set: %s", bool(GITHUB_CLIENT_ID))
logger.info("GITHUB_CLIENT_SECRET is set: %s", bool(GITHUB_CLIENT_SECRET))

if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET]):
    logger.error("One or more OAuth environment variables are missing.")
    raise ValueError("All OAuth credentials (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET) are required.")

google_oauth_client = GoogleOAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
github_oauth_client = GitHubOAuth2(GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET)

fastapi_users = FastAPIUsers[User, int](
    lambda: SQLAlchemyUserDatabase(User, SessionLocal(), oauth_account_table=OAuthAccount),
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True, optional=True)
