from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, JWTStrategy, AuthenticationBackend
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from api.database import User, OAuthAccount, get_user_db, get_db
from fastapi_users.manager import BaseUserManager, IntegerIDMixin
from fastapi import Depends, Request, FastAPI, HTTPException, status, APIRouter
from fastapi.responses import RedirectResponse
from fastapi_users.models import UP
from typing import Optional
import os
import logging
from api.database import User, OAuthAccount
from api.models import UserRead, UserCreate, UserUpdate
from sqlalchemy import select
from sqlalchemy.orm import Session
from datetime import datetime
import secrets
import json
import httpx

# Setup logging
logger = logging.getLogger(__name__)

# Cookie transport for JWT
cookie_transport = CookieTransport(cookie_max_age=3600)

# JWT Secret
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
logger.info("GITHUB_CLIENT_ID is set: %s", bool(GITHUB_CLIENT_ID))

if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET]):
    logger.error("One or more OAuth environment variables are missing.")
    raise ValueError("All OAuth credentials are required.")

# Create OAuth clients
google_oauth_client = GoogleOAuth2(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET
)

github_oauth_client = GitHubOAuth2(
    client_id=GITHUB_CLIENT_ID,
    client_secret=GITHUB_CLIENT_SECRET
)

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    # Standard user manager methods (for JWT only)
    pass

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

# Custom OAuth Router - SYNC VERSION
oauth_router = APIRouter(prefix="/auth", tags=["auth"])

@oauth_router.get("/google/authorize")
async def google_authorize(request: Request):
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    request.session["oauth_provider"] = "google"
    
    # SYNC call to get_authorization_url
    redirect_uri = "https://mgzon-mgzon-app.hf.space/auth/google/callback"
    url = google_oauth_client.get_authorization_url(
        redirect_uri, state=state, scope=["openid", "email", "profile"]
    )
    logger.info(f"Redirecting to Google OAuth: {url}")
    return RedirectResponse(str(url))

@oauth_router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    
    # Verify state
    if not state or state != request.session.get("oauth_state"):
        logger.error("OAuth state mismatch")
        return RedirectResponse("/login?error=Invalid state")
    
    provider = request.session.get("oauth_provider")
    if provider != "google":
        return RedirectResponse("/login?error=Invalid provider")
    
    try:
        # SYNC call to get_access_token
        redirect_uri = "https://mgzon-mgzon-app.hf.space/auth/google/callback"
        token = google_oauth_client.get_access_token(code, redirect_uri=redirect_uri)
        
        # SYNC call to get_id_email
        user_info = google_oauth_client.get_id_email(token)
        
        account_id = user_info["id"]
        account_email = user_info["email"]
        logger.info(f"Google OAuth success: account_id={account_id}, email={account_email}")
        
        # Find or create user - SYNC
        user = db.query(User).filter(User.email == account_email).first()
        if user is None:
            # Create new user
            user = User(
                email=account_email,
                hashed_password="dummy_hashed_password",  # We'll use JWT only, no password login
                is_active=True,
                is_verified=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new user: {user.email}")
        
        # Find or create OAuth account - SYNC
        oauth_account = db.query(OAuthAccount).filter(
            OAuthAccount.oauth_name == "google",
            OAuthAccount.account_id == account_id
        ).first()
        
        if oauth_account is None:
            oauth_account = OAuthAccount(
                oauth_name="google",
                access_token=token["access_token"],
                account_id=account_id,
                account_email=account_email,
                user_id=user.id
            )
            db.add(oauth_account)
            db.commit()
            logger.info(f"Created OAuth account for user: {user.email}")
        else:
            # Update existing OAuth account
            oauth_account.access_token = token["access_token"]
            oauth_account.account_email = account_email
            db.commit()
            logger.info(f"Updated OAuth account for user: {user.email}")
        
        # Create JWT token using fastapi_users - ASYNC
        user_db = get_user_db(db)
        user_manager = UserManager(user_db)
        jwt_token = await user_manager.create_access_token(user)
        
        # Set cookie
        response = RedirectResponse("/chat")
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            max_age=3600,
            httponly=True,
            secure=True,
            samesite="lax"
        )
        
        # Clear session
        request.session.clear()
        
        logger.info(f"OAuth login successful for user: {user.email}")
        return response
        
    except Exception as e:
        logger.error(f"Google OAuth callback failed: {e}")
        request.session.clear()
        return RedirectResponse(f"/login?error={str(e)}")

@oauth_router.get("/github/authorize")
async def github_authorize(request: Request):
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    request.session["oauth_provider"] = "github"
    
    # SYNC call to get_authorization_url
    redirect_uri = "https://mgzon-mgzon-app.hf.space/auth/github/callback"
    url = github_oauth_client.get_authorization_url(
        redirect_uri, state=state, scope=["user:email"]
    )
    logger.info(f"Redirecting to GitHub OAuth: {url}")
    return RedirectResponse(str(url))

@oauth_router.get("/github/callback")
async def github_callback(request: Request, db: Session = Depends(get_db)):
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    
    # Verify state
    if not state or state != request.session.get("oauth_state"):
        logger.error("OAuth state mismatch")
        return RedirectResponse("/login?error=Invalid state")
    
    provider = request.session.get("oauth_provider")
    if provider != "github":
        return RedirectResponse("/login?error=Invalid provider")
    
    try:
        # SYNC call to get_access_token
        redirect_uri = "https://mgzon-mgzon-app.hf.space/auth/github/callback"
        token = github_oauth_client.get_access_token(code, redirect_uri=redirect_uri)
        
        # SYNC call to get_id_email
        user_info = github_oauth_client.get_id_email(token)
        
        account_id = user_info["id"]
        account_email = user_info["email"]
        logger.info(f"GitHub OAuth success: account_id={account_id}, email={account_email}")
        
        # Find or create user - SYNC
        user = db.query(User).filter(User.email == account_email).first()
        if user is None:
            # Create new user
            user = User(
                email=account_email,
                hashed_password="dummy_hashed_password",  # We'll use JWT only, no password login
                is_active=True,
                is_verified=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new user: {user.email}")
        
        # Find or create OAuth account - SYNC
        oauth_account = db.query(OAuthAccount).filter(
            OAuthAccount.oauth_name == "github",
            OAuthAccount.account_id == account_id
        ).first()
        
        if oauth_account is None:
            oauth_account = OAuthAccount(
                oauth_name="github",
                access_token=token["access_token"],
                account_id=account_id,
                account_email=account_email,
                user_id=user.id
            )
            db.add(oauth_account)
            db.commit()
            logger.info(f"Created OAuth account for user: {user.email}")
        else:
            # Update existing OAuth account
            oauth_account.access_token = token["access_token"]
            oauth_account.account_email = account_email
            db.commit()
            logger.info(f"Updated OAuth account for user: {user.email}")
        
        # Create JWT token using fastapi_users - ASYNC
        user_db = get_user_db(db)
        user_manager = UserManager(user_db)
        jwt_token = await user_manager.create_access_token(user)
        
        # Set cookie
        response = RedirectResponse("/chat")
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            max_age=3600,
            httponly=True,
            secure=True,
            samesite="lax"
        )
        
        # Clear session
        request.session.clear()
        
        logger.info(f"OAuth login successful for user: {user.email}")
        return response
        
    except Exception as e:
        logger.error(f"GitHub OAuth callback failed: {e}")
        request.session.clear()
        return RedirectResponse(f"/login?error={str(e)}")

@oauth_router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    response = RedirectResponse("/login")
    response.delete_cookie("access_token")
    return response

# Standard fastapi_users setup for JWT only (no OAuth)
fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True, optional=True)

def get_auth_router(app: FastAPI):
    # Add custom OAuth router
    app.include_router(oauth_router)
    
    # Add standard fastapi_users routes (without OAuth)
    app.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"])
    app.include_router(fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"])
