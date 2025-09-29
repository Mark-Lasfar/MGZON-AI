# SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
# SPDX-License-Identifier: Apache-2.0

from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, JWTStrategy, AuthenticationBackend
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from fastapi_users.manager import BaseUserManager, IntegerIDMixin
from fastapi import Depends, Request, FastAPI, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi_users.models import UP
from typing import Optional
import os
import logging
import secrets
import httpx
from datetime import datetime
import jwt  # إضافة مكتبة pyjwt لتوليد الـ token يدويًا

from api.database import User, OAuthAccount, CustomSQLAlchemyUserDatabase, get_user_db
from api.models import UserRead, UserCreate, UserUpdate

# إعداد اللوقينج
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

# OAuth بيانات الاعتماد
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# تحقق من توافر بيانات الاعتماد
if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET]):
    logger.error("One or more OAuth environment variables are missing.")
    raise ValueError("All OAuth credentials are required.")

GOOGLE_REDIRECT_URL = os.getenv("GOOGLE_REDIRECT_URL", "https://mgzon-mgzon-app.hf.space/auth/google/callback")
GITHUB_REDIRECT_URL = os.getenv("GITHUB_REDIRECT_URL", "https://mgzon-mgzon-app.hf.space/auth/github/callback")

google_oauth_client = GoogleOAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
github_oauth_client = GitHubOAuth2(GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET)
github_oauth_client._access_token_url = "https://github.com/login/oauth/access_token"
github_oauth_client._access_token_params = {"headers": {"Accept": "application/json"}}

# مدير المستخدمين
class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def get_by_oauth_account(self, oauth_name: str, account_id: str):
        logger.info(f"Checking OAuth account: {oauth_name}/{account_id}")
        statement = select(OAuthAccount).where(
            OAuthAccount.oauth_name == oauth_name,
            OAuthAccount.account_id == account_id
        )
        result = await self.user_db.session.execute(statement)
        return result.scalar_one_or_none()

    async def add_oauth_account(self, oauth_account: OAuthAccount):
        logger.info(f"Adding OAuth account for user {oauth_account.user_id}")
        self.user_db.session.add(oauth_account)
        await self.user_db.session.commit()
        await self.user_db.session.refresh(oauth_account)

    async def oauth_callback(
        self,
        oauth_name: str,
        access_token: str,
        account_id: str,
        account_email: str,
        expires_at: Optional[int] = None,
        refresh_token: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        associate_by_email: bool = False,
        is_verified_by_default: bool = False,
    ) -> UP:
        logger.info(f"OAuth callback for {oauth_name} account {account_id}")

        oauth_account = OAuthAccount(
            oauth_name=oauth_name,
            access_token=access_token,
            account_id=account_id,
            account_email=account_email,
            expires_at=expires_at,
            refresh_token=refresh_token,
        )

        existing_oauth_account = await self.get_by_oauth_account(oauth_name, account_id)
        if existing_oauth_account:
            logger.info(f"Fetching user for OAuth account with user_id: {existing_oauth_account.user_id}")
            statement = select(User).where(User.id == existing_oauth_account.user_id)
            result = await self.user_db.session.execute(statement)
            user = result.scalar_one_or_none()

            if user:
                logger.info(f"User found: {user.email}, proceeding with on_after_login")
                await self.on_after_login(user, request)
                return user
            else:
                logger.error(f"No user found for OAuth account with user_id: {existing_oauth_account.user_id}")
                raise ValueError("User not found for existing OAuth account")

        if associate_by_email:
            logger.info(f"Associating OAuth account by email: {account_email}")
            user = await self.user_db.get_by_email(account_email)
            if user:
                oauth_account.user_id = user.id
                await self.add_oauth_account(oauth_account)
                logger.info(f"User associated: {user.email}, proceeding with on_after_login")
                await self.on_after_login(user, request)
                return user

        logger.info(f"Creating new user for email: {account_email}")
        user_dict = {
            "email": account_email,
            "hashed_password": self.password_helper.hash(secrets.token_hex(32)),
            "is_active": True,
            "is_verified": is_verified_by_default,
        }

        user = await self.user_db.create(user_dict)
        oauth_account.user_id = user.id
        await self.add_oauth_account(oauth_account)
        logger.info(f"New user created: {user.email}, proceeding with on_after_login")
        await self.on_after_login(user, request)
        return user

# استدعاء user manager من get_user_db
async def get_user_manager(user_db: CustomSQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

fastapi_users = FastAPIUsers[User, int](
    get_user_db,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True, optional=True)

# دالة مساعدة لتوليد JWT token يدويًا
async def generate_jwt_token(user: User, secret: str, lifetime_seconds: int) -> str:
    payload = {
        "sub": str(user.id),
        "aud": "fastapi-users:auth",
        "exp": int(datetime.utcnow().timestamp()) + lifetime_seconds,
    }
    return jwt.encode(payload, secret, algorithm="HS256")

# --- إضافة custom authorize endpoints (يرجع JSON مع authorization_url) ---

async def custom_google_authorize(
    state: Optional[str] = None,
    oauth_client=Depends(lambda: google_oauth_client),
):
    logger.debug("Generating Google authorization URL")
    state_data = secrets.token_urlsafe(32) if state is None else state
    authorization_url = await oauth_client.get_authorization_url(
        redirect_uri=GOOGLE_REDIRECT_URL,
        state=state_data,
    )
    return JSONResponse(content={
        "authorization_url": authorization_url
    }, status_code=200)

async def custom_github_authorize(
    state: Optional[str] = None,
    oauth_client=Depends(lambda: github_oauth_client),
):
    logger.debug("Generating GitHub authorization URL")
    state_data = secrets.token_urlsafe(32) if state is None else state
    authorization_url = await oauth_client.get_authorization_url(
        redirect_uri=GITHUB_REDIRECT_URL,
        state=state_data,
    )
    return JSONResponse(content={
        "authorization_url": authorization_url
    }, status_code=200)

# --- Custom Callback endpoints ---

async def custom_oauth_callback(
    code: str,
    state: Optional[str] = None,
    user_manager: UserManager = Depends(get_user_manager),
    oauth_client=Depends(lambda: google_oauth_client),
    redirect_url: str = GOOGLE_REDIRECT_URL,
    response: Response = None,
):
    logger.debug(f"Processing Google callback with code: {code}, state: {state}")
    try:
        # تحقق من الـ state لو موجود (اختياري لـ CSRF protection)
        if state:
            logger.debug(f"Received state: {state}")
            # إضافة تحقق من الـ state لو بتستخدمه لمنع CSRF
            # مثال: if state != stored_state: raise ValueError("Invalid state parameter")

        # Get access token
        token_data = await oauth_client.get_access_token(code, redirect_url)
        access_token = token_data["access_token"]
        
        # Manually fetch user info from Google API
        async with httpx.AsyncClient() as client:
            user_info_response = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if user_info_response.status_code != 200:
                raise ValueError(f"Failed to fetch user info: {user_info_response.text}")
            user_info = user_info_response.json()

        user = await user_manager.oauth_callback(
            oauth_name="google",
            access_token=access_token,
            account_id=user_info["sub"],
            account_email=user_info["email"],
            expires_at=token_data.get("expires_in"),
            refresh_token=token_data.get("refresh_token"),
            associate_by_email=True,
            is_verified_by_default=True,
        )

        # توليد الـ JWT token يدويًا
        token = await generate_jwt_token(user, SECRET, 3600)

        # ضبط الـ cookie
        cookie_transport.set_cookie(response, token)

        return JSONResponse(content={
            "message": "Google login successful",
            "access_token": token
        }, status_code=200)
    except Exception as e:
        logger.error(f"Error in Google OAuth callback: {str(e)}")
        if "400" in str(e):
            return JSONResponse(
                content={"detail": "Invalid authorization code. Please try logging in again."},
                status_code=400
            )
        return JSONResponse(content={"detail": str(e)}, status_code=400)

async def custom_github_oauth_callback(
    code: str,
    state: Optional[str] = None,
    user_manager: UserManager = Depends(get_user_manager),
    oauth_client=Depends(lambda: github_oauth_client),
    redirect_url: str = GITHUB_REDIRECT_URL,
    response: Response = None,
):
    logger.debug(f"Processing GitHub callback with code: {code}, state: {state}")
    try:
        # تحقق من الـ state لو موجود (اختياري لـ CSRF protection)
        if state:
            logger.debug(f"Received state: {state}")
            # إضافة تحقق من الـ state لو بتستخدمه لمنع CSRF
            # مثال: if state != stored_state: raise ValueError("Invalid state parameter")

        # Get access token
        token_data = await oauth_client.get_access_token(code, redirect_url)
        access_token = token_data["access_token"]
        
        # Manually fetch user info from GitHub API
        async with httpx.AsyncClient() as client:
            user_info_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            if user_info_response.status_code != 200:
                raise ValueError(f"Failed to fetch user info: {user_info_response.text}")
            user_info = user_info_response.json()

            # Get email if not in user info
            email = user_info.get("email")
            if not email:
                email_response = await client.get(
                    "https://api.github.com/user/emails",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
                )
                if email_response.status_code == 200:
                    emails = email_response.json()
                    primary_email = next((e["email"] for e in emails if e["primary"] and e["verified"]), None)
                    email = primary_email or f"{user_info['login']}@github.com"

        user = await user_manager.oauth_callback(
            oauth_name="github",
            access_token=access_token,
            account_id=str(user_info["id"]),
            account_email=email,
            expires_at=token_data.get("expires_in"),
            refresh_token=token_data.get("refresh_token"),
            associate_by_email=True,
            is_verified_by_default=True,
        )

        # توليد الـ JWT token يدويًا
        token = await generate_jwt_token(user, SECRET, 3600)

        # ضبط الـ cookie
        cookie_transport.set_cookie(response, token)

        return JSONResponse(content={
            "message": "GitHub login successful",
            "access_token": token
        }, status_code=200)
    except Exception as e:
        logger.error(f"Error in GitHub OAuth callback: {str(e)}")
        if "400" in str(e):
            return JSONResponse(
                content={"detail": "Invalid authorization code. Please try logging in again."},
                status_code=400
            )
        return JSONResponse(content={"detail": str(e)}, status_code=400)

# تضمين الراوترات داخل التطبيق
def get_auth_router(app: FastAPI):
    app.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"])
    app.include_router(fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"])

    # إضافة الـ custom OAuth endpoints يدويًا
    app.get("/auth/google/authorize")(custom_google_authorize)
    app.get("/auth/google/callback")(custom_oauth_callback)
    app.get("/auth/github/authorize")(custom_github_authorize)
    app.get("/auth/github/callback")(custom_github_oauth_callback)
