# api/auth.py
# SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
# SPDX-License-License: Apache-2.0

from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, JWTStrategy, AuthenticationBackend
from fastapi_users.router.oauth import get_oauth_router
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from fastapi_users.manager import BaseUserManager, IntegerIDMixin
from fastapi import Depends, Request, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi_users.models import UP
from typing import Optional, Dict
import os
import logging
import secrets

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
                if request:
                    request.session["user_id"] = str(user.id)
                    response = RedirectResponse(url="/chat", status_code=302)
                    return response
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
                if request:
                    request.session["user_id"] = str(user.id)
                    response = RedirectResponse(url="/chat", status_code=302)
                    return response
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
        if request:
            request.session["user_id"] = str(user.id)
            response = RedirectResponse(url="/chat", status_code=302)
            return response
        return user

async def get_user_manager(user_db: CustomSQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

google_oauth_router = get_oauth_router(
    google_oauth_client,
    auth_backend,
    get_user_manager,
    state_secret=SECRET,
    associate_by_email=True,
    redirect_url=GOOGLE_REDIRECT_URL,
)

github_oauth_client._access_token_url = "https://github.com/login/oauth/access_token"
github_oauth_client._access_token_params = {"headers": {"Accept": "application/json"}}
github_oauth_router = get_oauth_router(
    github_oauth_client,
    auth_backend,
    get_user_manager,
    state_secret=SECRET,
    associate_by_email=True,
    redirect_url=GITHUB_REDIRECT_URL,
)

fastapi_users = FastAPIUsers[User, int](
    get_user_db,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True, optional=True)

def get_auth_router(app: FastAPI):
    app.include_router(google_oauth_router, prefix="/auth/google", tags=["auth"])
    app.include_router(github_oauth_router, prefix="/auth/github", tags=["auth"])
    app.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"])
    app.include_router(fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"])
    app.include_router(fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"])