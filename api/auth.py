from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, JWTStrategy, AuthenticationBackend
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from fastapi_users.router.oauth import get_oauth_router
from api.database import User, OAuthAccount, get_user_db
from api.models import UserRead, UserCreate, UserUpdate
from fastapi_users.manager import BaseUserManager, IntegerIDMixin
from fastapi import Depends, Request, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi_users.models import UP
from typing import Optional, Dict, Any
import os
import logging
import secrets

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

# OAuth redirect URLs
GOOGLE_REDIRECT_URL = os.getenv("GOOGLE_REDIRECT_URL", "https://mgzon-mgzon-app.hf.space/auth/google/callback")
GITHUB_REDIRECT_URL = os.getenv("GITHUB_REDIRECT_URL", "https://mgzon-mgzon-app.hf.space/auth/github/callback")

google_oauth_client = GoogleOAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
github_oauth_client = GitHubOAuth2(GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET)

class CustomSQLAlchemyUserDatabase(SQLAlchemyUserDatabase):
    async def get_by_email(self, email: str) -> Optional[User]:
        """Override to fix ChunkedIteratorResult issue for get_by_email"""
        logger.info(f"Checking for user with email: {email}")
        try:
            statement = select(self.user_table).where(self.user_table.email == email)
            result = await self.session.execute(statement)
            user = result.scalar_one_or_none()
            if user:
                logger.info(f"Found user with email: {email}")
            else:
                logger.info(f"No user found with email: {email}")
            return user
        except Exception as e:
            logger.error(f"Error in get_by_email: {e}")
            raise

    async def create(self, create_dict: Dict[str, Any]) -> User:
        """Override to fix potential async issues in create"""
        logger.info(f"Creating user with email: {create_dict.get('email')}")
        try:
            user = self.user_table(**create_dict)
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            logger.info(f"Created user with email: {create_dict.get('email')}")
            return user
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            await self.session.rollback()
            raise

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def get_by_oauth_account(self, oauth_name: str, account_id: str):
        """Override to fix ChunkedIteratorResult issue in SQLAlchemy 2.0+"""
        logger.info(f"Checking for existing OAuth account: {oauth_name}/{account_id}")
        try:
            statement = select(OAuthAccount).where(
                OAuthAccount.oauth_name == oauth_name, OAuthAccount.account_id == account_id
            )
            result = await self.session.execute(statement)
            oauth_account = result.scalar_one_or_none()
            if oauth_account:
                logger.info(f"Found existing OAuth account for {account_id}")
            else:
                logger.info(f"No existing OAuth account found for {account_id}")
            return oauth_account
        except Exception as e:
            logger.error(f"Error in get_by_oauth_account: {e}")
            raise

    async def add_oauth_account(self, oauth_account: OAuthAccount):
        """Override to fix potential async issues"""
        logger.info(f"Adding OAuth account for user {oauth_account.user_id}")
        try:
            self.session.add(oauth_account)
            await self.session.commit()
            await self.session.refresh(oauth_account)
            logger.info(f"Successfully added OAuth account for user {oauth_account.user_id}")
        except Exception as e:
            logger.error(f"Error adding OAuth account: {e}")
            await self.session.rollback()
            raise

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
        logger.info(f"OAuth callback for {oauth_name} with account_id {account_id}")
        oauth_account_dict = {
            "oauth_name": oauth_name,
            "access_token": access_token,
            "account_id": account_id,
            "account_email": account_email,
            "expires_at": expires_at,
            "refresh_token": refresh_token,
        }
        oauth_account = OAuthAccount(**oauth_account_dict)
        existing_oauth_account = await self.get_by_oauth_account(oauth_name, account_id)
        if existing_oauth_account is not None:
            logger.info(f"Existing account found, logging in user {existing_oauth_account.user.email}")
            return await self.on_after_login(existing_oauth_account.user, request)

        if associate_by_email:
            user = await self.user_db.get_by_email(account_email)
            if user is not None:
                oauth_account.user_id = user.id
                await self.add_oauth_account(oauth_account)
                logger.info(f"Associated with existing user {user.email}")
                return await self.on_after_login(user, request)

        user_dict = {
            "email": account_email,
            "hashed_password": self.password_helper.hash(secrets.token_hex(32)),
            "is_active": True,
            "is_verified": is_verified_by_default,
        }
        user = await self.user_db.create(user_dict)
        oauth_account.user_id = user.id
        await self.add_oauth_account(oauth_account)
        logger.info(f"Created new user {user.email}")
        return await self.on_after_login(user, request)

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

google_oauth_router = get_oauth_router(
    google_oauth_client,
    auth_backend,
    get_user_manager,
    state_secret=SECRET,
    associate_by_email=True,
    redirect_url=GOOGLE_REDIRECT_URL,
)
logger.info("Google OAuth router initialized successfully")

github_oauth_router = get_oauth_router(
    github_oauth_client,
    auth_backend,
    get_user_manager,
    state_secret=SECRET,
    associate_by_email=True,
    redirect_url=GITHUB_REDIRECT_URL,
)
logger.info("GitHub OAuth router initialized successfully")

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
