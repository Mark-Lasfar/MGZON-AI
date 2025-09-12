from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, JWTStrategy, AuthenticationBackend
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from api.database import User, OAuthAccount, get_user_db
from fastapi_users.manager import BaseUserManager, IntegerIDMixin
from fastapi import Depends, Request, FastAPI
from fastapi_users.models import UP
from typing import Optional
import os
import logging
from api.database import User, OAuthAccount
from api.models import UserRead, UserCreate, UserUpdate
from sqlalchemy import select

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
logger.info("GOOGLE_CLIENT_SECRET is set: %s", bool(GOOGLE_CLIENT_SECRET))
logger.info("GITHUB_CLIENT_ID is set: %s", bool(GITHUB_CLIENT_ID))
logger.info("GITHUB_CLIENT_SECRET is set: %s", bool(GITHUB_CLIENT_SECRET))

if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET]):
    logger.error("One or more OAuth environment variables are missing.")
    raise ValueError("All OAuth credentials (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET) are required.")

google_oauth_client = GoogleOAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
github_oauth_client = GitHubOAuth2(GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET)

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

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
        logger.info(f"Processing OAuth callback for {oauth_name} with account_id: {account_id}, email: {account_email}")
        
        # Validate inputs
        if not account_email or not account_id:
            logger.error(f"Invalid OAuth callback data: email={account_email}, account_id={account_id}")
            raise ValueError("Invalid OAuth callback data: email and account_id are required.")

        oauth_account_dict = {
            "oauth_name": oauth_name,
            "access_token": access_token,
            "account_id": account_id,
            "account_email": account_email,
            "expires_at": expires_at,
            "refresh_token": refresh_token,
        }
        oauth_account = OAuthAccount(**oauth_account_dict)

        # Custom query to fetch OAuth account
        statement = select(OAuthAccount).where(
            (OAuthAccount.oauth_name == oauth_name) & (OAuthAccount.account_id == account_id)
        )
        result = self.user_db.session.execute(statement)
        existing_oauth_account = result.scalars().first()

        if existing_oauth_account is not None:
            logger.info(f"Found existing OAuth account for {oauth_name}, account_id: {account_id}")
            user = existing_oauth_account.user
            if user is None:
                logger.error(f"No user associated with OAuth account {account_id}. Creating new user.")
                # Create new user if user is None
                user_dict = {
                    "email": account_email,
                    "hashed_password": self.password_helper.hash("dummy_password"),
                    "is_active": True,
                    "is_verified": is_verified_by_default,
                }
                user = User(**user_dict)
                try:
                    self.user_db.session.add(user)
                    self.user_db.session.commit()
                    self.user_db.session.refresh(user)
                    existing_oauth_account.user_id = user.id
                    self.user_db.session.commit()
                    logger.info(f"Created new user and linked to existing OAuth account: {user.email}")
                except Exception as e:
                    self.user_db.session.rollback()
                    logger.error(f"Failed to create user for OAuth account {account_id}: {e}")
                    raise ValueError(f"Failed to create user: {e}")
            logger.info(f"Returning existing user: {user.email}")
            return await self.on_after_login(user, request)

        if associate_by_email:
            logger.info(f"Associating by email: {account_email}")
            statement = select(User).where(User.email == account_email)
            result = self.user_db.session.execute(statement)
            user = result.scalars().first()
            if user is None:
                logger.info(f"No user found for email {account_email}. Creating new user.")
                # Create new user if not found
                user_dict = {
                    "email": account_email,
                    "hashed_password": self.password_helper.hash("dummy_password"),
                    "is_active": True,
                    "is_verified": is_verified_by_default,
                }
                user = User(**user_dict)
                try:
                    self.user_db.session.add(user)
                    self.user_db.session.commit()
                    self.user_db.session.refresh(user)
                    logger.info(f"Created new user for email: {user.email}")
                except Exception as e:
                    self.user_db.session.rollback()
                    logger.error(f"Failed to create user for email {account_email}: {e}")
                    raise ValueError(f"Failed to create user: {e}")
            oauth_account.user_id = user.id
            try:
                self.user_db.session.add(oauth_account)
                self.user_db.session.commit()
                logger.info(f"Associated OAuth account with user: {user.email}")
            except Exception as e:
                self.user_db.session.rollback()
                logger.error(f"Failed to associate OAuth account with user {user.email}: {e}")
                raise ValueError(f"Failed to associate OAuth account: {e}")
            return await self.on_after_login(user, request)

        # Create new user
        logger.info(f"Creating new user for email: {account_email}")
        user_dict = {
            "email": account_email,
            "hashed_password": self.password_helper.hash("dummy_password"),
            "is_active": True,
            "is_verified": is_verified_by_default,
        }
        user = User(**user_dict)
        try:
            self.user_db.session.add(user)
            self.user_db.session.commit()
            self.user_db.session.refresh(user)
            logger.info(f"Created new user: {user.email}")
        except Exception as e:
            self.user_db.session.rollback()
            logger.error(f"Failed to create user for email {account_email}: {e}")
            raise ValueError(f"Failed to create user: {e}")

        oauth_account.user_id = user.id
        try:
            self.user_db.session.add(oauth_account)
            self.user_db.session.commit()
            logger.info(f"Linked OAuth account to new user: {user.email}")
        except Exception as e:
            self.user_db.session.rollback()
            logger.error(f"Failed to link OAuth account to user {user.email}: {e}")
            raise ValueError(f"Failed to link OAuth account: {e}")
        return await self.on_after_login(user, request)

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

from fastapi_users.router.oauth import get_oauth_router

google_oauth_router = get_oauth_router(
    google_oauth_client,
    auth_backend,
    get_user_manager,
    state_secret=SECRET,
    associate_by_email=True,
    redirect_url="https://mgzon-mgzon-app.hf.space/auth/google/callback",
)

github_oauth_router = get_oauth_router(
    github_oauth_client,
    auth_backend,
    get_user_manager,
    state_secret=SECRET,
    associate_by_email=True,
    redirect_url="https://mgzon-mgzon-app.hf.space/auth/github/callback",
)

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
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
