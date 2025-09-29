# api/database.py
# SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
# SPDX-License-License: Apache-2.0

import os
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import Document, init_beanie
from fastapi_users_db_mongobeanie import MongoBeanieUserDatabase
from fastapi_users import models as fu_models
from fastapi_users_db_mongobeanie.oauth import BeanieBaseOAuthAccount

# إعداد اللوج
logger = logging.getLogger(__name__)

# جلب MONGO_URI من متغيرات البيئة
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    logger.error("MONGODB_URI is not set in environment variables.")
    raise ValueError("MONGODB_URI is required for MongoDB.")

# إعداد MongoDB
client = AsyncIOMotorClient(MONGO_URI)
mongo_db = client["hager"]

# تعريف نموذج OAuthAccount
class OAuthAccount(BeanieBaseOAuthAccount, Document):
    class Settings:
        name = "oauth_account"

# تعريف نموذج المستخدم
class User(fu_models.UP, fu_models.OAP, Document):
    class Settings:
        name = "user"

    id: fu_models.ID
    email: str
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False
    display_name: Optional[str] = None
    preferred_model: Optional[str] = None
    job_title: Optional[str] = None
    education: Optional[str] = None
    interests: Optional[str] = None
    additional_info: Optional[str] = None
    conversation_style: Optional[str] = None
    oauth_accounts: list[OAuthAccount] = []

# تعريف نموذج المحادثة
class Conversation(Document):
    class Settings:
        name = "conversation"

    conversation_id: str
    user_id: fu_models.ID
    title: str
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()

# تعريف نموذج الرسالة
class Message(Document):
    class Settings:
        name = "message"

    conversation_id: str
    role: str
    content: str
    created_at: datetime = datetime.utcnow()

# دالة لتهيئة قاعدة البيانات
async def init_db():
    try:
        await init_beanie(database=mongo_db, document_models=[User, OAuthAccount, Conversation, Message])
        logger.info("MongoDB initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {e}")
        raise

# دالة لجلب قاعدة بيانات المستخدمين
async def get_user_db() -> AsyncGenerator[MongoBeanieUserDatabase, None]:
    yield MongoBeanieUserDatabase(User, OAuthAccount)
