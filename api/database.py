# api/database.py
# SPDX-FileCopyrightText: Hadad <hadad@linuxmail.org>
# SPDX-License-License: Apache-2.0

import os
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any

from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Boolean, Text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
import aiosqlite

# إعداد اللوج
logger = logging.getLogger(__name__)

# استخدم القيمة مباشرة إذا لم يكن هناك متغير بيئة
SQLALCHEMY_DATABASE_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL"
) or "sqlite+aiosqlite:///./data/mgzon_users.db"

# تأكد أن الدرايفر async
if "aiosqlite" not in SQLALCHEMY_DATABASE_URL:
    raise ValueError("Database URL must use 'sqlite+aiosqlite' for async support")

# إنشاء محرك async
async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,
    connect_args={"check_same_thread": False}
)

# إعداد الجلسة async
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession
)

# القاعدة الأساسية للنماذج
Base = declarative_base()

# النماذج (Models)
class OAuthAccount(Base):
    __tablename__ = "oauth_account"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    oauth_name = Column(String, nullable=False)
    access_token = Column(String, nullable=False)
    expires_at = Column(Integer, nullable=True)
    refresh_token = Column(String, nullable=True)
    account_id = Column(String, index=True, nullable=False)
    account_email = Column(String, nullable=False)

    user = relationship("User", back_populates="oauth_accounts", lazy="selectin")

class User(SQLAlchemyBaseUserTable[int], Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    display_name = Column(String, nullable=True)
    preferred_model = Column(String, nullable=True)
    job_title = Column(String, nullable=True)
    education = Column(String, nullable=True)
    interests = Column(String, nullable=True)
    additional_info = Column(Text, nullable=True)
    conversation_style = Column(String, nullable=True)

    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversation"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "message"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversation.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

# قاعدة بيانات المستخدم المخصصة (نقلناها من user_db.py)
class CustomSQLAlchemyUserDatabase(SQLAlchemyUserDatabase[User, int]):
    """
    قاعدة بيانات مخصَّصة لمكتبة fastapi-users.
    تضيف طريقة parse_id التي تُحوِّل الـ ID من str → int.
    """
    def parse_id(self, value: Any) -> int:
        logger.debug(f"Parsing user id: {value} (type={type(value)})")
        return int(value) if isinstance(value, str) else value

    async def get_by_email(self, email: str) -> Optional[User]:
        logger.info(f"Looking for user with email: {email}")
        stmt = select(self.user_table).where(self.user_table.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, create_dict: Dict[str, Any]) -> User:
        logger.info(f"Creating new user: {create_dict.get('email')}")
        user = self.user_table(**create_dict)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

# دالة لجلب الجلسة async
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# دالة لجلب قاعدة بيانات المستخدمين لـ fastapi-users
async def get_user_db(session: AsyncSession = Depends(get_db)) -> AsyncGenerator[CustomSQLAlchemyUserDatabase, None]:
    yield CustomSQLAlchemyUserDatabase(session, User, OAuthAccount)

# دالة لإنشاء الجداول
async def init_db():
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise
