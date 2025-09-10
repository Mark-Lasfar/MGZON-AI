# api/database.py
import os
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Boolean
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
from sqlalchemy.orm import Session
from typing import AsyncGenerator
from fastapi import Depends
from datetime import datetime

# جلب URL قاعدة البيانات من المتغيرات البيئية
SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    raise ValueValue("SQLALCHEMY_DATABASE_URL is not set in environment variables.")

# إنشاء المحرك
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# إعداد الجلسة
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# قاعدة أساسية للنماذج
Base = declarative_base()

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

    user = relationship("User", back_populates="oauth_accounts")

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
    additional_info = Column(String, nullable=True)
    conversation_style = Column(String, nullable=True)
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversation"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    title = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversation.id"))
    role = Column(String)
    content = Column(String)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=engine)

async def get_user_db(session: Session = Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User, OAuthAccount)
