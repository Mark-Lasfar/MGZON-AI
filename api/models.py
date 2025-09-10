# models.py
from fastapi_users.db import SQLAlchemyBaseUserTable
from sqlalchemy import Column, Integer, String, Boolean, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi_users import schemas
from datetime import datetime
import uuid

Base = declarative_base()

# جدول OAuth Accounts لتخزين بيانات تسجيل الدخول الخارجي
class OAuthAccount(Base):
    __tablename__ = "oauth_accounts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    oauth_name = Column(String, nullable=False)
    access_token = Column(String, nullable=False)
    expires_at = Column(Integer, nullable=True)
    refresh_token = Column(String, nullable=True)
    account_id = Column(String, index=True, nullable=False)
    account_email = Column(String, nullable=False)
    user = relationship("User", back_populates="oauth_accounts")

# نموذج المستخدم
class User(SQLAlchemyBaseUserTable, Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    display_name = Column(String, nullable=True)  # الاسم المستعار للمستخدم
    preferred_model = Column(String, nullable=True)  # النموذج المفضل (اسم وهمي)
    job_title = Column(String, nullable=True)  # الوظيفة
    education = Column(String, nullable=True)  # التعليم
    interests = Column(Text, nullable=True)  # الاهتمامات
    additional_info = Column(Text, nullable=True)  # معلومات إضافية
    conversation_style = Column(String, nullable=True)  # نمط المحادثة (مثل: موجز، تحليلي)
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

# نموذج المحادثة
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

# نموذج الرسالة
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String)
    content = Column(Text)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

# Pydantic schemas for fastapi-users
class UserRead(schemas.BaseUser[int]):
    id: int
    email: str
    is_active: bool = True
    is_superuser: bool = False
    display_name: Optional[str] = None
    preferred_model: Optional[str] = None
    job_title: Optional[str] = None
    education: Optional[str] = None
    interests: Optional[str] = None
    additional_info: Optional[str] = None
    conversation_style: Optional[str] = None

class UserCreate(schemas.BaseUserCreate):
    email: str
    password: str
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False
    display_name: Optional[str] = None
    preferred_model: Optional[str] = None
    job_title: Optional[str] = None
    education: Optional[str] = None
    interests: Optional[str] = None
    additional_info: Optional[str] = None
    conversation_style: Optional[str] = None

# Pydantic schema for updating user settings
class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    preferred_model: Optional[str] = None
    job_title: Optional[str] = None
    education: Optional[str] = None
    interests: Optional[str] = None
    additional_info: Optional[str] = None
    conversation_style: Optional[str] = None

# Pydantic schemas للمحادثات
class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

class ConversationCreate(BaseModel):
    title: Optional[str] = None

class ConversationOut(BaseModel):
    id: int
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageOut] = []

# نموذج طلب الاستعلام
class QueryRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = "You are an expert assistant providing detailed, comprehensive, and well-structured responses."
    history: Optional[List[dict]] = []
    temperature: Optional[float] = 0.7
    max_new_tokens: Optional[int] = 2048
    enable_browsing: Optional[bool] = True
    output_format: Optional[str] = "text"
    title: Optional[str] = None  

ConversationOut.model_revalidate = True
MessageOut.model_revalidate = True
