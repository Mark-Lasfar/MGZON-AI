# api/models.py
from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyBaseOAuthAccountTable
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from typing import List, Optional

Base = declarative_base()

# جدول OAuth Accounts لتخزين بيانات تسجيل الدخول الخارجي
class OAuthAccount(SQLAlchemyBaseOAuthAccountTable, Base):
    __tablename__ = "oauth_accounts"
    id = Column(Integer, primary_key=True)
    user = relationship("User", back_populates="oauth_accounts")
# نموذج المستخدم
class User(SQLAlchemyBaseUserTable, Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # nullable لأن OAuth ممكن ما يحتاج باسورد
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    oauth_accounts = relationship("OAuthAccount", back_populates="user")

# نموذج طلب الاستعلام (كما هو)
class QueryRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = "You are an expert assistant providing detailed, comprehensive, and well-structured responses."
    history: Optional[List[dict]] = []
    temperature: Optional[float] = 0.7
    max_new_tokens: Optional[int] = 128000
    enable_browsing: Optional[bool] = True
    output_format: Optional[str] = "text"
