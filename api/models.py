# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi_users import schemas
from datetime import datetime

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

    model_config = {"from_attributes": True}

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

    model_config = {"from_attributes": True}

# Pydantic schema for updating user settings
class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    preferred_model: Optional[str] = None
    job_title: Optional[str] = None
    education: Optional[str] = None
    interests: Optional[str] = None
    additional_info: Optional[str] = None
    conversation_style: Optional[str] = None

    model_config = {"from_attributes": True}

# Pydantic schemas للمحادثات
class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}

class ConversationCreate(BaseModel):
    title: Optional[str] = None

class ConversationOut(BaseModel):
    id: int
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageOut] = []

    model_config = {"from_attributes": True}

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
