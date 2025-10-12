from pydantic import BaseModel, EmailStr, validator
from datetime import datetime
from typing import Optional
import re

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str
    full_name: str
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username must be less than 50 characters')
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Username can only contain letters, numbers, dots, dashes, and underscores')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 72:
            raise ValueError('Password must be less than 72 characters')
        return v
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Full name must be at least 2 characters long')
        if len(v) > 100:
            raise ValueError('Full name must be less than 100 characters')
        return v.strip()

class UserResponse(UserBase):
    id: str
    full_name: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if v is not None:
            if len(v.strip()) < 2:
                raise ValueError('Full name must be at least 2 characters long')
            if len(v) > 100:
                raise ValueError('Full name must be less than 100 characters')
            return v.strip()
        return v

class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 72:
            raise ValueError('Password must be less than 72 characters')
        return v

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

# Conversation schemas
class ConversationBase(BaseModel):
    title: Optional[str] = None

class ConversationCreate(ConversationBase):
    pass

class ConversationResponse(ConversationBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Message schemas
class MessageBase(BaseModel):
    content: str
    role: str  # "user" or "assistant"

class MessageCreate(MessageBase):
    pass  # conversation_id comes from URL path

class MessageResponse(MessageBase):
    id: str
    conversation_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True
