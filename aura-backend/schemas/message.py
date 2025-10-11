from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageCreate(BaseModel):
    content: str
    role: MessageRole = MessageRole.USER
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Message content cannot be empty')
        if len(v) > 10000:
            raise ValueError('Message content must be less than 10,000 characters')
        return v.strip()

class MessageUpdate(BaseModel):
    content: str
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Message content cannot be empty')
        if len(v) > 10000:
            raise ValueError('Message content must be less than 10,000 characters')
        return v.strip()

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    content: str
    role: MessageRole
    sender_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    edited_at: Optional[datetime] = None
    read_status: bool = False
    
    class Config:
        from_attributes = True
        use_enum_values = True

class MessageWithSender(BaseModel):
    id: str
    conversation_id: str
    content: str
    role: MessageRole
    sender_id: Optional[str] = None
    sender_username: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    edited_at: Optional[datetime] = None
    read_status: bool = False
    
    class Config:
        from_attributes = True
        use_enum_values = True
