from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional, List

class ConversationCreate(BaseModel):
    title: Optional[str] = None
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError('Title cannot be empty')
            if len(v) > 200:
                raise ValueError('Title must be less than 200 characters')
            return v.strip()
        return v

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError('Title cannot be empty')
            if len(v) > 200:
                raise ValueError('Title must be less than 200 characters')
            return v.strip()
        return v

class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0
    
    class Config:
        from_attributes = True

class ConversationWithMessages(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List['MessageResponse'] = []
    
    class Config:
        from_attributes = True

# Forward reference for messages
from .schemas import MessageResponse
ConversationWithMessages.model_rebuild()
