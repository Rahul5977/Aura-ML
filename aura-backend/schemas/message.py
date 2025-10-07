from pydantic import BaseModel
from datetime import datetime

class MessageCreate(BaseModel):
    content: str
    role: str

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    content: str
    role: str
    created_at: datetime
