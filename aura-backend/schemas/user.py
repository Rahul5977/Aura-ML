from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    username: str
    is_active: bool
    created_at: datetime

class UserLogin(BaseModel):
    username: str
    password: str

