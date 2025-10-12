from pydantic import BaseModel
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: Optional[int] = None  # Token expiration time in seconds

class TokenData(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None

class RefreshToken(BaseModel):
    refresh_token: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int  # Token expiration time in seconds
