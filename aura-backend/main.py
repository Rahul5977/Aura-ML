from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta
import os
from dotenv import load_dotenv

# Import from existing working modules
from auth import create_access_token, verify_token, verify_password, ACCESS_TOKEN_EXPIRE_MINUTES
from database import (
    connect_db, disconnect_db, create_user, authenticate_user, 
    get_user_by_id, get_user_by_username, get_user_by_email,
    create_conversation, get_user_conversations, get_conversation_by_id,
    create_message, get_conversation_messages,
    update_user_profile, update_user_password
)
from schemas import (
    UserCreate, UserResponse, UserLogin, Token,
    ConversationCreate, ConversationResponse,
    MessageCreate, MessageResponse,
    UserUpdate, PasswordChange
)

load_dotenv()

app = FastAPI(title="Aura Backend API", version="1.0.0")
security = HTTPBearer()

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    await connect_db()

@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    token_data = verify_token(token)
    user = await get_user_by_id(token_data["user_id"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user

@app.get("/health")
def health():
    return {"status": "ok"}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    # Check if user already exists
    existing_user = await get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    existing_email = await get_user_by_email(user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user = await create_user(user_data)
    return user

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    user = await authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoints
@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    return current_user

@app.get("/conversations", response_model=list[ConversationResponse])
async def get_conversations(current_user = Depends(get_current_user)):
    conversations = await get_user_conversations(current_user.id)
    return conversations

@app.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_new_conversation(
    conversation_data: ConversationCreate,
    current_user = Depends(get_current_user)
):
    conversation = await create_conversation(current_user.id, conversation_data.title)
    return conversation

@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user = Depends(get_current_user)
):
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation

@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: str,
    current_user = Depends(get_current_user)
):
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = await get_conversation_messages(conversation_id)
    return messages

@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_new_message(
    conversation_id: str,
    message_data: MessageCreate,
    current_user = Depends(get_current_user)
):
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    message = await create_message(
        conversation_id=conversation_id,
        content=message_data.content,
        role=message_data.role
    )
    return message

# Additional auth routes for testing
@app.put("/auth/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user = Depends(get_current_user)
):
    """Update current user's profile information."""
    # Check if email is being updated and if it's already taken
    if hasattr(user_update, 'email') and user_update.email and user_update.email != current_user.email:
        existing_email = await get_user_by_email(user_update.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    try:
        updated_user = await update_user_profile(current_user.id, user_update)
        return updated_user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )

@app.post("/auth/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user = Depends(get_current_user)
):
    """Change user's password."""
    # Verify current password
    if not verify_password(password_change.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    try:
        await update_user_password(current_user.id, password_change.new_password)
        return {"message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )

@app.post("/auth/logout")
async def logout(current_user = Depends(get_current_user)):
    """Logout user (client should discard the JWT token)."""
    return {"message": "Logged out successfully"}
