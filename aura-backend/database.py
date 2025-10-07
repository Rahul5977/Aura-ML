from prisma import Prisma
from typing import Optional
from auth import get_password_hash, verify_password
from schemas import UserCreate
import asyncio

prisma = Prisma()

async def connect_db():
    """Connect to the database."""
    await prisma.connect()

async def disconnect_db():
    """Disconnect from the database."""
    await prisma.disconnect()

# User operations
async def create_user(user_data: UserCreate):
    """Create a new user."""
    hashed_password = get_password_hash(user_data.password)
    user = await prisma.user.create(
        data={
            "email": user_data.email,
            "username": user_data.username,
            "hashed_password": hashed_password,
        }
    )
    return user

async def get_user_by_username(username: str):
    """Get user by username."""
    user = await prisma.user.find_unique(
        where={"username": username}
    )
    return user

async def get_user_by_email(email: str):
    """Get user by email."""
    user = await prisma.user.find_unique(
        where={"email": email}
    )
    return user

async def get_user_by_id(user_id: str):
    """Get user by ID."""
    user = await prisma.user.find_unique(
        where={"id": user_id}
    )
    return user

async def authenticate_user(username: str, password: str):
    """Authenticate user with username and password."""
    user = await get_user_by_username(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Conversation operations
async def create_conversation(user_id: str, title: Optional[str] = None):
    """Create a new conversation."""
    conversation = await prisma.conversation.create(
        data={
            "user_id": user_id,
            "title": title,
        }
    )
    return conversation

async def get_user_conversations(user_id: str):
    """Get all conversations for a user."""
    conversations = await prisma.conversation.find_many(
        where={"user_id": user_id},
        order={"created_at": "desc"}
    )
    return conversations

async def get_conversation_by_id(conversation_id: str, user_id: str):
    """Get conversation by ID if it belongs to the user."""
    conversation = await prisma.conversation.find_first(
        where={
            "id": conversation_id,
            "user_id": user_id
        },
        include={"messages": True}
    )
    return conversation

# Message operations
async def create_message(conversation_id: str, content: str, role: str):
    """Create a new message in a conversation."""
    message = await prisma.message.create(
        data={
            "conversation_id": conversation_id,
            "content": content,
            "role": role,
        }
    )
    return message

async def get_conversation_messages(conversation_id: str):
    """Get all messages for a conversation."""
    messages = await prisma.message.find_many(
        where={"conversation_id": conversation_id},
        order={"created_at": "asc"}
    )
    return messages
