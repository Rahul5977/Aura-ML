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
            "password_hash": hashed_password,
            "full_name": user_data.full_name,
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
    if not verify_password(password, user.password_hash):
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
async def create_message(conversation_id: str, content: str, role: str, sender_id: Optional[str] = None):
    """Create a new message in a conversation."""
    message = await prisma.message.create(
        data={
            "conversation_id": conversation_id,
            "content": content,
            "role": role,
            "sender_id": sender_id,
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

async def update_user_profile(user_id: str, user_update):
    """Update user profile information."""
    # Build update data dynamically based on provided fields
    update_data = {}
    if hasattr(user_update, 'email') and user_update.email:
        update_data['email'] = user_update.email
    if hasattr(user_update, 'full_name') and user_update.full_name:
        update_data['full_name'] = user_update.full_name
    
    if not update_data:
        # No fields to update, return current user
        return await get_user_by_id(user_id)
    
    updated_user = await prisma.user.update(
        where={'id': user_id},
        data=update_data
    )
    return updated_user

async def update_user_password(user_id: str, new_password: str):
    """Update user password with new hashed password."""
    hashed_password = get_password_hash(new_password)
    
    await prisma.user.update(
        where={'id': user_id},
        data={'password_hash': hashed_password}
    )
    return True
