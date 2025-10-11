from prisma import Prisma
from db.prisma import User, Conversation, Message
from core.security import get_password_hash, verify_password
from schemas.user import UserCreate, UserUpdate # type: ignore
from datetime import datetime
from typing import Optional, List

async def get_db():
    """Get database instance"""
    from db.prisma import db
    return db

# User CRUD operations
async def create_user(user_data: UserCreate) -> User:
    """Create a new user with hashed password"""
    db = await get_db()
    
    hashed_password = get_password_hash(user_data.password)
    
    user = await db.user.create(
        data={
            'username': user_data.username,
            'email': user_data.email,
            'password_hash': hashed_password,
            'full_name': user_data.full_name,
            'is_active': True,
        }
    )
    return user

async def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    db = await get_db()
    
    user = await db.user.find_first(
        where={
            'OR': [
                {'username': username},
                {'email': username}
            ]
        }
    )
    
    if not user or not verify_password(password, user.password_hash):
        return None
    
    return user

async def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID"""
    db = await get_db()
    return await db.user.find_unique(where={'id': user_id})

async def get_user_by_username(username: str) -> Optional[User]:
    """Get user by username"""
    db = await get_db()
    return await db.user.find_unique(where={'username': username})

async def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email"""
    db = await get_db()
    return await db.user.find_unique(where={'email': email})

async def update_user_profile(user_id: str, user_data: UserUpdate) -> Optional[User]:
    """Update user profile information"""
    db = await get_db()
    
    # Build update data dynamically based on provided fields
    update_data = {}
    if user_data.email:
        update_data['email'] = user_data.email
    if user_data.full_name:
        update_data['full_name'] = user_data.full_name
    
    if not update_data:
        # No fields to update, return current user
        return await get_user_by_id(user_id)
    
    updated_user = await db.user.update(
        where={'id': user_id},
        data=update_data
    )
    return updated_user

async def update_user_password(user_id: str, new_password: str) -> bool:
    """Update user password with new hashed password"""
    db = await get_db()
    
    hashed_password = get_password_hash(new_password)
    
    await db.user.update(
        where={'id': user_id},
        data={'password_hash': hashed_password}
    )
    return True

# Conversation CRUD operations
async def create_conversation(user_id: str, title: Optional[str] = None) -> Conversation:
    """Create a new conversation for a user"""
    db = await get_db()
    
    conversation = await db.conversation.create(
        data={
            'user_id': user_id,
            'title': title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        }
    )
    return conversation

async def get_user_conversations(user_id: str) -> List[Conversation]:
    """Get all conversations for a user"""
    db = await get_db()
    
    conversations = await db.conversation.find_many(
        where={'user_id': user_id},
        order={'created_at': 'desc'}
    )
    return conversations

async def get_conversation_by_id(conversation_id: str, user_id: str) -> Optional[Conversation]:
    """Get a specific conversation by ID (only if it belongs to the user)"""
    db = await get_db()
    
    conversation = await db.conversation.find_first(
        where={
            'id': conversation_id,
            'user_id': user_id
        }
    )
    return conversation

async def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Delete a conversation (only if it belongs to the user)"""
    db = await get_db()
    
    # First verify the conversation belongs to the user
    conversation = await get_conversation_by_id(conversation_id, user_id)
    if not conversation:
        return False
    
    # Delete all messages in the conversation first
    await db.message.delete_many(where={'conversation_id': conversation_id})
    
    # Delete the conversation
    await db.conversation.delete(where={'id': conversation_id})
    return True

# Message CRUD operations
async def create_message(conversation_id: str, content: str, role: str, sender_id: Optional[str] = None) -> Message:
    """Create a new message in a conversation"""
    db = await get_db()
    
    message = await db.message.create(
        data={
            'conversation_id': conversation_id,
            'content': content,
            'role': role,
            'sender_id': sender_id,
        }
    )
    return message

async def get_conversation_messages(conversation_id: str) -> List[Message]:
    """Get all messages in a conversation"""
    db = await get_db()
    
    messages = await db.message.find_many(
        where={'conversation_id': conversation_id},
        order={'created_at': 'asc'}
    )
    return messages

async def update_message(message_id: str, content: str, user_id: str) -> Optional[Message]:
    """Update a message (only if user owns the conversation)"""
    db = await get_db()
    
    # First verify the user owns the conversation containing this message
    message = await db.message.find_first(
        where={'id': message_id},
        include={'conversation': True}
    )
    
    if not message or message.conversation.user_id != user_id:
        return None
    
    updated_message = await db.message.update(
        where={'id': message_id},
        data={
            'content': content,
            'edited_at': datetime.now()
        }
    )
    return updated_message

async def delete_message(message_id: str, user_id: str) -> bool:
    """Delete a message (only if user owns the conversation)"""
    db = await get_db()
    
    # First verify the user owns the conversation containing this message
    message = await db.message.find_first(
        where={'id': message_id},
        include={'conversation': True}
    )
    
    if not message or message.conversation.user_id != user_id:
        return False
    
    await db.message.delete(where={'id': message_id})
    return True
