from fastapi import APIRouter, Depends, HTTPException, status
from core.dependencies import get_current_user
from db.crud import (
    create_conversation, get_user_conversations, get_conversation_by_id, delete_conversation
)
from schemas.conversation import ConversationCreate, ConversationResponse, ConversationUpdate
from typing import List

router = APIRouter(prefix="/conversations", tags=["conversations"])

@router.get("/", response_model=List[ConversationResponse])
async def get_conversations(current_user=Depends(get_current_user)):
    """
    Get all conversations for the current user.
    
    Returns conversations ordered by creation date (newest first).
    """
    conversations = await get_user_conversations(current_user.id)
    return conversations

@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_new_conversation(
    conversation_data: ConversationCreate, 
    current_user=Depends(get_current_user)
):
    """
    Create a new conversation for the current user.
    
    - **title**: Optional conversation title (auto-generated if not provided)
    """
    conversation = await create_conversation(current_user.id, conversation_data.title)
    return conversation

@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, current_user=Depends(get_current_user)):
    """
    Get a specific conversation by ID.
    
    Only the conversation owner can access it.
    """
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Conversation not found"
        )
    return conversation

@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    conversation_update: ConversationUpdate,
    current_user=Depends(get_current_user)
):
    """
    Update a conversation's title.
    
    Only the conversation owner can update it.
    """
    # Verify conversation exists and belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Update conversation (would need to implement this in crud.py)
    try:
        updated_conversation = await update_conversation_title(
            conversation_id, 
            conversation_update.title, 
            current_user.id
        )
        return updated_conversation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update conversation"
        )

@router.delete("/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: str, 
    current_user=Depends(get_current_user)
):
    """
    Delete a conversation and all its messages.
    
    Only the conversation owner can delete it.
    """
    success = await delete_conversation(conversation_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return {"message": "Conversation deleted successfully"}

# Helper function that would need to be implemented in crud.py
async def update_conversation_title(conversation_id: str, title: str, user_id: str):
    """Update conversation title - to be implemented in crud.py"""
    pass
