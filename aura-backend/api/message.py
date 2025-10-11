from fastapi import APIRouter, Depends, HTTPException, status
from core.dependencies import get_current_user
from db.crud import get_conversation_by_id, get_conversation_messages, create_message, update_message, delete_message
from schemas.message import MessageCreate, MessageResponse, MessageUpdate
from typing import List

router = APIRouter(tags=["messages"])

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(conversation_id: str, current_user=Depends(get_current_user)):
    """
    Get all messages in a conversation.
    
    Only the conversation owner can access messages.
    """
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Conversation not found"
        )
    
    # Get messages
    messages = await get_conversation_messages(conversation_id)
    return messages

@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_new_message(
    conversation_id: str, 
    message_data: MessageCreate, 
    current_user=Depends(get_current_user)
):
    """
    Create a new message in a conversation.
    
    Only the conversation owner can create messages.
    """
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Conversation not found"
        )
    
    # Create message
    message = await create_message(
        conversation_id=conversation_id,
        content=message_data.content,
        role=message_data.role.value,  # Get enum value
        sender_id=current_user.id
    )
    return message

@router.put("/conversations/{conversation_id}/messages/{message_id}", response_model=MessageResponse)
async def update_message_content(
    conversation_id: str,
    message_id: str,
    message_update: MessageUpdate,
    current_user=Depends(get_current_user)
):
    """
    Update a message in a conversation.
    
    Only the conversation owner can update messages.
    """
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Conversation not found"
        )
    
    # Update message
    updated_message = await update_message(message_id, message_update.content, current_user.id)
    if not updated_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    return updated_message

@router.delete("/conversations/{conversation_id}/messages/{message_id}")
async def delete_message_from_conversation(
    conversation_id: str,
    message_id: str,
    current_user=Depends(get_current_user)
):
    """
    Delete a message from a conversation.
    
    Only the conversation owner can delete messages.
    """
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Conversation not found"
        )
    
    # Delete message
    success = await delete_message(message_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    return {"message": "Message deleted successfully"}
