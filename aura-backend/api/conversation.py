from fastapi import APIRouter, Depends, HTTPException, status
from core.dependencies import get_current_user
from db.crud import (
    create_conversation, get_user_conversations, get_conversation_by_id
)
from schemas.conversation import ConversationCreate, ConversationResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])

@router.get("/", response_model=list[ConversationResponse])
async def get_conversations(current_user=Depends(get_current_user)):
    conversations = await get_user_conversations(current_user.id)
    return conversations

@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_new_conversation(conversation_data: ConversationCreate, current_user=Depends(get_current_user)):
    conversation = await create_conversation(current_user.id, conversation_data.title)
    return conversation

@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, current_user=Depends(get_current_user)):
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation
