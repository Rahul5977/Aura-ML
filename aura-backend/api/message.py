from fastapi import APIRouter, Depends, HTTPException, status
from core.dependencies import get_current_user
from db.crud import get_conversation_by_id, get_conversation_messages, create_message
from schemas.message import MessageCreate, MessageResponse

router = APIRouter(prefix="/conversations/{conversation_id}/messages", tags=["messages"])

@router.get("/", response_model=list[MessageResponse])
async def get_messages(conversation_id: str, current_user=Depends(get_current_user)):
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    messages = await get_conversation_messages(conversation_id)
    return messages

@router.post("/", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_new_message(conversation_id: str, message_data: MessageCreate, current_user=Depends(get_current_user)):
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    message = await create_message(conversation_id=conversation_id, content=message_data.content, role=message_data.role)
    return message
