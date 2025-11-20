"""
Chat API Router
"""

from fastapi import APIRouter, Depends, HTTPException
import logging

from api.models.schemas import ChatRequest, ChatResponse
from api.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service():
    """Dependency injection for chat service"""
    from api.main import get_chat_service
    return get_chat_service()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Chat with Aura AI
    
    - **message**: User's message
    - **emotion**: Optional current emotion
    - **cause**: Optional cause of the emotion
    - **max_tokens**: Maximum tokens to generate (default: 128)
    - **temperature**: Sampling temperature (default: 0.7)
    """
    try:
        response = await service.chat(
            message=request.message,
            emotion=request.emotion,
            cause=request.cause,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response,
            emotion_context={
                "emotion": request.emotion,
                "cause": request.cause
            } if request.emotion else None
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Chat with Aura AI (streaming response)
    
    Returns Server-Sent Events (SSE) stream
    """
    # TODO: Implement streaming with SSE
    raise HTTPException(status_code=501, detail="Streaming not yet implemented")
