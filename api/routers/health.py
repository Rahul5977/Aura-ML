"""
Health Check Router
"""

from fastapi import APIRouter
import torch

from api.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns service status, model status, and GPU availability
    """
    from api.main import chat_service
    
    return HealthResponse(
        status="healthy" if chat_service and chat_service.is_ready() else "initializing",
        model_loaded=chat_service is not None and chat_service.is_ready(),
        gpu_available=torch.cuda.is_available()
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong"}
