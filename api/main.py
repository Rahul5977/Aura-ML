"""
FastAPI Application for Aura Emotional Support API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.routers import chat, emotion, health
from api.services.chat_service import ChatService
from aura_ml.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance
chat_service: ChatService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app"""
    global chat_service
    
    # Startup
    logger.info("🚀 Starting Aura API...")
    logger.info(f"Environment: {settings.ENV}")
    
    # Initialize chat service
    try:
        chat_service = ChatService()
        await chat_service.initialize()
        logger.info("✅ Chat service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chat service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Aura API...")
    if chat_service:
        await chat_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Aura API",
    description="Emotional Support AI Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(emotion.router, prefix="/api/v1", tags=["emotion"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Aura API",
        "version": "1.0.0",
        "docs": "/docs"
    }


def get_chat_service() -> ChatService:
    """Dependency to get chat service"""
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return chat_service
