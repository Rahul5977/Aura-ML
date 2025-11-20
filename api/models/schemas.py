"""Request/Response models for API"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User's message")
    emotion: Optional[str] = Field(None, description="Current emotion")
    cause: Optional[str] = Field(None, description="Cause of the emotion")
    max_tokens: Optional[int] = Field(128, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "I'm feeling anxious about my exam",
                "emotion": "anxious",
                "cause": "upcoming exam",
                "max_tokens": 128,
                "temperature": 0.7
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI's response")
    emotion_context: Optional[dict] = Field(None, description="Emotion context used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "I understand you're feeling anxious about your exam...",
                "emotion_context": {
                    "emotion": "anxious",
                    "cause": "upcoming exam"
                }
            }
        }


class EmotionDetectionRequest(BaseModel):
    """Emotion detection request"""
    text: str = Field(..., description="Text to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I'm so worried about my presentation tomorrow"
            }
        }


class EmotionDetectionResponse(BaseModel):
    """Emotion detection response"""
    emotion: str = Field(..., description="Detected emotion")
    confidence: float = Field(..., description="Confidence score")
    cause: Optional[str] = Field(None, description="Extracted cause if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "emotion": "anxious",
                "confidence": 0.92,
                "cause": "presentation tomorrow"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "gpu_available": True
            }
        }
