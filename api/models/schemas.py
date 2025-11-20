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


class AudioAnalysisRequest(BaseModel):
    """Audio analysis request"""
    return_prosodic: bool = Field(
        default=True, 
        description="Whether to include prosodic features"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "return_prosodic": True
            }
        }


class AudioAnalysisResponse(BaseModel):
    """Audio analysis response with transcription and emotion"""
    transcription: str = Field(..., description="Transcribed text from audio")
    emotion: str = Field(..., description="Detected emotion from speech")
    emotion_confidence: float = Field(..., description="Confidence score for emotion")
    emotion_scores: dict = Field(..., description="Scores for all emotion classes")
    duration: float = Field(..., description="Audio duration in seconds")
    prosodic_features: Optional[dict] = Field(
        None, 
        description="Prosodic features (pitch, energy, speaking rate)"
    )
    model_info: Optional[dict] = Field(None, description="Model specifications")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transcription": "I'm feeling really anxious about my presentation tomorrow",
                "emotion": "fearful",
                "emotion_confidence": 0.78,
                "emotion_scores": {
                    "neutral": 0.05,
                    "calm": 0.03,
                    "happy": 0.02,
                    "sad": 0.08,
                    "angry": 0.02,
                    "fearful": 0.78,
                    "disgust": 0.01,
                    "surprised": 0.01
                },
                "duration": 4.5,
                "prosodic_features": {
                    "pitch_mean_hz": 185.3,
                    "pitch_std_hz": 42.7,
                    "energy_mean": 0.045,
                    "energy_std": 0.012,
                    "zero_crossing_rate": 0.082,
                    "spectral_centroid_hz": 2847.5,
                    "duration_sec": 4.5
                },
                "model_info": {
                    "stt_model": "whisper-base",
                    "stt_params": "74M",
                    "ser_model": "wav2vec2-ravdess",
                    "ser_params": "95M encoder + 0.2M head",
                    "ser_accuracy": 0.681,
                    "whisper_wer": "<10%"
                }
            }
        }


class AudioStreamResponse(BaseModel):
    """Real-time audio stream processing response"""
    transcription: str = Field(..., description="Transcribed text from audio chunk")
    emotion: str = Field(..., description="Detected emotion from chunk")
    emotion_confidence: float = Field(..., description="Confidence score")
    duration: float = Field(..., description="Actual chunk duration")
    chunk_duration: float = Field(..., description="Expected chunk duration")
    latency_target: str = Field(..., description="Target latency")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transcription": "I'm not sure what to do",
                "emotion": "sad",
                "emotion_confidence": 0.65,
                "duration": 5.0,
                "chunk_duration": 5.0,
                "latency_target": "<500ms"
            }
        }
