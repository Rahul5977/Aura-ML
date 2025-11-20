"""
Emotion Detection Router
"""

from fastapi import APIRouter, HTTPException
import logging

from api.models.schemas import EmotionDetectionRequest, EmotionDetectionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/emotion/detect", response_model=EmotionDetectionResponse)
async def detect_emotion(request: EmotionDetectionRequest):
    """
    Detect emotion from text using ECE model
    
    - **text**: Text to analyze for emotion
    """
    # TODO: Implement ECE model integration
    raise HTTPException(status_code=501, detail="Emotion detection not yet implemented")
