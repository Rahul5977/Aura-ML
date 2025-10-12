"""
Speech Emotion Recognition (SER) service using Wav2Vec2-based models
Handles model loading and emotion inference
"""

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import numpy as np
from typing import Optional, Dict, List
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class EmotionRecognitionService:
    """
    Service for recognizing emotions from audio using Wav2Vec2-based models.
    Uses pre-trained model from Hugging Face for emotion classification.
    """
    
    # Emotion labels mapping (model-specific)
    EMOTION_LABELS = [
        "angry",
        "disgust", 
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]
    
    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize emotion recognition service.
        
        Args:
            model_name: Hugging Face model identifier
                       Default: wav2vec2-lg-xlsr-en-speech-emotion-recognition
                       Alternative options:
                       - "superb/wav2vec2-base-superb-er" (SUPERB emotion recognition)
                       - "facebook/wav2vec2-large-xlsr-53"
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        logger.info(f"Initializing EmotionRecognitionService with model: {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load Wav2Vec2 emotion recognition model and processor.
        This is a blocking operation and should be called during startup.
        """
        try:
            logger.info(f"Loading emotion recognition model: {self.model_name}")
            
            # Load processor and model
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading emotion recognition model: {e}")
            raise
    
    async def recognize_emotion(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        return_all_scores: bool = False
    ) -> Dict[str, any]:
        """
        Recognize emotion from audio array.
        
        Args:
            audio_array: Audio data as numpy array (16kHz, mono)
            sampling_rate: Sample rate of the audio (default: 16000)
            return_all_scores: Whether to return scores for all emotions
            
        Returns:
            Dictionary with emotion recognition results:
            {
                "emotion": "happy",
                "confidence": 0.87,
                "timestamp": "ISO 8601 timestamp",
                "all_scores": {  # if return_all_scores=True
                    "angry": 0.05,
                    "happy": 0.87,
                    "neutral": 0.08,
                    ...
                }
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Emotion recognition model not loaded. Call load_model() first.")
        
        try:
            # Run emotion recognition in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._recognize_sync,
                audio_array,
                sampling_rate,
                return_all_scores
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during emotion recognition: {e}")
            raise
    
    def _recognize_sync(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
        return_all_scores: bool
    ) -> Dict[str, any]:
        """
        Synchronous emotion recognition (runs in thread pool).
        
        Args:
            audio_array: Audio data as numpy array
            sampling_rate: Sample rate of the audio
            return_all_scores: Whether to return all emotion scores
            
        Returns:
            Dictionary with emotion recognition results
        """
        start_time = datetime.now()
        
        # Ensure audio is float32 and normalized
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize audio to [-1, 1] range if not already
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val
        
        # Process audio input
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        scores = probabilities[0].cpu().numpy()
        
        # Get predicted emotion
        predicted_idx = scores.argmax()
        predicted_emotion = self.EMOTION_LABELS[predicted_idx]
        confidence = float(scores[predicted_idx])
        
        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "inference_time_ms": int(inference_time * 1000)
        }
        
        # Add all emotion scores if requested
        if return_all_scores:
            all_scores = {
                emotion: float(score)
                for emotion, score in zip(self.EMOTION_LABELS, scores)
            }
            result["all_scores"] = all_scores
        
        logger.info(
            f"Emotion recognition completed: {predicted_emotion} "
            f"(confidence: {confidence:.2f}, time: {inference_time*1000:.0f}ms)"
        )
        
        return result
    
    def get_emotion_labels(self) -> List[str]:
        """
        Get list of supported emotion labels.
        
        Returns:
            List of emotion label strings
        """
        return self.EMOTION_LABELS.copy()
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.is_loaded


# Global service instance
emotion_service = EmotionRecognitionService()


def initialize_emotion_service(model_name: Optional[str] = None) -> None:
    """
    Initialize and load the emotion recognition service.
    Should be called during application startup.
    
    Args:
        model_name: Optional custom model name to use
    """
    global emotion_service
    
    if model_name:
        emotion_service = EmotionRecognitionService(model_name=model_name)
    
    emotion_service.load_model()


async def recognize_emotion_from_audio(
    audio_array: np.ndarray,
    sampling_rate: int = 16000,
    return_all_scores: bool = False
) -> Dict[str, any]:
    """
    Convenience function to recognize emotion from audio.
    
    Args:
        audio_array: Audio data as numpy array
        sampling_rate: Sample rate of the audio
        return_all_scores: Whether to return all emotion scores
        
    Returns:
        Dictionary with emotion recognition results
    """
    return await emotion_service.recognize_emotion(
        audio_array,
        sampling_rate,
        return_all_scores
    )
