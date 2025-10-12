"""
Speech-to-Text transcription service using Whisper
Handles model loading and transcription
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from typing import Optional, Dict
import logging
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Service for transcribing audio using Whisper model.
    Uses a smaller model for faster inference.
    """
    
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        """
        Initialize transcription service.
        
        Args:
            model_name: Hugging Face model identifier
                       Options: whisper-tiny, whisper-base, whisper-small
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        logger.info(f"Initializing TranscriptionService with model: {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load Whisper model and processor.
        This is a blocking operation and should be called during startup.
        """
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_array: np.ndarray,
        language: str = "en",
        return_timestamps: bool = False
    ) -> Dict[str, any]:
        """
        Transcribe audio array to text.
        
        Args:
            audio_array: Audio data as numpy array (16kHz, mono)
            language: Language code (default: "en" for English)
            return_timestamps: Whether to return word-level timestamps
            
        Returns:
            Dictionary with transcription results:
            {
                "text": "transcribed text",
                "language": "en",
                "confidence": 0.95,  # if available
                "timestamps": [...] # if requested
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Transcription model not loaded. Call load_model() first.")
        
        try:
            # Run transcription in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._transcribe_sync,
                audio_array,
                language,
                return_timestamps
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return {
                "text": "",
                "language": language,
                "error": str(e)
            }
    
    def _transcribe_sync(
        self,
        audio_array: np.ndarray,
        language: str,
        return_timestamps: bool
    ) -> Dict[str, any]:
        """
        Synchronous transcription (runs in thread pool).
        
        Args:
            audio_array: Audio data as numpy array
            language: Language code
            return_timestamps: Whether to return timestamps
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Prepare audio input
            input_features = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            
            # Move to device
            input_features = input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                # Force language if specified
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe"
                )
                
                predicted_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    return_timestamps=return_timestamps
                )
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Transcription successful: '{transcription[:50]}...'")
            
            return {
                "text": transcription.strip(),
                "language": language,
                "duration": len(audio_array) / 16000  # duration in seconds
            }
            
        except Exception as e:
            logger.error(f"Error in sync transcription: {e}")
            raise
    
    def transcribe_sync(
        self,
        audio_array: np.ndarray,
        language: str = "en"
    ) -> str:
        """
        Synchronous transcription (for non-async contexts).
        
        Args:
            audio_array: Audio data as numpy array
            language: Language code
            
        Returns:
            Transcribed text
        """
        result = self._transcribe_sync(audio_array, language, False)
        return result.get("text", "")
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        if self.model:
            del self.model
            del self.processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.is_loaded = False
            logger.info("Unloaded Whisper model")


# Global transcription service instance
# Use whisper-tiny for fastest inference, upgrade to whisper-base for better accuracy
transcription_service = TranscriptionService(model_name="openai/whisper-tiny")


def initialize_transcription_service():
    """Initialize and load the transcription model (call during startup)"""
    try:
        transcription_service.load_model()
        logger.info("Transcription service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize transcription service: {e}")
        raise
