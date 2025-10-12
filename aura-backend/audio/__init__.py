"""
Audio transcription and emotion recognition module for real-time speech processing
"""

from .audio_utils import (
    preprocess_audio_for_whisper,
    bytes_to_audio_array,
    resample_audio,
    calculate_audio_duration,
    detect_silence
)
from .buffer_manager import AudioBufferManager, audio_buffer_manager
from .transcription import TranscriptionService, transcription_service, initialize_transcription_service
from .emotion import EmotionRecognitionService, emotion_service, initialize_emotion_service, recognize_emotion_from_audio

__all__ = [
    'preprocess_audio_for_whisper',
    'bytes_to_audio_array',
    'resample_audio',
    'calculate_audio_duration',
    'detect_silence',
    'AudioBufferManager',
    'audio_buffer_manager',
    'TranscriptionService',
    'transcription_service',
    'initialize_transcription_service',
    'EmotionRecognitionService',
    'emotion_service',
    'initialize_emotion_service',
    'recognize_emotion_from_audio',
]
