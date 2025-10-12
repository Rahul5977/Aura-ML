"""
Audio transcription module for real-time speech-to-text
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
]
