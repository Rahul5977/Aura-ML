"""
Audio processing utilities for real-time transcription
Handles audio buffering, preprocessing, and resampling
"""

import numpy as np
import librosa
import io
import wave
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Standard sample rate for Whisper and most STT models
TARGET_SAMPLE_RATE = 16000


def bytes_to_audio_array(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Convert raw audio bytes to numpy array.
    
    Args:
        audio_bytes: Raw audio data in bytes
        sample_rate: Expected sample rate of the audio
        
    Returns:
        Numpy array of audio samples, or None if conversion fails
    """
    try:
        # Try to load as WAV format first
        audio_io = io.BytesIO(audio_bytes)
        
        # Try reading as WAV
        try:
            with wave.open(audio_io, 'rb') as wav_file:
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                framerate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                
                # Convert bytes to numpy array based on sample width
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16
                
                audio_array = np.frombuffer(frames, dtype=dtype)
                
                # Convert to float32 and normalize to [-1, 1]
                if dtype == np.uint8:
                    audio_array = (audio_array.astype(np.float32) - 128) / 128
                else:
                    max_val = np.iinfo(dtype).max
                    audio_array = audio_array.astype(np.float32) / max_val
                
                # Handle stereo by averaging channels
                if channels > 1:
                    audio_array = audio_array.reshape(-1, channels).mean(axis=1)
                
                # Resample if necessary
                if framerate != sample_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=framerate,
                        target_sr=sample_rate
                    )
                
                logger.debug(f"Converted audio: {len(audio_array)} samples at {sample_rate}Hz")
                return audio_array
                
        except wave.Error:
            # Not a WAV file, try raw PCM
            pass
        
        # Try as raw PCM (16-bit signed integers)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        logger.debug(f"Converted raw PCM audio: {len(audio_array)} samples")
        return audio_array
        
    except Exception as e:
        logger.error(f"Error converting audio bytes to array: {e}")
        return None


def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Resample audio to target sample rate (16kHz for Whisper).
    
    Args:
        audio_array: Audio data as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio_array
    
    try:
        resampled = librosa.resample(
            audio_array,
            orig_sr=orig_sr,
            target_sr=target_sr
        )
        logger.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
        return resampled
    except Exception as e:
        logger.error(f"Error resampling audio: {e}")
        return audio_array


def normalize_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio_array: Audio data as numpy array
        
    Returns:
        Normalized audio array
    """
    max_val = np.abs(audio_array).max()
    if max_val > 0:
        return audio_array / max_val
    return audio_array


def preprocess_audio_for_whisper(
    audio_bytes: bytes,
    sample_rate: int = 16000
) -> Optional[np.ndarray]:
    """
    Complete preprocessing pipeline for Whisper STT model.
    Converts bytes to numpy array and resamples to 16kHz.
    
    Args:
        audio_bytes: Raw audio data in bytes
        sample_rate: Original sample rate of the audio
        
    Returns:
        Preprocessed audio array ready for Whisper, or None if processing fails
    """
    try:
        # Convert bytes to array
        audio_array = bytes_to_audio_array(audio_bytes, sample_rate)
        if audio_array is None:
            return None
        
        # Ensure it's 16kHz
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_array = resample_audio(audio_array, sample_rate, TARGET_SAMPLE_RATE)
        
        # Normalize
        audio_array = normalize_audio(audio_array)
        
        logger.info(f"Preprocessed audio: {len(audio_array)} samples at {TARGET_SAMPLE_RATE}Hz")
        return audio_array
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return None


def calculate_audio_duration(audio_array: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    """
    Calculate duration of audio in seconds.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        Duration in seconds
    """
    return len(audio_array) / sample_rate


def detect_silence(audio_array: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Simple silence detection based on RMS energy.
    
    Args:
        audio_array: Audio data as numpy array
        threshold: RMS threshold below which audio is considered silent
        
    Returns:
        True if audio is silent, False otherwise
    """
    rms = np.sqrt(np.mean(audio_array ** 2))
    return rms < threshold
