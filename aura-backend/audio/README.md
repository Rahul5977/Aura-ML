# Audio Processing Module

Real-time speech-to-text transcription and emotion recognition using OpenAI Whisper and Wav2Vec2.

## Overview

This module provides a complete multi-modal audio processing pipeline:

- **Speech-to-Text (STT)**: OpenAI Whisper for accurate transcription
- **Speech Emotion Recognition (SER)**: Wav2Vec2 for 7-class emotion detection
- **Parallel Processing**: STT and SER run concurrently for optimal performance
- Audio format conversion and preprocessing
- Per-client audio buffering with silence detection
- Async/await support for non-blocking operations

## Quick Start

```python
from audio import (
    audio_buffer_manager,
    transcription_service,
    emotion_service,
    initialize_transcription_service,
    initialize_emotion_service,
    preprocess_audio_for_whisper
)

# 1. Initialize services (at startup)
initialize_transcription_service()
initialize_emotion_service()

# 2. Create buffer for a user
buffer = audio_buffer_manager.create_buffer(user_id="user_123")

# 3. Add audio chunks as they arrive
buffer.add_chunk(audio_bytes)

# 4. Check for silence timeout
if buffer.is_silent_timeout():
    audio_data = buffer.clear_buffer()

    # 5. Preprocess audio
    audio_array = preprocess_audio_for_whisper(audio_data)

    # 6. Run STT and SER in parallel
    import asyncio
    stt_result, ser_result = await asyncio.gather(
        transcription_service.transcribe_audio(audio_array),
        emotion_service.recognize_emotion(audio_array)
    )

    print(f"Transcript: {stt_result['text']}")
    print(f"Emotion: {ser_result['emotion']} ({ser_result['confidence']:.2f})")
```

## Module Structure

```
audio/
├── __init__.py           # Public API exports
├── audio_utils.py        # Audio preprocessing utilities
├── buffer_manager.py     # Client buffer management
├── transcription.py      # Whisper STT service
└── emotion.py            # Wav2Vec2 SER service (NEW in Week 4.2)
```

## Components

### Audio Utils (`audio_utils.py`)

**Core Functions:**

```python
def preprocess_audio_for_whisper(
    audio_bytes: bytes,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Complete preprocessing pipeline for Whisper and Wav2Vec2.

    - Converts bytes to numpy array
    - Handles WAV and raw PCM formats
    - Resamples to 16kHz
    - Converts stereo to mono
    - Normalizes amplitude

    Returns:
        Audio as float32 numpy array (16kHz, mono, normalized)
    """
```

```python
def bytes_to_audio_array(
    audio_bytes: bytes,
    sample_rate: int = 16000
) -> Optional[np.ndarray]:
    """
    Convert audio bytes to numpy array.

    Supports:
    - WAV format (auto-detected)
    - Raw PCM 16-bit
    - Multiple sample rates (auto-resampled)
    - Stereo (converted to mono)
    """
```

```python
def resample_audio(
    audio_array: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    """Resample audio using librosa."""
```

```python
def calculate_audio_duration(
    audio_array: np.ndarray,
    sample_rate: int = 16000
) -> float:
    """Calculate duration in seconds."""
```

```python
def detect_silence(
    audio_array: np.ndarray,
    threshold: float = 0.01,
    min_silence_duration: float = 0.5,
    sample_rate: int = 16000
) -> bool:
    """Detect if audio segment is mostly silence."""
```

### Buffer Manager (`buffer_manager.py`)

**AudioBuffer Class:**

```python
class AudioBuffer:
    """Per-client audio buffer with timeout detection."""

    def __init__(self, user_id: str, silence_timeout: float = 1.5):
        """
        Args:
            user_id: Unique user identifier
            silence_timeout: Seconds of silence before triggering transcription
        """

    def add_chunk(self, chunk: bytes) -> None:
        """Add audio chunk to buffer."""

    def get_buffer(self) -> bytes:
        """Get current buffer contents."""

    def clear_buffer(self) -> bytes:
        """Get and clear buffer."""

    def is_silent_timeout(self) -> bool:
        """Check if silence timeout reached."""

    def has_data(self) -> bool:
        """Check if buffer has data."""
```

**AudioBufferManager Class:**

```python
class AudioBufferManager:
    """Global manager for all client buffers."""

    def create_buffer(
        self,
        user_id: str,
        silence_timeout: float = 1.5
    ) -> AudioBuffer:
        """Create buffer for user."""

    def get_buffer(self, user_id: str) -> Optional[AudioBuffer]:
        """Get user's buffer."""

    def remove_buffer(self, user_id: str) -> None:
        """Remove user's buffer."""

    async def monitor_buffers(
        self,
        callback: Callable[[str, bytes], Awaitable[None]]
    ) -> None:
        """Monitor all buffers for timeout (internal use)."""
```

**Global Instance:**

```python
# Use this singleton instance
from audio import audio_buffer_manager
```

### Transcription Service (`transcription.py`)

**TranscriptionService Class:**

```python
class TranscriptionService:
    """Whisper-based speech-to-text service."""

    def __init__(self, model_name: str = "openai/whisper-tiny"):
        """
        Initialize service.

        Model options:
        - openai/whisper-tiny (fastest, ~40MB)
        - openai/whisper-base (balanced, ~150MB)
        - openai/whisper-small (accurate, ~500MB)
        """

    def load_model(self) -> None:
        """
        Load Whisper model (blocking).
        Call once during startup.
        """

    async def transcribe_audio(
        self,
        audio_array: np.ndarray,
        language: str = "en",
        return_timestamps: bool = False
    ) -> Dict[str, any]:
        """
        Transcribe audio (async, non-blocking).

        Args:
            audio_array: Audio as numpy array (16kHz, mono)
            language: Language code ("en", "es", "fr", etc.)
            return_timestamps: Whether to return word timestamps

        Returns:
            {
                "text": "transcribed text",
                "language": "en",
                "duration": 2.5  # audio duration in seconds
            }
        """

    def unload_model(self) -> None:
        """Free model from memory."""
```

**Global Instance:**

```python
# Use this singleton instance
from audio import transcription_service, initialize_transcription_service

# At startup
initialize_transcription_service()

# In code
result = await transcription_service.transcribe_audio(audio)
```

### Emotion Recognition Service (`emotion.py`)

**EmotionRecognitionService Class:**

```python
class EmotionRecognitionService:
    """Wav2Vec2-based speech emotion recognition service."""

    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize service.

        Model options:
        - ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (recommended)
        - superb/wav2vec2-base-superb-er (alternative)
        """

    def load_model(self) -> None:
        """
        Load Wav2Vec2 model (blocking).
        Call once during startup.
        """

    async def recognize_emotion(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        return_all_scores: bool = False
    ) -> Dict[str, any]:
        """
        Recognize emotion from audio (async, non-blocking).

        Args:
            audio_array: Audio as numpy array (16kHz, mono)
            sampling_rate: Sample rate of audio (default: 16000)
            return_all_scores: Return scores for all emotions

        Returns:
            {
                "emotion": "happy",
                "confidence": 0.87,
                "timestamp": "ISO 8601 timestamp",
                "inference_time_ms": 250,
                "all_scores": {  # if return_all_scores=True
                    "angry": 0.02,
                    "disgust": 0.01,
                    "fear": 0.03,
                    "happy": 0.87,
                    "neutral": 0.05,
                    "sad": 0.01,
                    "surprise": 0.01
                }
            }
        """

    def get_emotion_labels(self) -> List[str]:
        """Get list of supported emotions."""
        # Returns: ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

**Supported Emotions:**

1. **angry** - Anger, frustration
2. **disgust** - Disgust, revulsion
3. **fear** - Fear, anxiety
4. **happy** - Happiness, joy
5. **neutral** - Neutral, calm
6. **sad** - Sadness, sorrow
7. **surprise** - Surprise, shock

**Global Instance:**

```python
# Use this singleton instance
from audio import emotion_service, initialize_emotion_service

# At startup
initialize_emotion_service()

# In code
result = await emotion_service.recognize_emotion(audio)
```

## Usage Examples

### Basic Transcription

```python
import asyncio
from audio import transcription_service, preprocess_audio_for_whisper

async def transcribe_file(file_path: str):
    # Load audio file
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    # Preprocess
    audio_array = preprocess_audio_for_whisper(audio_bytes)

    # Transcribe
    result = await transcription_service.transcribe_audio(audio_array)

    print(f"Transcription: {result['text']}")
    print(f"Duration: {result['duration']:.2f}s")

asyncio.run(transcribe_file("speech.wav"))
```

### Basic Emotion Recognition

```python
import asyncio
from audio import emotion_service, preprocess_audio_for_whisper

async def recognize_emotion_file(file_path: str):
    # Load audio file
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    # Preprocess
    audio_array = preprocess_audio_for_whisper(audio_bytes)

    # Recognize emotion
    result = await emotion_service.recognize_emotion(
        audio_array,
        return_all_scores=True
    )

    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Inference time: {result['inference_time_ms']}ms")

    print("\nAll emotion scores:")
    for emotion, score in result['all_scores'].items():
        print(f"  {emotion:10s}: {score:.2%}")

asyncio.run(recognize_emotion_file("speech.wav"))
```

### Parallel STT + SER Processing (Recommended)

```python
import asyncio
from audio import (
    transcription_service,
    emotion_service,
    preprocess_audio_for_whisper
)

async def analyze_audio_file(file_path: str):
    # Load and preprocess audio
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    audio_array = preprocess_audio_for_whisper(audio_bytes)

    # Run STT and SER in parallel for faster processing
    stt_result, ser_result = await asyncio.gather(
        transcription_service.transcribe_audio(audio_array),
        emotion_service.recognize_emotion(audio_array, return_all_scores=True)
    )

    # Build unified response
    response = {
        "transcript": {
            "text": stt_result["text"],
            "language": stt_result.get("language", "en")
        },
        "emotion": {
            "primary": ser_result["emotion"],
            "confidence": ser_result["confidence"],
            "all_scores": ser_result["all_scores"]
        },
        "processing": {
            "stt_time_ms": stt_result.get("inference_time_ms", 0),
            "ser_time_ms": ser_result["inference_time_ms"]
        }
    }

    print(f"Transcript: {response['transcript']['text']}")
    print(f"Emotion: {response['emotion']['primary']} ({response['emotion']['confidence']:.2%})")
    print(f"Processing: STT={response['processing']['stt_time_ms']}ms, "
          f"SER={response['processing']['ser_time_ms']}ms")

    return response

asyncio.run(analyze_audio_file("speech.wav"))
```

### WebSocket Integration

```python
from fastapi import WebSocket
from audio import audio_buffer_manager, preprocess_audio_for_whisper, transcription_service

@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()

    # Create buffer
    buffer = audio_buffer_manager.create_buffer(user_id)

    # Transcription callback
    async def on_timeout(uid: str, audio_data: bytes):
        audio_array = preprocess_audio_for_whisper(audio_data)
        result = await transcription_service.transcribe_audio(audio_array)
        await websocket.send_json({"text": result["text"]})

    # Start monitoring
    monitor_task = asyncio.create_task(
        audio_buffer_manager._monitor_buffers(on_timeout)
    )

    try:
        while True:
            # Receive audio chunks
            data = await websocket.receive_bytes()
            buffer.add_chunk(data)

    finally:
        monitor_task.cancel()
        audio_buffer_manager.remove_buffer(user_id)
```

### Batch Processing

```python
from audio import transcription_service
import asyncio

async def transcribe_batch(audio_arrays: list):
    """Transcribe multiple audio arrays concurrently."""
    tasks = [
        transcription_service.transcribe_audio(audio)
        for audio in audio_arrays
    ]
    results = await asyncio.gather(*tasks)
    return results

# Usage
results = await transcribe_batch([audio1, audio2, audio3])
```

## Configuration

### Model Selection

Change model in `transcription.py`:

```python
# Faster, less accurate
transcription_service = TranscriptionService("openai/whisper-tiny")

# Balanced (recommended for production)
transcription_service = TranscriptionService("openai/whisper-base")

# Slower, more accurate
transcription_service = TranscriptionService("openai/whisper-small")
```

### Silence Timeout

Adjust timeout when creating buffer:

```python
# Faster response (may cut off speech)
buffer = audio_buffer_manager.create_buffer(user_id, silence_timeout=1.0)

# More patient (better for slow speakers)
buffer = audio_buffer_manager.create_buffer(user_id, silence_timeout=2.5)
```

### Audio Parameters

In `audio_utils.py`:

```python
TARGET_SAMPLE_RATE = 16000  # Required for Whisper
MIN_AUDIO_DURATION = 0.3    # Skip shorter audio
SILENCE_THRESHOLD = 0.01    # Silence detection sensitivity
```

## Performance

### Latency Benchmarks

**whisper-tiny (CPU):**

- 1s audio: ~300-500ms
- 3s audio: ~500-800ms
- 5s audio: ~800-1200ms

**whisper-tiny (GPU):**

- 1s audio: ~100-150ms
- 3s audio: ~150-250ms
- 5s audio: ~250-400ms

**whisper-base (CPU):**

- 1s audio: ~600-900ms
- 3s audio: ~1000-1500ms
- 5s audio: ~1500-2500ms

### Memory Usage

- whisper-tiny: ~200MB RAM
- whisper-base: ~400MB RAM
- whisper-small: ~1GB RAM
- Per-client buffer: ~1-10MB (depends on audio length)

### Optimization Tips

1. **Use GPU**: Install CUDA-enabled PyTorch

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Batch Processing**: Process multiple users concurrently

   ```python
   results = await asyncio.gather(*[
       transcription_service.transcribe_audio(audio1),
       transcription_service.transcribe_audio(audio2),
   ])
   ```

3. **Model Quantization**: Use INT8 models (future)
   ```python
   model = WhisperForConditionalGeneration.from_pretrained(
       "openai/whisper-tiny",
       load_in_8bit=True  # Requires bitsandbytes
   )
   ```

## Error Handling

```python
from audio import transcription_service, preprocess_audio_for_whisper

try:
    # Preprocess
    audio_array = preprocess_audio_for_whisper(audio_bytes)
    if audio_array is None:
        raise ValueError("Failed to process audio")

    # Transcribe
    result = await transcription_service.transcribe_audio(audio_array)

    if "error" in result:
        print(f"Transcription error: {result['error']}")
    else:
        print(f"Success: {result['text']}")

except Exception as e:
    print(f"Error: {e}")
```

## Testing

### Unit Tests

```python
import pytest
from audio import preprocess_audio_for_whisper, AudioBuffer

def test_audio_preprocessing():
    """Test audio preprocessing."""
    # Load test audio
    with open("test_audio.wav", "rb") as f:
        audio_bytes = f.read()

    # Preprocess
    audio_array = preprocess_audio_for_whisper(audio_bytes)

    assert audio_array is not None
    assert len(audio_array) > 0
    assert audio_array.dtype == np.float32

def test_buffer_timeout():
    """Test silence timeout detection."""
    buffer = AudioBuffer("test_user", silence_timeout=0.5)

    # Add chunk
    buffer.add_chunk(b"audio data")

    # Should not timeout immediately
    assert not buffer.is_silent_timeout()

    # Wait for timeout
    import time
    time.sleep(0.6)

    # Should timeout now
    assert buffer.is_silent_timeout()

@pytest.mark.asyncio
async def test_transcription():
    """Test transcription service."""
    from audio import transcription_service

    # Load and preprocess
    with open("test_audio.wav", "rb") as f:
        audio_bytes = f.read()
    audio_array = preprocess_audio_for_whisper(audio_bytes)

    # Transcribe
    result = await transcription_service.transcribe_audio(audio_array)

    assert "text" in result
    assert len(result["text"]) > 0
```

### Integration Test

Use `test_audio_client.py`:

```bash
python test_audio_client.py -u testuser -p testpass -a sample.wav
```

## Troubleshooting

**Issue: No audio detected**

- Check format (should be 16kHz, 16-bit PCM or WAV)
- Verify audio duration (min 0.3 seconds)
- Check buffer is receiving chunks

**Issue: Poor transcription accuracy**

- Upgrade to whisper-base or whisper-small
- Ensure audio quality (clear speech, low noise)
- Check sample rate (should be 16kHz)

**Issue: Slow performance**

- Use GPU if available
- Switch to whisper-tiny
- Check CPU/memory usage
- Consider batch processing

**Issue: Memory leak**

- Ensure buffers are removed on disconnect
- Call `audio_buffer_manager.remove_buffer(user_id)`
- Monitor buffer count: `len(audio_buffer_manager.buffers)`

## Dependencies

```txt
torch>=2.1.0          # PyTorch
transformers>=4.35.0  # Whisper models
librosa>=0.10.0       # Audio processing
numpy>=1.24.0         # Array operations
soundfile>=0.12.0     # Audio I/O
```

## License

Part of the Aura project. See main project LICENSE.

---

**Version:** 1.0.0  
**Last Updated:** October 12, 2025  
**Status:** Production Ready ✅
