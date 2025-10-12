# Testing Guide for Week 4.2: Speech Emotion Recognition

## Overview

This guide covers testing the new Speech Emotion Recognition (SER) functionality integrated with the Speech-to-Text (STT) pipeline.

## Prerequisites

### System Requirements

- Docker and Docker Compose installed
- At least 6GB RAM available
- 4GB free disk space (for models)
- Working internet connection (first run only, to download models)

### Audio Requirements

- 16kHz sample rate (or will be resampled)
- Mono audio (stereo will be converted)
- WAV or PCM format
- Clean speech (minimal background noise for best results)

## Testing Methods

### Method 1: Quick Model Verification (No Docker Required)

Test the emotion recognition model directly without starting the full application.

```bash
cd aura-backend

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the test script
python test_emotion_service.py
```

**Expected Output:**

```
============================================================
Testing Speech Emotion Recognition Service
============================================================

1. Importing emotion service...
✅ Import successful

2. Initializing emotion service...
✅ Service initialized with model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
   Device: cpu

3. Loading emotion recognition model...
   (This may take a few minutes on first run)
✅ Model loaded successfully

4. Checking model status...
✅ Model loaded: True

5. Getting emotion labels...
✅ Supported emotions: angry, disgust, fear, happy, neutral, sad, surprise

6. Testing inference with dummy audio...
   Audio shape: (32000,)
   Sample rate: 16000 Hz
   Duration: 2.0 seconds
✅ Inference successful!

7. Inference Results:
   Primary Emotion: neutral
   Confidence: 45.67%
   Inference Time: 250ms

   All Emotion Scores:
      neutral   : 45.67%
      happy     : 20.12%
      sad       : 15.34%
      ...

============================================================
✅ ALL TESTS PASSED!
============================================================
```

**Troubleshooting:**

- If download fails: Check internet connection
- If model loading fails: Verify sufficient RAM (4GB+)
- If import fails: Run `pip install -r requirements.txt`

### Method 2: Full Docker Integration Test

Test the complete pipeline with Docker containers.

```bash
# Build and start all services
docker-compose up --build

# Wait for startup messages:
# ✅ Transcription service loaded successfully
# ✅ Emotion recognition service loaded successfully
```

**First Build Notes:**

- Takes 10-15 minutes (downloads models ~1.5GB)
- Subsequent builds take 2-3 minutes (cached)
- Models are pre-downloaded during build

### Method 3: WebSocket Client Test

Test with real audio streaming using the test client.

#### 3.1: Using Built-in Test Client

```bash
# In a new terminal (while docker-compose is running)
docker exec -it ml_proj-backend-1 python test_audio_client.py
```

**Expected Output:**

```
Connecting to ws://localhost:8000/ws/v1/audio...
Connected to audio transcription service
Streaming audio file: test_audio.wav

Received message (analysis):
{
  "type": "analysis",
  "transcript": {
    "text": "Hello, how are you doing today?",
    "language": "en"
  },
  "emotion": {
    "primary": "happy",
    "confidence": 0.87,
    "all_scores": {
      "angry": 0.02,
      "disgust": 0.01,
      "fear": 0.03,
      "happy": 0.87,
      "neutral": 0.05,
      "sad": 0.01,
      "surprise": 0.01
    }
  },
  "audio": {
    "duration": 2.5,
    "sample_rate": 16000
  },
  "processing": {
    "total_time_ms": 450,
    "transcription_time_ms": 380,
    "emotion_time_ms": 250
  },
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

#### 3.2: Using Custom Audio File

```bash
# Copy your audio file into the container
docker cp your_audio.wav ml_proj-backend-1:/app/

# Run test with custom audio
docker exec -it ml_proj-backend-1 python test_audio_client.py \
  --audio /app/your_audio.wav \
  --username testuser \
  --password testpass
```

### Method 4: Interactive Testing with curl/wscat

For advanced testing with custom WebSocket clients.

```bash
# Install wscat (if not already installed)
npm install -g wscat

# 1. First, get a JWT token
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass"}' \
  | jq -r '.access_token')

# 2. Connect to WebSocket
wscat -c "ws://localhost:8000/ws/v1/audio?token=$TOKEN"

# Now you can send binary audio data
# (This requires a custom client that can send binary WebSocket messages)
```

## Test Cases

### 1. Happy Speech Test

**Input:** Audio of someone saying "I'm so excited about this project!"  
**Expected Emotion:** happy (confidence > 0.7)  
**Expected Transcript:** Close match to input text

### 2. Sad Speech Test

**Input:** Audio of someone saying "I'm feeling really down today"  
**Expected Emotion:** sad (confidence > 0.6)  
**Expected Transcript:** Close match to input text

### 3. Angry Speech Test

**Input:** Audio of someone saying "This is so frustrating!"  
**Expected Emotion:** angry (confidence > 0.6)  
**Expected Transcript:** Close match to input text

### 4. Neutral Speech Test

**Input:** Audio of someone saying "The meeting is scheduled for 3 PM"  
**Expected Emotion:** neutral (confidence > 0.5)  
**Expected Transcript:** Close match to input text

### 5. Short Audio Test

**Input:** Audio less than 0.3 seconds  
**Expected Response:**

```json
{
  "type": "status",
  "content": "Audio too short, skipping transcription",
  "timestamp": "..."
}
```

### 6. Silence Test

**Input:** Audio with only background noise, no speech  
**Expected Response:**

```json
{
  "type": "status",
  "content": "No speech detected in audio",
  "timestamp": "..."
}
```

### 7. Error Handling Test

**Input:** Invalid audio data (corrupted file)  
**Expected Response:**

```json
{
  "type": "error",
  "content": "Failed to process audio data",
  "timestamp": "..."
}
```

## Performance Benchmarks

### Target Performance Metrics

| Metric                | Target  | Acceptable |
| --------------------- | ------- | ---------- |
| Total Processing Time | < 500ms | < 1000ms   |
| STT Inference Time    | < 400ms | < 800ms    |
| SER Inference Time    | < 300ms | < 600ms    |
| Parallel Speedup      | > 1.5x  | > 1.2x     |

### How to Measure

Check the logs or response for timing information:

```json
"processing": {
  "total_time_ms": 450,
  "transcription_time_ms": 380,
  "emotion_time_ms": 250
}
```

**Interpreting Results:**

- `total_time_ms` ≈ max(transcription_time_ms, emotion_time_ms) + overhead
- Overhead should be < 50ms (asyncio coordination)
- If total_time > transcription_time + emotion_time, services are NOT running in parallel

## Verification Checklist

### ✅ Pre-deployment Checklist

- [ ] Both models load successfully on startup
- [ ] Audio preprocessing works correctly
- [ ] STT produces accurate transcripts
- [ ] SER detects emotions with reasonable confidence
- [ ] STT and SER run in parallel (verify timing)
- [ ] Unified response format is correct
- [ ] Error handling works for invalid audio
- [ ] Short audio is handled gracefully
- [ ] Silence detection works correctly
- [ ] Logs contain all expected information
- [ ] No memory leaks after multiple requests
- [ ] Container restarts successfully

### ✅ Production Readiness Checklist

- [ ] Models pre-downloaded in Docker image
- [ ] Environment variables configured correctly
- [ ] Logging level set appropriately (INFO in prod)
- [ ] Error messages don't expose sensitive info
- [ ] Performance metrics meet targets
- [ ] Resource usage is acceptable (CPU < 80%, RAM < 4GB)
- [ ] Documentation is complete
- [ ] All tests pass consistently

## Common Issues and Solutions

### Issue 1: Models Not Loading

**Symptoms:**

```
⚠️  Failed to load emotion recognition service
```

**Solutions:**

1. Check internet connection (first run needs to download ~1.5GB)
2. Verify disk space: `df -h`
3. Check Docker logs: `docker-compose logs backend`
4. Ensure transformers version is correct: `pip show transformers`

### Issue 2: Slow Processing

**Symptoms:**

- Processing time > 2 seconds
- High CPU usage

**Solutions:**

1. Use smaller models (whisper-tiny is already the smallest)
2. Enable GPU if available
3. Reduce audio quality/sample rate (not recommended)
4. Scale horizontally (multiple containers)

### Issue 3: Incorrect Emotions

**Symptoms:**

- All audio detected as "neutral"
- Confidence always < 0.5

**Solutions:**

1. Check audio quality (clear speech, minimal noise)
2. Ensure audio is in correct format (16kHz mono)
3. Verify model loaded correctly
4. Try with known test samples (emotion datasets)

### Issue 4: WebSocket Connection Fails

**Symptoms:**

```
Authentication failed
```

**Solutions:**

1. Verify JWT token is valid
2. Check token expiration
3. Ensure user exists in database
4. Check database connection

### Issue 5: Container Out of Memory

**Symptoms:**

- Container crashes
- `docker stats` shows high memory usage

**Solutions:**

1. Increase Docker memory limit (Settings > Resources)
2. Use CPU-only mode (no GPU)
3. Close other applications
4. Consider using model quantization (future enhancement)

## Debugging Tips

### Enable Debug Logging

```python
# In main.py, change:
logging.basicConfig(level=logging.DEBUG)
```

### Monitor Container Resources

```bash
# Real-time resource monitoring
docker stats ml_proj-backend-1
```

### Check Logs

```bash
# Follow logs
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Search for errors
docker-compose logs backend | grep ERROR
```

### Test Components Individually

```bash
# Test only transcription
docker exec -it ml_proj-backend-1 python -c "
from audio import initialize_transcription_service, transcription_service
initialize_transcription_service()
print('STT OK')
"

# Test only emotion recognition
docker exec -it ml_proj-backend-1 python -c "
from audio import initialize_emotion_service, emotion_service
initialize_emotion_service()
print('SER OK')
"
```

## Test Data Recommendations

### Creating Test Audio

Use a tool like Audacity or ffmpeg to create test samples:

```bash
# Convert any audio to correct format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav

# Generate test tones (for quick testing)
ffmpeg -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 test_tone.wav
```

### Recommended Test Samples

1. **Clear speech** - Single speaker, quiet environment
2. **Emotional speech** - Clear emotions (happy, sad, angry)
3. **Conversational** - Natural conversation style
4. **Noisy audio** - Background noise (to test robustness)
5. **Multiple speakers** - To test how system handles overlapping speech

## Next Steps

After successful testing:

1. **Collect Performance Metrics**: Monitor in production for 1 week
2. **Fine-tune Models**: Consider domain-specific fine-tuning
3. **Optimize Performance**: Profile and optimize bottlenecks
4. **Scale Testing**: Test with multiple concurrent users
5. **Security Audit**: Review authentication and data handling

## Support

If you encounter issues not covered in this guide:

1. Check the main documentation: `docs/WEEK_4_2_SER_INTEGRATION.md`
2. Review logs for error messages
3. Test components individually to isolate the issue
4. Verify all dependencies are installed correctly

---

**Happy Testing! 🎉**
