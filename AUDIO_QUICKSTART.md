# Audio Transcription Quick Start Guide

Get up and running with real-time audio transcription in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- 2GB+ free RAM (for Whisper model)
- A WAV audio file for testing (optional)

## 🚀 Quick Start

### 1. Start the Backend

```bash
cd /Users/rahulraj/Desktop/ML_Proj
docker-compose up --build
```

Wait for:

```
✅ Database connected
✅ Transcription service loaded successfully
```

### 2. Create a Test User

```bash
# In a new terminal
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "audiotest",
    "email": "audio@test.com",
    "password": "testpass123",
    "full_name": "Audio Tester"
  }'
```

### 3. Test Audio Transcription

**Option A: With a WAV file**

```bash
# Copy your WAV file into the container
docker cp your_audio.wav ml_proj-backend-1:/app/test_audio.wav

# Run test client
docker exec -it ml_proj-backend-1 python test_audio_client.py \
  --username audiotest \
  --password testpass123 \
  --audio test_audio.wav
```

**Option B: Create a test audio file**

```bash
# Access container
docker exec -it ml_proj-backend-1 bash

# Install required tools
pip install pydub

# Create 5-second test audio with speech
python3 << 'EOF'
import wave
import numpy as np

# Generate 5 seconds of audio at 16kHz
sample_rate = 16000
duration = 5
t = np.linspace(0, duration, sample_rate * duration)

# Create a simple tone (simulating speech frequency)
audio = np.sin(2 * np.pi * 200 * t) * 0.3

# Convert to 16-bit PCM
audio_int = (audio * 32767).astype(np.int16)

# Save as WAV
with wave.open('test_audio.wav', 'wb') as wf:
    wf.setnchannels(1)  # Mono
    wf.setsampwidth(2)  # 2 bytes (16-bit)
    wf.setframerate(sample_rate)
    wf.writeframes(audio_int.tobytes())

print("Created test_audio.wav")
EOF

# Test transcription
python test_audio_client.py -u audiotest -p testpass123 -a test_audio.wav
```

### 4. Expected Output

```
🎵 Loading audio file: test_audio.wav
✅ Audio loaded:
   Sample rate: 16000 Hz
   Channels: 1
   Duration: 5.00 seconds
   Size: 160000 bytes
   Chunk size: 4096 bytes

🔌 Connecting to: ws://localhost:8000/ws/v1/audio?token=eyJ...
✅ Connected to audio transcription service

🎤 Streaming audio (160000 bytes in 4096-byte chunks)...
============================================================
Sent chunk 1/39 (4096 bytes) - 2.6%
Sent chunk 2/39 (4096 bytes) - 5.1%
...
Sent chunk 39/39 (3072 bytes) - 100.0%

✅ Finished streaming audio
⏳ Waiting for transcription...

ℹ️  Status: Processing audio...

📝 TRANSCRIPTION:
   Text: "Hello, this is a test of the audio transcription system."
   Duration: 5.00s
```

## 📖 Usage Examples

### Python Client

```python
import asyncio
import websockets
import json
import requests

async def transcribe_audio(username, password, audio_file):
    # 1. Login
    response = requests.post("http://localhost:8000/auth/login", json={
        "username": username,
        "password": password
    })
    token = response.json()["access_token"]

    # 2. Connect to WebSocket
    uri = f"ws://localhost:8000/ws/v1/audio?token={token}"

    async with websockets.connect(uri) as ws:
        # 3. Read and stream audio
        with open(audio_file, 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                await ws.send(chunk)
                await asyncio.sleep(0.05)  # Simulate real-time

        # 4. Wait for transcription
        await asyncio.sleep(3)

        # 5. Receive results
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "transcription":
                print(f"Transcription: {data['text']}")
                break

# Run
asyncio.run(transcribe_audio("audiotest", "testpass123", "test_audio.wav"))
```

### JavaScript/TypeScript Client

```typescript
// 1. Login
const loginResponse = await fetch("http://localhost:8000/auth/login", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username: "audiotest", password: "testpass123" }),
});
const { access_token } = await loginResponse.json();

// 2. Connect to WebSocket
const ws = new WebSocket(
  `ws://localhost:8000/ws/v1/audio?token=${access_token}`
);

ws.onopen = async () => {
  console.log("Connected");

  // 3. Get microphone access
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mediaRecorder = new MediaRecorder(stream);

  // 4. Send audio chunks
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      ws.send(event.data);
    }
  };

  mediaRecorder.start(100); // Send chunks every 100ms
};

// 5. Receive transcriptions
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "transcription") {
    console.log("Transcription:", data.text);
    displayTranscription(data.text);
  }
};
```

### cURL (REST API only)

```bash
# Login
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"audiotest","password":"testpass123"}' \
  | jq -r '.access_token')

# Note: WebSocket requires a proper WebSocket client
# cURL doesn't support WebSocket, use test_audio_client.py instead
```

## 🎤 Recording Audio for Testing

### macOS (using QuickTime)

1. Open QuickTime Player
2. File → New Audio Recording
3. Record your speech
4. Save as `.mov` file
5. Convert to WAV:
   ```bash
   ffmpeg -i recording.mov -ar 16000 -ac 1 recording.wav
   ```

### Linux (using arecord)

```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 test.wav
```

### Windows (using Audacity)

1. Download Audacity
2. Record audio
3. Set sample rate to 16000 Hz
4. Export as WAV (16-bit PCM)

### Online TTS (Text-to-Speech)

Use a TTS service to generate test audio:

```bash
# Using gTTS (Google Text-to-Speech)
pip install gTTS
python -c "from gtts import gTTS; gTTS('Hello, this is a test').save('test.mp3')"

# Convert to WAV
ffmpeg -i test.mp3 -ar 16000 -ac 1 test.wav
```

## 🔍 Troubleshooting

### Issue: "Transcription service not loaded"

**Check logs:**

```bash
docker logs ml_proj-backend-1 | grep -i whisper
```

**Solution:**

```bash
# Restart container
docker-compose restart backend

# Or rebuild
docker-compose up --build
```

### Issue: "No speech detected in audio"

**Causes:**

- Audio is silence or noise
- Audio too short (< 0.3 seconds)
- Wrong format

**Solution:**

- Record clear speech
- Ensure audio is at least 1 second
- Convert to 16kHz, mono, 16-bit WAV

### Issue: "Connection timeout"

**Check:**

```bash
# Is backend running?
curl http://localhost:8000/health

# Response: {"status":"ok"}
```

**Solution:**

```bash
# Restart backend
docker-compose restart backend
```

### Issue: "Authentication failed"

**Check token:**

```bash
# Login again
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"audiotest","password":"testpass123"}'
```

**Solution:**

- Ensure user exists
- Check password is correct
- Token may have expired (re-login)

### Issue: Slow transcription (> 5 seconds)

**Check:**

- Using CPU or GPU?
- Model size (whisper-tiny is fastest)
- System resources

**Solution:**

```bash
# Check GPU availability
docker exec ml_proj-backend-1 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If using CPU, consider smaller model or use GPU
```

## 🎛️ Configuration

### Change Whisper Model

Edit `/aura-backend/audio/transcription.py`:

```python
# Line 197
transcription_service = TranscriptionService(
    model_name="openai/whisper-base"  # Options: whisper-tiny, whisper-base, whisper-small
)
```

### Adjust Silence Timeout

Edit `/aura-backend/main.py` in audio WebSocket endpoint:

```python
# Around line 496
buffer = audio_buffer_manager.create_buffer(
    user.id,
    silence_timeout=1.0  # Default: 1.5 seconds
)
```

### Change Audio Chunk Size

In `test_audio_client.py`:

```bash
python test_audio_client.py -u user -p pass -a audio.wav --chunk-size 8192
```

## 📊 Performance Tips

### For Faster Transcription

1. **Use GPU** (if available)

   ```bash
   # Check CUDA
   nvidia-smi
   ```

2. **Use smaller model**

   - whisper-tiny: Fastest (500-1000ms)
   - whisper-base: Balanced (1000-2000ms)
   - whisper-small: Accurate (2000-3000ms)

3. **Reduce silence timeout**
   - Lower = faster response
   - Too low = cuts off speech

### For Better Accuracy

1. **Use larger model** (whisper-base or whisper-small)
2. **Clear audio** (minimal background noise)
3. **Longer timeout** (2-3 seconds for slow speakers)

## 🚀 Next Steps

- ✅ Week 4.1 Complete: Audio transcription working
- 📝 Week 4.2: LLM integration for AI responses
- 🎨 Week 5: Frontend UI development
- 🚀 Week 6: Production deployment

## 📚 Additional Resources

- [Full Documentation](WEEK4_1_AUDIO_TRANSCRIPTION.md)
- [Architecture Guide](ARCHITECTURE.md)
- [WebSocket Chat Guide](WEBSOCKET_QUICKSTART.md)
- [Audio Module README](aura-backend/audio/README.md)

---

**Need Help?**

- Check logs: `docker logs ml_proj-backend-1 -f`
- Verify health: `curl http://localhost:8000/health`
- Review docs: See links above

**Last Updated:** October 12, 2025  
**Status:** Week 4.1 Complete ✅
