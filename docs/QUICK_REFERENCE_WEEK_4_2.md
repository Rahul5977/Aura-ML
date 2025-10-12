# Week 4.2 Quick Reference Card

## 🎯 What Changed?

The audio pipeline now performs **both** Speech-to-Text (STT) **and** Speech Emotion Recognition (SER) in parallel.

## 🚀 Quick Start

```bash
# Build and start (first time: 10-15 min for model downloads)
docker-compose up --build

# Test the endpoint
docker exec -it ml_proj-backend-1 python test_audio_client.py
```

## 📊 New Response Format

```json
{
  "type": "analysis",
  "transcript": {
    "text": "Hello, how are you?",
    "language": "en"
  },
  "emotion": {
    "primary": "happy",
    "confidence": 0.87,
    "all_scores": {...}
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

## 🎭 Supported Emotions

1. **angry** - Anger, frustration
2. **disgust** - Disgust, revulsion
3. **fear** - Fear, anxiety
4. **happy** - Happiness, joy
5. **neutral** - Neutral, calm
6. **sad** - Sadness, sorrow
7. **surprise** - Surprise, shock

## 🧪 Testing Commands

```bash
# Standalone emotion model test
cd aura-backend
python test_emotion_service.py

# Full integration test
docker-compose up --build
docker exec -it ml_proj-backend-1 python test_audio_client.py

# Check logs
docker-compose logs -f backend

# Monitor resources
docker stats ml_proj-backend-1
```

## 🔧 Configuration

### Change Models

Edit `main.py`:

```python
# Different emotion model
initialize_emotion_service(
    model_name="superb/wav2vec2-base-superb-er"
)

# Different STT model
initialize_transcription_service(
    model_name="openai/whisper-base"
)
```

### Audio Format Requirements

- **Sample Rate**: 16kHz (auto-resampled if different)
- **Channels**: Mono (auto-converted if stereo)
- **Format**: WAV or PCM
- **Bit Depth**: 16-bit

## ⚡ Performance Targets

| Metric           | Target    | Max    |
| ---------------- | --------- | ------ |
| Total Processing | 400-800ms | 1000ms |
| STT Time         | 300-500ms | 800ms  |
| SER Time         | 150-350ms | 600ms  |

## 🐛 Common Issues

### Models Not Loading

```bash
# Check internet, disk space, logs
docker-compose logs backend | grep ERROR
```

### Slow Processing

```bash
# Check resource usage
docker stats ml_proj-backend-1

# Consider using smaller models or GPU
```

### WebSocket Connection Failed

```bash
# Verify JWT token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass"}'
```

## 📁 Key Files

```
aura-backend/
├── audio/
│   ├── emotion.py           # NEW: SER service
│   ├── transcription.py     # STT service
│   ├── audio_utils.py       # Audio preprocessing
│   └── buffer_manager.py    # Buffering logic
├── main.py                  # MODIFIED: Parallel processing
├── test_emotion_service.py  # NEW: Standalone test
├── test_audio_client.py     # WebSocket test client
├── requirements.txt         # MODIFIED: +sentencepiece
└── Dockerfile               # MODIFIED: +models, +deps

docs/
├── WEEK_4_2_SER_INTEGRATION.md       # Complete docs
├── TESTING_GUIDE_WEEK_4_2.md         # Testing guide
└── WEEK_4_2_IMPLEMENTATION_SUMMARY.md # Summary
```

## 🔐 API Endpoints (Unchanged)

```
POST /auth/login
  → Get JWT token

WS /ws/v1/audio?token={jwt}
  → Stream audio, receive transcript + emotion
```

## 💾 Resource Requirements

- **RAM**: 4-6GB (models in memory)
- **Disk**: 4GB free (model downloads)
- **CPU**: 2-4 cores (parallel processing)
- **Network**: Fast connection for first build

## 📖 Documentation

- **Full Docs**: `docs/WEEK_4_2_SER_INTEGRATION.md`
- **Testing**: `docs/TESTING_GUIDE_WEEK_4_2.md`
- **Summary**: `docs/WEEK_4_2_IMPLEMENTATION_SUMMARY.md`
- **Architecture**: `ARCHITECTURE.md`

## 🎓 Learning Resources

- [Hugging Face Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [asyncio.gather()](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)

## ✅ Verification Checklist

- [ ] `docker-compose up --build` succeeds
- [ ] See: `✅ Transcription service loaded successfully`
- [ ] See: `✅ Emotion recognition service loaded successfully`
- [ ] `test_audio_client.py` returns analysis response
- [ ] Response contains `transcript` and `emotion` fields
- [ ] Processing time < 1 second
- [ ] No errors in logs

## 🚨 Breaking Changes

⚠️ **Response format changed** from:

```json
{"type": "transcription", "text": "...", ...}
```

to:

```json
{"type": "analysis", "transcript": {...}, "emotion": {...}, ...}
```

Old clients need to be updated to parse new format.

## 🔮 What's Next?

Week 4.3+ possibilities:

- Real-time streaming (no silence detection)
- Multi-language emotion recognition
- Emotion history tracking
- GPU acceleration
- Model quantization for smaller size

---

**Questions?** Check the full documentation or test scripts.

**Need Help?** Run diagnostic: `python test_emotion_service.py`

**Status**: ✅ Week 4.2 Complete - Ready to Test!
