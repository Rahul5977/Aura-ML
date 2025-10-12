# Aura - AI-Powered Chat Application

## Project Overview

Aura is a modern, real-time chat application built with FastAPI (backend) and React (frontend), featuring WebSocket-based communication, user authentication, and AI-powered conversations.

## 🚀 Current Status: Week 4.2 COMPLETED ✅

### Week 1: Project Setup & Database Design ✅

- PostgreSQL database with Prisma ORM
- User, Conversation, and Message models
- Docker containerization (backend, frontend, database)
- Database migrations and schema management

### Week 2: User Authentication & Basic API ✅

- JWT-based authentication system
- User registration and login endpoints
- Password hashing with bcrypt
- Protected API routes
- Conversation and message CRUD operations
- Comprehensive API testing

### Week 3: Real-time Communication & Chat Logic ✅

- **WebSocket endpoint for real-time chat**
- **Connection manager for multiple simultaneous users**
- **Message broadcasting to all connected clients**
- **Graceful connection/disconnection handling**
- **Message persistence to database**
- **System notifications (join/leave events)**
- **Active users tracking**
- **Comprehensive test suite and interactive client**

### Week 4.1: Real-Time Audio Transcription Pipeline ✅

- **WebSocket audio endpoint** (`/ws/v1/audio`)
- **Live audio streaming from clients**
- **Per-client audio buffering system**
- **OpenAI Whisper integration** (speech-to-text)
- **Smart silence detection** (1.5s timeout triggers transcription)
- **Audio preprocessing pipeline** (resampling, normalization)
- **Support for WAV and PCM formats**
- **Python test client** for audio streaming
- **Async/await non-blocking processing**
- **GPU/CPU automatic detection**

### Week 4.2: Speech Emotion Recognition (SER) ✅

- **Wav2Vec2-based emotion recognition model**
- **7-class emotion detection** (angry, disgust, fear, happy, neutral, sad, surprise)
- **Parallel STT + SER processing** with asyncio.gather()
- **Unified JSON response** with transcript, emotion, confidence, and timing
- **All emotion scores** with detailed confidence metrics
- **Enhanced error handling** for individual service failures
- **Performance monitoring** with detailed timing information
- **Docker optimization** with model pre-downloading
- **Production-ready logging** and error tracking

## 🔧 Tech Stack

### Backend

- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server with WebSocket support
- **Prisma**: Next-generation ORM for Python
- **PostgreSQL**: Relational database
- **JWT**: Authentication tokens
- **WebSockets**: Real-time bidirectional communication
- **OpenAI Whisper**: State-of-the-art speech recognition
- **Wav2Vec2**: Speech emotion recognition
- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing library
- **Transformers**: Hugging Face model hub

## 🚀 Quick Start

```bash
# Start all services
docker-compose up --build

# Backend API: http://localhost:8000
# WebSocket Chat: ws://localhost:8000/ws/conversations/{id}
# WebSocket Audio: ws://localhost:8000/ws/v1/audio
```

## 📡 Features

### Real-time Chat

```bash
# Interactive chat client
docker exec -it ml_proj-backend-1 python interactive_chat_client.py username password
```

See [WEBSOCKET_QUICKSTART.md](WEBSOCKET_QUICKSTART.md) for details.

### Audio Transcription & Emotion Recognition

```bash
# Test audio transcription with emotion detection
docker exec -it ml_proj-backend-1 python test_audio_client.py \
  --username testuser \
  --password testpass \
  --audio sample.wav

# Expected response format:
# {
#   "type": "analysis",
#   "transcript": {"text": "...", "language": "en"},
#   "emotion": {"primary": "happy", "confidence": 0.87, "all_scores": {...}},
#   "audio": {"duration": 2.5, "sample_rate": 16000},
#   "processing": {"total_time_ms": 450, ...},
#   "timestamp": "..."
# }
```

See [WEEK_4_2_SER_INTEGRATION.md](docs/WEEK_4_2_SER_INTEGRATION.md) for complete documentation.

---

**Last Updated**: January 15, 2024  
**Status**: Week 4.2 Complete - Speech Emotion Recognition Integrated ✅
