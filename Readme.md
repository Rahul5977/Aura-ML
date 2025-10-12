# Aura - AI-Powered Chat Application

## Project Overview

Aura is a modern, real-time chat application built with FastAPI (backend) and React (frontend), featuring WebSocket-based communication, user authentication, and AI-powered conversations.

## 🚀 Current Status: Week 4.1 COMPLETED ✅

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

## 🔧 Tech Stack

### Backend

- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server with WebSocket support
- **Prisma**: Next-generation ORM for Python
- **PostgreSQL**: Relational database
- **JWT**: Authentication tokens
- **WebSockets**: Real-time bidirectional communication
- **OpenAI Whisper**: State-of-the-art speech recognition
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

### Audio Transcription

```bash
# Test audio transcription
docker exec -it ml_proj-backend-1 python test_audio_client.py \
  --username testuser \
  --password testpass \
  --audio sample.wav
```

See [WEEK4_1_AUDIO_TRANSCRIPTION.md](WEEK4_1_AUDIO_TRANSCRIPTION.md) for complete documentation.

---

**Last Updated**: October 12, 2025  
**Status**: Week 4.1 Complete - Real-time Audio Transcription Functional ✅
