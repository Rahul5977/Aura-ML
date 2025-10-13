# Aura - AI-Powered Chat Application with Contextual Intelligence

## Project Overview

Aura is a modern, real-time chat application built with FastAPI (backend) and React (frontend), featuring WebSocket-based communication, user authentication, AI-powered conversations, and advanced contextual understanding.

## 🚀 Current Status: Week 5 COMPLETED ✅

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

### Week 5: Contextual Analysis & Knowledge Graph ✅ **NEW!**

- **Named Entity Recognition (NER)** with spaCy
  - Extracts people, places, organizations, concepts, and dates
  - Position tracking with character indices
  - Batch processing support
- **COMET Emotional Reasoning** (AllenAI)
  - Commonsense inference of emotional effects
  - Understanding of wants, needs, and reactions
  - 6 inference types: xReact, oReact, xWant, oWant, xEffect, oEffect
- **Dynamic Knowledge Graph**
  - Structures entities and relationships
  - Tracks entity occurrences across conversations
  - Emotional relationship modeling (FEELS, WANTS)
  - Graph traversal and querying
  - JSON export functionality
- **REST API Endpoints**
  - `POST /analyze/text` - Comprehensive text analysis
  - `GET /analyze/conversation/{id}` - Conversation context
  - `GET /knowledge-graph/summary` - Graph statistics
  - `GET /knowledge-graph/export` - Graph data export
- **Processing Performance**: ~400-650ms per analysis
- **Comprehensive test suite**: `test_week5.py`

## 🔧 Tech Stack

### Backend

- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server with WebSocket support
- **Prisma**: Next-generation ORM for Python
- **PostgreSQL**: Relational database
- **JWT**: Authentication tokens
- **WebSockets**: Real-time bidirectional communication

### Audio & ML

- **OpenAI Whisper**: State-of-the-art speech recognition
- **Wav2Vec2**: Speech emotion recognition
- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing library
- **Transformers**: Hugging Face model hub

### NLP & Contextual Analysis (Week 5)

- **spaCy**: Named Entity Recognition
- **COMET (AllenAI)**: Commonsense emotional reasoning
- **Custom Knowledge Graph**: Entity and relationship modeling

## 🚀 Quick Start

```bash
# Start all services
docker-compose up --build

# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
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

### Contextual Analysis (Week 5)

```bash
# Run Week 5 test suite
python test_week5.py

# Or use the API directly:
# POST /analyze/text?text=John met Sarah at Starbucks&conversation_id=conv_001
# Returns: entities, emotions, knowledge graph updates
```

See [WEEK5_QUICKSTART.md](WEEK5_QUICKSTART.md) for quick start guide.  
See [WEEK5_CONTEXTUAL_ANALYSIS.md](WEEK5_CONTEXTUAL_ANALYSIS.md) for full documentation.

## 📚 Documentation

### Quick Starts

- [WEEK5_QUICKSTART.md](WEEK5_QUICKSTART.md) - 5-minute contextual analysis guide
- [AUDIO_QUICKSTART.md](AUDIO_QUICKSTART.md) - Audio transcription guide
- [WEBSOCKET_QUICKSTART.md](WEBSOCKET_QUICKSTART.md) - Real-time chat guide

### Complete Documentation

- [WEEK5_CONTEXTUAL_ANALYSIS.md](WEEK5_CONTEXTUAL_ANALYSIS.md) - Contextual analysis features
- [WEEK5_COMPLETION_SUMMARY.md](WEEK5_COMPLETION_SUMMARY.md) - Week 5 implementation summary
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [TEST_RESULTS.md](TEST_RESULTS.md) - Test results and verification

### Week-by-Week Progress

- [WEEK3_COMPLETION_REPORT.md](WEEK3_COMPLETION_REPORT.md) - Real-time chat
- [WEEK4_1_AUDIO_TRANSCRIPTION.md](WEEK4_1_AUDIO_TRANSCRIPTION.md) - Audio features
- [WEEK4_1_COMPLETION_SUMMARY.md](WEEK4_1_COMPLETION_SUMMARY.md) - Week 4.1 summary
- Week 4.2 documentation (in docs/)
- [WEEK5_COMPLETION_SUMMARY.md](WEEK5_COMPLETION_SUMMARY.md) - Week 5 summary

## 🎯 What Aura Can Do

### Understanding Conversations

- 🎤 **Transcribe Speech**: Convert audio to text with Whisper
- 😊 **Detect Emotions**: Recognize 7 emotions from voice
- 🏷️ **Extract Entities**: Identify people, places, organizations
- 💭 **Infer Feelings**: Understand emotional states with COMET
- 🕸️ **Build Knowledge**: Structure information in a graph
- 🔍 **Query Context**: Retrieve conversation history and relationships

### Real-time Communication

- 💬 **Live Chat**: WebSocket-based messaging
- 🔊 **Audio Streaming**: Real-time audio processing
- 👥 **Multi-user**: Concurrent connections
- 📡 **Broadcast**: Message distribution

### Intelligence & Context

- 🧠 **Commonsense Reasoning**: Understand social dynamics
- 📊 **Structured Data**: Transform text into knowledge
- 🔗 **Entity Linking**: Track mentions across conversations
- 📈 **Emotion Tracking**: Monitor emotional states

## 🧪 Testing

```bash
# Test all features
python test_system.py       # Week 3 + 4 features
python test_week5.py        # Week 5 contextual analysis

# Test specific features
python test_audio_client.py # Audio transcription
python test_websocket_chat.py # Chat functionality
```

---

**Last Updated**: October 13, 2025  
**Status**: Week 5 Complete - Contextual Analysis & Knowledge Graph ✅  
**Next**: Week 6 - Advanced Graph Features & LLM Integration
