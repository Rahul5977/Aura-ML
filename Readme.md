# Aura - AI-Powered Chat Application with Contextual Intelligence

## Project Overview

Aura is a modern, real-time chat application built with FastAPI (backend) and Streamlit (frontend), featuring WebSocket-based communication, AI-powered conversations, and advanced contextual understanding with **complete ML pipeline transparency**.

## 🚀 Current Status: Week 8 - Production Frontend Complete ✅

### NEW: Streamlit Real-Time UI (Production-Grade) ✅

- **Production-grade Streamlit interface** in `aura-frontend/`
- **Real-time chat** with text and audio support
- **Live ML pipeline visualization** (all 5 stages visible)
- **Session management** with conversation history
- **Audio file upload** (WAV, MP3, M4A, OGG)
- **Complete transparency** into ML processing stages:
  - Speech-to-Text (STT) - Whisper
  - Speech Emotion Recognition (SER) - Wav2Vec2
  - Named Entity Recognition (NER) - spaCy
  - Commonsense Reasoning (COMET)
  - Knowledge Graph Update (Neo4j)
- **Backend health monitoring**
- **Session statistics** and analytics
- **Color-coded emotion and entity badges**
- **Expandable analysis details**
- **Docker support** with compose file
- **Comprehensive documentation**

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

### Week 6: Chat Orchestrator - Unified AI Pipeline ✅ **NEW!**

- **Single Unified Endpoint** (`POST /orchestrate/analyze-audio`)
  - One API call processes audio through complete AI pipeline
  - Returns aggregated JSON with all analysis results
  - Replaces multiple sequential API calls
- **Optimized Pipeline Execution**
  - **Phase 1**: STT + SER run in parallel (both need audio)
  - **Phase 2**: NER + COMET run sequentially (both need text)
  - **Phase 3**: Knowledge graph automatically updated
  - 40% faster than sequential processing
- **Complete AI Model Integration**
  - Speech-to-Text (Whisper)
  - Speech Emotion Recognition (Wav2Vec2)
  - Named Entity Recognition (spaCy)
  - Commonsense Reasoning (COMET)
- **Comprehensive Response Format**
  - Transcript with language detection
  - Emotion from audio (primary + all scores)
  - Emotion from text (detected feelings)
  - Entities (people, places, dates, concepts)
  - Commonsense inferences (feelings, wants, effects)
  - Knowledge graph updates
  - Processing metrics for monitoring
- **Production Features**
  - Graceful error handling per model
  - Processing time: 600-900ms
  - Built-in monitoring and logging
  - Comprehensive test suite: `test_week6.py`
  - Demo script: `chat_orchestrator_demo.py`

### Week 7: Neo4j & LLM Integration ✅ **NEW!**

- **Neo4j Graph Database** (Persistent Storage)
  - Enterprise-grade graph database
  - Cypher query language for graph traversal
  - Millions of nodes and relationships supported
  - Built-in graph algorithms and analytics
  - Real-time graph updates from AI pipeline
- **LLM Integration** (OpenAI GPT-4)
  - Context-aware response generation
  - Graph-powered AI responses
  - Emotional intelligence in conversations
  - Personalized interactions using knowledge graph
  - Fallback responses when API unavailable
- **Enhanced Orchestrator**
  - Single endpoint returns analysis + AI response
  - Graph context enriches LLM prompts
  - Conversation history maintained in Neo4j
  - Entity relationships tracked persistently
  - Production-ready error handling
- **Week 7 Features**
  - `POST /orchestrate/analyze-audio-v2` - Enhanced endpoint
  - Neo4j Browser at `http://localhost:7474`
  - Graph visualization and querying
  - Comprehensive test suite: `test_week7.py`
  - Quick start guide: `WEEK7_QUICKSTART.md`

## 🔧 Tech Stack

### Frontend (NEW!)

- **Streamlit**: Interactive web UI framework
- **Real-time updates**: Session state management
- **Responsive design**: Custom CSS styling
- **Docker support**: Containerized deployment

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

### NLP & Contextual Analysis (Week 5-6)

- **spaCy**: Named Entity Recognition
- **COMET (AllenAI)**: Commonsense emotional reasoning
- **Custom Knowledge Graph**: Entity and relationship modeling

### Graph Database & LLM (Week 7)

- **Neo4j**: Graph database for persistent knowledge storage
- **OpenAI GPT-4**: Large language model for intelligent responses
- **Cypher**: Graph query language
- **Graph Data Science**: Neo4j algorithms library

## 🚀 Quick Start

### Option 1: Streamlit Frontend (Recommended) **NEW!**

```bash
# Terminal 1: Start Backend
cd aura-backend
python main.py

# Terminal 2: Start Frontend
cd aura-frontend
./start.sh  # or start.bat on Windows

# Frontend UI: http://localhost:8501
# Backend API: http://localhost:8000
```

See [aura-frontend/QUICKSTART.md](aura-frontend/QUICKSTART.md) for detailed setup.

### Option 2: Docker Full Stack

```bash
# Start all services (Backend + Frontend + Neo4j)
cd aura-frontend
docker-compose up --build

# Frontend UI: http://localhost:8501
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
# Neo4j Browser: http://localhost:7474 (neo4j / password)
```

### Option 3: Backend Only

```bash
# Start all services (including Neo4j)
docker-compose up --build

# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
# Neo4j Browser: http://localhost:7474 (neo4j / aura_neo4j_pass)
# WebSocket Chat: ws://localhost:8000/ws/conversations/{id}
# WebSocket Audio: ws://localhost:8000/ws/v1/audio
```

## 📡 Features

### Streamlit UI (Production Frontend) **NEW!**

```bash
# Start the UI
cd aura-frontend
./start.sh

# Access at http://localhost:8501
```

**Features:**

- 💬 Real-time text and audio chat
- 🎤 Audio file upload (WAV, MP3, M4A, OGG)
- 👁️ Complete ML pipeline visualization
- 📊 Live session statistics
- 😊 Color-coded emotion badges
- 🏷️ Entity highlighting with types
- 🧠 Commonsense inference display
- 🔗 Knowledge graph updates
- ⚡ Performance metrics
- 🔄 Conversation history

See [aura-frontend/README.md](aura-frontend/README.md) for complete documentation.

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

### Chat Orchestrator (Week 6) **NEW!**

```bash
# Run the demo (no ML models required)
python3 chat_orchestrator_demo.py

# Or test with full backend:
python test_week6.py

# Or use the API directly:
# POST /orchestrate/analyze-audio
# Upload audio file, get complete analysis in one response
```

See [WEEK6_QUICKSTART.md](WEEK6_QUICKSTART.md) for quick start guide.  
See [WEEK6_IMPLEMENTATION_COMPLETE.md](WEEK6_IMPLEMENTATION_COMPLETE.md) for full documentation.

### Neo4j & LLM Integration (Week 7) **NEW!**

```bash
# Access Neo4j Browser
http://localhost:7474 (username: neo4j, password: aura_neo4j_pass)

# Run the enhanced demo
python3 chat_orchestrator_demo.py --week 7

# Or test with full backend:
python test_week7.py

# Or use the API directly:
# POST /orchestrate/analyze-audio-v2
# Upload audio file, get analysis + AI response in one response
```

See [WEEK7_QUICKSTART.md](WEEK7_QUICKSTART.md) for quick start guide.  
See [WEEK7_IMPLEMENTATION_COMPLETE.md](WEEK7_IMPLEMENTATION_COMPLETE.md) for full documentation.

## 📚 Documentation

### Frontend Documentation **NEW!**

- [aura-frontend/QUICKSTART.md](aura-frontend/QUICKSTART.md) - 5-minute frontend setup
- [aura-frontend/README.md](aura-frontend/README.md) - Complete frontend documentation
- [aura-frontend/IMPLEMENTATION_SUMMARY.md](aura-frontend/IMPLEMENTATION_SUMMARY.md) - Implementation details

### Quick Starts

- [WEEK5_QUICKSTART.md](WEEK5_QUICKSTART.md) - 5-minute contextual analysis guide
- [AUDIO_QUICKSTART.md](AUDIO_QUICKSTART.md) - Audio transcription guide
- [WEBSOCKET_QUICKSTART.md](WEBSOCKET_QUICKSTART.md) - Real-time chat guide
- [WEEK6_QUICKSTART.md](WEEK6_QUICKSTART.md) - Unified AI pipeline guide
- [WEEK7_QUICKSTART.md](WEEK7_QUICKSTART.md) - Neo4j & LLM integration guide

### Complete Documentation

- [WEEK5_CONTEXTUAL_ANALYSIS.md](WEEK5_CONTEXTUAL_ANALYSIS.md) - Contextual analysis features
- [WEEK5_COMPLETION_SUMMARY.md](WEEK5_COMPLETION_SUMMARY.md) - Week 5 implementation summary
- [WEEK6_CHAT_ORCHESTRATOR.md](WEEK6_CHAT_ORCHESTRATOR.md) - Chat orchestrator features
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [TEST_RESULTS.md](TEST_RESULTS.md) - Test results and verification
- [WEEK7_COMPLETION_SUMMARY.md](WEEK7_COMPLETION_SUMMARY.md) - Week 7 summary

### Week-by-Week Progress

- [WEEK3_COMPLETION_REPORT.md](WEEK3_COMPLETION_REPORT.md) - Real-time chat
- [WEEK4_1_AUDIO_TRANSCRIPTION.md](WEEK4_1_AUDIO_TRANSCRIPTION.md) - Audio features
- [WEEK4_1_COMPLETION_SUMMARY.md](WEEK4_1_COMPLETION_SUMMARY.md) - Week 4.1 summary
- Week 4.2 documentation (in docs/)
- [WEEK5_COMPLETION_SUMMARY.md](WEEK5_COMPLETION_SUMMARY.md) - Week 5 summary
- [WEEK6_COMPLETION_SUMMARY.md](WEEK6_COMPLETION_SUMMARY.md) - Week 6 summary
- [WEEK7_COMPLETION_SUMMARY.md](WEEK7_COMPLETION_SUMMARY.md) - Week 7 summary

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
python test_week6.py        # Week 6 unified AI pipeline
python test_week7.py        # Week 7 Neo4j & LLM integration

# Test specific features
python test_audio_client.py # Audio transcription
python test_websocket_chat.py # Chat functionality
```

---

**Last Updated**: October 13, 2025  
**Status**: Week 8 Complete - Production Frontend Complete ✅  
**Next**: Week 9 - Advanced Personalization & Recommendations
