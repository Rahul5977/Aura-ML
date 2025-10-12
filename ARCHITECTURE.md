# Aura Backend - Architecture & Folder Structure

## 📁 Project Structure

```
aura-backend/
├── audio/                          # Audio transcription module (Week 4.1)
│   ├── __init__.py                # Module exports
│   ├── audio_utils.py             # Audio preprocessing (librosa, numpy)
│   ├── buffer_manager.py          # Per-client buffering & timeout
│   ├── transcription.py           # Whisper STT service
│   └── README.md                  # Audio module documentation
│
├── tests/                          # Test suite
│   └── __init__.py
│
├── main.py                         # FastAPI application entry point
├── auth.py                         # Authentication (JWT, bcrypt)
├── database.py                     # Database operations (Prisma)
├── schemas.py                      # Pydantic data models
├── websocket_manager.py            # WebSocket connection manager
│
├── interactive_chat_client.py      # Interactive WebSocket chat client
├── test_audio_client.py           # Audio streaming test client
├── test_websocket_chat.py         # WebSocket test suite
│
├── schema.prisma                   # Prisma database schema
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container config
├── .env                           # Environment variables (not in git)
├── .env.example                   # Example environment config
├── .gitignore                     # Git ignore rules
└── pytest.ini                     # Pytest configuration
```

## 🏗️ Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                         │
├─────────────────┬───────────────────┬───────────────────────┤
│  Web Frontend   │   Mobile Apps     │   Python Test Clients │
│  (React/Next)   │  (iOS/Android)    │   (WebSocket)         │
└────────┬────────┴─────────┬─────────┴───────────┬───────────┘
         │                  │                     │
         └──────────────────┼─────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   API Gateway  │
                    │   (FastAPI)    │
                    └───────┬────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼─────┐    ┌──────▼──────┐    ┌─────▼─────┐
    │   REST   │    │  WebSocket  │    │ WebSocket │
    │   API    │    │    Chat     │    │   Audio   │
    └────┬─────┘    └──────┬──────┘    └─────┬─────┘
         │                 │                  │
         └─────────────────┼──────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐    ┌──────▼──────┐    ┌─────▼──────┐
    │  Auth   │    │  Database   │    │   Audio    │
    │ Service │    │  (Prisma)   │    │  Processing│
    └─────────┘    └──────┬──────┘    └─────┬──────┘
                           │                 │
                    ┌──────▼──────┐   ┌──────▼──────┐
                    │ PostgreSQL  │   │   Whisper   │
                    │  Database   │   │   (PyTorch) │
                    └─────────────┘   └─────────────┘
```

### Module Interactions

```
┌──────────────────────────────────────────────────────────────┐
│                         main.py                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application                                    │  │
│  │  - HTTP REST endpoints                                  │  │
│  │  - WebSocket endpoints (chat, audio)                   │  │
│  │  - Startup/shutdown lifecycle                          │  │
│  └──┬──────────────┬──────────────┬──────────────┬────────┘  │
│     │              │              │              │            │
│  ┌──▼──────┐  ┌───▼────┐  ┌──────▼──────┐  ┌───▼──────┐    │
│  │ auth.py │  │schemas │  │ database.py │  │ ws_mgr   │    │
│  │         │  │  .py   │  │             │  │   .py    │    │
│  │- JWT    │  │- Models│  │- Prisma ORM │  │- WS Pool │    │
│  │- bcrypt │  │- Valid │  │- CRUD ops   │  │- Broadcast│   │
│  └─────────┘  └────────┘  └─────────────┘  └──────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              audio/ module (Week 4.2)                 │   │
│  │  ┌──────────────┐  ┌────────────────┐  ┌──────────┐ │   │
│  │  │audio_utils.py│  │buffer_manager  │  │transcrip │ │   │
│  │  │              │  │      .py       │  │ tion.py  │ │   │
│  │  │- Preprocess  │  │                │  │          │ │   │
│  │  │- Resample    │  │- AudioBuffer   │  │- Whisper │ │   │
│  │  │- Normalize   │  │- BufferManager │  │- STT     │ │   │
│  │  │- librosa     │  │- Timeout detect│  │- Async   │ │   │
│  │  └──────────────┘  └────────────────┘  └──────────┘ │   │
│  │                                                       │   │
│  │  ┌──────────────┐                                    │   │
│  │  │  emotion.py  │      NEW in Week 4.2               │   │
│  │  │              │                                    │   │
│  │  │- Wav2Vec2    │                                    │   │
│  │  │- SER         │                                    │   │
│  │  │- 7 emotions  │                                    │   │
│  │  │- Parallel    │                                    │   │
│  │  └──────────────┘                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 🔌 API Endpoints

### REST API

| Method | Endpoint                       | Description                  | Auth |
| ------ | ------------------------------ | ---------------------------- | ---- |
| GET    | `/health`                      | Health check                 | No   |
| POST   | `/auth/register`               | Register new user            | No   |
| POST   | `/auth/login`                  | Login (get JWT)              | No   |
| GET    | `/auth/me`                     | Get current user             | Yes  |
| PUT    | `/auth/me`                     | Update profile               | Yes  |
| POST   | `/auth/change-password`        | Change password              | Yes  |
| POST   | `/auth/logout`                 | Logout                       | Yes  |
| GET    | `/conversations`               | List user's conversations    | Yes  |
| POST   | `/conversations`               | Create conversation          | Yes  |
| GET    | `/conversations/{id}`          | Get conversation details     | Yes  |
| GET    | `/conversations/{id}/messages` | Get messages in conversation | Yes  |
| POST   | `/conversations/{id}/messages` | Create message               | Yes  |

### WebSocket API

| Endpoint                          | Description                | Auth |
| --------------------------------- | -------------------------- | ---- |
| `ws://host/ws/conversations/{id}` | Real-time chat             | Yes  |
| `ws://host/ws/v1/audio`           | Real-time audio transcribe | Yes  |

**Authentication:** JWT token via query parameter `?token={jwt}`

## 📊 Data Flow

### REST API Request Flow

```
Client Request
    │
    ▼
FastAPI Router
    │
    ▼
Auth Middleware (JWT verification)
    │
    ▼
Route Handler (main.py)
    │
    ▼
Schema Validation (schemas.py)
    │
    ▼
Business Logic
    │
    ├─► Database Operation (database.py)
    │       │
    │       ▼
    │   Prisma ORM
    │       │
    │       ▼
    │   PostgreSQL
    │
    ▼
Response (JSON)
    │
    ▼
Client
```

### WebSocket Chat Flow

```
Client connects
    │
    ▼
JWT authentication
    │
    ▼
WebSocket accepted
    │
    ▼
Connection added to manager
    │
    ▼
System message: "User joined"
    │
    ▼
┌───► Receive message from client
│       │
│       ▼
│   Parse JSON
│       │
│       ▼
│   Save to database (Prisma)
│       │
│       ▼
│   Broadcast to all clients in room
│       │
│       ▼
└───── Wait for next message
    │
    ▼
Disconnect
    │
    ▼
Remove from manager
    │
    ▼
System message: "User left"
```

### WebSocket Audio Transcription Flow

```
Client connects with JWT
    │
    ▼
WebSocket accepted
    │
    ▼
Create AudioBuffer for user
    │
    ▼
Start buffer monitoring
    │
    ▼
┌───► Receive audio chunk (binary)
│       │
│       ▼
│   Add to buffer
│       │
│       ▼
│   Update last_chunk_time
│       │
│       ▼
│   Check silence timeout (1.5s)
│       │
│       ├─► No timeout: continue receiving
│       │
│       └─► Timeout detected:
│               │
│               ▼
│           Get buffer contents
│               │
│               ▼
│           Preprocess audio (audio_utils)
│               │
│               ├─► bytes_to_audio_array()
│               ├─► resample_audio() to 16kHz
│               └─► normalize()
│               │
│               ▼
│           ┌──────────────┴──────────────┐
│           │                             │
│           ▼                             ▼
│       Transcribe (Whisper)      Recognize Emotion (Wav2Vec2)
│           │                             │
│           ├─► Load model                ├─► Load model
│           ├─► Generate transcript       ├─► Generate emotion scores
│           └─► Return text               └─► Return emotion + confidence
│           │                             │
│           └──────────────┬──────────────┘
│                          ▼
│               asyncio.gather() - wait for both
│                          │
│                          ▼
│               Aggregate results into unified JSON
│                          │
│                          ▼
│           Send result to client (JSON)
│               {
│                 "type": "analysis",
│                 "transcript": {...},
│                 "emotion": {...},
│                 "audio": {...},
│                 "processing": {...}
│               }
│               │
│               ▼
└───────────── Clear buffer, continue
    │
    ▼
Disconnect
    │
    ▼
Process remaining audio
    │
    ▼
Remove buffer
```

## 🗄️ Database Schema

### Prisma Models

```prisma
model User {
  id            String         @id @default(cuid())
  username      String         @unique
  email         String         @unique
  password_hash String
  full_name     String?
  created_at    DateTime       @default(now())
  updated_at    DateTime       @updatedAt

  conversations Conversation[]
}

model Conversation {
  id         String    @id @default(cuid())
  title      String?
  user_id    String
  created_at DateTime  @default(now())
  updated_at DateTime  @updatedAt

  user       User      @relation(fields: [user_id], references: [id])
  messages   Message[]
}

model Message {
  id              String       @id @default(cuid())
  conversation_id String
  content         String
  role            String       // "user" or "assistant"
  created_at      DateTime     @default(now())

  conversation    Conversation @relation(fields: [conversation_id], references: [id])
}
```

### Entity Relationships

```
User (1) ────< Conversation (N)
                    │
                    │
                    ▼
              Message (N)
```

## 🔐 Security

### Authentication Flow

```
1. User Registration
   ├─► Validate input (schemas.py)
   ├─► Hash password (bcrypt in auth.py)
   ├─► Store in database (database.py)
   └─► Return user data (no password)

2. User Login
   ├─► Verify credentials (database.py)
   ├─► Check password (bcrypt.checkpw)
   ├─► Generate JWT token (auth.py)
   └─► Return token

3. Protected Request
   ├─► Extract Bearer token from header
   ├─► Verify JWT signature (jose)
   ├─► Extract user_id from token
   ├─► Fetch user from database
   └─► Pass user to route handler
```

### Security Features

- **Password Hashing**: bcrypt with salt
- **JWT Tokens**: HS256 algorithm, configurable expiry
- **Input Validation**: Pydantic schemas
- **SQL Injection Protection**: Prisma ORM parameterization
- **WebSocket Auth**: Token-based connection authentication

## 🎯 Key Design Decisions

### 1. Flat Module Structure

**Decision:** Keep core files in root (`auth.py`, `database.py`, `schemas.py`)  
**Reason:** Simplicity for small-medium projects, easy navigation  
**Trade-off:** Less modular than `/api`, `/core`, `/db` structure

### 2. Single Schema File

**Decision:** All Pydantic models in `schemas.py`  
**Reason:** Easy to find and maintain for current project size  
**Future:** Split when file exceeds ~500 lines

### 3. Audio Module Separation

**Decision:** Audio transcription in dedicated `/audio` module  
**Reason:** Complex ML code, reusable, clear separation of concerns  
**Benefits:** Can be extracted as library, easier testing

### 4. Global Singleton Services

**Decision:** `transcription_service`, `audio_buffer_manager` as singletons  
**Reason:** Model loading is expensive, shared state needed  
**Implementation:** Module-level instances

### 5. WebSocket Connection Manager

**Decision:** Centralized `WebSocketManager` class  
**Reason:** Track connections, broadcast messages, handle disconnects  
**Benefits:** Clean API, easier to add features (typing indicators, etc.)

## 🚀 Scalability Considerations

### Current Limitations

1. **In-Memory Buffer Storage**

   - Buffers stored in Python process memory
   - Lost on server restart
   - Doesn't scale across multiple servers

2. **Single Whisper Model Instance**

   - One model per server
   - Sequential processing (though async)
   - Memory-bound

3. **No Load Balancing**
   - Single server deployment
   - No horizontal scaling

### Future Improvements

1. **Distributed Buffers**

   - Use Redis for buffer storage
   - Persist across restarts
   - Share across servers

2. **Model Serving**

   - Separate transcription service
   - GPU-optimized servers
   - Queue-based processing (Celery/RQ)

3. **Horizontal Scaling**

   - Multiple API servers behind load balancer
   - Sticky sessions for WebSockets
   - Distributed database (read replicas)

4. **Caching**
   - Redis for session storage
   - Cache frequent queries
   - Rate limiting per user

## 🧪 Testing Strategy

### Test Files

```
test_websocket_chat.py       # WebSocket chat integration tests
test_audio_client.py         # Audio transcription client
interactive_chat_client.py   # Manual WebSocket testing
```

### Test Coverage

- ✅ User registration and login
- ✅ JWT token generation and validation
- ✅ WebSocket connection and messaging
- ✅ Message persistence
- ✅ Audio streaming and transcription
- ⏳ Unit tests for individual modules (future)
- ⏳ Load testing (future)

## 📦 Dependencies

### Core Web Framework

```
fastapi>=0.104.1           # Modern Python web framework
uvicorn[standard]>=0.24.0  # ASGI server with WebSocket support
```

### Database

```
prisma>=0.11.0             # Type-safe database ORM
asyncpg>=0.29.0            # PostgreSQL async driver
```

### Authentication

```
python-jose[cryptography]>=3.3.0  # JWT tokens
bcrypt>=4.1.2                      # Password hashing
passlib[bcrypt]>=1.7.4            # Password utilities
```

### Audio & ML

```
torch>=2.1.1               # PyTorch ML framework
torchaudio>=2.1.1         # Audio processing
transformers>=4.35.2       # Hugging Face Whisper
librosa>=0.10.1           # Audio preprocessing
numpy>=1.24.3             # Numerical computing
soundfile>=0.12.1         # Audio file I/O
accelerate>=0.25.0        # Optimized model loading
```

### Validation

```
pydantic[email]>=2.5.0    # Data validation
email-validator>=2.1.0    # Email validation
```

### Testing

```
pytest>=7.4.3              # Testing framework
pytest-asyncio>=0.21.1     # Async test support
httpx>=0.25.2             # HTTP client for tests
websockets>=12.0          # WebSocket client
```

## 🔄 Request-Response Cycles

### 1. User Registration

```
POST /auth/register
Body: {username, email, password, full_name}
  │
  ▼
Validate input (Pydantic)
  │
  ▼
Check username/email uniqueness
  │
  ▼
Hash password (bcrypt)
  │
  ▼
Create user in DB (Prisma)
  │
  ▼
Return user data (no password)
Status: 201 CREATED
```

### 2. User Login

```
POST /auth/login
Body: {username, password}
  │
  ▼
Fetch user by username
  │
  ▼
Verify password (bcrypt.checkpw)
  │
  ▼
Generate JWT token
  │
  ▼
Return {access_token, token_type}
Status: 200 OK
```

### 3. WebSocket Chat Message

```
WS /ws/conversations/{id}?token=JWT
Send: {type: "message", content: "Hello", role: "user"}
  │
  ▼
Verify JWT token
  │
  ▼
Validate message format
  │
  ▼
Save message to DB
  │
  ▼
Broadcast to all connected clients
  │
  ▼
Receive: {type: "message", message_id, content, sender, timestamp}
```

### 4. Audio Transcription

```
WS /ws/v1/audio?token=JWT
Send: <binary audio chunk>
  │
  ▼
Verify JWT token
  │
  ▼
Add chunk to buffer
  │
  ▼
Wait for silence timeout (1.5s)
  │
  ▼
Preprocess audio (16kHz, mono, normalized)
  │
  ▼
Transcribe with Whisper
  │
  ▼
Receive: {type: "transcription", text, duration, timestamp}
```

## 🛠️ Development Workflow

### 1. Setup

```bash
# Clone repository
git clone <repo>

# Start services
docker-compose up --build

# Database migrations
docker exec ml_proj-backend-1 python -m prisma db push
```

### 2. Development

```bash
# Hot reload enabled in Docker
# Edit files locally, server auto-restarts

# View logs
docker logs ml_proj-backend-1 -f

# Access container
docker exec -it ml_proj-backend-1 bash
```

### 3. Testing

```bash
# Run test suite
docker exec ml_proj-backend-1 python test_websocket_chat.py

# Test audio transcription
docker exec ml_proj-backend-1 python test_audio_client.py -u user -p pass -a audio.wav

# Interactive chat
docker exec ml_proj-backend-1 python interactive_chat_client.py user pass
```

### 4. Production Deployment

```bash
# Build production image
docker build -t aura-backend:latest .

# Run with production settings
docker run -e DATABASE_URL=... -e SECRET_KEY=... aura-backend:latest

# Or use docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d
```

## 📝 Code Style & Standards

### Code Organization

- **Imports**: Standard library → Third-party → Local
- **Functions**: Type hints for parameters and returns
- **Classes**: Docstrings with Args, Returns, Raises
- **Constants**: UPPER_CASE
- **Private**: Leading underscore `_private_function()`

### Documentation

- All functions have docstrings
- Complex logic has inline comments
- README.md in each module
- Architecture diagrams in this file

### Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.info("User logged in: %s", username)
logger.error("Failed to process audio: %s", error)
logger.debug("Buffer size: %d bytes", len(buffer))
```

---

**Last Updated:** January 15, 2024  
**Version:** 1.1.0 (Week 4.2)  
**Status:** Production Ready ✅

**Week 4.2 Updates:**

- Added Speech Emotion Recognition (SER) with Wav2Vec2
- Implemented parallel STT + SER processing with asyncio.gather()
- Updated audio flow diagram to show multi-modal analysis
- Enhanced response format with transcript and emotion data
