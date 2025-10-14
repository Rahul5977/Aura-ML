# Aura Project Structure

Clean, organized, and production-ready folder structure.

## 📁 Root Directory

\`\`\`
aura/
├── aura-backend/          # FastAPI backend (Python)
├── aura-frontend/         # React frontend (placeholder)
├── ml_scripts/            # ML/data science scripts
│
├── docker-compose.yml     # Docker orchestration
│
├── Readme.md             # Project overview (start here!)
├── DOCUMENTATION_INDEX.md # All documentation (navigation hub)
│
├── WEEK4_1_COMPLETION_SUMMARY.md      # Week 4.1 summary
├── WEEK4_1_AUDIO_TRANSCRIPTION.md     # Audio feature docs
├── AUDIO_QUICKSTART.md                # Audio quick start (5 min)
├── WEBSOCKET_QUICKSTART.md            # Chat quick start (5 min)
├── ARCHITECTURE.md                    # System architecture
├── WEEK3_COMPLETION_REPORT.md         # Week 3 history
│
└── ...                    # Other config files
\`\`\`

**Total**: 7 documentation files (~70KB)

---

## 🔧 Backend Structure (aura-backend/)

\`\`\`
aura-backend/
│
├── 📦 CORE APPLICATION
│   ├── main.py                    # FastAPI app (entry point)
│   ├── auth.py                    # Authentication (JWT, bcrypt)
│   ├── database.py                # Database ops (Prisma ORM)
│   ├── schemas.py                 # Pydantic data models
│   └── websocket_manager.py       # WebSocket connection manager
│
├── 🎤 AUDIO MODULE (Week 4.1)
│   └── audio/
│       ├── __init__.py           # Module exports
│       ├── audio_utils.py        # Audio preprocessing (librosa)
│       ├── buffer_manager.py     # Per-client buffering
│       ├── transcription.py      # Whisper STT service
│       └── README.md             # Audio module docs (13KB)
│
├── 🧪 TEST CLIENTS
│   ├── test_audio_client.py      # Audio streaming client
│   ├── test_websocket_chat.py    # WebSocket test suite
│   └── interactive_chat_client.py # Interactive chat client
│
├── 🗄️ DATABASE
│   └── schema.prisma             # Database schema (PostgreSQL)
│
├── ⚙️ CONFIGURATION
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile               # Container configuration
│   ├── .env                     # Environment variables
│   ├── .env.example             # Example configuration
│   └── pytest.ini               # Test configuration
│
└── 📁 OTHER
    └── tests/                    # Test directory
        └── __init__.py
\`\`\`

---

## 🗺️ Module Map

### Entry Point Flow

\`\`\`
main.py (FastAPI)
  │
  ├─► auth.py          → JWT authentication
  ├─► database.py      → Prisma database operations
  ├─► schemas.py       → Data validation (Pydantic)
  ├─► websocket_manager.py → Chat connections
  │
  └─► audio/           → Audio transcription module
      ├─► audio_utils.py      → Preprocessing
      ├─► buffer_manager.py   → Buffering
      └─► transcription.py    → Whisper STT
\`\`\`

### Test Clients

\`\`\`
test_audio_client.py           → Tests /ws/v1/audio endpoint
test_websocket_chat.py         → Tests /ws/conversations/{id} endpoint
interactive_chat_client.py     → Manual testing tool
\`\`\`

---

## 📊 File Statistics

### Code Files

| Type | Count | Total Lines |
|------|-------|-------------|
| Python modules | 8 | ~1,800 |
| Test clients | 3 | ~700 |
| Configuration | 5 | ~200 |
| **Total** | **16** | **~2,700** |

### Documentation Files

| File | Size | Purpose |
|------|------|---------|
| WEEK4_1_AUDIO_TRANSCRIPTION.md | 18KB | Audio feature docs |
| ARCHITECTURE.md | 21KB | System architecture |
| audio/README.md | 13KB | Audio module API |
| AUDIO_QUICKSTART.md | 9KB | Quick start guide |
| WEEK4_1_COMPLETION_SUMMARY.md | 12KB | Week 4.1 summary |
| WEBSOCKET_QUICKSTART.md | 6KB | Chat quick start |
| DOCUMENTATION_INDEX.md | 10KB | Navigation hub |
| **Total** | **~70KB** | **Complete docs** |

---

## 🎯 Key Components

### 1. FastAPI Application (main.py)
- REST API endpoints (auth, conversations, messages)
- WebSocket chat endpoint
- WebSocket audio endpoint
- Startup/shutdown lifecycle

### 2. Authentication (auth.py)
- JWT token generation/validation
- Password hashing (bcrypt)
- Token expiry management

### 3. Database (database.py + schema.prisma)
- Prisma ORM integration
- User, Conversation, Message models
- CRUD operations
- PostgreSQL backend

### 4. WebSocket Manager (websocket_manager.py)
- Connection pool management
- Message broadcasting
- Per-conversation rooms
- Active user tracking

### 5. Audio Module (audio/)
- Audio preprocessing pipeline
- Per-client buffer management
- Whisper speech-to-text
- Async transcription

---

## 🔗 Dependencies

### Production

\`\`\`txt
Core Web:
- fastapi>=0.104.1
- uvicorn[standard]>=0.24.0

Database:
- prisma>=0.11.0
- asyncpg>=0.29.0

Auth:
- python-jose[cryptography]>=3.3.0
- bcrypt>=4.1.2

ML/Audio:
- torch>=2.1.1
- transformers>=4.35.2
- librosa>=0.10.1
- numpy>=1.24.3
\`\`\`

### Development

\`\`\`txt
Testing:
- pytest>=7.4.3
- pytest-asyncio>=0.21.1
- httpx>=0.25.2
- websockets>=12.0
\`\`\`

**Total**: 43 dependencies

---

## 🏗️ Architecture Highlights

### Modular Design

\`\`\`
┌─────────────────────────────────────┐
│           main.py                   │
│         (FastAPI App)               │
└───┬────────┬────────┬────────┬─────┘
    │        │        │        │
    ▼        ▼        ▼        ▼
┌──────┐ ┌─────┐ ┌────────┐ ┌───────┐
│ auth │ │ db  │ │schemas │ │ audio │
└──────┘ └─────┘ └────────┘ └───────┘
    │        │                   │
    ▼        ▼                   ▼
┌──────┐ ┌──────────┐      ┌─────────┐
│ JWT  │ │PostgreSQL│      │ Whisper │
└──────┘ └──────────┘      └─────────┘
\`\`\`

### Clean Separation

- **Core Logic**: main.py, auth.py, database.py
- **Data Models**: schemas.py, schema.prisma
- **Real-time**: websocket_manager.py
- **Audio**: audio/ module (isolated)
- **Testing**: test_*.py files
- **Config**: .env, requirements.txt

---

## ✅ What's Included

### Features

- ✅ User authentication (JWT)
- ✅ Real-time chat (WebSocket)
- ✅ Audio transcription (Whisper)
- ✅ Database persistence (PostgreSQL)
- ✅ Docker containerization

### Testing

- ✅ Integration test suite
- ✅ Audio streaming client
- ✅ Interactive chat client
- ✅ Manual testing tools

### Documentation

- ✅ Quick start guides (2)
- ✅ Full feature docs (2)
- ✅ Architecture guide (1)
- ✅ Module API docs (1)
- ✅ Navigation index (1)

---

## 🚫 What's NOT Included (Cleaned Up)

### Removed Legacy Code

- ❌ /api folder (unused API routes)
- ❌ /core folder (duplicate auth code)
- ❌ /db folder (duplicate database code)
- ❌ /schemas folder (duplicate Pydantic models)
- ❌ /models folder (empty directory)
- ❌ /utils folder (empty directory)
- ❌ *.bak backup files
- ❌ Old test files (test_auth.py)

**Result**: Clean, maintainable codebase with clear structure

---

## 📈 Project Evolution

### Week 1: Foundation
- Database design
- Docker setup
- Prisma ORM

### Week 2: Authentication
- JWT implementation
- User management
- API endpoints

### Week 3: Real-time Chat
- WebSocket implementation
- Connection management
- Message broadcasting

### Week 4.1: Audio Transcription ✅
- WebSocket audio endpoint
- Audio preprocessing pipeline
- Whisper integration
- Per-client buffering
- **Code cleanup and documentation**

### Next: Week 4.2
- LLM integration
- AI response generation
- Voice Activity Detection

---

## 🎓 Learning the Codebase

### 5-Minute Tour

1. Read [Readme.md](Readme.md) - Overview
2. Look at \`aura-backend/main.py\` - Entry point
3. Check \`aura-backend/audio/\` - Audio module
4. Run test client - See it work

### 30-Minute Deep Dive

1. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
2. \`aura-backend/main.py\` - All endpoints
3. \`aura-backend/audio/\` - Audio implementation
4. [WEEK4_1_AUDIO_TRANSCRIPTION.md](WEEK4_1_AUDIO_TRANSCRIPTION.md) - Full docs

### Contributing

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) - Design patterns
2. Check [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Find docs
3. Review \`requirements.txt\` - Dependencies
4. Run tests - Ensure everything works

---

## 🎯 Navigation Shortcuts

### Want to...

**Get started quickly?**
→ [AUDIO_QUICKSTART.md](AUDIO_QUICKSTART.md)

**Understand the system?**
→ [ARCHITECTURE.md](ARCHITECTURE.md)

**Integrate audio?**
→ [WEEK4_1_AUDIO_TRANSCRIPTION.md](WEEK4_1_AUDIO_TRANSCRIPTION.md)

**Find a specific doc?**
→ [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

**See what's complete?**
→ [WEEK4_1_COMPLETION_SUMMARY.md](WEEK4_1_COMPLETION_SUMMARY.md)

**Test the chat?**
→ [WEBSOCKET_QUICKSTART.md](WEBSOCKET_QUICKSTART.md)

---

**Last Updated**: October 12, 2025  
**Status**: Week 4.1 Complete ✅  
**Structure**: Clean and Production-Ready 🎉
