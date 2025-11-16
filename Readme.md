# Aura ML — Multi-Modal AI Conversational System

An intelligent chat application combining real-time audio/video processing with advanced AI models for context-aware, emotionally intelligent conversations.

# Aura ML — Multi-Modal AI Conversational System

An intelligent chat application combining real-time audio/video processing with advanced AI models for context-aware, emotionally intelligent conversations.

## What It Does

- **Speech Recognition** → Convert audio to text (Whisper)
- **Emotion Detection** → Detect emotions from voice and text (Wav2Vec2, COMET)
- **Named Entities** → Extract people, places, dates (spaCy)
- **Contextual Reasoning** → Understand emotional context (COMET)
- **Knowledge Graph** → Build persistent conversation memory (Neo4j)
- **LLM Responses** → Generate intelligent replies (GPT-4)

## Quick Start

### Docker (Recommended)

```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
# Neo4j: http://localhost:7474
```

### Local Setup

```bash
# Backend
cd aura-backend
pip install -r requirements.txt
python main.py

# Frontend (in new terminal)
cd aura-frontend
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI + WebSocket
- **Database**: PostgreSQL (Prisma) + Neo4j
- **AI Models**: Whisper, Wav2Vec2, spaCy, COMET, GPT-4, LLaVA
- **Deployment**: Docker, Docker Compose

## Project Structure

```
aura-ml/
├── aura-backend/          # FastAPI server + ML pipeline
│   ├── main.py
│   ├── chat_orchestrator.py
│   ├── audio/             # STT, emotion recognition
│   ├── contextual/        # NER, COMET, knowledge graph
│   └── video/             # Video analysis (LLaVA, face detection)
├── aura-frontend/         # Streamlit UI
├── docker-compose.yml
└── README.md
```

## Key Features

✅ Real-time text and audio chat  
✅ Multi-modal video analysis  
✅ Emotion recognition from voice  
✅ Named entity extraction  
✅ Persistent knowledge graph  
✅ Context-aware LLM responses  
✅ Production UI with full transparency

## API Endpoints

| Endpoint                     | Method | Purpose                       |
| ---------------------------- | ------ | ----------------------------- |
| `/health`                    | GET    | Service health check          |
| `/orchestrate/analyze-audio` | POST   | Full ML pipeline              |
| `/transcribe`                | POST   | Speech-to-text                |
| `/recognize-emotion`         | POST   | Emotion from audio            |
| `/analyze/text`              | POST   | Entity extraction + reasoning |
| `/knowledge-graph/summary`   | GET    | Graph statistics              |
| `/ws/conversations/{id}`     | WS     | Real-time chat                |

## Documentation

- **[Backend README](aura-backend/README.md)** — API details, setup, troubleshooting
- **[Frontend README](aura-frontend/README.md)** — UI features, configuration

## Environment Setup

Create `.env` in `aura-backend/`:

```env
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost:5432/aura
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=aura_neo4j_pass
OPENAI_API_KEY=your-api-key
```

## Testing

```bash
cd aura-backend
pytest -q
```

## Troubleshooting

**Models fail to load?**  
Check disk space, CUDA version, internet connection

**Database error?**  
Verify `DATABASE_URL` and that PostgreSQL/Neo4j are running

**WebSocket issues?**  
Check CORS in `main.py` and frontend connection URL

---

**Status**: Week 8+ Complete ✅  
**Last Updated**: November 2025  
**See detailed docs in [backend](aura-backend/) and [frontend](aura-frontend/) folders.**
