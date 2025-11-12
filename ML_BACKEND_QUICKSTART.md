# 🚀 Aura ML Backend - Quick Start Guide

## Overview

This is the **simplified, ML-focused version** of the Aura backend with **authentication removed**. It provides a pure machine learning pipeline for conversational analysis.

## What's Inside

### ML Models & Services

1. **Speech-to-Text (STT)** - OpenAI Whisper
2. **Speech Emotion Recognition (SER)** - Wav2Vec2
3. **Named Entity Recognition (NER)** - spaCy
4. **Commonsense Reasoning** - COMET (AllenAI)
5. **Knowledge Graph** - Neo4j

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AUDIO INPUT                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 1: PARALLEL                          │
│  ┌──────────────────────┬──────────────────────┐       │
│  │  STT (Whisper)       │  SER (Wav2Vec2)      │       │
│  │  → Transcript        │  → Emotion           │       │
│  └──────────────────────┴──────────────────────┘       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 2: SEQUENTIAL                        │
│  ┌──────────────────────┬──────────────────────┐       │
│  │  NER (spaCy)         │  COMET (BART)        │       │
│  │  → Entities          │  → Commonsense       │       │
│  └──────────────────────┴──────────────────────┘       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              KNOWLEDGE GRAPH (Neo4j)                    │
│         Persistent conversational memory                │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Docker & Docker Compose** (for Neo4j)
- **Python 3.10+**
- **OpenAI API Key** (for advanced features, optional)

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone/navigate to project
cd /path/to/ML_Proj

# 2. Set environment variables
export OPENAI_API_KEY="your-api-key-here"  # Optional

# 3. Start services
docker-compose up -d

# 4. Check health
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
# 1. Install dependencies
cd aura-backend
pip install -r requirements.txt

# 2. Start Neo4j separately
docker run -d \
  --name aura-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4jpassword \
  neo4j:5.18-community

# 3. Start backend
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing the API

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Expected Output:**

```json
{
  "status": "healthy",
  "services": {
    "transcription": true,
    "emotion_recognition": true,
    "contextual_analysis": true,
    "chat_orchestrator": true
  },
  "timestamp": "2025-11-13T..."
}
```

### 2. Transcribe Audio

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample_audio.wav"
```

**Expected Output:**

```json
{
  "text": "I'm meeting Sarah at the coffee shop tomorrow",
  "language": "en",
  "duration": 3.5,
  "timestamp": "2025-11-13T..."
}
```

### 3. Recognize Emotion

```bash
curl -X POST http://localhost:8000/recognize-emotion \
  -F "file=@sample_audio.wav"
```

**Expected Output:**

```json
{
  "dominant_emotion": "neutral",
  "confidence": 0.85,
  "all_emotions": [
    { "label": "neutral", "score": 0.85 },
    { "label": "happy", "score": 0.1 },
    { "label": "sad", "score": 0.05 }
  ]
}
```

### 4. Analyze Text

```bash
curl -X POST "http://localhost:8000/analyze/text" \
  -G \
  --data-urlencode "text=I am meeting Sarah in Mumbai tomorrow" \
  --data-urlencode "conversation_id=test123"
```

**Expected Output:**

```json
{
  "entities": {
    "people": [{"text": "Sarah", "start": 15, "end": 20}],
    "places": [{"text": "Mumbai", "start": 24, "end": 30}],
    "dates": [{"text": "tomorrow", "start": 31, "end": 39}]
  },
  "emotional_context": {
    "xReact": ["interested", "hopeful"],
    "xWant": ["to meet friend"],
    ...
  },
  "graph_updates": {
    "nodes_created": 3,
    "relationships_created": 3
  }
}
```

### 5. Complete ML Pipeline

```bash
curl -X POST "http://localhost:8000/orchestrate/analyze-audio?conversation_id=test123" \
  -F "file=@sample_audio.wav"
```

**Expected Output:**

```json
{
  "transcript": {
    "text": "I'm meeting Sarah at the coffee shop in Mumbai tomorrow",
    "language": "en"
  },
  "emotion": {
    "from_audio": {
      "primary": "neutral",
      "confidence": 0.85
    },
    "from_text": {
      "detected": ["hopeful", "excited"]
    }
  },
  "entities": {
    "people": [{"text": "Sarah", ...}],
    "places": [{"text": "Mumbai", ...}],
    "dates": [{"text": "tomorrow", ...}]
  },
  "commonsense": {
    "inferences": {
      "xReact": ["interested", "hopeful"],
      "oReact": ["receptive"],
      "xWant": ["to meet friend", "to have coffee"],
      "oWant": ["to engage"],
      "xEffect": ["gains knowledge"],
      "oEffect": ["receives message"]
    }
  },
  "processing": {
    "total_time_ms": 650,
    "all_models_completed": true
  }
}
```

### 6. Knowledge Graph Query

```bash
# Get conversation context
curl http://localhost:8000/analyze/conversation/test123

# Get graph summary
curl http://localhost:8000/knowledge-graph/summary

# Export graph
curl http://localhost:8000/knowledge-graph/export?format=json
```

## Automated Testing

Run the comprehensive test suite:

```bash
./test_ml_backend.sh
```

This will test all endpoints and display results.

## Interactive API Documentation

Visit http://localhost:8000/docs for Swagger UI with interactive API testing.

## Key Endpoints

| Endpoint                     | Method | Description                    |
| ---------------------------- | ------ | ------------------------------ |
| `/`                          | GET    | API information                |
| `/health`                    | GET    | Service health check           |
| `/models/status`             | GET    | Detailed model status          |
| `/transcribe`                | POST   | Audio transcription (Whisper)  |
| `/recognize-emotion`         | POST   | Emotion recognition (Wav2Vec2) |
| `/analyze/text`              | POST   | Text analysis (NER + COMET)    |
| `/orchestrate/analyze-audio` | POST   | **Complete ML pipeline**       |
| `/analyze/conversation/{id}` | GET    | Get conversation context       |
| `/knowledge-graph/summary`   | GET    | Graph statistics               |
| `/knowledge-graph/export`    | GET    | Export graph data              |

## Configuration

### Environment Variables

Create `.env` file in `aura-backend/`:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpassword

# Optional: OpenAI API Key (for LLM features)
OPENAI_API_KEY=sk-...
```

## File Structure

```
aura-backend/
├── main.py                    # 🆕 Simplified ML-only API
├── main_backup.py            # Backup of auth version
├── chat_orchestrator.py      # ML pipeline coordinator
├── audio/
│   ├── __init__.py
│   ├── transcription.py      # Whisper STT
│   └── emotion.py            # Wav2Vec2 SER
├── contextual/
│   ├── __init__.py
│   ├── ner.py                # spaCy NER
│   ├── comet_service.py      # COMET reasoning
│   └── knowledge_graph.py    # Neo4j integration
├── requirements.txt
└── Dockerfile
```

## Performance Notes

### Model Loading Times

- **Whisper (base)**: ~2-5 seconds
- **Wav2Vec2**: ~3-7 seconds
- **spaCy**: ~1-2 seconds
- **COMET**: ~10-15 seconds (first time)

### Processing Times (per audio file)

- **STT**: 100-500ms (depends on audio length)
- **SER**: 50-200ms
- **NER**: 10-50ms
- **COMET**: 100-300ms
- **Total Pipeline**: 500-1000ms

## Troubleshooting

### Models Not Loading

```bash
# Check logs
docker-compose logs aura-backend

# Or if running locally
tail -f backend.log
```

### Neo4j Connection Issues

```bash
# Check Neo4j status
docker ps | grep neo4j

# Check Neo4j logs
docker logs aura-neo4j

# Restart Neo4j
docker restart aura-neo4j
```

### Audio File Errors

- Ensure audio is in a supported format (WAV, MP3, etc.)
- Check audio sample rate (16kHz recommended)
- Verify file is not corrupted

## Development

### Adding New Features

The modular structure makes it easy to add new ML models:

1. Create new service in appropriate directory
2. Add initialization in `main.py` startup
3. Create endpoint using FastAPI
4. Update health check

### Hot Reload

Run with auto-reload during development:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## What Changed from Previous Version?

### ❌ Removed

- Authentication system (JWT, passwords)
- User management
- PostgreSQL database
- Protected endpoints
- WebSocket authentication

### ✅ Kept

- All ML models (Whisper, Wav2Vec2, spaCy, COMET)
- Knowledge graph (Neo4j)
- ML pipeline orchestrator
- All analysis endpoints

See [AUTH_REMOVAL_SUMMARY.md](./docs/AUTH_REMOVAL_SUMMARY.md) for complete details.

## Production Considerations

**⚠️ This version has NO authentication!**

For production deployment:

1. Add API key authentication
2. Implement rate limiting
3. Set up proper CORS policies
4. Use HTTPS/TLS
5. Add monitoring and logging
6. Configure firewall rules

## Next Steps

1. ✅ Test all endpoints with `./test_ml_backend.sh`
2. 📖 Read API docs at `/docs`
3. 🧪 Experiment with different audio files
4. 📊 Explore knowledge graph in Neo4j browser (http://localhost:7474)
5. 🔧 Customize models and pipelines

## Resources

- **Swagger UI**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474
- **GitHub Issues**: Report bugs and request features

## Support

For questions or issues:

1. Check `/health` endpoint for service status
2. Review logs: `docker-compose logs aura-backend`
3. Consult documentation in `docs/` directory

---

**Version:** 2.0.0 (ML-Only)  
**Status:** ✅ Production-ready for research/demo use  
**License:** MIT
