# Aura Backend — ML Pipeline & Real-time Chat API

## Overview

This repository contains the Aura backend: a FastAPI-based service that implements a unified ML pipeline and real-time chat support. The service focuses on multi-modal conversational analysis and includes:

- Speech-to-Text (STT) using Whisper
- Speech Emotion Recognition (SER) using Wav2Vec2
- Named Entity Recognition (NER) with spaCy
- Commonsense emotional reasoning (COMET)
- A Knowledge Graph service (Neo4j integration via contextual services)
- A Chat Orchestrator that runs the full ML pipeline and returns an aggregated JSON analysis
- WebSocket connection management for real-time chat

This README documents how to run and develop the backend located at `aura-backend/`.

## Key Files

- `main.py` — FastAPI application and HTTP endpoints (health, transcription, emotion recognition, analysis, orchestrator)
- `chat_orchestrator.py` — Coordinates STT, SER, NER, COMET and builds aggregated responses
- `websocket_manager.py` — `ConnectionManager` for WebSocket lifecycle, broadcasts and active-user tracking
- `audio/transcription.py` — `TranscriptionService` (Whisper wrapper)
- `audio/transcription.py` & other `audio/*` — audio preprocessing and transcription helpers
- `contextual/*` — contextual analysis (NER, COMET, knowledge-graph services)
- `interactive_chat_client.py` — example/test client for WebSocket interactions
- `auth.py`, `database.py`, `db/*` — authentication helpers, Prisma DB helpers and CRUD utilities
- `Dockerfile` — image definition for containerized deployments
- `.env.example` — sample environment variables

## Features & Architecture

- Startup loads ML models (Whisper, Wav2Vec2, spaCy, COMET). If models fail to load, services gracefully degrade and `/health` reports status.
- `ChatOrchestrator` runs STT and SER in parallel on audio, then performs textual contextual analysis (NER + COMET), and optionally updates the knowledge graph.
- Real-time chat uses WebSockets managed by `ConnectionManager` to handle user lifecycle, broadcasting, and system messages.
- Database CRUD operations and user management use Prisma (see `db/prisma.py` and `db/crud.py`).

## Main HTTP Endpoints (summary)

- `GET /` — Basic service information and list of available endpoints.
- `GET /health` — Health status showing whether transcription, emotion, contextual services, and orchestrator are loaded.
- `POST /transcribe` — Upload an audio file (WAV/MP3/etc.) and receive Whisper transcription.
- `POST /recognize-emotion` — Upload audio file and receive emotion detection results.
- `POST /analyze/text` — Submit text for NER + COMET analysis and optional knowledge-graph updates.
- `GET /analyze/conversation/{conversation_id}` — Fetch accumulated context for a conversation (queries knowledge graph).
- `GET /knowledge-graph/summary` — Get summary statistics for the graph (node/relationship counts by type).
- `GET /knowledge-graph/export?format=json` — Export full graph data.
- `POST /orchestrate/analyze-audio` — Unified ML pipeline: upload audio and receive aggregated response including transcript, audio/text emotions, entities, commonsense inferences, graph updates, and timing metrics.
- `GET /models/status` — Detailed status of the loaded ML models and capabilities.
- `POST /test/echo` — Simple echo endpoint used for testing.

Note: WebSocket endpoints are used for real-time chat. See `websocket_manager.py` and `interactive_chat_client.py` for usage patterns. Typical WebSocket endpoints used by the frontend include `/ws/conversations/{conversation_id}` and `/ws/v1/audio` (audio streaming); confirm exact route definitions in your deployed app if routes are moved.

## Quickstart — Local (development)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy environment variables:

```powershell
copy .env.example .env
# Then edit .env as needed (SECRET_KEY, DATABASE_URL, NEO4J_URL, etc.)
```

4. Run the API locally (the project includes a `__main__` entry in `main.py` which runs uvicorn):

```powershell
python main.py
# or, using uvicorn directly for hot-reload during development:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. Visit API docs: `http://localhost:8000/docs` (OpenAPI UI)

## Quickstart — Docker

Build and run the Docker image for the backend:

```powershell
docker build -t aura-backend -f Dockerfile .
docker run --rm -p 8000:8000 --env-file .env aura-backend
```

(There is a root-level `docker-compose.yml` in the project that orchestrates backend, frontend and Neo4j; see repository root docs.)

## Environment Variables

Important variables are provided in `.env.example`. Common keys:

- `SECRET_KEY` — JWT secret for auth (change in production)
- `ALGORITHM` — JWT algorithm (default HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES` — token expiry
- `DATABASE_URL` — Prisma/Postgres connection string
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — Neo4j connection details

Ensure GPU drivers and CUDA are configured if you intend to run model inference on GPU.

## Development Notes

- Model loading occurs on FastAPI startup (`@app.on_event('startup')`). If a model fails to load the service will still run but corresponding endpoints will return 503 until the model is available.
- The `TranscriptionService` uses `openai/whisper-tiny` by default (fast inference). Change the model string in `audio/transcription.py` to use a larger model.
- The Chat Orchestrator (`chat_orchestrator.py`) executes STT and SER in parallel, then runs contextual analysis — helpful when optimizing latencies.
- WebSocket behavior and concurrency are implemented in `websocket_manager.py` (ConnectionManager) — it handles connection lifecycle, broadcasting and active user lists.

## Testing

There are test scripts in the backend folder such as:

- `test_integrated_analysis.py` — integrated pipeline tests
- `test_scene_captioner.py` — video/scene tests (if video components are used)

Run tests with pytest:

```powershell
pip install -r requirements.txt
pytest -q
```

## Troubleshooting

- If model loading fails, check logs printed at startup — models may fail due to missing weights, network issues, or insufficient memory.
- For GPU-related issues ensure `torch` is installed with matching CUDA version for your drivers.
- If Prisma/DB fails to connect, verify `DATABASE_URL` and that Postgres is up (or run with Docker Compose from repo root).
- If WebSocket connections immediately close, ensure the frontend is connecting to the correct WS route and that CORS and allowed origins are configured properly.

## Next Steps & Notes

- Confirm exact WebSocket route definitions in your deployment and adapt frontend accordingly (`interactive_chat_client.py` is a good reference for usage).
- Consider preloading models in a separate process or container when moving to production to avoid long startup times in the API container.
- Add healthchecks and readiness probes in your container orchestration (Kubernetes/Docker Compose) to ensure dependent services (Neo4j, Postgres) are available before heavy operations run.

---

If you'd like, I can:

- Add example `curl` requests for the major endpoints,
- Add a small `dev.sh` / `dev.ps1` that runs the backend and tail logs,
- Or run the test suite and share any failing tests.

File created: `aura-backend/README.md`
