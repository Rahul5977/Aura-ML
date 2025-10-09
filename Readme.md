# Project Aura: End-to-End AI Workflow

Project Aura is an ambitious endeavor to create an empathetic and context-aware AI system that processes multi-modal user input (audio and video streams) to generate intelligent responses. This project integrates various AI/ML models for feature extraction, contextual analysis, and advanced generative AI, all orchestrated through a robust backend and presented via a dynamic frontend.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Workflow](#workflow)
  - [Phase 0: Pre-Development Setup](#phase-0-pre-development-setup-before-week-1)
  - [Phase 1: Foundation & Backend Architecture](#phase-1-foundation--backend-architecture-weeks-1-3)
  - [Phase 2: Core AI/ML Integration & Prototyping](#phase-2-core-aiml-integration--prototyping-weeks-4-7)
  - [Phase 3: Advanced AI Development & Fine-Tuning](#phase-3-advanced-ai-development--fine-tuning-weeks-8-11)
  - [Phase 4: Frontend Development](#phase-4-frontend-development-weeks-12-14)
  - [Phase 5: Finalization & Deployment](#phase-5-finalization--deployment-weeks-15-16)
- [Key Technologies](#key-technologies)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Suggested Improvements](#suggested-improvements)
- [Team Tasks](#team-tasks)
- [References](#references)

---

## Project Structure

The project is organized as a monorepo with the following key subfolders:

```
aura/
├── aura-backend/         # FastAPI backend
├── aura-frontend/        # React frontend
├── ml_scripts/           # Training & experimentation scripts
├── infra/                # Deployment configs, Docker, CI/CD
├── docs/                 # Documentation, diagrams, papers
├── tests/                # End-to-end & integration tests
└── README.md
```

---

## Workflow

### Phase 0: Pre-Development Setup (Before Week 1)

**Goal:** Ensure everyone on the team can quickly onboard and follow the same development standards.

**Key Tasks:**

- Project Repo Setup: Create a monorepo, set up Git branching strategy, add .gitignore, CONTRIBUTING.md, CODE_OF_CONDUCT.md.
- Dev Environment Standardization: Define requirements.txt/pyproject.toml, use nvm/.node-version, configure pre-commit hooks, set up .editorconfig.
- CI/CD Bootstrap: Configure GitHub Actions/GitLab CI for linting, type-checking, and running tests.

**Milestone:** Any new developer can clone the repo, run `docker-compose up`, and see backend/frontend skeletons running.

---

### Phase 1: Foundation & Backend Architecture (Weeks 1-3)

#### Week 1: Project Setup & Containerization

- Initialize Git, define branching strategy.
- Set up top-level project structure.
- Create docker-compose.yml for backend, frontend, and PostgreSQL.
- Build Dockerfile for FastAPI.
- Initialize FastAPI with a `/health` endpoint.
- Add Makefile for common tasks.

**Milestone:** `docker-compose up` runs backend+frontend+db, `/health` endpoint returns "OK".

#### Week 2: User Authentication & Database

- Finalize database schema (User, Conversation, Message, MemorySummary).
- Implement user registration/login with JWT.
- Create secure API endpoints.
- Write unit tests for authentication.
- Add Alembic for migrations.

**Milestone:** User can register, log in to receive JWT, and access protected API route.

#### Week 3: Real-time Communication & Chat Logic

- Implement WebSocket endpoint in FastAPI.
- Develop backend logic for message exchange via WebSocket.
- Handle connection/disconnection events.
- (Optional: Add Redis for scalable pub/sub).

**Milestone:** Basic text-based chat application with real-time message exchange.

---

### Phase 2: Core AI/ML Integration & Prototyping (Weeks 4-7)

#### Week 4: Multi-Modal Input Processing (Audio)

- Create endpoint for audio data over WebSocket.
- Integrate Speech-to-Text (Whisper) and Speech Emotion Recognition (SER) models.
- Use librosa for audio preprocessing.

**Milestone:** Backend receives audio stream and returns transcribed text and detected emotion.

#### Week 5: Initial Contextual Analysis

- Integrate COMET for commonsense emotional effects.
- Integrate Named Entity Recognition (NER) model (spaCy).
- Build Dynamic Knowledge Graph service.

**Milestone:** Backend enriches text with entities and commonsense inferences.

#### Week 6: Custom Model Data Preparation

- Data cleaning/preprocessing scripts for ESConv and ECF datasets.
- Train baseline Strategy Predictor model (XGBoost).
- Version datasets using DVC.

**Milestone:** Clean, versioned datasets ready for training.

#### Week 7: Backend AI Orchestration

- Design and implement chat_orchestrator service in FastAPI.
- Aggregate outputs into a single "analysis packet".

**Milestone:** Single function call triggers analysis pipeline and produces comprehensive JSON object.

---

### Phase 3: Advanced AI Development & Fine-Tuning (Weeks 8-11)

#### Week 8: Advanced Video Feature Extraction

- Integrate LLaVA for video frame descriptions.
- Implement face analysis pipeline (MTCNN, MobileFaceNets, VGG19).

**Milestone:** Backend processes video stream for facial emotion features and captions.

#### Week 9: Custom Model Fine-Tuning (ECE)

- Implement and fine-tune Emotion Cause Extraction (ECE) module.
- Generate "Hyper-Contextual Prompt" dataset.

**Milestone:** Trained ECE model saved and benchmarked.

#### Week 10: GenAI Fine-Tuning (LLM)

- Finalize "Hyper-Contextual Prompt" dataset.
- Run LoRA fine-tuning on base LLM (e.g., Llama 3 8B).

**Milestone:** Specialized, fine-tuned LLM for emotional support created and saved.

#### Week 11: Final AI Integration & Memory System

- Integrate ECE and fine-tuned LLM into chat_orchestrator.
- Implement Persistent Memory service.

**Milestone:** Backend is feature-complete with advanced analysis, memory, and context-aware responses.

---

### Phase 4: Frontend Development (Weeks 12-14)

#### Week 12: UI Foundation & Real-time Connection

- Set up React project (Vite), state management.
- Build core chat UI components.
- Implement WebSocket client.

**Milestone:** User can send/receive messages in real-time.

#### Week 13: Advanced UI Features

- Capture audio/video and stream over WebSocket.
- Build XAI dashboard for reasoning display.

**Milestone:** User can speak to app, XAI dashboard displays analysis.

#### Week 14: Visualization & Polish

- Integrate graph visualization for Dynamic Knowledge Graph.
- Add loading indicators, error messages, refine styling.

**Milestone:** Frontend is feature-complete and polished.

---

### Phase 5: Finalization & Deployment (Weeks 15-16)

#### Week 15: End-to-End Testing & Deployment

- Conduct end-to-end testing (pytest + Playwright).
- Fix bugs, optimize performance.
- Write deployment scripts, deploy full stack.
- Conduct load testing (locust).
- Set up Nginx reverse proxy and HTTPS.

**Milestone:** Project "Aura" is live and stable.

#### Week 16: Documentation & Presentation

- Write final report in LaTeX.
- Record video demo.
- Prepare final presentation.

**Milestone:** Project is fully documented and submitted.

---

## Key Technologies

- **Backend:** FastAPI, PostgreSQL, SQLAlchemy, Docker, Docker Compose, Redis (optional)
- **Frontend:** React, Vite, socket.io-client, CSS
- **AI/ML:** Hugging Face transformers (Whisper, SER), spaCy (NER), COMET, LLaVA VLM, MTCNN, MobileFaceNets, VGG19, PyTorch, LoRA, peft, bitsandbytes, accelerate, pandas, scikit-learn, XGBoost
- **DevOps:** Git, GitHub Actions/GitLab CI, DVC, Nginx, Let's Encrypt, AWS/GCP/Hugging Face Spaces
- **Testing:** pytest, Playwright, locust
- **Documentation:** LaTeX

---

## Setup and Installation

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd aura
   ```

2. **Set up development environment:**  
   Follow instructions in `CONTRIBUTING.md` for pre-commit hooks and environment variables.

3. **Run with Docker Compose:**
   ```sh
   docker-compose up --build
   ```
   This will bring up the backend, frontend, and PostgreSQL database.

---

## Usage

Once the Docker containers are running, you can access:

- **FastAPI Backend:** [http://localhost:8000](http://localhost:8000)
- **React Frontend:** [http://localhost:3000](http://localhost:3000)

See the `docs/` folder for detailed API usage and frontend interaction.

---

## Suggested Improvements

- **File/Folder Structure:** Define early for scalability.
- **CI/CD Pipelines:** Integrate from Week 1 (lint, test, build).
- **Artifact Management:** Use DVC for datasets and model versions.
- **Config Management:** Store secrets in `.env` and use Pydantic BaseSettings.
- **Scalability Prep:** Plan for Redis pub/sub in chat, GPU inference via microservices.
- **Security:** Enforce HTTPS, JWT refresh tokens, rate limiting.
- **Monitoring:** Add logging (structlog), metrics (Prometheus + Grafana).

---

## Team Tasks

**Rahul (Deadline: Thursday EOD)**

- Implement Week 1 and Week 2 tasks.
- Backend server development.
- Research models mentioned in workflow.
- Create ChatLogic (WebSocket logic).

**Rishi (Deadline: Thursday EOD)**

- Research rationale behind chosen models, alternatives, and usage.
- Understand datasets used in project.
- Integrate COMET and pre-trained NER model (spaCy).
- Explore ESConv and ECF datasets (cleaning if needed).

---

## References

- [Transformer-based-SER](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)
- [Unbabel COMET](https://github.com/Unbabel/COMET)
- [arXiv:2504.15681](https://arxiv.org/abs/2504.15681)
- [RECCON - dataset for ECE/ECPE](https://github.com/declare-lab/RECCON)
- [thu-coai/esconv dataset](https://github.com/thu-coai/ESConv)
