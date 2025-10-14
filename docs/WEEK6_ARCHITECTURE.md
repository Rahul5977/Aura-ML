# Week 6: Chat Orchestrator Architecture

## System Overview

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    AURA CHAT ORCHESTRATOR                      ┃
┃                  Unified AI Pipeline (Week 6)                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌──────────────────────────────────────────────────────────────┐
│                        CLIENT                                 │
│  (Web App, Mobile App, CLI Tool)                             │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ HTTP POST /orchestrate/analyze-audio
                     │ Authorization: Bearer <token>
                     │ Content-Type: multipart/form-data
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVER                            │
│                   (main.py - Port 8000)                       │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  POST /orchestrate/analyze-audio                      │   │
│  │  ↓                                                     │   │
│  │  1. Validate authentication (JWT token)               │   │
│  │  2. Read audio file from upload                       │   │
│  │  3. Call chat_orchestrator.process_audio()            │   │
│  │  4. Return aggregated JSON response                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│              CHAT ORCHESTRATOR SERVICE                        │
│            (chat_orchestrator.py)                             │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ChatOrchestrator.process_audio()                     │   │
│  │                                                        │   │
│  │  Pipeline Execution:                                  │   │
│  │                                                        │   │
│  │  PHASE 1: Parallel Audio Processing                   │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │   run_stt()         │    run_ser()             │  │   │
│  │  │   ↓                 │    ↓                      │  │   │
│  │  │   STT Service       │    SER Service           │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                                                        │   │
│  │  PHASE 2: Sequential Text Processing                  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │   run_contextual_analysis()                    │  │   │
│  │  │   ↓                                            │  │   │
│  │  │   Contextual Analyzer                          │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                                                        │   │
│  │  PHASE 3: Response Aggregation                        │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │   _build_response()                            │  │   │
│  │  │   - Merge all results                          │  │   │
│  │  │   - Calculate metrics                          │  │   │
│  │  │   - Format JSON                                │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└────┬───────────────┬────────────────────────┬────────────────┘
     │               │                        │
     ↓               ↓                        ↓
┌─────────┐   ┌──────────┐          ┌────────────────────┐
│   STT   │   │   SER    │          │   CONTEXTUAL       │
│ Service │   │ Service  │          │   ANALYZER         │
└─────────┘   └──────────┘          └────────────────────┘
     │               │                        │
     ↓               ↓                        ↓
┌─────────┐   ┌──────────┐          ┌─────────┬──────────┐
│ Whisper │   │ Wav2Vec2 │          │   NER   │  COMET   │
│  Model  │   │  Model   │          │ Service │ Service  │
└─────────┘   └──────────┘          └─────────┴──────────┘
                                            │
                                            ↓
                                    ┌───────────────┐
                                    │  KNOWLEDGE    │
                                    │  GRAPH        │
                                    │  SERVICE      │
                                    └───────────────┘
```

## Detailed Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT                               │
│                    (WAV, MP3, etc.)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ audio_bytes
                         │
                         ↓
    ╔════════════════════════════════════════════════════════════╗
    ║              PHASE 1: PARALLEL PROCESSING                  ║
    ║              (Both models need audio)                      ║
    ╚════════════════════════════════════════════════════════════╝
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ↓                           ↓
    ┌─────────────┐            ┌─────────────┐
    │  STT Model  │            │  SER Model  │
    │  (Whisper)  │            │ (Wav2Vec2)  │
    │             │            │             │
    │ 400-600ms   │            │ 100-200ms   │
    └──────┬──────┘            └──────┬──────┘
           │                           │
           │ transcript                │ emotion
           │                           │
           └─────────────┬─────────────┘
                         │
                         ↓
               ┌─────────────────┐
               │  Wait for both  │
               │    to complete  │
               │  (max: 600ms)   │
               └────────┬────────┘
                        │
                        ↓
    ╔════════════════════════════════════════════════════════════╗
    ║            PHASE 2: SEQUENTIAL PROCESSING                  ║
    ║            (Both models need transcript)                   ║
    ╚════════════════════════════════════════════════════════════╝
                        │
                        │ transcript.text
                        │
                        ↓
            ┌───────────────────────┐
            │  CONTEXTUAL ANALYZER  │
            │                       │
            │  ┌─────────────────┐ │
            │  │  NER (spaCy)    │ │
            │  │  50-100ms       │ │
            │  └────────┬────────┘ │
            │           │          │
            │           ↓          │
            │  ┌─────────────────┐ │
            │  │  COMET (BART)   │ │
            │  │  500-800ms      │ │
            │  └────────┬────────┘ │
            │           │          │
            │           ↓          │
            │  ┌─────────────────┐ │
            │  │  Knowledge      │ │
            │  │  Graph Update   │ │
            │  └────────┬────────┘ │
            └───────────┼──────────┘
                        │
                        ↓
    ╔════════════════════════════════════════════════════════════╗
    ║              PHASE 3: RESPONSE BUILDING                    ║
    ╚════════════════════════════════════════════════════════════╝
                        │
                        ↓
            ┌───────────────────────┐
            │  Aggregate Results    │
            │  ┌─────────────────┐  │
            │  │ Transcript      │  │
            │  │ Emotion (audio) │  │
            │  │ Emotion (text)  │  │
            │  │ Entities        │  │
            │  │ Commonsense     │  │
            │  │ Graph Updates   │  │
            │  │ Processing Time │  │
            │  │ Metadata        │  │
            │  └─────────────────┘  │
            └───────────┬───────────┘
                        │
                        ↓
            ┌───────────────────────┐
            │  UNIFIED JSON         │
            │  RESPONSE             │
            └───────────────────────┘
```

## Performance Characteristics

```
SEQUENTIAL EXECUTION (Before Week 6):
═══════════════════════════════════════

STT ───────────> [450ms]
                    ↓
SER ───────────────> [120ms]
                        ↓
NER ───────────────────> [80ms]
                            ↓
COMET ─────────────────────> [600ms]

Total: 450 + 120 + 80 + 600 = 1250ms


PARALLEL EXECUTION (Week 6):
═══════════════════════════════════════

┌─ STT ──> [450ms] ─┐
│                   │─── max(450, 120) = 450ms
└─ SER ──> [120ms] ─┘
            ↓
        NER ──> [80ms]
            ↓
        COMET ──> [600ms]

Total: 450 + 80 + 600 = 1130ms

IMPROVEMENT: 120ms faster (9.6% reduction)
            + Better worst-case (570ms saved if SER slow)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
├─────────────────────────────────────────────────────────────────┤
│  audio_file: Binary audio data                                  │
│  conversation_id: "conv_001"                                     │
│  speaker_id: "user_123"                                          │
│  include_graph: true                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STT RESULT                                    │
├─────────────────────────────────────────────────────────────────┤
│  {                                                               │
│    "text": "I'm meeting Sarah at coffee shop in Mumbai",        │
│    "language": "en",                                             │
│    "confidence": 0.95                                            │
│  }                                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SER RESULT                                    │
├─────────────────────────────────────────────────────────────────┤
│  {                                                               │
│    "emotion": "neutral",                                         │
│    "confidence": 0.85,                                           │
│    "all_scores": {                                               │
│      "neutral": 0.85, "happy": 0.08, ...                        │
│    }                                                             │
│  }                                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                CONTEXTUAL ANALYSIS RESULT                        │
├─────────────────────────────────────────────────────────────────┤
│  {                                                               │
│    "entities": {                                                 │
│      "people": [{"text": "Sarah", ...}],                         │
│      "places": [{"text": "Mumbai", ...}]                         │
│    },                                                            │
│    "emotional_context": {                                        │
│      "subject_emotions": ["excited", "hopeful"],                 │
│      "subject_wants": ["to meet friend"]                         │
│    },                                                            │
│    "graph_updates": { ... }                                      │
│  }                                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                  UNIFIED OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  {                                                               │
│    "transcript": { ... },                                        │
│    "emotion": {                                                  │
│      "from_audio": { ... },                                      │
│      "from_text": { ... }                                        │
│    },                                                            │
│    "entities": { ... },                                          │
│    "commonsense": { ... },                                       │
│    "graph_updates": { ... },                                     │
│    "processing": {                                               │
│      "total_time_ms": 1130,                                      │
│      "stt_completed": true,                                      │
│      "ser_completed": true,                                      │
│      "ner_completed": true,                                      │
│      "comet_completed": true                                     │
│    },                                                            │
│    "metadata": { ... }                                           │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Error Handling

```
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR SCENARIOS                               │
└─────────────────────────────────────────────────────────────────┘

SCENARIO 1: STT Fails
═════════════════════
STT ──X── [ERROR]
         ↓
SER ──✓── [SUCCESS]
         ↓
Response: {
  "transcript": {"error": "STT failed"},
  "emotion": {...}, // SER results OK
  "processing": {"stt_completed": false}
}


SCENARIO 2: SER Fails
═════════════════════
STT ──✓── [SUCCESS]
         ↓
SER ──X── [ERROR]
         ↓
NER ──✓── [SUCCESS] // Uses STT output
         ↓
Response: {
  "transcript": {...}, // STT results OK
  "emotion": {"error": "SER failed"},
  "entities": {...}, // NER results OK
  "processing": {"ser_completed": false}
}


SCENARIO 3: All Succeed
═══════════════════════
STT ──✓── [SUCCESS]
         ↓
SER ──✓── [SUCCESS]
         ↓
NER ──✓── [SUCCESS]
         ↓
COMET ──✓── [SUCCESS]
         ↓
Response: Complete with all results
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRODUCTION DEPLOYMENT                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         LOAD BALANCER                             │
│                      (nginx / ALB / etc.)                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ↓             ↓             ↓
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ FastAPI  │  │ FastAPI  │  │ FastAPI  │
    │ Instance │  │ Instance │  │ Instance │
    │    1     │  │    2     │  │    3     │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │
         │  Each with Chat Orchestrator loaded
         │  + All ML models in memory (~3.5GB)
         │
         └─────────────┼─────────────┘
                       │
                       ↓
            ┌──────────────────┐
            │    DATABASE       │
            │   (PostgreSQL)    │
            └──────────────────┘
```

## Resource Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│                      RESOURCE USAGE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PER INSTANCE:                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Memory:                                               │    │
│  │  - Whisper Model:         ~1.5 GB                      │    │
│  │  - Wav2Vec2 Model:        ~400 MB                      │    │
│  │  - spaCy Model:           ~50 MB                       │    │
│  │  - COMET Model:           ~1.6 GB                      │    │
│  │  - Python Runtime:        ~200 MB                      │    │
│  │  - FastAPI + Libraries:   ~150 MB                      │    │
│  │  ────────────────────────────────                      │    │
│  │  TOTAL:                   ~3.9 GB                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  CPU:                                                            │
│  - Inference: High (4-8 cores recommended)                       │
│  - Idle: Low (~1-2%)                                             │
│                                                                  │
│  GPU (Optional but recommended):                                 │
│  - VRAM: 6-8 GB                                                  │
│  - Speeds up inference 3-5x                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Architecture Version**: 1.0  
**Last Updated**: October 13, 2025  
**Status**: Production Ready ✅
