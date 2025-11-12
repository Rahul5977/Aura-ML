"""
Aura ML Pipeline - Mock Backend for Testing
============================================
A lightweight mock backend that simulates the ML pipeline responses
without requiring heavy ML models. Perfect for UI development and testing.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import random
import time

app = FastAPI(
    title="Aura ML Pipeline (Mock Backend)",
    description="Lightweight mock backend for testing the Streamlit UI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"]
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "DATE", "TIME", "EVENT", "PRODUCT"]

SAMPLE_TEXTS = [
    "Hello, I'm feeling great today!",
    "I met Sarah Johnson at Google headquarters yesterday",
    "The meeting is scheduled for tomorrow at 2 PM",
    "We discussed the new AI project in the conference room"
]

SAMPLE_ENTITIES = [
    {"text": "Sarah Johnson", "label": "PERSON", "start": 6, "end": 19},
    {"text": "Google", "label": "ORG", "start": 23, "end": 29},
    {"text": "yesterday", "label": "DATE", "start": 43, "end": 52},
    {"text": "2 PM", "label": "TIME", "start": 23, "end": 27},
]

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Aura ML Pipeline API (Mock)",
        "version": "1.0.0",
        "status": "running",
        "mode": "mock",
        "description": "Mock backend for UI testing without ML models",
        "endpoints": {
            "health": "/health",
            "models": "/models/status",
            "transcribe": "/transcribe",
            "emotion": "/recognize-emotion",
            "text_analysis": "/analyze/text",
            "orchestrate": "/orchestrate/analyze-audio",
            "knowledge_graph": "/knowledge-graph/*"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "mock",
        "timestamp": time.time(),
        "message": "Mock backend is running (no ML models required)"
    }

# Models status
@app.get("/models/status")
async def get_models_status():
    return {
        "whisper": {"loaded": True, "status": "ready (mock)", "model": "mock-whisper"},
        "wav2vec2": {"loaded": True, "status": "ready (mock)", "model": "mock-wav2vec2"},
        "spacy": {"loaded": True, "status": "ready (mock)", "model": "mock-spacy"},
        "comet": {"loaded": True, "status": "ready (mock)", "model": "mock-comet"},
        "neo4j": "connected (mock)",
        "note": "This is a mock backend - responses are simulated"
    }

# Transcribe audio
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Simulate processing delay
    await asyncio_sleep(1)
    
    # Return mock transcription
    return {
        "transcription": random.choice(SAMPLE_TEXTS),
        "language": "en",
        "confidence": round(random.uniform(0.85, 0.99), 2),
        "duration": round(random.uniform(2.0, 10.0), 2),
        "model": "mock-whisper"
    }

# Recognize emotion
@app.post("/recognize-emotion")
async def recognize_emotion(file: UploadFile = File(...)):
    # Simulate processing delay
    await asyncio_sleep(0.5)
    
    # Generate mock emotion scores
    emotions = {}
    remaining = 1.0
    for emotion in EMOTIONS[:-1]:
        score = round(random.uniform(0.05, remaining * 0.8), 3)
        emotions[emotion] = score
        remaining -= score
    emotions[EMOTIONS[-1]] = round(remaining, 3)
    
    # Normalize
    total = sum(emotions.values())
    emotions = {k: round(v/total, 3) for k, v in emotions.items()}
    
    return {
        "emotions": emotions,
        "dominant_emotion": max(emotions, key=emotions.get),
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "model": "mock-wav2vec2"
    }

# Analyze text
@app.post("/analyze/text")
async def analyze_text(
    text: str,
    conversation_id: str = "default",
    speaker_id: str = "speaker_001",
    include_graph: bool = True
):
    # Simulate processing delay
    await asyncio_sleep(0.8)
    
    # Generate mock entities based on text
    entities = []
    words = text.split()
    if len(words) > 3:
        # Add some random entities
        for _ in range(random.randint(1, min(4, len(words)//2))):
            entity = random.choice(SAMPLE_ENTITIES)
            if entity not in entities:
                entities.append(entity.copy())
    
    # Generate mock COMET embedding
    embedding_dims = 768
    embedding = [round(random.uniform(-1, 1), 4) for _ in range(embedding_dims)]
    
    result = {
        "text": text,
        "conversation_id": conversation_id,
        "speaker_id": speaker_id,
        "entities": entities,
        "comet_embedding": {
            "dimensions": embedding_dims,
            "model": "mock-comet",
            "embedding_preview": embedding[:10]
        },
        "processing_time": round(random.uniform(0.5, 1.5), 2)
    }
    
    if include_graph:
        result["graph_summary"] = {
            "nodes_created": len(entities),
            "relationships_created": random.randint(1, len(entities) * 2),
            "conversation_updated": True
        }
    
    return result

# Complete audio pipeline
@app.post("/orchestrate/analyze-audio")
async def orchestrate_audio_analysis(
    file: UploadFile = File(...),
    conversation_id: str = "default",
    speaker_id: str = "speaker_001",
    include_graph: bool = True
):
    # Simulate processing delay for complete pipeline
    await asyncio_sleep(2)
    
    # Combine all mock results
    transcription = random.choice(SAMPLE_TEXTS)
    
    # Emotions
    emotions = {}
    remaining = 1.0
    for emotion in EMOTIONS[:-1]:
        score = round(random.uniform(0.05, remaining * 0.8), 3)
        emotions[emotion] = score
        remaining -= score
    emotions[EMOTIONS[-1]] = round(remaining, 3)
    total = sum(emotions.values())
    emotions = {k: round(v/total, 3) for k, v in emotions.items()}
    
    # Entities
    entities = [e.copy() for e in SAMPLE_ENTITIES[:random.randint(2, 4)]]
    
    # COMET embedding
    embedding_dims = 768
    embedding = [round(random.uniform(-1, 1), 4) for _ in range(embedding_dims)]
    
    result = {
        "transcription": transcription,
        "language": "en",
        "confidence": round(random.uniform(0.85, 0.99), 2),
        "emotions": emotions,
        "dominant_emotion": max(emotions, key=emotions.get),
        "entities": entities,
        "comet_embedding": {
            "dimensions": embedding_dims,
            "model": "mock-comet",
            "embedding_preview": embedding[:10]
        },
        "conversation_id": conversation_id,
        "speaker_id": speaker_id,
        "processing_time": round(random.uniform(1.5, 3.0), 2),
        "pipeline_stages": ["STT", "SER", "NER", "COMET", "Graph"]
    }
    
    if include_graph:
        result["graph_summary"] = {
            "nodes_created": len(entities) + 2,  # entities + conversation + speaker
            "relationships_created": random.randint(len(entities), len(entities) * 3),
            "conversation_updated": True
        }
    
    return result

# Get conversation context
@app.get("/analyze/conversation/{conversation_id}")
async def get_conversation_context(conversation_id: str):
    return {
        "conversation_id": conversation_id,
        "messages": random.randint(3, 10),
        "speakers": random.randint(1, 3),
        "entities": [e.copy() for e in SAMPLE_ENTITIES[:random.randint(3, len(SAMPLE_ENTITIES))]],
        "timeline": "mock timeline data",
        "note": "This is mock data"
    }

# Knowledge graph summary
@app.get("/knowledge-graph/summary")
async def get_graph_summary():
    return {
        "total_nodes": random.randint(50, 200),
        "total_relationships": random.randint(100, 500),
        "conversations": random.randint(5, 20),
        "speakers": random.randint(3, 15),
        "entities": random.randint(30, 150),
        "node_types": {
            "Conversation": random.randint(5, 20),
            "Speaker": random.randint(3, 15),
            "Entity": random.randint(30, 150)
        },
        "note": "This is mock data"
    }

# Export knowledge graph
@app.get("/knowledge-graph/export")
async def export_graph(format: str = "json"):
    nodes = []
    relationships = []
    
    # Generate some mock nodes
    for i in range(random.randint(10, 20)):
        nodes.append({
            "id": f"node_{i}",
            "type": random.choice(["Entity", "Speaker", "Conversation"]),
            "properties": {"name": f"MockNode{i}"}
        })
    
    # Generate some mock relationships
    for i in range(random.randint(15, 30)):
        if len(nodes) >= 2:
            source = random.choice(nodes)
            target = random.choice([n for n in nodes if n != source])
            relationships.append({
                "source": source["id"],
                "target": target["id"],
                "type": random.choice(["MENTIONED", "RELATED_TO", "PARTICIPATED"])
            })
    
    return {
        "nodes": nodes,
        "relationships": relationships,
        "format": format,
        "exported_at": time.time(),
        "note": "This is mock data"
    }

# Echo test endpoint
@app.post("/test/echo")
async def echo_test(data: dict):
    return {
        "echo": data,
        "timestamp": time.time(),
        "message": "Mock backend received your data"
    }

# Async sleep helper
async def asyncio_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  AURA ML PIPELINE - MOCK BACKEND")
    print("=" * 70)
    print("\nThis is a lightweight mock backend for testing the Streamlit UI.")
    print("It simulates ML pipeline responses without requiring ML models.\n")
    print("Starting server on http://localhost:8000\n")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
