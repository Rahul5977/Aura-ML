"""
Aura ML Backend API - Simplified Version without Auth
Multi-Modal AI System for Conversational Analysis

Features:
- Speech-to-Text (STT) with Whisper
- Speech Emotion Recognition (SER) with Wav2Vec2
- Named Entity Recognition (NER) with spaCy
- Commonsense Reasoning with COMET
- Knowledge Graph Integration with Neo4j
- Unified ML Pipeline Orchestrator
"""

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import json
import logging
import asyncio
import uuid

# Import from existing working modules
from audio import (
    transcription_service,
    initialize_transcription_service,
    emotion_service,
    initialize_emotion_service,
    preprocess_audio_for_whisper
)
from contextual import (
    contextual_analyzer,
    initialize_contextual_services
)
from chat_orchestrator import (
    chat_orchestrator,
    initialize_chat_orchestrator
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aura ML Backend API", 
    version="2.0.0",
    description="Multi-Modal AI System for Conversational Analysis - ML Pipeline Only"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting Aura ML Backend...")
    
    # Initialize transcription service
    try:
        initialize_transcription_service()
        logger.info("✅ Transcription service (Whisper) loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to load transcription service: {e}")
        logger.info("Audio transcription will not be available")
    
    # Initialize emotion recognition service
    try:
        initialize_emotion_service()
        logger.info("✅ Emotion recognition service (Wav2Vec2) loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to load emotion recognition service: {e}")
        logger.info("Audio emotion recognition will not be available")
    
    # Initialize contextual analysis services (NER + COMET)
    try:
        await initialize_contextual_services()
        logger.info("✅ Contextual analysis services (NER + COMET) loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to load contextual analysis services: {e}")
        logger.info("Contextual analysis will not be available")
    
    # Initialize chat orchestrator
    try:
        initialize_chat_orchestrator(
            transcription_service=transcription_service,
            emotion_service=emotion_service,
            contextual_analyzer=contextual_analyzer
        )
        logger.info("✅ Chat orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to initialize chat orchestrator: {e}")
        logger.info("Chat orchestrator will not be available")
    
    logger.info("✅ Aura ML Backend is ready!")
    logger.info("📚 API documentation available at: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown():
    logger.info("👋 Shutting down Aura ML Backend...")

# ============================================================================
# Health Check & Info Endpoints
# ============================================================================

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "Aura ML Backend API",
        "version": "2.0.0",
        "description": "Multi-Modal AI System for Conversational Analysis",
        "status": "operational",
        "features": [
            "Speech-to-Text (Whisper)",
            "Speech Emotion Recognition (Wav2Vec2)",
            "Named Entity Recognition (spaCy)",
            "Commonsense Reasoning (COMET)",
            "Knowledge Graph (Neo4j)",
            "Unified ML Pipeline"
        ],
        "endpoints": {
            "health": "/health",
            "documentation": "/docs",
            "ml_pipeline": "/orchestrate/analyze-audio",
            "transcription": "/transcribe",
            "emotion_recognition": "/recognize-emotion",
            "text_analysis": "/analyze/text",
            "conversation_context": "/analyze/conversation/{conversation_id}",
            "knowledge_graph_summary": "/knowledge-graph/summary",
            "export_graph": "/knowledge-graph/export"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint with service status"""
    services_status = {
        "transcription": transcription_service.is_loaded if transcription_service else False,
        "emotion_recognition": emotion_service.is_loaded if emotion_service else False,
        "contextual_analysis": contextual_analyzer.is_ready() if contextual_analyzer else False,
        "chat_orchestrator": chat_orchestrator.is_ready() if chat_orchestrator else False
    }
    
    all_healthy = all(services_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services_status,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Audio Processing REST Endpoints
# ============================================================================

@app.post("/transcribe")
async def transcribe_audio_file(
    file: UploadFile = File(...)
):
    """
    Transcribe an audio file using Whisper.
    
    **Input:** Audio file (WAV, MP3, etc.)
    
    **Output:** Transcription with language detection
    
    Supports WAV, MP3, and other common audio formats.
    """
    if not transcription_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transcription service not available"
        )
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Preprocess audio
        audio_array = preprocess_audio_for_whisper(audio_bytes)
        if audio_array is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process audio file"
            )
        
        # Transcribe (Whisper expects 16kHz audio)
        sample_rate = 16000
        result = await transcription_service.transcribe_audio(audio_array)
        
        return {
            "text": result["text"],
            "language": result.get("language", "en"),
            "duration": len(audio_array) / sample_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )

@app.post("/recognize-emotion")
async def recognize_emotion_file(
    file: UploadFile = File(...)
):
    """
    Recognize emotion from an audio file.
    
    **Input:** Audio file (WAV, MP3, etc.)
    
    **Output:** Emotion classification with confidence scores
    
    Supports WAV, MP3, and other common audio formats.
    """
    if not emotion_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion recognition service not available"
        )
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Preprocess audio
        audio_array = preprocess_audio_for_whisper(audio_bytes)
        if audio_array is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process audio file"
            )
        
        # Recognize emotion (model expects 16kHz audio)
        sample_rate = 16000
        result = await emotion_service.recognize_emotion(
            audio_array, 
            sample_rate,
            return_all_scores=True
        )
        
        return result
    except Exception as e:
        logger.error(f"Emotion recognition error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emotion recognition failed: {str(e)}"
        )

# ============================================================================
# Contextual Analysis REST Endpoints
# ============================================================================

@app.post("/analyze/text")
async def analyze_text(
    text: str = Query(..., description="Text to analyze"),
    conversation_id: Optional[str] = Query(None, description="Conversation identifier"),
    speaker_id: Optional[str] = Query(None, description="Speaker identifier"),
    include_graph: bool = Query(True, description="Update knowledge graph")
):
    """
    Perform comprehensive contextual analysis on text.
    
    **Features:**
    - Named Entity Recognition (people, places, organizations, concepts)
    - Emotional context inference (using COMET)
    - Knowledge graph updates (Neo4j)
    
    **Input:** Text string
    
    **Output:** Entities, emotions, commonsense inferences
    """
    if not contextual_analyzer.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Contextual analysis services not available"
        )
    
    try:
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        # Generate speaker ID if not provided
        if not speaker_id:
            speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
        
        result = await contextual_analyzer.analyze(
            text=text,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            include_graph_updates=include_graph
        )
        
        return result
    except Exception as e:
        logger.error(f"Contextual analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Contextual analysis failed: {str(e)}"
        )

@app.get("/analyze/conversation/{conversation_id}")
async def get_conversation_context(
    conversation_id: str
):
    """
    Get accumulated contextual knowledge for a conversation.
    
    **Returns:** Entities, emotions, and relationships from the knowledge graph
    
    This endpoint queries the Neo4j knowledge graph to retrieve all
    entities, emotions, and commonsense inferences associated with
    a specific conversation.
    """
    try:
        context = await contextual_analyzer.get_conversation_context(conversation_id)
        return context
    except Exception as e:
        logger.error(f"Error getting conversation context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation context: {str(e)}"
        )

@app.get("/knowledge-graph/summary")
async def get_knowledge_graph_summary():
    """
    Get summary statistics of the knowledge graph.
    
    **Returns:** Counts of nodes and relationships by type
    
    This provides an overview of the entire knowledge graph,
    showing how many entities, conversations, emotions, and
    relationships have been stored.
    """
    try:
        from contextual import knowledge_graph_service
        summary = await knowledge_graph_service.get_graph_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting graph summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph summary: {str(e)}"
        )

@app.get("/knowledge-graph/export")
async def export_knowledge_graph(
    format: str = Query("json", description="Export format (json)")
):
    """
    Export the knowledge graph in various formats.
    
    **Supported formats:** json
    
    **Returns:** Complete graph data structure
    """
    try:
        from contextual import knowledge_graph_service
        graph_data = await knowledge_graph_service.export_graph(format=format)
        
        return {
            "format": format,
            "data": graph_data,
            "exported_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exporting graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export graph: {str(e)}"
        )

# ============================================================================
# Chat Orchestrator Endpoint - Unified ML Pipeline
# ============================================================================

@app.post("/orchestrate/analyze-audio")
async def orchestrate_audio_analysis(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Query(None, description="Conversation identifier"),
    speaker_id: Optional[str] = Query(None, description="Speaker identifier"),
    include_graph: bool = Query(True, description="Update knowledge graph")
):
    """
    **Unified AI Pipeline - Complete Audio Analysis**
    
    Process audio through the complete ML pipeline in a single request:
    
    1. **Speech-to-Text (STT)** - Transcribe audio with Whisper
    2. **Speech Emotion Recognition (SER)** - Detect emotion with Wav2Vec2
    3. **Named Entity Recognition (NER)** - Extract entities with spaCy
    4. **Commonsense Reasoning (COMET)** - Infer emotional context
    5. **Knowledge Graph** - Update Neo4j graph with results
    
    **Input:** Audio file (WAV, MP3, etc.)
    
    **Output:** Aggregated JSON response with:
    - Transcript (text, language)
    - Emotion (from audio and text)
    - Entities (people, places, organizations, concepts)
    - Commonsense inferences (feelings, wants, effects)
    - Knowledge graph updates
    - Processing metrics
    
    **Example Response:**
    ```json
    {
      "transcript": {
        "text": "I'm meeting Sarah at the coffee shop in Mumbai tomorrow.",
        "language": "en"
      },
      "emotion": {
        "from_audio": {"primary": "neutral", "confidence": 0.85},
        "from_text": {"detected": ["hopeful", "excited"]}
      },
      "entities": {
        "people": [{"text": "Sarah", "start": 13, "end": 18}],
        "places": [{"text": "Mumbai", "start": 42, "end": 48}],
        "dates": [{"text": "tomorrow", "start": 49, "end": 57}]
      },
      "commonsense": {
        "inferences": {
          "subject": {
            "feelings": ["interested", "hopeful"],
            "wants": ["to meet friend", "to have coffee"]
          }
        }
      },
      "processing": {
        "total_time_ms": 650,
        "all_models_completed": true
      }
    }
    ```
    """
    if not chat_orchestrator or not chat_orchestrator.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat orchestrator service not available"
        )
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Validate audio file
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file"
            )
        
        # Generate IDs if not provided
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        if not speaker_id:
            speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
        
        # Process through orchestrator
        result = await chat_orchestrator.process_audio(
            audio_bytes=audio_bytes,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            sample_rate=16000,
            include_graph_updates=include_graph
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat orchestrator error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio analysis failed: {str(e)}"
        )

# ============================================================================
# Additional Utility Endpoints
# ============================================================================

@app.get("/models/status")
def get_models_status():
    """Get detailed status of all loaded ML models"""
    return {
        "models": {
            "whisper": {
                "loaded": transcription_service.is_loaded if transcription_service else False,
                "model_name": "openai/whisper-base",
                "capabilities": ["transcription", "language_detection"]
            },
            "wav2vec2": {
                "loaded": emotion_service.is_loaded if emotion_service else False,
                "model_name": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                "capabilities": ["emotion_recognition"]
            },
            "spacy": {
                "loaded": contextual_analyzer.is_ready() if contextual_analyzer else False,
                "model_name": "en_core_web_sm",
                "capabilities": ["named_entity_recognition"]
            },
            "comet": {
                "loaded": contextual_analyzer.is_ready() if contextual_analyzer else False,
                "model_name": "comet-atomic_2020",
                "capabilities": ["commonsense_reasoning"]
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/test/echo")
async def echo_test(data: Dict[str, Any]):
    """Simple echo endpoint for testing"""
    return {
        "echo": data,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": f"The endpoint {request.url.path} does not exist",
        "available_endpoints": [
            "/",
            "/health",
            "/docs",
            "/transcribe",
            "/recognize-emotion",
            "/analyze/text",
            "/orchestrate/analyze-audio",
            "/knowledge-graph/summary"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
