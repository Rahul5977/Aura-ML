from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Query, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta, datetime
from typing import Optional
import os
from dotenv import load_dotenv
import json
import logging
import asyncio

# Import from existing working modules
from auth import create_access_token, verify_token, verify_password, ACCESS_TOKEN_EXPIRE_MINUTES
from websocket_manager import manager as ws_manager
from audio import (
    audio_buffer_manager,
    transcription_service,
    initialize_transcription_service,
    emotion_service,
    initialize_emotion_service,
    preprocess_audio_for_whisper
)
# Week 5: Contextual Analysis imports
from contextual import (
    contextual_analyzer,
    initialize_contextual_services
)
# Week 6: Chat Orchestrator import
from chat_orchestrator import (
    chat_orchestrator,
    initialize_chat_orchestrator
)
from database import (
    connect_db, disconnect_db, create_user, authenticate_user, 
    get_user_by_id, get_user_by_username, get_user_by_email,
    create_conversation, get_user_conversations, get_conversation_by_id,
    create_message, get_conversation_messages,
    update_user_profile, update_user_password
)
from schema_demo import (
    UserCreate, UserResponse, UserLogin, Token,
    ConversationCreate, ConversationResponse,
    MessageCreate, MessageResponse,
    UserUpdate, PasswordChange
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aura Backend API", version="1.0.0")
security = HTTPBearer()

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    await connect_db()
    # Initialize transcription service in background
    try:
        initialize_transcription_service()
        logger.info("✅ Transcription service loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to load transcription service: {e}")
        logger.info("Audio transcription will not be available")
    
    # Initialize emotion recognition service
    try:
        initialize_emotion_service()
        logger.info("✅ Emotion recognition service loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to load emotion recognition service: {e}")
        logger.info("Audio emotion recognition will not be available")
    
    # Week 5: Initialize contextual analysis services
    try:
        await initialize_contextual_services()
        logger.info("✅ Contextual analysis services loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to load contextual analysis services: {e}")
        logger.info("Contextual analysis will not be available")
    
    # Week 6: Initialize chat orchestrator
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

@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    token_data = verify_token(token)
    user = await get_user_by_id(token_data["user_id"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user

@app.get("/health")
def health():
    return {"status": "ok"}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    # Check if user already exists
    existing_user = await get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    existing_email = await get_user_by_email(user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user = await create_user(user_data)
    return user

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    user = await authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoints
@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    return current_user

@app.get("/conversations", response_model=list[ConversationResponse])
async def get_conversations(current_user = Depends(get_current_user)):
    conversations = await get_user_conversations(current_user.id)
    return conversations

@app.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_new_conversation(
    conversation_data: ConversationCreate,
    current_user = Depends(get_current_user)
):
    conversation = await create_conversation(current_user.id, conversation_data.title)
    return conversation

@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user = Depends(get_current_user)
):
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation

@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: str,
    current_user = Depends(get_current_user)
):
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = await get_conversation_messages(conversation_id)
    return messages

@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_new_message(
    conversation_id: str,
    message_data: MessageCreate,
    current_user = Depends(get_current_user)
):
    # Verify conversation belongs to user
    conversation = await get_conversation_by_id(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    message = await create_message(
        conversation_id=conversation_id,
        content=message_data.content,
        role=message_data.role
    )
    return message

# Additional auth routes for testing
@app.put("/auth/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user = Depends(get_current_user)
):
    """Update current user's profile information."""
    # Check if email is being updated and if it's already taken
    if hasattr(user_update, 'email') and user_update.email and user_update.email != current_user.email:
        existing_email = await get_user_by_email(user_update.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    try:
        updated_user = await update_user_profile(current_user.id, user_update)
        return updated_user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )

@app.post("/auth/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user = Depends(get_current_user)
):
    """Change user's password."""
    # Verify current password
    if not verify_password(password_change.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    try:
        await update_user_password(current_user.id, password_change.new_password)
        return {"message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )

@app.post("/auth/logout")
async def logout(current_user = Depends(get_current_user)):
    """Logout user (client should discard the JWT token)."""
    return {"message": "Logged out successfully"}

# ============================================================================
# Audio Processing REST Endpoints
# ============================================================================

@app.post("/transcribe")
async def transcribe_audio_file(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    Transcribe an audio file using Whisper.
    
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
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    Recognize emotion from an audio file.
    
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
# Week 5: Contextual Analysis REST Endpoints
# ============================================================================

@app.post("/analyze/text")
async def analyze_text(
    text: str,
    conversation_id: str,
    speaker_id: Optional[str] = None,
    include_graph: bool = True,
    current_user = Depends(get_current_user)
):
    """
    Perform comprehensive contextual analysis on text.
    
    Extracts:
    - Named entities (people, places, organizations, concepts)
    - Emotional context (using COMET)
    - Knowledge graph updates
    
    **Week 5 Feature**
    """
    if not contextual_analyzer.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Contextual analysis services not available"
        )
    
    try:
        result = await contextual_analyzer.analyze(
            text=text,
            conversation_id=conversation_id,
            speaker_id=speaker_id or str(current_user.id),
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
    conversation_id: str,
    current_user = Depends(get_current_user)
):
    """
    Get accumulated contextual knowledge for a conversation.
    
    Returns entities, emotions, and relationships from the knowledge graph.
    
    **Week 5 Feature**
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
async def get_knowledge_graph_summary(
    current_user = Depends(get_current_user)
):
    """
    Get summary statistics of the knowledge graph.
    
    Returns counts of nodes and relationships by type.
    
    **Week 5 Feature**
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
    format: str = "json",
    current_user = Depends(get_current_user)
):
    """
    Export the knowledge graph in various formats.
    
    Supported formats: json
    
    **Week 5 Feature**
    """
    try:
        from contextual import knowledge_graph_service
        graph_data = await knowledge_graph_service.export_graph(format=format)
        
        return {
            "format": format,
            "data": graph_data
        }
    except Exception as e:
        logger.error(f"Error exporting graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export graph: {str(e)}"
        )

# ============================================================================
# Week 6: Chat Orchestrator Endpoint - Unified AI Pipeline
# ============================================================================

@app.post("/orchestrate/analyze-audio")
async def orchestrate_audio_analysis(
    file: UploadFile = File(...),
    conversation_id: str = Query(..., description="Conversation identifier"),
    speaker_id: Optional[str] = Query(None, description="Speaker identifier"),
    include_graph: bool = Query(True, description="Update knowledge graph"),
    current_user = Depends(get_current_user)
):
    """
    **Week 6: Chat Orchestrator - Unified AI Pipeline**
    
    Process audio through the complete AI pipeline in a single request:
    1. Speech-to-Text (STT) - Transcribe audio with Whisper
    2. Speech Emotion Recognition (SER) - Detect emotion with Wav2Vec2
    3. Named Entity Recognition (NER) - Extract entities with spaCy
    4. Commonsense Reasoning (COMET) - Infer emotional context
    5. Knowledge Graph - Update graph with results
    
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
        
        # Process through orchestrator
        result = await chat_orchestrator.process_audio(
            audio_bytes=audio_bytes,
            conversation_id=conversation_id,
            speaker_id=speaker_id or str(current_user.id),
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
# WebSocket Endpoints for Real-time Chat
# ============================================================================

@app.websocket("/ws/conversations/{conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    conversation_id: str,
    token: str = Query(...)
):
    """
    WebSocket endpoint for real-time chat in a conversation.
    
    Clients must provide a valid JWT token as a query parameter.
    Messages are broadcasted to all connected clients in the same conversation.
    
    Args:
        websocket: WebSocket connection
        conversation_id: ID of the conversation to join
        token: JWT authentication token (query parameter)
    
    Message Format (Client to Server):
        {
            "type": "message",
            "content": "message text",
            "role": "user" or "assistant"
        }
    
    Message Format (Server to Client):
        {
            "type": "message" | "system" | "active_users" | "error",
            "message_id": "...",
            "content": "...",
            "role": "user" | "assistant",
            "sender": {
                "user_id": "...",
                "username": "...",
                "full_name": "..."
            },
            "timestamp": "ISO 8601 timestamp"
        }
    """
    # Verify token and get user
    try:
        token_data = verify_token(token)
        user = await get_user_by_id(token_data["user_id"])
        if not user:
            await websocket.close(code=1008, reason="Invalid authentication token")
            return
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return
    
    # Verify conversation exists and user has access
    try:
        conversation = await get_conversation_by_id(conversation_id, user.id)
        if not conversation:
            await websocket.close(code=1008, reason="Conversation not found or access denied")
            return
    except Exception as e:
        logger.error(f"Error accessing conversation: {e}")
        await websocket.close(code=1008, reason="Error accessing conversation")
        return
    
    # Connect to WebSocket
    user_data = {
        "user_id": user.id,
        "username": user.username,
        "full_name": user.full_name
    }
    
    await ws_manager.connect(websocket, conversation_id, user.id, user_data)
    
    # Send connection success message
    await ws_manager.send_personal_message(
        {
            "type": "system",
            "content": f"Connected to conversation: {conversation.title or 'Untitled'}",
            "timestamp": datetime.now().isoformat()
        },
        websocket
    )
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "message")
                
                if message_type == "message":
                    # Extract message content and role
                    content = message_data.get("content", "").strip()
                    role = message_data.get("role", "user")
                    
                    if not content:
                        await ws_manager.send_personal_message(
                            {
                                "type": "error",
                                "content": "Message content cannot be empty",
                                "timestamp": datetime.now().isoformat()
                            },
                            websocket
                        )
                        continue
                    
                    # Validate role
                    if role not in ["user", "assistant"]:
                        role = "user"
                    
                    # Save message to database
                    try:
                        saved_message = await create_message(
                            conversation_id=conversation_id,
                            content=content,
                            role=role
                        )
                        
                        # Prepare broadcast message
                        broadcast_data = {
                            "type": "message",
                            "message_id": saved_message.id,
                            "content": saved_message.content,
                            "role": saved_message.role,
                            "sender": {
                                "user_id": user.id,
                                "username": user.username,
                                "full_name": user.full_name
                            },
                            "timestamp": saved_message.created_at.isoformat(),
                            "conversation_id": conversation_id
                        }
                        
                        # Send confirmation to sender
                        await ws_manager.send_personal_message(broadcast_data, websocket)
                        
                        # Broadcast to other users in the conversation
                        await ws_manager.broadcast_message(
                            conversation_id,
                            broadcast_data,
                            exclude_user_id=user.id
                        )
                        
                        logger.info(
                            f"Message from {user.username} in conversation {conversation_id}: "
                            f"{content[:50]}..."
                        )
                        
                    except Exception as e:
                        logger.error(f"Error saving message: {e}")
                        await ws_manager.send_personal_message(
                            {
                                "type": "error",
                                "content": "Failed to save message",
                                "timestamp": datetime.now().isoformat()
                            },
                            websocket
                        )
                
                elif message_type == "ping":
                    # Handle ping/keepalive
                    await ws_manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )
                
                elif message_type == "get_active_users":
                    # Send active users list
                    await ws_manager.send_active_users(websocket, conversation_id)
                
                else:
                    # Unknown message type
                    await ws_manager.send_personal_message(
                        {
                            "type": "error",
                            "content": f"Unknown message type: {message_type}",
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )
            
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from user {user.username}")
                await ws_manager.send_personal_message(
                    {
                        "type": "error",
                        "content": "Invalid message format. Expected JSON.",
                        "timestamp": datetime.now().isoformat()
                    },
                    websocket
                )
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await ws_manager.send_personal_message(
                    {
                        "type": "error",
                        "content": "Error processing message",
                        "timestamp": datetime.now().isoformat()
                    },
                    websocket
                )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user.username}")
        await ws_manager.disconnect(conversation_id, user.id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user.username}: {e}")
        await ws_manager.disconnect(conversation_id, user.id)

# ============================================================================
# WebSocket Audio Endpoint for Real-time Transcription
# ============================================================================

@app.websocket("/ws/v1/audio")
async def audio_websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    """
    WebSocket endpoint for real-time audio transcription.
    
    Accepts binary audio data chunks, buffers them, and transcribes
    when end-of-speech is detected (silence timeout).
    
    Args:
        websocket: WebSocket connection
        token: JWT authentication token (query parameter)
    
    Audio Format Expected:
        - WAV format (preferred) or raw PCM
        - Sample rate: 16kHz (will be resampled if different)
        - Channels: Mono (stereo will be converted)
        - Bit depth: 16-bit
    
    Message Format (Server to Client):
        {
            "type": "transcription" | "status" | "error",
            "text": "transcribed text",
            "duration": 2.5,  # audio duration in seconds
            "timestamp": "ISO 8601 timestamp"
        }
    """
    # Verify token and get user
    try:
        token_data = verify_token(token)
        user = await get_user_by_id(token_data["user_id"])
        if not user:
            await websocket.close(code=1008, reason="Invalid authentication token")
            return
    except Exception as e:
        logger.error(f"Audio WebSocket authentication failed: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return
    
    # Accept WebSocket connection
    await websocket.accept()
    
    # Create audio buffer for this user
    buffer = audio_buffer_manager.create_buffer(user.id)
    
    # Send connection confirmation
    await websocket.send_json({
        "type": "status",
        "content": "Connected to audio transcription service",
        "user_id": user.id,
        "username": user.username,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info(f"User {user.username} ({user.id}) connected to audio WebSocket")
    
    # Define transcription callback for silence timeout
    async def transcription_callback(user_id: str, audio_data: bytes):
        """Called when silence timeout is detected"""
        try:
            logger.info(f"Processing audio for user {user_id}: {len(audio_data)} bytes")
            
            # Send processing status
            await websocket.send_json({
                "type": "status",
                "content": "Processing audio...",
                "timestamp": datetime.now().isoformat()
            })
            
            # Preprocess audio
            audio_array = preprocess_audio_for_whisper(audio_data, sample_rate=16000)
            
            if audio_array is None or len(audio_array) == 0:
                await websocket.send_json({
                    "type": "error",
                    "content": "Failed to process audio data",
                    "timestamp": datetime.now().isoformat()
                })
                return
            
            # Calculate duration
            duration = len(audio_array) / 16000
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Skip very short audio (less than 0.3 seconds)
            if duration < 0.3:
                logger.info(f"Audio too short ({duration:.2f}s), skipping transcription")
                return
            
            # Run STT and SER in parallel for faster processing
            start_time = datetime.now()
            
            try:
                # Run transcription and emotion recognition concurrently
                transcription_task = transcription_service.transcribe_audio(audio_array, language="en")
                emotion_task = emotion_service.recognize_emotion(audio_array, sampling_rate=16000, return_all_scores=True)
                
                # Wait for both tasks to complete
                transcription_result, emotion_result = await asyncio.gather(
                    transcription_task,
                    emotion_task,
                    return_exceptions=True
                )
                
                # Handle transcription errors
                if isinstance(transcription_result, Exception):
                    logger.error(f"Transcription error: {transcription_result}")
                    transcription_result = {
                        "text": "",
                        "language": "en",
                        "error": str(transcription_result)
                    }
                
                # Handle emotion recognition errors
                if isinstance(emotion_result, Exception):
                    logger.error(f"Emotion recognition error: {emotion_result}")
                    emotion_result = {
                        "emotion": "unknown",
                        "confidence": 0.0,
                        "error": str(emotion_result)
                    }
                
            except Exception as e:
                logger.error(f"Error during parallel processing: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"Processing error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                return
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            transcribed_text = transcription_result.get("text", "")
            
            # Build unified response
            response = {
                "type": "analysis",
                "transcript": {
                    "text": transcribed_text,
                    "language": transcription_result.get("language", "en"),
                },
                "emotion": {
                    "primary": emotion_result.get("emotion", "unknown"),
                    "confidence": emotion_result.get("confidence", 0.0),
                    "all_scores": emotion_result.get("all_scores", {})
                },
                "audio": {
                    "duration": duration,
                    "sample_rate": 16000
                },
                "processing": {
                    "total_time_ms": int(processing_time * 1000),
                    "transcription_time_ms": transcription_result.get("inference_time_ms", 0),
                    "emotion_time_ms": emotion_result.get("inference_time_ms", 0)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Only send if we have meaningful results
            if transcribed_text or emotion_result.get("emotion") != "unknown":
                await websocket.send_json(response)
                
                logger.info(
                    f"Analysis for {user.username}: "
                    f"'{transcribed_text}' | Emotion: {emotion_result.get('emotion')} "
                    f"({emotion_result.get('confidence', 0):.2f}) | "
                    f"Processing: {processing_time*1000:.0f}ms"
                )
            else:
                # No speech detected or both services failed
                await websocket.send_json({
                    "type": "status",
                    "content": "No speech detected in audio",
                    "timestamp": datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.error(f"Error in transcription callback: {e}")
            await websocket.send_json({
                "type": "error",
                "content": f"Analysis error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    # Start monitoring buffer for silence timeout
    monitor_task = asyncio.create_task(
        audio_buffer_manager._monitor_buffers(transcription_callback)
    )
    
    try:
        while True:
            # Receive audio chunk (binary data)
            data = await websocket.receive_bytes()
            
            # Add chunk to buffer
            buffer.add_chunk(data)
            
            logger.debug(f"Received audio chunk from {user.username}: {len(data)} bytes")
    
    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket disconnected for user {user.username}")
    except Exception as e:
        logger.error(f"Audio WebSocket error for user {user.username}: {e}")
    finally:
        # Cleanup
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Process any remaining audio in buffer
        if buffer.has_data():
            final_audio = buffer.get_buffer()
            await transcription_callback(user.id, final_audio)
        
        # Remove buffer
        audio_buffer_manager.remove_buffer(user.id)
        logger.info(f"Cleaned up audio resources for user {user.username}")
