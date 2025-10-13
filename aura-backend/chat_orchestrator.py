"""
Chat Orchestrator Service (Week 6)

Unified endpoint that processes audio through the complete AI pipeline:
1. Speech-to-Text (STT) - Whisper
2. Speech Emotion Recognition (SER) - Wav2Vec2
3. Named Entity Recognition (NER) - spaCy
4. Commonsense Reasoning (COMET) - AllenAI COMET

Returns a single aggregated JSON response with all analysis results.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Orchestrates the complete AI pipeline for audio-based chat analysis.
    Coordinates STT, SER, NER, and COMET models in the correct order.
    """
    
    def __init__(
        self,
        transcription_service,
        emotion_service,
        contextual_analyzer
    ):
        """
        Initialize chat orchestrator with required services.
        
        Args:
            transcription_service: Audio transcription service (Whisper)
            emotion_service: Emotion recognition service (Wav2Vec2)
            contextual_analyzer: Contextual analysis service (NER + COMET)
        """
        self.transcription_service = transcription_service
        self.emotion_service = emotion_service
        self.contextual_analyzer = contextual_analyzer
        
        logger.info("Initialized ChatOrchestrator")
    
    async def process_audio(
        self,
        audio_bytes: bytes,
        conversation_id: str,
        speaker_id: Optional[str] = None,
        sample_rate: int = 16000,
        include_graph_updates: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio through the complete AI pipeline.
        
        Pipeline:
        1. STT (Speech-to-Text) - Extract text from audio
        2. SER (Speech Emotion Recognition) - Detect emotion from audio
        3. NER (Named Entity Recognition) - Extract entities from text
        4. COMET (Commonsense Reasoning) - Infer emotional context from text
        5. Knowledge Graph - Update graph with results
        
        Args:
            audio_bytes: Raw audio data
            conversation_id: Conversation identifier
            speaker_id: Optional speaker identifier
            sample_rate: Audio sample rate (default: 16000 Hz)
            include_graph_updates: Whether to update knowledge graph
            
        Returns:
            Aggregated JSON response with all analysis results:
            {
                "transcript": {...},
                "emotion": {...},
                "entities": {...},
                "emotional_context": {...},
                "graph_updates": {...},
                "processing": {...},
                "metadata": {...}
            }
        """
        start_time = datetime.utcnow()
        
        logger.info(f"Starting audio processing for conversation {conversation_id}")
        
        # Step 1 & 2: Run STT and SER in parallel (both need audio)
        try:
            stt_task = self._run_stt(audio_bytes, sample_rate)
            ser_task = self._run_ser(audio_bytes, sample_rate)
            
            transcript_result, emotion_result = await asyncio.gather(
                stt_task,
                ser_task,
                return_exceptions=True
            )
            
            # Handle STT errors
            if isinstance(transcript_result, Exception):
                logger.error(f"STT failed: {transcript_result}")
                transcript_result = {
                    "text": "",
                    "language": "unknown",
                    "error": str(transcript_result)
                }
            
            # Handle SER errors
            if isinstance(emotion_result, Exception):
                logger.error(f"SER failed: {emotion_result}")
                emotion_result = {
                    "emotion": "unknown",
                    "confidence": 0.0,
                    "error": str(emotion_result)
                }
            
        except Exception as e:
            logger.error(f"Error in audio processing phase: {e}")
            return self._create_error_response(
                "Audio processing failed",
                str(e),
                conversation_id
            )
        
        # Step 3 & 4: Run NER and COMET on transcribed text
        text = transcript_result.get("text", "")
        
        if text:
            try:
                contextual_result = await self._run_contextual_analysis(
                    text=text,
                    conversation_id=conversation_id,
                    speaker_id=speaker_id,
                    include_graph_updates=include_graph_updates
                )
            except Exception as e:
                logger.error(f"Contextual analysis failed: {e}")
                contextual_result = {
                    "entities": {},
                    "emotional_context": {},
                    "emotions_detected": [],
                    "error": str(e)
                }
        else:
            logger.warning("No text transcribed, skipping contextual analysis")
            contextual_result = {
                "entities": {},
                "emotional_context": {},
                "emotions_detected": [],
                "message": "No text to analyze"
            }
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Build aggregated response
        response = self._build_response(
            transcript=transcript_result,
            emotion=emotion_result,
            contextual=contextual_result,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            processing_time_ms=processing_time_ms,
            timestamp=end_time.isoformat()
        )
        
        logger.info(
            f"Audio processing complete for {conversation_id}: "
            f"{len(text)} chars, {processing_time_ms}ms"
        )
        
        return response
    
    async def _run_stt(self, audio_bytes: bytes, sample_rate: int) -> Dict[str, Any]:
        """
        Run Speech-to-Text model.
        
        Args:
            audio_bytes: Raw audio data
            sample_rate: Audio sample rate
            
        Returns:
            Transcription result with text and language
        """
        logger.info("Running STT (Speech-to-Text)...")
        
        # Import preprocessing function
        from audio import preprocess_audio_for_whisper
        
        # Preprocess audio
        audio_array = preprocess_audio_for_whisper(audio_bytes)
        if audio_array is None:
            raise ValueError("Failed to preprocess audio for STT")
        
        # Run transcription
        result = await self.transcription_service.transcribe_audio(audio_array)
        
        logger.info(f"STT complete: {len(result.get('text', ''))} chars")
        return result
    
    async def _run_ser(self, audio_bytes: bytes, sample_rate: int) -> Dict[str, Any]:
        """
        Run Speech Emotion Recognition model.
        
        Args:
            audio_bytes: Raw audio data
            sample_rate: Audio sample rate
            
        Returns:
            Emotion detection result with emotion, confidence, and scores
        """
        logger.info("Running SER (Speech Emotion Recognition)...")
        
        # Import preprocessing function
        from audio import preprocess_audio_for_whisper
        
        # Preprocess audio (same preprocessing for SER)
        audio_array = preprocess_audio_for_whisper(audio_bytes)
        if audio_array is None:
            raise ValueError("Failed to preprocess audio for SER")
        
        # Run emotion recognition
        result = await self.emotion_service.recognize_emotion(
            audio_array,
            sample_rate,
            return_all_scores=True
        )
        
        logger.info(f"SER complete: {result.get('emotion', 'unknown')}")
        return result
    
    async def _run_contextual_analysis(
        self,
        text: str,
        conversation_id: str,
        speaker_id: Optional[str],
        include_graph_updates: bool
    ) -> Dict[str, Any]:
        """
        Run contextual analysis (NER + COMET).
        
        Args:
            text: Transcribed text
            conversation_id: Conversation identifier
            speaker_id: Speaker identifier
            include_graph_updates: Whether to update graph
            
        Returns:
            Contextual analysis result with entities and emotional context
        """
        logger.info("Running contextual analysis (NER + COMET)...")
        
        result = await self.contextual_analyzer.analyze(
            text=text,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            include_graph_updates=include_graph_updates
        )
        
        logger.info(
            f"Contextual analysis complete: "
            f"{sum(len(v) for v in result.get('entities', {}).values())} entities"
        )
        return result
    
    def _build_response(
        self,
        transcript: Dict[str, Any],
        emotion: Dict[str, Any],
        contextual: Dict[str, Any],
        conversation_id: str,
        speaker_id: Optional[str],
        processing_time_ms: int,
        timestamp: str
    ) -> Dict[str, Any]:
        """
        Build aggregated response from all model results.
        
        Returns:
            Complete analysis result in structured format
        """
        return {
            "transcript": {
                "text": transcript.get("text", ""),
                "language": transcript.get("language", "unknown"),
                "confidence": transcript.get("confidence"),
                "error": transcript.get("error")
            },
            "emotion": {
                "from_audio": {
                    "primary": emotion.get("emotion", "unknown"),
                    "confidence": emotion.get("confidence", 0.0),
                    "all_scores": emotion.get("all_scores", {}),
                    "inference_time_ms": emotion.get("inference_time_ms")
                },
                "from_text": {
                    "detected": contextual.get("emotions_detected", []),
                    "context": contextual.get("emotional_context", {})
                }
            },
            "entities": contextual.get("entities", {}),
            "commonsense": {
                "emotional_context": contextual.get("emotional_context", {}),
                "inferences": self._extract_inferences(
                    contextual.get("emotional_context", {})
                )
            },
            "graph_updates": contextual.get("graph_updates"),
            "processing": {
                "total_time_ms": processing_time_ms,
                "stt_completed": "error" not in transcript,
                "ser_completed": "error" not in emotion,
                "ner_completed": bool(contextual.get("entities")),
                "comet_completed": bool(contextual.get("emotional_context")),
                "graph_updated": contextual.get("graph_updates") is not None
            },
            "metadata": {
                "conversation_id": conversation_id,
                "speaker_id": speaker_id,
                "timestamp": timestamp,
                "text_length": len(transcript.get("text", "")),
                "entity_count": sum(
                    len(v) for v in contextual.get("entities", {}).values()
                )
            }
        }
    
    def _extract_inferences(self, emotional_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure commonsense inferences.
        
        Args:
            emotional_context: Emotional context from COMET
            
        Returns:
            Structured inferences
        """
        return {
            "subject": {
                "feelings": emotional_context.get("subject_emotions", []),
                "wants": emotional_context.get("subject_wants", []),
                "effects": emotional_context.get("subject_effects", [])
            },
            "others": {
                "feelings": emotional_context.get("other_emotions", []),
                "wants": emotional_context.get("other_wants", []),
                "effects": emotional_context.get("other_effects", [])
            }
        }
    
    def _create_error_response(
        self,
        error_type: str,
        error_message: str,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Create error response structure.
        
        Args:
            error_type: Type of error
            error_message: Error message
            conversation_id: Conversation ID
            
        Returns:
            Error response structure
        """
        return {
            "error": error_type,
            "message": error_message,
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "transcript": None,
            "emotion": None,
            "entities": None,
            "commonsense": None,
            "graph_updates": None
        }
    
    def is_ready(self) -> bool:
        """
        Check if all required services are loaded and ready.
        
        Returns:
            True if all services are ready
        """
        return (
            self.transcription_service.is_loaded and
            self.emotion_service.is_loaded and
            self.contextual_analyzer.is_ready()
        )


# Global singleton instance (initialized in main.py)
chat_orchestrator: Optional[ChatOrchestrator] = None


def initialize_chat_orchestrator(
    transcription_service,
    emotion_service,
    contextual_analyzer
) -> None:
    """
    Initialize the global chat orchestrator.
    Should be called during application startup.
    
    Args:
        transcription_service: Transcription service instance
        emotion_service: Emotion recognition service instance
        contextual_analyzer: Contextual analyzer instance
    """
    global chat_orchestrator
    
    chat_orchestrator = ChatOrchestrator(
        transcription_service=transcription_service,
        emotion_service=emotion_service,
        contextual_analyzer=contextual_analyzer
    )
    
    logger.info("✅ Chat orchestrator initialized successfully")
