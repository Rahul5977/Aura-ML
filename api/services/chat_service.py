"""
Chat Service - Business logic for chat functionality
"""

import sys
from pathlib import Path
import logging
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aura_ml.models.llm_wrapper import AuraLLM
from aura_ml.inference.chatbot import AuraChatbot
from aura_ml.config.model_config import LLMConfig, InferenceConfig
from aura_ml.config.settings import settings

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat operations"""
    
    def __init__(self):
        """Initialize chat service"""
        self.llm: Optional[AuraLLM] = None
        self.chatbot: Optional[AuraChatbot] = None
        self._ready = False
        
    async def initialize(self):
        """Initialize the service (load models)"""
        try:
            logger.info("Initializing chat service...")
            
            # Create configurations
            llm_config = LLMConfig()
            inference_config = InferenceConfig(enable_streaming=False)  # No streaming in API
            
            # Initialize LLM
            logger.info(f"Loading model from {settings.LLM_MODEL_PATH}")
            self.llm = AuraLLM(
                model_path=settings.LLM_MODEL_PATH,
                config=llm_config,
                inference_config=inference_config
            )
            self.llm.load_model()
            
            # Create chatbot
            self.chatbot = AuraChatbot(self.llm, inference_config)
            
            self._ready = True
            logger.info("Chat service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up chat service...")
        self.llm = None
        self.chatbot = None
        self._ready = False
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._ready and self.chatbot is not None
    
    async def chat(
        self,
        message: str,
        emotion: Optional[str] = None,
        cause: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.7
    ) -> str:
        """
        Generate chat response
        
        Args:
            message: User's message
            emotion: Current emotion
            cause: Cause of the emotion
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            AI's response
        """
        if not self.is_ready():
            raise RuntimeError("Chat service not initialized")
        
        # Set emotion context if provided
        if emotion:
            self.chatbot.set_emotion_context(emotion, cause)
        
        # Generate response
        response = self.chatbot.chat(message, stream=False)
        
        return response
