"""
LLM Service (Week 7)

Generates intelligent, context-aware responses using Large Language Models.
Supports OpenAI GPT models and local LLMs.
"""

from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM service for generating intelligent responses.
    Uses graph context and analysis data to provide enriched, context-aware answers.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize LLM service.
        
        Args:
            api_key: OpenAI API key (or None for local LLM)
            model: Model name (e.g., gpt-4, gpt-3.5-turbo, or local model)
            base_url: Optional base URL for local LLM endpoints
            temperature: Response creativity (0-1)
            max_tokens: Maximum response length
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        if api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.is_available = True
            logger.info(f"Initialized LLM service with model: {model}")
        else:
            self.client = None
            self.is_available = False
            logger.warning("LLM service initialized without API key (disabled)")
    
    async def generate_response(
        self,
        user_message: str,
        analysis_packet: Dict[str, Any],
        graph_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate intelligent response using analysis and context.
        
        Args:
            user_message: Transcribed user message
            analysis_packet: Complete analysis from orchestrator
            graph_context: Knowledge graph context
            conversation_history: Previous messages
            
        Returns:
            Response dictionary with text and metadata
        """
        if not self.is_available:
            return {
                "text": "I understand. How can I help you further?",
                "error": "LLM service not available",
                "fallback": True
            }
        
        try:
            # Build enriched context prompt
            context_prompt = self._build_context_prompt(
                user_message,
                analysis_packet,
                graph_context,
                conversation_history
            )
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": context_prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            return {
                "text": response_text,
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "text": "I'm having trouble processing your request right now. Could you try rephrasing?",
                "error": str(e),
                "fallback": True
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the AI assistant."""
        return """You are Aura, an empathetic and intelligent AI assistant with deep contextual understanding.

You have access to:
- Speech transcription and emotion recognition from voice
- Text-based emotional analysis and commonsense reasoning
- Knowledge graphs showing relationships between entities
- Conversation history for context

Your responses should:
1. Be warm, empathetic, and conversational
2. Acknowledge the user's emotions and feelings
3. Use context from the knowledge graph when relevant
4. Be helpful and actionable when appropriate
5. Ask clarifying questions if needed
6. Keep responses concise but complete (2-4 sentences usually)

Never:
- Ignore emotional cues
- Make assumptions without context
- Provide medical, legal, or financial advice
- Be overly formal or robotic"""
    
    def _build_context_prompt(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        graph_context: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]]
    ) -> str:
        """
        Build enriched prompt with all available context.
        
        Args:
            user_message: User's transcribed message
            analysis: Complete analysis packet
            graph_context: Graph data
            history: Conversation history
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # User message
        prompt_parts.append(f"User said: \"{user_message}\"")
        
        # Emotional context from audio
        emotion_audio = analysis.get("emotion", {}).get("from_audio", {})
        if emotion_audio.get("primary") and emotion_audio.get("primary") != "unknown":
            confidence = emotion_audio.get("confidence", 0)
            prompt_parts.append(
                f"\n[AUDIO EMOTION] {emotion_audio['primary']} "
                f"(confidence: {confidence:.0%})"
            )
            
            # Include all emotions if available
            all_scores = emotion_audio.get("all_scores", {})
            if all_scores:
                top_emotions = sorted(
                    all_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                emotions_str = ", ".join([f"{e}: {s:.0%}" for e, s in top_emotions])
                prompt_parts.append(f"  All scores: {emotions_str}")
        
        # Emotional context from text
        emotion_text = analysis.get("emotion", {}).get("from_text", {})
        if emotion_text.get("detected"):
            prompt_parts.append(
                f"\n[TEXT EMOTIONS] {', '.join(emotion_text['detected'])}"
            )
        
        # Entities mentioned
        entities = analysis.get("entities", {})
        if entities:
            prompt_parts.append("\n[ENTITIES MENTIONED]")
            for category, items in entities.items():
                if items:
                    names = [e["text"] for e in items]
                    prompt_parts.append(f"  {category.title()}: {', '.join(names)}")
        
        # Commonsense understanding
        commonsense = analysis.get("commonsense", {}).get("inferences", {})
        if commonsense:
            subject = commonsense.get("subject", {})
            
            if subject.get("feelings"):
                feelings_str = ", ".join(subject["feelings"][:3])
                prompt_parts.append(f"\n[USER LIKELY FEELS] {feelings_str}")
            
            if subject.get("wants"):
                wants_str = ", ".join(subject["wants"][:3])
                prompt_parts.append(f"[USER LIKELY WANTS] {wants_str}")
            
            if subject.get("effects"):
                effects_str = ", ".join(subject["effects"][:2])
                prompt_parts.append(f"[LIKELY EFFECTS] {effects_str}")
        
        # Graph context
        if graph_context:
            nodes = graph_context.get("nodes", [])
            relationships = graph_context.get("relationships", [])
            
            if nodes:
                prompt_parts.append(
                    f"\n[KNOWLEDGE GRAPH] {len(nodes)} related concepts, "
                    f"{len(relationships)} relationships"
                )
                
                # Sample some relevant nodes
                if len(nodes) > 0:
                    prompt_parts.append("[RELATED CONTEXT]")
                    for node in nodes[:5]:  # Show up to 5 nodes
                        if isinstance(node, dict) and "label" in node:
                            prompt_parts.append(f"  - {node.get('label')}")
        
        # Conversation history
        if history and len(history) > 0:
            prompt_parts.append("\n[RECENT CONVERSATION]")
            for msg in history[-3:]:  # Last 3 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:100]  # Truncate long messages
                prompt_parts.append(f"  {role.title()}: {content}")
        
        # Instructions
        prompt_parts.append(
            "\n[INSTRUCTIONS]"
            "\nRespond in a helpful, empathetic way that:"
            "\n- Acknowledges the user's emotions"
            "\n- Uses relevant context from entities and knowledge graph"
            "\n- Is conversational and natural"
            "\n- Offers support or helpful suggestions when appropriate"
            "\n- Asks clarifying questions if more context is needed"
        )
        
        return "\n".join(prompt_parts)
    
    async def generate_simple_response(
        self,
        user_message: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate simple response without full analysis.
        Useful for quick replies or fallback scenarios.
        
        Args:
            user_message: User's message
            context: Optional context string
            
        Returns:
            Response text
        """
        if not self.is_available:
            return "I understand. How can I help you?"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": f"{context}\n\nUser: {user_message}" if context else user_message
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Simple response generation failed: {e}")
            return "I'm having trouble right now. Could you try again?"
    
    def is_ready(self) -> bool:
        """Check if LLM service is ready to use."""
        return self.is_available and self.client is not None


# Global singleton (initialized in main.py)
llm_service: Optional[LLMService] = None


def initialize_llm_service(
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    base_url: Optional[str] = None
) -> None:
    """
    Initialize the global LLM service.
    Should be called during application startup.
    
    Args:
        api_key: OpenAI API key
        model: Model to use
        base_url: Optional base URL for local LLMs
    """
    global llm_service
    
    llm_service = LLMService(
        api_key=api_key,
        model=model,
        base_url=base_url
    )
    
    if llm_service.is_ready():
        logger.info(f"✅ LLM service initialized with {model}")
    else:
        logger.warning("⚠️  LLM service initialized but not available")
