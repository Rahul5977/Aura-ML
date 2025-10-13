"""
Contextual Analyzer

Main orchestrator for contextual analysis.
Coordinates NER, COMET, and Knowledge Graph services to provide
comprehensive understanding of conversational dynamics.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from .ner_service import ner_service
from .comet_service import comet_service
from .knowledge_graph_service import knowledge_graph_service

logger = logging.getLogger(__name__)


class ContextualAnalyzer:
    """
    Main analyzer for contextual understanding of conversations.
    Orchestrates multiple services to extract entities, emotions, and knowledge.
    """
    
    def __init__(self):
        """Initialize contextual analyzer."""
        self.ner = ner_service
        self.comet = comet_service
        self.kg = knowledge_graph_service
        
        logger.info("Initialized ContextualAnalyzer")
    
    async def analyze(
        self,
        text: str,
        conversation_id: str,
        speaker_id: Optional[str] = None,
        include_graph_updates: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive contextual analysis of conversational text.
        
        Args:
            text: Input text to analyze
            conversation_id: ID of the conversation
            speaker_id: Optional speaker identifier
            include_graph_updates: Whether to update knowledge graph
            
        Returns:
            Complete analysis including:
            - entities: Extracted named entities
            - emotional_context: COMET emotional analysis
            - graph_summary: Knowledge graph updates
            - metadata: Analysis metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Run NER and COMET analysis in parallel
            logger.info(f"Starting contextual analysis for conversation {conversation_id}")
            
            entities_task = self.ner.extract_entities(text, include_confidence=True)
            emotional_task = self.comet.analyze_emotional_context(text)
            
            entities, emotional_context = await asyncio.gather(
                entities_task,
                emotional_task,
                return_exceptions=True
            )
            
            # Handle potential errors
            if isinstance(entities, Exception):
                logger.error(f"NER failed: {entities}")
                entities = {"people": [], "places": [], "organizations": [], "concepts": []}
            
            if isinstance(emotional_context, Exception):
                logger.error(f"COMET failed: {emotional_context}")
                emotional_context = {
                    "subject_emotions": [],
                    "other_emotions": [],
                    "subject_wants": [],
                    "other_wants": [],
                    "subject_effects": [],
                    "other_effects": []
                }
            
            # Update knowledge graph if requested
            graph_updates = None
            if include_graph_updates:
                graph_updates = await self._update_knowledge_graph(
                    text=text,
                    conversation_id=conversation_id,
                    speaker_id=speaker_id,
                    entities=entities,
                    emotional_context=emotional_context
                )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Build result
            result = {
                "entities": entities,
                "emotional_context": emotional_context,
                "emotions_detected": self.comet.extract_emotions(emotional_context),
                "graph_updates": graph_updates,
                "metadata": {
                    "conversation_id": conversation_id,
                    "speaker_id": speaker_id,
                    "text_length": len(text),
                    "processing_time_ms": processing_time_ms,
                    "timestamp": end_time.isoformat()
                }
            }
            
            logger.info(
                f"Contextual analysis complete: "
                f"{sum(len(v) for v in entities.values())} entities, "
                f"{len(result['emotions_detected'])} emotions "
                f"({processing_time_ms}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in contextual analysis: {e}")
            raise
    
    async def _update_knowledge_graph(
        self,
        text: str,
        conversation_id: str,
        speaker_id: Optional[str],
        entities: Dict[str, List[Dict[str, Any]]],
        emotional_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update knowledge graph with analysis results.
        
        Returns:
            Summary of graph updates
        """
        try:
            # Add conversation node
            conv_node = await self.kg.add_conversation_context(
                conversation_id=conversation_id,
                text=text,
                speaker_id=speaker_id
            )
            
            # Add entity nodes
            entity_nodes = await self.kg.add_entity_nodes(
                entities=entities,
                conversation_id=conversation_id
            )
            
            # Add emotional relationships if speaker is identified
            emotional_rels = []
            if speaker_id:
                speaker_node_id = f"PERSON_{speaker_id}"
                emotional_rels = await self.kg.add_emotional_relationships(
                    emotional_analysis=emotional_context,
                    subject_node_id=speaker_node_id,
                    conversation_id=conversation_id
                )
            
            return {
                "conversation_node": conv_node.to_dict(),
                "entity_nodes_count": len(entity_nodes),
                "emotional_relationships_count": len(emotional_rels),
                "graph_summary": await self.kg.get_graph_summary()
            }
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}")
            return {
                "error": str(e),
                "conversation_node": None,
                "entity_nodes_count": 0,
                "emotional_relationships_count": 0
            }
    
    async def analyze_batch(
        self,
        texts: List[str],
        conversation_ids: List[str],
        speaker_ids: Optional[List[str]] = None,
        include_graph_updates: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            conversation_ids: List of conversation IDs
            speaker_ids: Optional list of speaker IDs
            include_graph_updates: Whether to update knowledge graph
            
        Returns:
            List of analysis results
        """
        if len(texts) != len(conversation_ids):
            raise ValueError("texts and conversation_ids must have same length")
        
        if speaker_ids and len(speaker_ids) != len(texts):
            raise ValueError("speaker_ids must match texts length")
        
        # Create analysis tasks
        tasks = []
        for i, (text, conv_id) in enumerate(zip(texts, conversation_ids)):
            speaker = speaker_ids[i] if speaker_ids else None
            tasks.append(
                self.analyze(
                    text=text,
                    conversation_id=conv_id,
                    speaker_id=speaker,
                    include_graph_updates=include_graph_updates
                )
            )
        
        # Run in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed for index {i}: {result}")
                results[i] = {
                    "error": str(result),
                    "conversation_id": conversation_ids[i]
                }
        
        return results
    
    async def get_conversation_context(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get accumulated context for a conversation from the knowledge graph.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation context with entities and emotions
        """
        try:
            node_id = f"CONV_{conversation_id}"
            
            # Query related entities and emotions
            related = await self.kg.query_related_entities(
                node_id=node_id,
                max_depth=2
            )
            
            return {
                "conversation_id": conversation_id,
                "related_entities": related["nodes"],
                "relationships": related["relationships"],
                "summary": await self.kg.get_graph_summary()
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {
                "conversation_id": conversation_id,
                "error": str(e)
            }
    
    def is_ready(self) -> bool:
        """
        Check if all services are loaded and ready.
        
        Returns:
            True if all services are ready
        """
        return self.ner.is_loaded and self.comet.is_loaded


# Global singleton instance
contextual_analyzer = ContextualAnalyzer()


async def initialize_contextual_services() -> None:
    """
    Initialize all contextual analysis services.
    Should be called during application startup.
    """
    logger.info("Initializing contextual analysis services...")
    
    # Initialize services in parallel
    from .ner_service import initialize_ner_service
    from .comet_service import initialize_comet_service
    
    try:
        # Load NER model
        ner_task = asyncio.get_event_loop().run_in_executor(
            None,
            initialize_ner_service,
            "en_core_web_sm"
        )
        
        # Load COMET model
        comet_task = asyncio.get_event_loop().run_in_executor(
            None,
            initialize_comet_service,
            "comet-atomic_2020_BART"
        )
        
        await asyncio.gather(ner_task, comet_task)
        
        logger.info("✅ All contextual analysis services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize contextual services: {e}")
        raise
