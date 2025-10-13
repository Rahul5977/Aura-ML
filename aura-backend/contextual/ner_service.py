"""
Named Entity Recognition (NER) Service

Uses spaCy to identify and categorize entities in conversational text:
- PERSON: People mentioned in the conversation
- GPE/LOC: Places and locations
- ORG: Organizations
- PRODUCT/EVENT/WORK_OF_ART: Concepts and topics
"""

import logging
from typing import Dict, List, Optional
import asyncio

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Language = None

logger = logging.getLogger(__name__)


class NERService:
    """
    Service for Named Entity Recognition using spaCy.
    Extracts and categorizes entities from conversational text.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize NER service.
        
        Args:
            model_name: spaCy model to use (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        self.model_name = model_name
        self.nlp: Optional[Language] = None
        self.is_loaded = False
        
        logger.info(f"Initializing NERService with model: {model_name}")
    
    def load_model(self) -> None:
        """
        Load spaCy NER model.
        Downloads the model if not already installed.
        """
        if not SPACY_AVAILABLE:
            logger.error("spaCy not installed. Install with: pip install spacy")
            raise RuntimeError("spaCy not available")
        
        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            
            # Try to load the model
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                # Model not found, try to download it
                logger.info(f"Model {self.model_name} not found, downloading...")
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", self.model_name],
                    check=True,
                    capture_output=True
                )
                self.nlp = spacy.load(self.model_name)
            
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
    
    async def extract_entities(
        self,
        text: str,
        include_confidence: bool = False
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to analyze
            include_confidence: Whether to include confidence scores (if available)
            
        Returns:
            Dictionary with categorized entities:
            {
                "people": [{"text": "John", "start": 0, "end": 4}],
                "places": [{"text": "Seattle", "start": 25, "end": 32}],
                "organizations": [...],
                "concepts": [...]
            }
        """
        if not self.is_loaded:
            raise RuntimeError("NER model not loaded. Call load_model() first.")
        
        try:
            # Run NER in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._extract_entities_sync,
                text,
                include_confidence
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            raise
    
    def _extract_entities_sync(
        self,
        text: str,
        include_confidence: bool
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Synchronous entity extraction (runs in thread pool).
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Categorize entities
        entities = {
            "people": [],
            "places": [],
            "organizations": [],
            "concepts": [],
            "dates": [],
            "other": []
        }
        
        for ent in doc.ents:
            entity_info = {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            }
            
            # Add confidence if available (spaCy 3.0+)
            if include_confidence and hasattr(ent, 'confidence'):
                entity_info["confidence"] = ent.confidence
            
            # Categorize by entity type
            if ent.label_ == "PERSON":
                entities["people"].append(entity_info)
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                entities["places"].append(entity_info)
            elif ent.label_ == "ORG":
                entities["organizations"].append(entity_info)
            elif ent.label_ in ["PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
                entities["concepts"].append(entity_info)
            elif ent.label_ in ["DATE", "TIME"]:
                entities["dates"].append(entity_info)
            else:
                entities["other"].append(entity_info)
        
        logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities from text")
        
        return entities
    
    async def extract_entities_batch(
        self,
        texts: List[str],
        include_confidence: bool = False
    ) -> List[Dict[str, List[Dict[str, any]]]]:
        """
        Extract entities from multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            include_confidence: Whether to include confidence scores
            
        Returns:
            List of entity dictionaries, one per input text
        """
        if not self.is_loaded:
            raise RuntimeError("NER model not loaded. Call load_model() first.")
        
        try:
            # Process all texts concurrently
            tasks = [
                self.extract_entities(text, include_confidence)
                for text in texts
            ]
            
            results = await asyncio.gather(*tasks)
            return results
            
        except Exception as e:
            logger.error(f"Error in batch entity extraction: {e}")
            raise
    
    def get_entity_summary(self, entities: Dict[str, List[Dict[str, any]]]) -> Dict[str, int]:
        """
        Get a summary count of entities by category.
        
        Args:
            entities: Entity dictionary from extract_entities()
            
        Returns:
            Count of entities by category
        """
        return {
            category: len(entity_list)
            for category, entity_list in entities.items()
        }


# Global singleton instance
ner_service = NERService()


def initialize_ner_service(model_name: str = "en_core_web_sm") -> None:
    """
    Initialize the global NER service.
    Should be called during application startup.
    
    Args:
        model_name: spaCy model to use
    """
    global ner_service
    
    if model_name != ner_service.model_name:
        ner_service = NERService(model_name)
    
    if not ner_service.is_loaded:
        ner_service.load_model()
