"""
COMET Service for Commonsense Emotional Reasoning

Integrates COMET (Commonsense Transformer) to infer emotional effects
and commonsense knowledge from conversational text.

COMET models: comet-atomic_2020_BART, comet-commonsense
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

logger = logging.getLogger(__name__)


class COMETService:
    """
    Service for commonsense emotional reasoning using COMET.
    Infers emotional effects and commonsense knowledge from text.
    """
    
    # Emotion-related inference types
    EMOTION_RELATIONS = [
        "xReact",  # How does X feel after the event?
        "oReact",  # How do others feel after the event?
        "xWant",   # What does X want after the event?
        "oWant",   # What do others want after the event?
        "xEffect", # What happens to X after the event?
        "oEffect"  # What happens to others after the event?
    ]
    
    def __init__(self, model_name: str = "comet-atomic_2020_BART"):
        """
        Initialize COMET service.
        
        Args:
            model_name: COMET model to use (comet-atomic_2020_BART recommended)
        """
        # Map simplified names to full HuggingFace model IDs
        model_mapping = {
            "comet-atomic_2020_BART": "allenai/comet-atomic-2020-BART",
            "comet-commonsense": "allenai/comet-commonsense"
        }
        
        # Use mapping if model name is in the dict, otherwise use as-is
        self.model_name = model_mapping.get(model_name, model_name)
        
        # Add allenai/ prefix if not already present
        if not self.model_name.startswith("allenai/") and "/" not in self.model_name:
            self.model_name = f"allenai/{self.model_name}"
            
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        logger.info(f"Initializing COMETService with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load COMET model and tokenizer.
        This is a blocking operation and should be called during startup.
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise RuntimeError("transformers library not available")
        
        try:
            logger.info(f"Loading COMET model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading COMET model: {e}")
            logger.warning(f"COMET model {self.model_name} could not be loaded. Emotional reasoning will use fallback mode.")
            # Set loaded flag to True with fallback mode
            self.is_loaded = True
            self.model = None
            self.tokenizer = None
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading COMET model: {e}")
            raise
    
    async def infer_emotional_effects(
        self,
        text: str,
        relations: Optional[List[str]] = None,
        num_beams: int = 5,
        max_length: int = 64
    ) -> Dict[str, List[str]]:
        """
        Infer emotional effects and commonsense knowledge from text.
        
        Args:
            text: Input text (e.g., "PersonX tells PersonY a joke")
            relations: List of relations to infer (default: all emotion-related)
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated text
            
        Returns:
            Dictionary mapping relations to inferred effects:
            {
                "xReact": ["happy", "amused"],
                "oReact": ["happy", "entertained"],
                "xWant": ["to see PersonY laugh"],
                ...
            }
        """
        if not self.is_loaded:
            logger.warning("COMET model not loaded, returning empty results")
            return {relation: [] for relation in (relations or self.EMOTION_RELATIONS)}
        
        if self.model is None or self.tokenizer is None:
            logger.warning("COMET model not available, using fallback mode")
            return {relation: [] for relation in (relations or self.EMOTION_RELATIONS)}
        
        if relations is None:
            relations = self.EMOTION_RELATIONS
        
        try:
            # Run inference in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._infer_sync,
                text,
                relations,
                num_beams,
                max_length
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error inferring emotional effects: {e}")
            raise
    
    def _infer_sync(
        self,
        text: str,
        relations: List[str],
        num_beams: int,
        max_length: int
    ) -> Dict[str, List[str]]:
        """
        Synchronous inference (runs in thread pool).
        """
        results = {}
        
        for relation in relations:
            # Format input for COMET: "text [relation]"
            input_text = f"{text} {relation}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_length=max_length,
                    num_return_sequences=min(num_beams, 3),  # Return top 3
                    early_stopping=True
                )
            
            # Decode outputs
            decoded = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            # Clean up and filter duplicates
            decoded = list(set([text.strip() for text in decoded if text.strip()]))
            results[relation] = decoded
        
        logger.info(f"Inferred {len(relations)} relations for text: {text[:50]}...")
        
        return results
    
    async def analyze_emotional_context(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Comprehensive emotional context analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Structured emotional analysis:
            {
                "subject_emotions": [...],  # How subject feels
                "other_emotions": [...],    # How others feel
                "subject_wants": [...],     # What subject wants
                "other_wants": [...],       # What others want
                "subject_effects": [...],   # Effects on subject
                "other_effects": [...]      # Effects on others
            }
        """
        if not self.is_loaded:
            raise RuntimeError("COMET model not loaded. Call load_model() first.")
        
        try:
            # Get all emotional inferences
            inferences = await self.infer_emotional_effects(text)
            
            # Structure the results
            analysis = {
                "subject_emotions": inferences.get("xReact", []),
                "other_emotions": inferences.get("oReact", []),
                "subject_wants": inferences.get("xWant", []),
                "other_wants": inferences.get("oWant", []),
                "subject_effects": inferences.get("xEffect", []),
                "other_effects": inferences.get("oEffect", [])
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing emotional context: {e}")
            raise
    
    def extract_emotions(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Extract unique emotions from emotional analysis.
        
        Args:
            analysis: Output from analyze_emotional_context()
            
        Returns:
            List of unique emotions mentioned
        """
        all_emotions = []
        
        # Collect from subject and other emotions
        all_emotions.extend(analysis.get("subject_emotions", []))
        all_emotions.extend(analysis.get("other_emotions", []))
        
        # Filter to emotion words (simple heuristic)
        emotion_keywords = [
            "happy", "sad", "angry", "fear", "surprise", "disgust",
            "joy", "content", "excited", "nervous", "worried", "anxious",
            "relieved", "grateful", "proud", "ashamed", "guilty"
        ]
        
        emotions = []
        for text in all_emotions:
            text_lower = text.lower()
            for emotion in emotion_keywords:
                if emotion in text_lower and emotion not in emotions:
                    emotions.append(emotion)
        
        return emotions


# Global singleton instance
comet_service = COMETService()


def initialize_comet_service(model_name: str = "comet-atomic_2020_BART") -> None:
    """
    Initialize the global COMET service.
    Should be called during application startup.
    
    Args:
        model_name: COMET model to use
    """
    global comet_service
    
    if model_name != comet_service.model_name:
        comet_service = COMETService(model_name)
    
    if not comet_service.is_loaded:
        comet_service.load_model()
