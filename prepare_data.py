"""
Emotion Cause Extraction (ECE) Training Data Preparation Script
================================================================

Purpose: Generate high-quality training dataset for ECE model by processing
         the ESConv emotional support conversation dataset with sophisticated
         linguistic heuristics.

Author: Senior Data Engineering Team
Date: November 19, 2025
Version: 2.0.0

Dataset Structure:
- Input: esconv_dataset/{train,validation,test}.jsonl
- Output: ece_training_data.json
- Target: 15,000+ labeled samples

Output Format:
{
    "text": str,           # The complete sentence/utterance
    "emotion": str,        # Detected emotion type
    "cause_span": str,     # Extracted cause text
    "bio_tags": list       # BIO tags for each token [B-CAUSE, I-CAUSE, O]
}

Linguistic Heuristics ("The Why Engine"):
==========================================

The script employs a sophisticated multi-strategy approach to identify emotional causes
without explicit labels. This "Why Engine" uses:

1. EXPLICIT CAUSAL MARKERS (Confidence: 0.85-0.95)
   - Direct causality: "because", "since", "due to", "owing to"
   - Explicit reasons: "the reason is", "the cause is"
   - Examples:
     * "I'm anxious BECAUSE I might lose my job" → cause: "I might lose my job"
     * "Depressed DUE TO my father's death" → cause: "my father's death"

2. TEMPORAL CAUSAL MARKERS (Confidence: 0.70-0.85)
   - Time-based causality: "when", "after", "while", "once", "ever since"
   - Examples:
     * "I felt sad WHEN she left" → cause: "she left"
     * "Stressed AFTER moving houses" → cause: "moving houses"

3. CONDITIONAL CAUSAL MARKERS (Confidence: 0.65-0.70)
   - Hypothetical causality: "if", "whenever", "in case"
   - Examples:
     * "Worried IF I fail the exam" → cause: "I fail the exam"
     * "Anxious WHENEVER I'm alone" → cause: "I'm alone"

4. EMOTIONAL CONTEXT MARKERS (Confidence: 0.75-0.85)
   - Domain-specific patterns: "worried about", "anxious that", "scared of"
   - Emotional verbs: "makes me feel", "I feel...because"
   - Examples:
     * "I'm WORRIED ABOUT losing my income" → cause: "losing my income"
     * "It MAKES ME FEEL upset" → cause preceding context

5. ADVERSATIVE MARKERS (Confidence: 0.60-0.65)
   - Contrast-based causality: "but", "however", "although", "even though"
   - Examples:
     * "I want to work BUT I'm sick" → cause: "I'm sick"
     * "Happy ALTHOUGH things are hard" → cause: "things are hard"

6. EXPLANATORY MARKERS (Confidence: 0.80-0.85)
   - Problem statements: "the problem is", "the issue is", "what happened was"
   - Examples:
     * "THE PROBLEM IS I lost my job" → cause: "I lost my job"

Intelligent Boundary Detection:
================================

The system uses multi-level boundary detection for precise cause extraction:

1. Sentence boundaries: Period, exclamation, question mark
2. Clause boundaries: Semicolons, coordinating conjunctions (and, but, or)
3. Soft boundaries: Commas (for complex sentences)
4. Logical breaks: Coordinating conjunctions that signal topic shift

Example:
"I'm anxious because I lost my job, and now I can't pay rent, so I'm stressed."
                    └──────────────┘
                    Extracted cause (stops at comma + conjunction)

Emotion Detection Strategy:
============================

Multi-layered emotion detection:

1. Context emotion (from conversation metadata)
2. Keyword matching with intensity weighting
3. Intensifier detection: "very", "extremely", "really", "so"
4. Default to "neutral" when no causal markers found (as per requirements)

Example:
"I'm REALLY anxious about the interview"
         └────┘ intensifier → boosts anxiety confidence score

Validation & Filtering:
========================

Strict quality controls ensure high-quality training data:

1. Minimum cause length: 10 characters, 2+ words
2. Semantic validation: Must contain verbs or nouns (spaCy POS tagging)
3. Exclude questions, pure punctuation, filler words
4. Confidence threshold: Default 0.6 (configurable)
5. BIO tag alignment: Ensures tags match token count exactly

Example of rejected causes:
❌ "it" (too short, no meaningful content)
❌ "the and or" (only filler words)
❌ "Is that true?" (question)
✅ "I lost my job" (valid: has verb, noun, meaningful)

BIO Tagging Algorithm:
=======================

Precise token-level labeling using spaCy tokenization:

B-CAUSE: Beginning of cause span (first token)
I-CAUSE: Inside cause span (continuation tokens)
O: Outside cause span (not part of cause)

Example:
Text: "I'm anxious because I lost my job last week"
Tokens: ["I", "'m", "anxious", "because", "I", "lost", "my", "job", "last", "week"]
Tags:   [O,   O,    O,         O,         B-CAUSE, I-CAUSE, I-CAUSE, I-CAUSE, I-CAUSE, I-CAUSE]
Cause span: "I lost my job last week"

Error Handling:
===============

Comprehensive error handling for production reliability:

1. Missing dataset files → Detailed error message with expected structure
2. spaCy model missing → Automatic download with fallback instructions
3. JSON parsing errors → Skip malformed records, log warnings
4. Tag-token mismatches → Validation and quality scoring
5. Empty results → Clear diagnostics and suggestions

Usage:
======

Basic usage:
    python prepare_data.py

Requirements:
    pip install spacy tqdm
    python -m spacy download en_core_web_sm

Output validation:
    - Quality score: 0-100 (based on validation checks)
    - Sample examples displayed for manual verification
    - Statistics: emotion distribution, cause length, etc.

Expected Output:
    ece_training_data.json (15,000+ samples if sufficient data available)

Notes:
======

1. The "neutral emotion" rule: If no causal markers are found in text,
   the emotion is classified as "neutral" and the sample is filtered out
   (as per requirement: "if there is no such words...keep is to neutral emotion")

2. Confidence tuning: Adjust min_confidence in process_all() to control
   quality vs quantity trade-off:
   - 0.8+: High precision, fewer samples
   - 0.6: Balanced (default)
   - 0.5: More samples, some noise

3. spaCy tokenization: Critical for BIO tag alignment. DO NOT use simple
   split() - it will cause misalignment with contractions and punctuation.

"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import spacy
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CausalMarker:
    """
    Represents a linguistic marker that signals causal relationships.
    
    Attributes:
        pattern: Regex pattern to match the marker
        type: Type of causality (explicit, temporal, conditional, etc.)
        confidence: Confidence score for this marker (0.0-1.0)
        capture_after: Whether to capture text after the marker
    """
    pattern: str
    type: str
    confidence: float
    capture_after: bool = True


class CauseExtractor:
    """
    Sophisticated cause extraction engine using linguistic heuristics.
    
    This class implements a multi-strategy approach to identifying emotional causes:
    1. Explicit causal markers (because, since, due to)
    2. Temporal markers (when, after, while)
    3. Conditional markers (if, whenever)
    4. Consequential markers (so that, therefore)
    5. Contextual patterns (specific to emotional conversations)
    """
    
    # Comprehensive causal marker taxonomy
    CAUSAL_MARKERS = [
        # Explicit causality (highest confidence)
        CausalMarker(r'\bbecause\b', 'explicit', 0.95, True),
        CausalMarker(r'\bsince\b', 'explicit', 0.90, True),
        CausalMarker(r'\bdue to\b', 'explicit', 0.95, True),
        CausalMarker(r'\bowning to\b', 'explicit', 0.90, True),
        CausalMarker(r'\bas a result of\b', 'explicit', 0.92, True),
        CausalMarker(r'\bon account of\b', 'explicit', 0.90, True),
        CausalMarker(r'\bthanks to\b', 'explicit', 0.85, True),
        CausalMarker(r'\bthe reason is\b', 'explicit', 0.93, True),
        CausalMarker(r'\bthe cause is\b', 'explicit', 0.93, True),
        
        # Temporal causality (medium-high confidence)
        CausalMarker(r'\bwhen\b', 'temporal', 0.75, True),
        CausalMarker(r'\bafter\b', 'temporal', 0.80, True),
        CausalMarker(r'\bwhile\b', 'temporal', 0.70, True),
        CausalMarker(r'\bonce\b', 'temporal', 0.72, True),
        CausalMarker(r'\bever since\b', 'temporal', 0.85, True),
        
        # Conditional causality (medium confidence)
        CausalMarker(r'\bif\b', 'conditional', 0.65, True),
        CausalMarker(r'\bwhenever\b', 'conditional', 0.70, True),
        CausalMarker(r'\bin case\b', 'conditional', 0.68, True),
        
        # Consequential markers (medium confidence)
        CausalMarker(r'\bso that\b', 'consequential', 0.75, True),
        CausalMarker(r'\btherefore\b', 'consequential', 0.70, False),
        CausalMarker(r'\bthus\b', 'consequential', 0.68, False),
        CausalMarker(r'\bhence\b', 'consequential', 0.68, False),
        
        # Emotional context markers (domain-specific)
        CausalMarker(r'\bmakes? me (feel|think)\b', 'emotional', 0.80, False),
        CausalMarker(r'\bi (feel|felt) .{0,20} (because|when|after)\b', 'emotional', 0.85, True),
        CausalMarker(r'\bworried (that|about)\b', 'emotional', 0.75, True),
        CausalMarker(r'\banxious (about|that)\b', 'emotional', 0.75, True),
        CausalMarker(r'\bscared (of|that)\b', 'emotional', 0.75, True),
        CausalMarker(r'\bfrustrated (by|with)\b', 'emotional', 0.75, True),
        CausalMarker(r'\bdepressed (about|over)\b', 'emotional', 0.75, True),
        CausalMarker(r'\bupset (about|by)\b', 'emotional', 0.75, True),
        
        # Adversative markers (but/however indicating contrast-cause)
        CausalMarker(r'\bbut\b', 'adversative', 0.60, True),
        CausalMarker(r'\bhowever\b', 'adversative', 0.58, True),
        CausalMarker(r'\balthough\b', 'adversative', 0.62, True),
        CausalMarker(r'\beven though\b', 'adversative', 0.65, True),
        
        # Comparative causality
        CausalMarker(r'\bgiven that\b', 'comparative', 0.80, True),
        CausalMarker(r'\bconsidering (that)?\b', 'comparative', 0.75, True),
        CausalMarker(r'\bseeing (that|as)\b', 'comparative', 0.77, True),
        
        # Explanatory markers
        CausalMarker(r'\bthe problem is\b', 'explanatory', 0.85, True),
        CausalMarker(r'\bthe issue is\b', 'explanatory', 0.85, True),
        CausalMarker(r'\bwhat happened (is|was)\b', 'explanatory', 0.82, True),
        CausalMarker(r'\bi\'m .{0,15} (because|since|as)\b', 'explanatory', 0.83, True),
    ]
    
    # Emotion keywords for emotion detection
    EMOTION_KEYWORDS = {
        'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'panic', 'fear', 'scared', 'afraid'],
        'depression': ['depressed', 'sad', 'down', 'hopeless', 'miserable', 'despair', 'unhappy'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'rage'],
        'fear': ['afraid', 'scared', 'terrified', 'frightened', 'fearful', 'worried'],
        'sadness': ['sad', 'unhappy', 'sorrowful', 'heartbroken', 'grief', 'mourning'],
        'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'grossed out'],
        'shame': ['ashamed', 'embarrassed', 'humiliated', 'guilty', 'mortified'],
        'joy': ['happy', 'joyful', 'delighted', 'pleased', 'glad', 'cheerful'],
        'neutral': ['okay', 'fine', 'alright', 'normal', 'stable']
    }
    
    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        """
        Initialize the CauseExtractor with spaCy NLP model.
        
        Args:
            spacy_model: Name of spaCy model to use for tokenization
        """
        logger.info(f"🔧 Loading spaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"✅ spaCy model '{spacy_model}' loaded successfully")
        except OSError:
            logger.warning(f"⚠️  Model '{spacy_model}' not found. Attempting to download...")
            try:
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, '-m', 'spacy', 'download', spacy_model],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    error_msg = (
                        f"Failed to download spaCy model '{spacy_model}'.\n"
                        f"Please install manually: python -m spacy download {spacy_model}\n"
                        f"Error: {result.stderr}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                self.nlp = spacy.load(spacy_model)
                logger.info(f"✅ Model '{spacy_model}' downloaded and loaded successfully")
            except Exception as e:
                error_msg = (
                    f"Failed to download/load spaCy model.\n"
                    f"Please install manually:\n"
                    f"  pip install spacy\n"
                    f"  python -m spacy download {spacy_model}\n"
                    f"Error: {str(e)}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        logger.info("✅ CauseExtractor initialized successfully")
    
    def detect_emotion(self, text: str, context_emotion: Optional[str] = None) -> str:
        """
        Detect emotion from text using keyword matching and context.
        
        Args:
            text: Input text to analyze
            context_emotion: Emotion from conversation metadata (if available)
        
        Returns:
            Detected emotion label
        """
        text_lower = text.lower()
        
        # Use context emotion if available and valid
        if context_emotion and context_emotion.lower() in self.EMOTION_KEYWORDS:
            # Verify the context emotion is actually present in text
            context_keywords = self.EMOTION_KEYWORDS.get(context_emotion.lower(), [])
            if any(keyword in text_lower for keyword in context_keywords):
                return context_emotion.lower()
        
        # Enhanced keyword-based emotion detection with intensity weighting
        emotion_scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Check for intensifiers
                    intensifiers = ['very', 'extremely', 'really', 'so', 'too', 'incredibly']
                    context_window = text_lower[max(0, text_lower.index(keyword)-20):text_lower.index(keyword)]
                    
                    base_score = 1.0
                    if any(intensifier in context_window for intensifier in intensifiers):
                        base_score = 1.5  # Boost score for intensified emotions
                    
                    score += base_score
            
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        return 'neutral'
    
    def extract_cause(self, text: str) -> Optional[Tuple[str, float, int, int]]:
        """
        Extract causal span from text using linguistic heuristics.
        
        Strategy:
        1. Scan text for causal markers
        2. Extract text following/preceding marker based on type
        3. Clean and validate extracted span
        4. Return highest confidence match
        
        Args:
            text: Input text to analyze
        
        Returns:
            Tuple of (cause_text, confidence, start_char, end_char) or None
        """
        best_match = None
        best_confidence = 0.0
        
        for marker in self.CAUSAL_MARKERS:
            matches = list(re.finditer(marker.pattern, text, re.IGNORECASE))
            
            for match in matches:
                marker_start, marker_end = match.span()
                
                if marker.capture_after:
                    # Extract text after the marker
                    cause_start = marker_end
                    
                    # Find intelligent boundary (sentence end, clause boundary, or logical break)
                    sentence_end = text.find('.', cause_start)
                    comma_end = text.find(',', cause_start)
                    semicolon_end = text.find(';', cause_start)
                    
                    # Look for coordinating conjunctions that might signal clause boundary
                    coord_conj_pattern = r'\b(and|but|or|so|yet)\b'
                    coord_match = re.search(coord_conj_pattern, text[cause_start:], re.IGNORECASE)
                    coord_end = cause_start + coord_match.start() if coord_match else -1
                    
                    # Choose nearest meaningful boundary
                    boundaries = [len(text)]
                    if sentence_end > 0:
                        boundaries.append(sentence_end)
                    if semicolon_end > 0:
                        boundaries.append(semicolon_end)
                    if comma_end > cause_start + 15:  # Avoid too-short spans
                        boundaries.append(comma_end)
                    if coord_end > cause_start + 10:
                        boundaries.append(coord_end)
                    
                    cause_end = min(boundaries)
                    cause_text = text[cause_start:cause_end].strip()
                else:
                    # Extract text before the marker (for consequential markers)
                    cause_end = marker_start
                    # Find previous sentence start or clause boundary
                    prev_period = text.rfind('.', 0, marker_start)
                    prev_comma = text.rfind(',', max(0, marker_start - 50), marker_start)
                    
                    # Choose better starting point
                    if prev_comma > prev_period and prev_comma > marker_start - 40:
                        cause_start = prev_comma + 1
                    else:
                        cause_start = prev_period + 1 if prev_period > 0 else 0
                    
                    cause_text = text[cause_start:cause_end].strip()
                
                # Validate cause span
                if self._is_valid_cause(cause_text):
                    confidence = marker.confidence
                    
                    # Boost confidence for longer, more detailed causes
                    word_count = len(cause_text.split())
                    if word_count >= 5:
                        confidence *= 1.1
                    elif word_count < 3:
                        confidence *= 0.8
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (cause_text, confidence, cause_start, cause_end)
        
        return best_match
    
    def _is_valid_cause(self, cause_text: str) -> bool:
        """
        Validate if extracted text is a legitimate cause span.
        
        Criteria:
        - Minimum length: 10 characters
        - Contains at least 2 words
        - Not purely punctuation
        - Not a question
        - Contains meaningful content (verbs, nouns)
        - Not just pronouns or articles
        
        Args:
            cause_text: Candidate cause text
        
        Returns:
            True if valid, False otherwise
        """
        if not cause_text or len(cause_text) < 10:
            return False
        
        words = cause_text.split()
        if len(words) < 2:
            return False
        
        # Check if it's purely punctuation
        if all(c in '.,!?;:' for c in cause_text):
            return False
        
        # Avoid questions
        if '?' in cause_text:
            return False
        
        # Reject if it's just filler words
        filler_words = {'i', 'me', 'my', 'the', 'a', 'an', 'and', 'or', 'but', 'it', 'is', 'am', 'are'}
        meaningful_words = [w.lower() for w in words if w.lower() not in filler_words]
        
        if len(meaningful_words) < 1:
            return False
        
        # Use spaCy for semantic validation
        try:
            doc = self.nlp(cause_text)
            has_verb = any(token.pos_ == 'VERB' for token in doc)
            has_noun = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
            
            # A valid cause should have at least a verb or noun
            if not (has_verb or has_noun):
                return False
        except:
            pass  # If spaCy fails, rely on other checks
        
        return True
    
    def generate_bio_tags(self, text: str, cause_start: int, cause_end: int) -> List[str]:
        """
        Generate BIO tags for tokenized text.
        
        BIO Tagging Scheme:
        - B-CAUSE: Beginning of cause span
        - I-CAUSE: Inside cause span
        - O: Outside cause span (not part of cause)
        
        Args:
            text: Full text
            cause_start: Character index where cause starts
            cause_end: Character index where cause ends
        
        Returns:
            List of BIO tags aligned with spaCy tokens
        """
        doc = self.nlp(text)
        bio_tags = []
        
        first_cause_token = True
        
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            # Check if token overlaps with cause span
            if token_start >= cause_start and token_end <= cause_end:
                if first_cause_token:
                    bio_tags.append('B-CAUSE')
                    first_cause_token = False
                else:
                    bio_tags.append('I-CAUSE')
            else:
                bio_tags.append('O')
        
        return bio_tags


class ESConvDataProcessor:
    """
    Process ESConv dataset and generate ECE training data.
    """
    
    def __init__(self, dataset_dir: str = 'esconv_dataset'):
        """
        Initialize the data processor.
        
        Args:
            dataset_dir: Directory containing ESConv dataset files
        """
        self.dataset_dir = Path(dataset_dir)
        self.cause_extractor = CauseExtractor()
        self.processed_samples = []
        
        logger.info(f"Initialized ESConvDataProcessor with dataset dir: {self.dataset_dir}")
    
    def load_dataset(self) -> List[Dict]:
        """
        Load all ESConv dataset files (train, validation, test).
        
        Returns:
            List of all conversation records
        
        Raises:
            FileNotFoundError: If dataset directory doesn't exist
        """
        # Validate dataset directory exists
        if not self.dataset_dir.exists():
            error_msg = (
                f"Dataset directory not found: {self.dataset_dir}\n"
                f"Please ensure the ESConv dataset is in the correct location.\n"
                f"Expected structure:\n"
                f"  {self.dataset_dir}/\n"
                f"    - train.jsonl\n"
                f"    - validation.jsonl\n"
                f"    - test.jsonl"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        all_data = []
        files_found = 0
        
        for split in ['train', 'validation', 'test']:
            file_path = self.dataset_dir / f"{split}.jsonl"
            
            if not file_path.exists():
                logger.warning(f"⚠️  File not found: {file_path}")
                continue
            
            files_found += 1
            
            logger.info(f"Loading {split}.jsonl...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            record = json.loads(line.strip())
                            all_data.append(record)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in {split}.jsonl line {line_num}: {e}")
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
        
        if files_found == 0:
            error_msg = (
                f"No dataset files found in {self.dataset_dir}!\n"
                f"Please download the ESConv dataset and place the following files:\n"
                f"  - train.jsonl\n"
                f"  - validation.jsonl\n"
                f"  - test.jsonl"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✅ Loaded {len(all_data)} conversation records from {files_found} file(s)")
        return all_data
    
    def extract_utterances(self, record: Dict) -> List[Dict]:
        """
        Extract individual utterances from conversation record.
        
        Args:
            record: Conversation record from ESConv dataset
        
        Returns:
            List of utterance dictionaries with metadata
        """
        utterances = []
        
        try:
            # Parse nested JSON in 'text' field
            conversation_data = json.loads(record['text'])
            
            # Extract metadata
            emotion_type = conversation_data.get('emotion_type', 'neutral')
            problem_type = conversation_data.get('problem_type', '')
            situation = conversation_data.get('situation', '')
            
            # Extract dialog turns
            dialog = conversation_data.get('dialog', [])
            
            for turn in dialog:
                text = turn.get('text', '').strip()
                speaker = turn.get('speaker', 'unknown')
                
                # Focus on seeker utterances (they express emotions)
                if speaker == 'usr' and len(text) > 20:  # Filter short utterances
                    utterances.append({
                        'text': text,
                        'emotion': emotion_type,
                        'context': {
                            'problem_type': problem_type,
                            'situation': situation
                        }
                    })
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error parsing record: {e}")
        
        return utterances
    
    def process_utterance(self, utterance: Dict) -> Optional[Dict]:
        """
        Process a single utterance to extract cause and generate BIO tags.
        
        Args:
            utterance: Utterance dictionary with text and metadata
        
        Returns:
            Processed sample dictionary or None if no cause found
        """
        text = utterance['text']
        context_emotion = utterance['emotion']
        
        # Detect emotion
        emotion = self.cause_extractor.detect_emotion(text, context_emotion)
        
        # Extract cause
        cause_result = self.cause_extractor.extract_cause(text)
        
        if cause_result is None:
            # No explicit causal markers found - mark as neutral emotion
            # This follows the requirement: "if there is no such words...keep it to neutral emotion"
            emotion = 'neutral'
            return None  # Skip this sample as per filtering requirement
        
        cause_text, confidence, cause_start, cause_end = cause_result
        
        # Generate BIO tags
        bio_tags = self.cause_extractor.generate_bio_tags(text, cause_start, cause_end)
        
        # Ensure BIO tags match token count
        doc = self.cause_extractor.nlp(text)
        tokens = [token.text for token in doc]
        
        if len(bio_tags) != len(tokens):
            logger.warning(f"BIO tag count mismatch: {len(bio_tags)} tags vs {len(tokens)} tokens")
            return None
        
        return {
            'text': text,
            'emotion': emotion,
            'cause_span': cause_text,
            'bio_tags': bio_tags,
            'tokens': tokens,  # Include tokens for debugging
            'confidence': confidence
        }
    
    def process_all(self, min_confidence: float = 0.6) -> List[Dict]:
        """
        Process all dataset records and generate training samples.
        
        Args:
            min_confidence: Minimum confidence threshold for including samples
        
        Returns:
            List of processed training samples
        """
        logger.info("=" * 80)
        logger.info("STARTING ECE TRAINING DATA GENERATION")
        logger.info("=" * 80)
        
        # Load dataset
        all_records = self.load_dataset()
        
        if not all_records:
            logger.error("No data loaded from dataset!")
            return []
        
        logger.info(f"Processing {len(all_records)} conversation records...")
        
        training_samples = []
        total_utterances = 0
        
        for record in tqdm(all_records, desc="Processing conversations"):
            utterances = self.extract_utterances(record)
            total_utterances += len(utterances)
            
            for utterance in utterances:
                sample = self.process_utterance(utterance)
                
                if sample and sample['confidence'] >= min_confidence:
                    # Remove tokens and confidence from final output
                    final_sample = {
                        'text': sample['text'],
                        'emotion': sample['emotion'],
                        'cause_span': sample['cause_span'],
                        'bio_tags': sample['bio_tags']
                    }
                    training_samples.append(final_sample)
        
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total conversation records: {len(all_records)}")
        logger.info(f"Total utterances extracted: {total_utterances}")
        logger.info(f"Samples with valid causes: {len(training_samples)}")
        logger.info(f"Success rate: {len(training_samples)/total_utterances*100:.2f}%")
        logger.info("=" * 80)
        
        return training_samples
    
    def save_to_json(self, samples: List[Dict], output_path: str = 'ece_training_data.json'):
        """
        Save processed samples to JSON file.
        
        Args:
            samples: List of training samples
            output_path: Output file path
        """
        output_file = Path(output_path)
        
        logger.info(f"Saving {len(samples)} samples to {output_file}...")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Successfully saved to {output_file}")
            logger.info(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        except Exception as e:
            logger.error(f"❌ Error saving file: {e}")
            raise
    
    def generate_statistics(self, samples: List[Dict]) -> Dict:
        """
        Generate statistics about the processed dataset.
        
        Args:
            samples: List of training samples
        
        Returns:
            Dictionary with statistics
        """
        if not samples:
            return {}
        
        emotion_counts = {}
        cause_lengths = []
        text_lengths = []
        
        for sample in samples:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            cause_lengths.append(len(sample['cause_span'].split()))
            text_lengths.append(len(sample['text'].split()))
        
        stats = {
            'total_samples': len(samples),
            'emotion_distribution': emotion_counts,
            'avg_cause_length': sum(cause_lengths) / len(cause_lengths),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'min_cause_length': min(cause_lengths),
            'max_cause_length': max(cause_lengths)
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"\nEmotion distribution:")
        for emotion, count in sorted(stats['emotion_distribution'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {emotion:15s}: {count:5d} ({count/stats['total_samples']*100:5.2f}%)")
        logger.info(f"\nCause span statistics:")
        logger.info(f"  Average length: {stats['avg_cause_length']:.2f} words")
        logger.info(f"  Min length: {stats['min_cause_length']} words")
        logger.info(f"  Max length: {stats['max_cause_length']} words")
        logger.info(f"\nText statistics:")
        logger.info(f"  Average length: {stats['avg_text_length']:.2f} words")
        logger.info("=" * 80 + "\n")
        
        return stats


def display_sample_examples(samples: List[Dict], num_examples: int = 5):
    """
    Display sample examples from the dataset for quality verification.
    
    Args:
        samples: List of training samples
        num_examples: Number of examples to display
    """
    if not samples:
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE EXAMPLES FOR QUALITY VERIFICATION")
    logger.info("=" * 80)
    
    import random
    examples = random.sample(samples, min(num_examples, len(samples)))
    
    for i, sample in enumerate(examples, 1):
        logger.info(f"\n{'─' * 80}")
        logger.info(f"EXAMPLE {i}:")
        logger.info(f"{'─' * 80}")
        logger.info(f"Text: {sample['text']}")
        logger.info(f"Emotion: {sample['emotion']}")
        logger.info(f"Cause Span: {sample['cause_span']}")
        logger.info(f"\nTokens: {' '.join([f'{i}' for i in range(len(sample['bio_tags']))])}")
        logger.info(f"         {' '.join(sample['text'].split()[:len(sample['bio_tags'])])}")
        logger.info(f"BIO Tags: {' '.join(sample['bio_tags'])}")
        
        # Highlight cause tokens
        cause_tokens = [token if tag.startswith('B-') or tag.startswith('I-') else '_' 
                       for token, tag in zip(sample['text'].split()[:len(sample['bio_tags'])], sample['bio_tags'])]
        logger.info(f"Causes:   {' '.join(cause_tokens)}")
    
    logger.info(f"\n{'=' * 80}\n")


def validate_dataset_quality(samples: List[Dict]) -> Dict[str, any]:
    """
    Perform quality checks on the generated dataset.
    
    Args:
        samples: List of training samples
    
    Returns:
        Dictionary with validation results
    """
    logger.info("\n" + "=" * 80)
    logger.info("DATASET QUALITY VALIDATION")
    logger.info("=" * 80)
    
    issues = {
        'tag_mismatch': 0,
        'no_cause_tags': 0,
        'empty_cause_span': 0,
        'duplicate_texts': 0
    }
    
    seen_texts = set()
    
    for i, sample in enumerate(samples):
        # Check for tag-token mismatch
        tokens = sample['text'].split()
        if len(sample['bio_tags']) > len(tokens):
            issues['tag_mismatch'] += 1
        
        # Check if there are any cause tags
        if not any(tag != 'O' for tag in sample['bio_tags']):
            issues['no_cause_tags'] += 1
        
        # Check for empty cause span
        if not sample['cause_span'].strip():
            issues['empty_cause_span'] += 1
        
        # Check for duplicates
        if sample['text'] in seen_texts:
            issues['duplicate_texts'] += 1
        seen_texts.add(sample['text'])
    
    # Calculate quality metrics
    total = len(samples)
    quality_score = 100.0
    
    if total > 0:
        quality_score -= (issues['tag_mismatch'] / total) * 20
        quality_score -= (issues['no_cause_tags'] / total) * 30
        quality_score -= (issues['empty_cause_span'] / total) * 25
        quality_score -= (issues['duplicate_texts'] / total) * 10
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Tag-token mismatches: {issues['tag_mismatch']} ({issues['tag_mismatch']/total*100:.2f}%)")
    logger.info(f"  Samples without cause tags: {issues['no_cause_tags']} ({issues['no_cause_tags']/total*100:.2f}%)")
    logger.info(f"  Empty cause spans: {issues['empty_cause_span']} ({issues['empty_cause_span']/total*100:.2f}%)")
    logger.info(f"  Duplicate texts: {issues['duplicate_texts']} ({issues['duplicate_texts']/total*100:.2f}%)")
    logger.info(f"\n  Overall Quality Score: {quality_score:.2f}/100")
    
    if quality_score >= 90:
        logger.info(f"  Status: ✅ EXCELLENT - Dataset is high quality")
    elif quality_score >= 75:
        logger.info(f"  Status: ✅ GOOD - Dataset is usable with minor issues")
    elif quality_score >= 60:
        logger.info(f"  Status: ⚠️  FAIR - Dataset has some quality concerns")
    else:
        logger.info(f"  Status: ❌ POOR - Dataset needs improvement")
    
    logger.info("=" * 80 + "\n")
    
    return {
        'issues': issues,
        'quality_score': quality_score,
        'total_samples': total
    }


def main():
    """
    Main execution function.
    """
    try:
        logger.info("\n" + "=" * 80)
        logger.info("🚀 EMOTION CAUSE EXTRACTION (ECE) TRAINING DATA GENERATOR")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {Path.cwd()}")
        logger.info(f"Working Directory: {Path.cwd()}")
        logger.info("=" * 80 + "\n")
        
        # Initialize processor
        logger.info("📦 Initializing Data Processor...")
        processor = ESConvDataProcessor(dataset_dir='esconv_dataset')
        
        # Process all data
        logger.info("\n🔄 Processing dataset with linguistic heuristics...")
        training_samples = processor.process_all(min_confidence=0.6)
        
        if not training_samples:
            logger.error("\n❌ No training samples generated!")
            logger.error("Possible reasons:")
            logger.error("  1. Dataset files are empty or corrupted")
            logger.error("  2. Confidence threshold is too high")
            logger.error("  3. No causal markers found in conversations")
            return
        
        # Generate statistics
        logger.info("\n📊 Generating Dataset Statistics...")
        stats = processor.generate_statistics(training_samples)
        
        # Validate quality
        logger.info("\n🔍 Validating Dataset Quality...")
        validation_results = validate_dataset_quality(training_samples)
        
        # Display sample examples
        logger.info("\n📝 Displaying Sample Examples...")
        display_sample_examples(training_samples, num_examples=5)
        
        # Save to file
        logger.info("\n💾 Saving Dataset...")
        processor.save_to_json(training_samples, output_path='ece_training_data.json')
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ ECE TRAINING DATA GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📁 Output File: ece_training_data.json")
        logger.info(f"📊 Total Samples: {len(training_samples)}")
        logger.info(f"🎯 Quality Score: {validation_results['quality_score']:.2f}/100")
        
        if len(training_samples) >= 15000:
            logger.info(f"✅ Target of 15,000+ samples achieved!")
        else:
            logger.info(f"⚠️  Generated {len(training_samples)} samples (target: 15,000+)")
            logger.info(f"   Consider lowering min_confidence or adding more data sources")
        
        logger.info("=" * 80 + "\n")
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ File not found error: {e}")
        logger.error("\nPlease ensure:")
        logger.error("  1. ESConv dataset is in the 'esconv_dataset' directory")
        logger.error("  2. Files are named: train.jsonl, validation.jsonl, test.jsonl")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
