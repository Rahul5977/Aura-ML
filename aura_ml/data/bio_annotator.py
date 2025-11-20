"""
BIO Annotator for Emotion-Cause Extraction

This module converts cause spans to token-level BIO (Begin-Inside-Outside) annotations
using RoBERTa byte-pair encoding tokenizer. Produces training data for sequence labeling
models (RoBERTa, BERT, etc.).

BIO Tags:
- B-CAUSE: Beginning of causal span
- I-CAUSE: Inside causal span
- O: Outside causal span (non-causal tokens)
"""

from transformers import RobertaTokenizerFast
from typing import List, Dict, Tuple, Optional
import re


class BIOAnnotator:
    """
    BIO annotator using RoBERTa tokenizer for token-level cause span labeling.
    Handles subword tokenization and alignment with character-level spans.
    """
    
    # BIO tag vocabulary
    TAG2ID = {
        "O": 0,
        "B-CAUSE": 1,
        "I-CAUSE": 2
    }
    
    ID2TAG = {v: k for k, v in TAG2ID.items()}
    
    def __init__(self, model_name: str = "roberta-base", max_length: int = 128):
        """
        Initialize BIO annotator with RoBERTa Fast tokenizer.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length
        
        self.stats = {
            "total_annotated": 0,
            "total_tokens": 0,
            "cause_tokens": 0,
            "truncated": 0
        }
    
    def _find_cause_char_span(self, text: str, cause: str) -> Optional[Tuple[int, int]]:
        """
        Find character-level start and end positions of cause in text.
        
        Args:
            text: Full utterance text
            cause: Causal phrase to locate
            
        Returns:
            Tuple of (start_char, end_char) or None if not found
        """
        # Normalize whitespace for better matching
        text_norm = re.sub(r'\s+', ' ', text.lower().strip())
        cause_norm = re.sub(r'\s+', ' ', cause.lower().strip())
        
        # Try exact match first
        start = text_norm.find(cause_norm)
        if start != -1:
            end = start + len(cause_norm)
            return (start, end)
        
        # Try fuzzy matching (allowing small variations)
        # Split cause into words and find approximate span
        cause_words = cause_norm.split()
        if not cause_words:
            return None
        
        # Search for first and last words
        first_word = cause_words[0]
        last_word = cause_words[-1]
        
        first_pos = text_norm.find(first_word)
        if first_pos == -1:
            return None
        
        last_pos = text_norm.find(last_word, first_pos)
        if last_pos == -1:
            return None
        
        start = first_pos
        end = last_pos + len(last_word)
        
        return (start, end)
    
    def annotate(self, text: str, cause: str) -> Dict:
        """
        Generate BIO annotations for text with cause span.
        
        Args:
            text: Full utterance text
            cause: Causal phrase within text
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: BIO tag IDs
                - tokens: List of token strings (for debugging)
                - bio_tags: List of BIO tag strings (for debugging)
        """
        self.stats["total_annotated"] += 1
        
        # Find cause character span
        cause_span = self._find_cause_char_span(text, cause)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors=None
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]
        
        # Initialize all labels as O (outside)
        labels = [self.TAG2ID["O"]] * len(input_ids)
        
        # If cause span found, annotate with B-CAUSE and I-CAUSE
        if cause_span:
            cause_start, cause_end = cause_span
            first_cause_token = True
            
            for idx, (start, end) in enumerate(offset_mapping):
                # Skip special tokens (CLS, SEP, PAD)
                if start == 0 and end == 0:
                    continue
                
                # Check if token overlaps with cause span
                if start < cause_end and end > cause_start:
                    if first_cause_token:
                        labels[idx] = self.TAG2ID["B-CAUSE"]
                        first_cause_token = False
                        self.stats["cause_tokens"] += 1
                    else:
                        labels[idx] = self.TAG2ID["I-CAUSE"]
                        self.stats["cause_tokens"] += 1
        
        # Convert to token strings and BIO tags for debugging
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        bio_tags = [self.ID2TAG[label] for label in labels]
        
        # Update statistics
        self.stats["total_tokens"] += sum(attention_mask)
        if len(text) > self.max_length:
            self.stats["truncated"] += 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tokens": tokens,
            "bio_tags": bio_tags
        }
    
    def annotate_batch(self, texts: List[str], causes: List[str]) -> List[Dict]:
        """
        Generate BIO annotations for a batch of texts.
        
        Args:
            texts: List of full utterance texts
            causes: List of causal phrases (aligned with texts)
            
        Returns:
            List of annotation dictionaries
        """
        assert len(texts) == len(causes), "texts and causes must have same length"
        return [self.annotate(text, cause) for text, cause in zip(texts, causes)]
    
    def get_statistics(self) -> Dict:
        """
        Get annotation statistics.
        
        Returns:
            Dictionary containing annotation statistics
        """
        avg_tokens = (
            self.stats["total_tokens"] / self.stats["total_annotated"]
            if self.stats["total_annotated"] > 0 else 0
        )
        
        cause_ratio = (
            self.stats["cause_tokens"] / self.stats["total_tokens"]
            if self.stats["total_tokens"] > 0 else 0
        )
        
        return {
            **self.stats,
            "avg_tokens_per_sample": avg_tokens,
            "cause_token_ratio": cause_ratio,
            "cause_token_percentage": f"{cause_ratio * 100:.2f}%"
        }
    
    def reset_statistics(self):
        """Reset annotation statistics."""
        self.stats = {
            "total_annotated": 0,
            "total_tokens": 0,
            "cause_tokens": 0,
            "truncated": 0
        }


# Example usage
if __name__ == "__main__":
    annotator = BIOAnnotator()
    
    # Test example
    text = "I have been with her for close to 2 years and now she has been cheating me right under my nose for the past few months!"
    cause = "been cheating me right under my nose for the past few months"
    
    print("Testing BIO Annotator\n" + "="*80)
    print(f"Text: {text}")
    print(f"Cause: {cause}\n")
    
    result = annotator.annotate(text, cause)
    
    print("Tokenized with BIO tags:")
    print("-" * 80)
    for token, bio_tag in zip(result["tokens"][:30], result["bio_tags"][:30]):
        if token.strip():  # Skip padding
            print(f"{token:20s} -> {bio_tag}")
    
    print("\n" + "="*80)
    print("Statistics:", annotator.get_statistics())
    
    # Count BIO tag distribution
    from collections import Counter
    tag_counts = Counter(result["bio_tags"])
    print(f"\nBIO Tag Distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")
