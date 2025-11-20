"""
ESConv Dataset Processor

This module handles loading and preprocessing of the ESConv (Emotional Support Conversations)
dataset. Extracts seeker utterances with emotion labels and applies emotion mapping to
standardize emotion categories.

ESConv Format:
- JSONL file with conversation-level structure
- Each conversation contains dialog list with speaker-tagged utterances
- Emotion types: anxiety, depression, fear, etc.
- Target: Extract only seeker ("usr") utterances with emotions

Emotion Mapping:
- Maps 48 fine-grained emotion terms to 7 categories:
  anxious/fear, sad, angry, happy, disgusted, surprised, neutral
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
import re


class ESConvProcessor:
    """
    Processor for ESConv dataset: loads, filters, and maps emotion categories.
    """
    
    def __init__(self, emotion_mapping_path: str):
        """
        Initialize ESConv processor with emotion mapping.
        
        Args:
            emotion_mapping_path: Path to emotion_mapping.json file
        """
        self.emotion_mapping = self._load_emotion_mapping(emotion_mapping_path)
        
        self.stats = {
            "total_conversations": 0,
            "total_utterances": 0,
            "seeker_utterances": 0,
            "mapped_emotions": {emotion: 0 for emotion in set(self.emotion_mapping.values())}
        }
    
    def _load_emotion_mapping(self, path: str) -> Dict[str, str]:
        """
        Load emotion mapping from JSON file.
        
        Args:
            path: Path to emotion_mapping.json
            
        Returns:
            Dictionary mapping source emotions to target categories
        """
        with open(path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        print(f"Loaded emotion mapping: {len(mapping)} emotion terms -> {len(set(mapping.values()))} categories")
        return mapping
    
    def _map_emotion(self, emotion: str) -> str:
        """
        Map fine-grained emotion to standardized category.
        
        Args:
            emotion: Original emotion label
            
        Returns:
            Mapped emotion category (or 'neutral' if not found)
        """
        emotion_lower = emotion.lower().strip()
        return self.emotion_mapping.get(emotion_lower, "neutral")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def load_esconv_file(self, file_path: str) -> List[Dict]:
        """
        Load ESConv JSONL file and extract seeker utterances.
        
        Args:
            file_path: Path to ESConv JSONL file (train.jsonl, validation.jsonl, test.jsonl)
            
        Returns:
            List of processed utterances with emotion labels
        """
        utterances = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                conversation = json.loads(line)
                self.stats["total_conversations"] += 1
                
                # Parse the nested JSON structure
                # ESConv format: {"text": "{json_string}"}
                if "text" in conversation:
                    conv_data = json.loads(conversation["text"])
                else:
                    conv_data = conversation
                
                # Extract emotion type and dialog
                emotion_type = conv_data.get("emotion_type", "neutral")
                mapped_emotion = self._map_emotion(emotion_type)
                
                dialog = conv_data.get("dialog", [])
                
                # Extract seeker utterances (speaker == "usr")
                for utterance in dialog:
                    self.stats["total_utterances"] += 1
                    
                    if utterance.get("speaker") == "usr":
                        text = self._clean_text(utterance.get("text", ""))
                        
                        # Skip very short utterances (< 3 words)
                        if len(text.split()) < 3:
                            continue
                        
                        utterances.append({
                            "text": text,
                            "emotion": mapped_emotion,
                            "original_emotion": emotion_type,
                            "conversation_id": self.stats["total_conversations"]
                        })
                        
                        self.stats["seeker_utterances"] += 1
                        self.stats["mapped_emotions"][mapped_emotion] += 1
        
        return utterances
    
    def load_esconv_split(
        self, 
        train_path: str, 
        val_path: Optional[str] = None, 
        test_path: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Load all ESConv dataset splits.
        
        Args:
            train_path: Path to train.jsonl
            val_path: Path to validation.jsonl (optional)
            test_path: Path to test.jsonl (optional)
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing utterance lists
        """
        splits = {}
        
        print("Loading ESConv training set...")
        splits["train"] = self.load_esconv_file(train_path)
        print(f"  Loaded {len(splits['train'])} training utterances")
        
        if val_path:
            print("Loading ESConv validation set...")
            splits["val"] = self.load_esconv_file(val_path)
            print(f"  Loaded {len(splits['val'])} validation utterances")
        
        if test_path:
            print("Loading ESConv test set...")
            splits["test"] = self.load_esconv_file(test_path)
            print(f"  Loaded {len(splits['test'])} test utterances")
        
        return splits
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        seeker_ratio = (
            self.stats["seeker_utterances"] / self.stats["total_utterances"]
            if self.stats["total_utterances"] > 0 else 0
        )
        
        return {
            **self.stats,
            "seeker_utterance_ratio": seeker_ratio,
            "seeker_percentage": f"{seeker_ratio * 100:.2f}%",
            "avg_utterances_per_conv": (
                self.stats["total_utterances"] / self.stats["total_conversations"]
                if self.stats["total_conversations"] > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            "total_conversations": 0,
            "total_utterances": 0,
            "seeker_utterances": 0,
            "mapped_emotions": {emotion: 0 for emotion in set(self.emotion_mapping.values())}
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Paths (update these to your actual paths)
    emotion_mapping_path = "/home/rishi/Desktop/Aura-ML/emotion_mapping.json"
    esconv_dir = "/home/rishi/Desktop/Aura-ML/esconv_dataset-20251120T185554Z-1-001/esconv_dataset"
    
    train_path = f"{esconv_dir}/train.jsonl"
    val_path = f"{esconv_dir}/validation.jsonl"
    test_path = f"{esconv_dir}/test.jsonl"
    
    # Check if files exist
    if not Path(emotion_mapping_path).exists():
        print(f"Error: emotion_mapping.json not found at {emotion_mapping_path}")
        sys.exit(1)
    
    if not Path(train_path).exists():
        print(f"Error: train.jsonl not found at {train_path}")
        sys.exit(1)
    
    print("Testing ESConv Processor\n" + "="*80)
    
    # Initialize processor
    processor = ESConvProcessor(emotion_mapping_path)
    
    # Load all splits
    splits = processor.load_esconv_split(train_path, val_path, test_path)
    
    print("\n" + "="*80)
    print("Dataset Statistics:")
    print(f"  Train: {len(splits['train'])} utterances")
    if 'val' in splits:
        print(f"  Val: {len(splits['val'])} utterances")
    if 'test' in splits:
        print(f"  Test: {len(splits['test'])} utterances")
    
    print("\n" + "="*80)
    print("Processing Statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        if key != "mapped_emotions":
            print(f"  {key}: {value}")
    
    print("\nEmotion Distribution:")
    for emotion, count in sorted(stats["mapped_emotions"].items()):
        percentage = (count / stats["seeker_utterances"] * 100) if stats["seeker_utterances"] > 0 else 0
        print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("Sample Utterances:")
    for i, utt in enumerate(splits["train"][:3]):
        print(f"\n{i+1}. [{utt['emotion']}] {utt['text'][:80]}...")
