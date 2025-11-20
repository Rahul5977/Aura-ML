"""
Hypercontextual Dataset Generator for LLM Fine-tuning

Enriches ESConv conversations with multi-modal analysis outputs:
- Emotion labels (from ESConv)
- Extracted causes (from ECE model)
- Named entities (from spaCy NER)
- Problem types (heuristic classification)
- Conversation history (sliding window)
- Support strategies (from ESConv)

Generates instruction-completion pairs for LLM fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

import torch
import spacy
from transformers import RobertaTokenizerFast

# Import ECE model for cause extraction
from aura_ml.models.ece_classifier import RoBERTaForECE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    speaker: str
    text: str
    emotion: Optional[str] = None
    strategy: Optional[str] = None


class HypercontextualDatasetGenerator:
    """
    Generates hypercontextual instruction-tuning dataset from ESConv.
    
    Features:
    - ECE model inference for cause extraction
    - spaCy NER for entity recognition
    - Heuristic problem type classification
    - Sliding window conversation history (default: 3 turns)
    - Instruction-completion pair generation
    """
    
    # Problem type keywords
    PROBLEM_KEYWORDS = {
        'relationship': ['friend', 'relationship', 'partner', 'family', 'girlfriend', 
                        'boyfriend', 'spouse', 'husband', 'wife', 'parent', 'sibling',
                        'breakup', 'divorce', 'argument', 'fight'],
        'work': ['job', 'work', 'career', 'boss', 'colleague', 'coworker', 'office',
                'manager', 'workplace', 'fired', 'quit', 'promotion', 'interview'],
        'health': ['health', 'sick', 'illness', 'hospital', 'doctor', 'disease',
                  'pain', 'medical', 'appointment', 'diagnosis', 'treatment'],
        'academic': ['school', 'exam', 'test', 'study', 'university', 'college',
                    'grade', 'homework', 'assignment', 'professor', 'class', 'course'],
        'financial': ['money', 'debt', 'financial', 'rent', 'bills', 'loan',
                     'payment', 'salary', 'income', 'expense', 'budget', 'afford'],
        'emotional_distress': []  # Will be inferred from emotions
    }
    
    # Emotion to distress mapping
    DISTRESS_EMOTIONS = {'sad', 'fear', 'angry', 'disgust', 'anxiety'}
    
    # ESConv emotion type mapping to our 7 emotions
    EMOTION_MAPPING = {
        'anxiety': 'fear',
        'fear': 'fear',
        'sadness': 'sad',
        'sad': 'sad',
        'anger': 'angry',
        'angry': 'angry',
        'depression': 'sad',
        'disgust': 'disgust',
        'joy': 'happy',
        'happiness': 'happy',
        'happy': 'happy',
        'surprise': 'surprise',
        'neutral': 'neutral',
        'excited': 'happy',
        'grateful': 'happy',
        'hopeful': 'happy',
        'proud': 'happy',
        'content': 'happy',
        'caring': 'happy',
        'prepared': 'neutral',
        'confident': 'happy',
        'trusting': 'happy',
        'lonely': 'sad',
        'ashamed': 'sad',
        'disappointed': 'sad',
        'embarrassed': 'sad',
        'guilty': 'sad',
        'devastated': 'sad',
        'terrified': 'fear',
        'apprehensive': 'fear',
        'afraid': 'fear',
        'furious': 'angry',
        'annoyed': 'angry',
        'jealous': 'angry',
    }
    
    # Support strategies description
    STRATEGY_DESCRIPTIONS = {
        'Question': 'asking open-ended questions to gather more information',
        'Restatement or Paraphrasing': 'restating what the user said to show understanding',
        'Reflection of feelings': 'reflecting and validating the user\'s emotions',
        'Self-disclosure': 'sharing personal experiences to build connection',
        'Affirmation and Reassurance': 'providing encouragement and reassurance',
        'Providing Suggestions': 'offering practical advice and solutions',
        'Information': 'providing educational content or information',
        'Others': 'providing general emotional support'
    }
    
    def __init__(
        self,
        ece_model_path: str,
        history_window: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the generator.
        
        Args:
            ece_model_path: Path to trained ECE model
            history_window: Number of conversation turns to include in history
            device: Device to run ECE model on ('cuda' or 'cpu')
        """
        self.history_window = history_window
        self.device = device
        
        # Load ECE model
        logger.info(f"Loading ECE model from {ece_model_path}")
        self.ece_model = RoBERTaForECE.from_pretrained(ece_model_path)
        self.ece_model.to(device)
        self.ece_model.eval()
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        
        # Load spaCy for NER
        logger.info("Loading spaCy NER model")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            logger.info("Attempting to download spaCy model...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        logger.info("Initialization complete")
    
    def extract_cause(self, text: str, max_length: int = 128) -> str:
        """
        Extract emotion cause using trained ECE model.
        
        Args:
            text: Input text
            max_length: Maximum tokenization length
            
        Returns:
            Extracted cause text or full text if no cause found
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.ece_model(input_ids, attention_mask)
            
            # Extract logits from output object
            clause_logits = outputs.clause_logits if hasattr(outputs, 'clause_logits') else outputs[0]
            token_logits = outputs.token_logits if hasattr(outputs, 'token_logits') else outputs[1]
            
            # Get predictions
            clause_pred = torch.argmax(clause_logits, dim=-1).item()
            token_preds = torch.argmax(token_logits, dim=-1)[0]  # [seq_len]
        
        # If no clause detected, return full text
        if clause_pred == 0:
            return text
        
        # Extract cause tokens (BIO tags: 0=O, 1=B-CAUSE, 2=I-CAUSE)
        cause_token_ids = []
        tokens = encoding['input_ids'][0]
        
        for i, (token_id, pred) in enumerate(zip(tokens, token_preds)):
            if attention_mask[0][i] == 0:  # Skip padding
                break
            if pred in [1, 2]:  # B-CAUSE or I-CAUSE
                cause_token_ids.append(token_id.item())
        
        # Decode cause
        if cause_token_ids:
            cause_text = self.tokenizer.decode(cause_token_ids, skip_special_tokens=True)
            return cause_text.strip()
        
        # Fallback to full text
        return text
    
    def extract_entities(self, text: str) -> str:
        """
        Extract named entities using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            Comma-separated entity list or "None"
        """
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'EVENT']:
                entities.append(f"{ent.text} ({ent.label_})")
        
        return ', '.join(entities) if entities else "None"
    
    def classify_problem_type(self, text: str, emotion: str) -> str:
        """
        Classify problem type using heuristic keyword matching.
        
        Args:
            text: Input text
            emotion: Detected emotion
            
        Returns:
            Problem type category
        """
        text_lower = text.lower()
        
        # Check each category
        for problem_type, keywords in self.PROBLEM_KEYWORDS.items():
            if problem_type == 'emotional_distress':
                continue  # Handle separately
            
            for keyword in keywords:
                if keyword in text_lower:
                    return problem_type
        
        # Check if emotional distress based on emotion
        if emotion in self.DISTRESS_EMOTIONS:
            return 'emotional_distress'
        
        return 'general'
    
    def format_history(self, history: List[ConversationTurn]) -> str:
        """
        Format conversation history as string.
        
        Args:
            history: List of conversation turns
            
        Returns:
            Formatted history string
        """
        if not history:
            return "None"
        
        formatted = []
        for turn in history:
            speaker_name = "User" if turn.speaker == "seeker" else "Aura"
            formatted.append(f"{speaker_name}: {turn.text}")
        
        return " | ".join(formatted)
    
    def create_instruction(
        self,
        emotion: str,
        user_message: str,
        cause: str,
        strategy: str
    ) -> str:
        """
        Create instruction prompt for LLM.
        
        Args:
            emotion: Detected emotion
            user_message: User's message
            cause: Extracted cause
            strategy: Support strategy to use
            
        Returns:
            Formatted instruction string
        """
        strategy_desc = self.STRATEGY_DESCRIPTIONS.get(strategy, 'providing emotional support')
        
        instruction = (
            f"You are Aura, an empathetic AI assistant specialized in emotional support. "
            f"Your user is feeling {emotion}. They are saying: '{user_message}'. "
            f"The main reason they feel this way is: '{cause}'. "
            f"Respond by {strategy_desc}."
        )
        
        return instruction
    
    def process_conversation(
        self,
        conversation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a single ESConv conversation into training samples.
        
        Args:
            conversation: ESConv conversation dictionary
            
        Returns:
            List of instruction-completion samples
        """
        samples = []
        history = deque(maxlen=self.history_window)
        
        # Parse nested JSON structure
        if 'text' in conversation and isinstance(conversation['text'], str):
            try:
                conv_data = json.loads(conversation['text'])
            except json.JSONDecodeError:
                logger.warning("Failed to parse conversation text")
                return samples
        else:
            conv_data = conversation
        
        # Extract conversation-level information
        conv_emotion_type = conv_data.get('emotion_type', 'neutral')
        # Map ESConv emotion to our emotion categories
        conv_emotion = self.EMOTION_MAPPING.get(conv_emotion_type.lower(), 'neutral')
        
        # Get conversation ID (use problem_type or generate one)
        conv_id = conv_data.get('problem_type', 'unknown')
        
        dialog = conv_data.get('dialog', [])
        
        for i, turn in enumerate(dialog):
            speaker = turn.get('speaker', '')
            text = turn.get('text', '').strip()
            
            if not text:
                continue
            
            # Process user turns (usr = seeker)
            if speaker == 'usr':
                emotion = conv_emotion  # Use conversation-level emotion
                
                # Extract features
                cause = self.extract_cause(text)
                entities = self.extract_entities(text)
                problem_type = self.classify_problem_type(text, emotion)
                history_str = self.format_history(list(history))
                
                # Find next supporter response
                supporter_response = None
                strategy_used = None
                
                for j in range(i + 1, len(dialog)):
                    if dialog[j].get('speaker') == 'sys':
                        supporter_response = dialog[j].get('text', '').strip()
                        strategy_used = dialog[j].get('strategy', 'Others')
                        break
                
                # Create sample if we have a response
                if supporter_response and strategy_used:
                    instruction = self.create_instruction(
                        emotion, text, cause, strategy_used
                    )
                    
                    sample = {
                        'instruction': instruction,
                        'input': {
                            'user_message': text,
                            'emotion': emotion,
                            'cause': cause,
                            'entities': entities,
                            'history': history_str,
                            'problem_type': problem_type
                        },
                        'output': supporter_response,
                        'metadata': {
                            'conversation_id': f"{conv_id}_{i}",
                            'strategy_used': strategy_used,
                            'turn_index': i
                        }
                    }
                    
                    samples.append(sample)
            
            # Update history
            turn_obj = ConversationTurn(
                speaker='User' if speaker == 'usr' else 'Aura',
                text=text,
                emotion=conv_emotion if speaker == 'usr' else None,
                strategy=turn.get('strategy')
            )
            history.append(turn_obj)
        
        return samples
    
    def generate_dataset(
        self,
        esconv_path: str,
        output_dir: str,
        train_split: float = 0.9
    ) -> Dict[str, int]:
        """
        Generate complete hypercontextual dataset from ESConv.
        
        Args:
            esconv_path: Path to ESConv dataset directory
            output_dir: Output directory for generated dataset
            train_split: Training data split ratio (default: 0.9)
            
        Returns:
            Statistics dictionary
        """
        esconv_path = Path(esconv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_samples = []
        
        # Process each split (ESConv uses .jsonl format)
        for split_name in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
            split_file = esconv_path / split_name
            
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue
            
            logger.info(f"Processing {split_name}...")
            
            # Read JSONL format (one JSON per line)
            conversations = []
            with open(split_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {split_name}: {e}")
                        continue
            
            # Process each conversation
            for conv in conversations:
                samples = self.process_conversation(conv)
                all_samples.extend(samples)
            
            logger.info(f"  Processed {len(conversations)} conversations from {split_name}")
        
        logger.info(f"Total samples generated: {len(all_samples)}")
        
        # Split into train/val
        split_idx = int(len(all_samples) * train_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        # Save datasets
        output_files = {
            'llm_training_data.json': all_samples,
            'llm_train.json': train_samples,
            'llm_val.json': val_samples
        }
        
        for filename, data in output_files.items():
            output_path = output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} samples to {output_path}")
        
        # Generate statistics
        stats = {
            'total_samples': len(all_samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'emotion_distribution': self._count_emotions(all_samples),
            'problem_type_distribution': self._count_problem_types(all_samples),
            'strategy_distribution': self._count_strategies(all_samples)
        }
        
        # Save statistics
        stats_path = output_dir / 'dataset_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
        
        return stats
    
    def _count_emotions(self, samples: List[Dict]) -> Dict[str, int]:
        """Count emotion distribution"""
        counts = {}
        for sample in samples:
            emotion = sample['input']['emotion']
            counts[emotion] = counts.get(emotion, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def _count_problem_types(self, samples: List[Dict]) -> Dict[str, int]:
        """Count problem type distribution"""
        counts = {}
        for sample in samples:
            problem_type = sample['input']['problem_type']
            counts[problem_type] = counts.get(problem_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def _count_strategies(self, samples: List[Dict]) -> Dict[str, int]:
        """Count support strategy distribution"""
        counts = {}
        for sample in samples:
            strategy = sample['metadata']['strategy_used']
            counts[strategy] = counts.get(strategy, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate hypercontextual dataset')
    parser.add_argument('--esconv-path', type=str, required=True,
                       help='Path to ESConv dataset directory')
    parser.add_argument('--ece-model', type=str, required=True,
                       help='Path to trained ECE model')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for generated dataset')
    parser.add_argument('--history-window', type=int, default=3,
                       help='Conversation history window size')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Training data split ratio')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = HypercontextualDatasetGenerator(
        ece_model_path=args.ece_model,
        history_window=args.history_window
    )
    
    # Generate dataset
    stats = generator.generate_dataset(
        esconv_path=args.esconv_path,
        output_dir=args.output_dir,
        train_split=args.train_split
    )
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Val samples: {stats['val_samples']}")
    print("\nTop 5 Emotions:")
    for emotion, count in list(stats['emotion_distribution'].items())[:5]:
        print(f"  {emotion}: {count}")
    print("\nProblem Types:")
    for ptype, count in stats['problem_type_distribution'].items():
        print(f"  {ptype}: {count}")
    print("="*60)


if __name__ == '__main__':
    main()
