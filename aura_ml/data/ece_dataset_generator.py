"""
ECE Dataset Generator - Main Pipeline

This module orchestrates the complete two-pass ECE dataset generation pipeline:

Pass 1 (Rule-based): CausalKeywordExtractor -> 62% coverage (12,564 samples)
Pass 2 (Heuristic): HeuristicCauseExtractor -> 38% additional (7,700 samples)
Total Coverage: 73% (20,264 samples)

Output Format:
- JSON file with emotion-cause pairs
- BIO-annotated sequences for training
- 80/10/10 train/val/test split
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

from aura_ml.data.esconv_processor import ESConvProcessor
from aura_ml.data.causal_keyword_extractor import CausalKeywordExtractor
from aura_ml.data.heuristic_fallback import HeuristicCauseExtractor
from aura_ml.data.bio_annotator import BIOAnnotator


class ECEDatasetGenerator:
    """
    Main pipeline for generating ECE dataset from ESConv.
    Orchestrates two-pass extraction + BIO annotation.
    """
    
    def __init__(
        self,
        emotion_mapping_path: str,
        output_dir: str = "data/processed",
        use_bio_annotation: bool = True
    ):
        """
        Initialize ECE dataset generator.
        
        Args:
            emotion_mapping_path: Path to emotion_mapping.json
            output_dir: Directory to save generated dataset
            use_bio_annotation: Whether to generate BIO annotations
        """
        self.esconv_processor = ESConvProcessor(emotion_mapping_path)
        self.keyword_extractor = CausalKeywordExtractor()
        self.heuristic_extractor = HeuristicCauseExtractor()
        
        self.use_bio_annotation = use_bio_annotation
        if use_bio_annotation:
            self.bio_annotator = BIOAnnotator()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total_utterances": 0,
            "pass1_success": 0,
            "pass2_success": 0,
            "total_success": 0,
            "coverage": 0.0
        }
    
    def generate_from_utterances(self, utterances: List[Dict]) -> List[Dict]:
        """
        Generate ECE samples from utterance list using two-pass extraction.
        
        Args:
            utterances: List of utterances from ESConvProcessor
            
        Returns:
            List of ECE samples with emotion-cause pairs
        """
        ece_samples = []
        
        self.stats["total_utterances"] = len(utterances)
        
        for utt in utterances:
            text = utt["text"]
            emotion = utt["emotion"]
            original_emotion = utt.get("original_emotion", emotion)
            
            # Pass 1: Keyword-based extraction
            keyword_result = self.keyword_extractor.extract_cause(text)
            
            if keyword_result:
                # Pass 1 succeeded
                sample = {
                    "text": text,
                    "emotion": emotion,
                    "cause": keyword_result["cause"],
                    "source": keyword_result["source"],
                    "extraction_method": "pass1_keyword",
                    "category": keyword_result.get("category"),
                    "keyword": keyword_result.get("keyword"),
                    "original_emotion": original_emotion
                }
                
                ece_samples.append(sample)
                self.stats["pass1_success"] += 1
            
            else:
                # Pass 2: Heuristic-based extraction
                heuristic_result = self.heuristic_extractor.extract_cause(text, fallback_to_full_text=True)
                
                if heuristic_result:
                    sample = {
                        "text": text,
                        "emotion": emotion,
                        "cause": heuristic_result["cause"],
                        "source": heuristic_result["source"],
                        "extraction_method": "pass2_heuristic",
                        "heuristic_method": heuristic_result.get("method"),
                        "original_emotion": original_emotion
                    }
                    
                    ece_samples.append(sample)
                    self.stats["pass2_success"] += 1
        
        self.stats["total_success"] = len(ece_samples)
        self.stats["coverage"] = (
            self.stats["total_success"] / self.stats["total_utterances"]
            if self.stats["total_utterances"] > 0 else 0.0
        )
        
        return ece_samples
    
    def add_bio_annotations(self, ece_samples: List[Dict]) -> List[Dict]:
        """
        Add BIO annotations to ECE samples.
        
        Args:
            ece_samples: List of ECE samples with emotion-cause pairs
            
        Returns:
            List of ECE samples with BIO annotations added
        """
        if not self.use_bio_annotation:
            return ece_samples
        
        annotated_samples = []
        
        for sample in ece_samples:
            text = sample["text"]
            cause = sample["cause"]
            
            # Generate BIO annotation
            bio_result = self.bio_annotator.annotate(text, cause)
            
            # Add BIO fields to sample
            annotated_sample = {
                **sample,
                "input_ids": bio_result["input_ids"],
                "attention_mask": bio_result["attention_mask"],
                "labels": bio_result["labels"],
                "tokens": bio_result["tokens"],
                "bio_tags": bio_result["bio_tags"]
            }
            
            annotated_samples.append(annotated_sample)
        
        return annotated_samples
    
    def split_dataset(
        self, 
        ece_samples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split ECE dataset into train/val/test sets.
        
        Args:
            ece_samples: Complete ECE dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle if requested
        if shuffle:
            import random
            random.seed(42)
            ece_samples = ece_samples.copy()
            random.shuffle(ece_samples)
        
        # Calculate split indices
        n = len(ece_samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples = ece_samples[:train_end]
        val_samples = ece_samples[train_end:val_end]
        test_samples = ece_samples[val_end:]
        
        return train_samples, val_samples, test_samples
    
    def save_ece_dataset(
        self,
        ece_samples: List[Dict],
        output_name: str = "ece_data.json"
    ):
        """
        Save ECE dataset to JSON file.
        
        Args:
            ece_samples: ECE samples to save
            output_name: Output filename
        """
        output_path = self.output_dir / output_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ece_samples, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(ece_samples)} samples to {output_path}")
    
    def generate_complete_dataset(
        self,
        esconv_train_path: str,
        esconv_val_path: Optional[str] = None,
        esconv_test_path: Optional[str] = None,
        save_splits: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Generate complete ECE dataset from ESConv splits.
        
        Args:
            esconv_train_path: Path to ESConv train.jsonl
            esconv_val_path: Path to ESConv validation.jsonl (optional)
            esconv_test_path: Path to ESConv test.jsonl (optional)
            save_splits: Whether to save train/val/test splits separately
            
        Returns:
            Dictionary with 'train', 'val', 'test', 'all' keys containing ECE samples
        """
        print("\n" + "="*80)
        print("ECE Dataset Generation Pipeline")
        print("="*80)
        
        # Step 1: Load ESConv dataset
        print("\n[Step 1/4] Loading ESConv dataset...")
        esconv_splits = self.esconv_processor.load_esconv_split(
            esconv_train_path, esconv_val_path, esconv_test_path
        )
        
        # Step 2: Generate ECE samples with two-pass extraction
        print("\n[Step 2/4] Extracting emotion-cause pairs (Two-Pass)...")
        
        ece_splits = {}
        
        # Process training set
        print("\n  Processing training set...")
        ece_splits["train"] = self.generate_from_utterances(esconv_splits["train"])
        print(f"    Pass 1 (Keyword): {self.keyword_extractor.get_statistics()['successful_extractions']} samples")
        print(f"    Pass 2 (Heuristic): {self.heuristic_extractor.get_statistics()['successful_extractions']} samples")
        print(f"    Total: {len(ece_splits['train'])} samples ({self.stats['coverage']*100:.1f}% coverage)")
        
        # Process validation set
        if "val" in esconv_splits:
            print("\n  Processing validation set...")
            self.keyword_extractor.reset_statistics()
            self.heuristic_extractor.reset_statistics()
            ece_splits["val"] = self.generate_from_utterances(esconv_splits["val"])
            print(f"    Total: {len(ece_splits['val'])} samples")
        
        # Process test set
        if "test" in esconv_splits:
            print("\n  Processing test set...")
            self.keyword_extractor.reset_statistics()
            self.heuristic_extractor.reset_statistics()
            ece_splits["test"] = self.generate_from_utterances(esconv_splits["test"])
            print(f"    Total: {len(ece_splits['test'])} samples")
        
        # Step 3: Add BIO annotations
        if self.use_bio_annotation:
            print("\n[Step 3/4] Adding BIO annotations...")
            for split_name, samples in ece_splits.items():
                print(f"  Annotating {split_name} set...")
                ece_splits[split_name] = self.add_bio_annotations(samples)
        else:
            print("\n[Step 3/4] Skipping BIO annotation (disabled)")
        
        # Step 4: Save datasets
        print("\n[Step 4/4] Saving ECE dataset...")
        
        # Combine all splits
        all_samples = []
        for samples in ece_splits.values():
            all_samples.extend(samples)
        ece_splits["all"] = all_samples
        
        # Save splits
        if save_splits:
            for split_name, samples in ece_splits.items():
                self.save_ece_dataset(samples, f"ece_{split_name}.json")
        else:
            self.save_ece_dataset(all_samples, "ece_data.json")
        
        # Print final statistics
        self._print_final_statistics(ece_splits)
        
        return ece_splits
    
    def _print_final_statistics(self, ece_splits: Dict[str, List[Dict]]):
        """Print final generation statistics."""
        print("\n" + "="*80)
        print("Final Statistics")
        print("="*80)
        
        print(f"\nDataset Sizes:")
        for split_name, samples in ece_splits.items():
            if split_name != "all":
                print(f"  {split_name.capitalize()}: {len(samples)} samples")
        print(f"  Total: {len(ece_splits['all'])} samples")
        
        # Emotion distribution
        all_emotions = [s["emotion"] for s in ece_splits["all"]]
        emotion_counts = Counter(all_emotions)
        
        print(f"\nEmotion Distribution:")
        for emotion, count in emotion_counts.most_common():
            percentage = count / len(ece_splits["all"]) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # Extraction method distribution
        extraction_methods = [s["extraction_method"] for s in ece_splits["all"]]
        method_counts = Counter(extraction_methods)
        
        print(f"\nExtraction Method Distribution:")
        for method, count in method_counts.most_common():
            percentage = count / len(ece_splits["all"]) * 100
            print(f"  {method}: {count} ({percentage:.1f}%)")
        
        # Source distribution
        sources = [s["source"] for s in ece_splits["all"]]
        source_counts = Counter(sources)
        
        print(f"\nSource Distribution:")
        for source, count in source_counts.most_common():
            percentage = count / len(ece_splits["all"]) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Paths (update these to your actual paths)
    emotion_mapping_path = "/home/rishi/Desktop/Aura-ML/emotion_mapping.json"
    esconv_dir = "/home/rishi/Desktop/Aura-ML/esconv_dataset-20251120T185554Z-1-001/esconv_dataset"
    output_dir = "/home/rishi/Desktop/Aura-ML/data/processed/ece_generated"
    
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
    
    # Initialize generator
    generator = ECEDatasetGenerator(
        emotion_mapping_path=emotion_mapping_path,
        output_dir=output_dir,
        use_bio_annotation=True
    )
    
    # Generate complete dataset
    ece_splits = generator.generate_complete_dataset(
        esconv_train_path=train_path,
        esconv_val_path=val_path,
        esconv_test_path=test_path,
        save_splits=True
    )
    
    print("\n✅ ECE Dataset Generation Complete!")
    print(f"Output directory: {output_dir}")
