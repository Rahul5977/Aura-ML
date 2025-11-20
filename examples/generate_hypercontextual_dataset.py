#!/usr/bin/env python3
"""
Generate Hypercontextual Dataset for LLM Fine-tuning

This script generates a comprehensive instruction-tuning dataset by enriching
ESConv conversations with multi-modal analysis outputs:
- Emotion labels (from ESConv annotations)
- Extracted causes (from trained ECE model)
- Named entities (from spaCy NER)
- Problem types (heuristic classification)
- Conversation history (sliding window)
- Support strategies (from ESConv annotations)

The output is formatted as instruction-completion pairs suitable for LLM fine-tuning.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aura_ml.data.hypercontextual_dataset_generator import HypercontextualDatasetGenerator


def main():
    """Generate hypercontextual dataset from ESConv"""
    
    # Configuration
    ESCONV_PATH = "esconv_dataset-20251120T185554Z-1-001/esconv_dataset"
    ECE_MODEL_PATH = "data/models/ece/ece_roberta_model"  # Update with your model path
    OUTPUT_DIR = "data/processed/hypercontextual"
    HISTORY_WINDOW = 3
    TRAIN_SPLIT = 0.9
    
    print("="*70)
    print("HYPERCONTEXTUAL DATASET GENERATOR")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  ESConv Dataset: {ESCONV_PATH}")
    print(f"  ECE Model:      {ECE_MODEL_PATH}")
    print(f"  Output Dir:     {OUTPUT_DIR}")
    print(f"  History Window: {HISTORY_WINDOW} turns")
    print(f"  Train Split:    {TRAIN_SPLIT:.0%}")
    print()
    
    # Check paths
    if not Path(ESCONV_PATH).exists():
        print(f"❌ Error: ESConv dataset not found at {ESCONV_PATH}")
        print("\nPlease update ESCONV_PATH in this script to point to your ESConv dataset.")
        return
    
    if not Path(ECE_MODEL_PATH).exists():
        print(f"❌ Error: ECE model not found at {ECE_MODEL_PATH}")
        print("\nPlease update ECE_MODEL_PATH to point to your trained ECE model.")
        print("You can train an ECE model first using the ECE dataset generation pipeline.")
        return
    
    # Initialize generator
    print("🔧 Initializing generator...")
    generator = HypercontextualDatasetGenerator(
        ece_model_path=ECE_MODEL_PATH,
        history_window=HISTORY_WINDOW
    )
    
    # Generate dataset
    print("\n📊 Processing ESConv conversations...")
    print("This may take several minutes...\n")
    
    stats = generator.generate_dataset(
        esconv_path=ESCONV_PATH,
        output_dir=OUTPUT_DIR,
        train_split=TRAIN_SPLIT
    )
    
    # Print summary
    print("\n" + "="*70)
    print("✅ DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"\n📈 Statistics:")
    print(f"  Total samples:       {stats['total_samples']:,}")
    print(f"  Training samples:    {stats['train_samples']:,}")
    print(f"  Validation samples:  {stats['val_samples']:,}")
    
    print(f"\n😊 Emotion Distribution (Top 5):")
    for emotion, count in list(stats['emotion_distribution'].items())[:5]:
        percentage = (count / stats['total_samples']) * 100
        print(f"  {emotion:<12} {count:>6,} ({percentage:>5.1f}%)")
    
    print(f"\n🎯 Problem Type Distribution:")
    for ptype, count in stats['problem_type_distribution'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"  {ptype:<20} {count:>6,} ({percentage:>5.1f}%)")
    
    print(f"\n💬 Support Strategy Distribution (Top 5):")
    for strategy, count in list(stats['strategy_distribution'].items())[:5]:
        percentage = (count / stats['total_samples']) * 100
        print(f"  {strategy:<30} {count:>6,} ({percentage:>5.1f}%)")
    
    print(f"\n📁 Output Files:")
    print(f"  {OUTPUT_DIR}/llm_training_data.json  (all samples)")
    print(f"  {OUTPUT_DIR}/llm_train.json          (training split)")
    print(f"  {OUTPUT_DIR}/llm_val.json            (validation split)")
    print(f"  {OUTPUT_DIR}/dataset_statistics.json (statistics)")
    
    print("\n🚀 Next Steps:")
    print("  1. Review the generated dataset samples")
    print("  2. Use llm_train.json and llm_val.json for LLM fine-tuning")
    print("  3. Fine-tune your LLM (LLaMA, Mistral, Phi, etc.) on this dataset")
    print("="*70)


if __name__ == '__main__':
    main()
