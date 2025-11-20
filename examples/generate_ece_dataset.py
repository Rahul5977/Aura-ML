#!/usr/bin/env python3
"""
ECE Dataset Generation - Example Usage

This script demonstrates how to generate the ECE dataset from ESConv
using the complete two-pass extraction pipeline.

Usage:
    python examples/generate_ece_dataset.py
    
    or
    
    python examples/generate_ece_dataset.py --esconv-dir /path/to/esconv --output-dir /path/to/output
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_ml.data import ECEDatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate ECE dataset from ESConv")
    
    parser.add_argument(
        "--esconv-dir",
        type=str,
        default="/home/rishi/Desktop/Aura-ML/esconv_dataset-20251120T185554Z-1-001/esconv_dataset",
        help="Path to ESConv dataset directory"
    )
    
    parser.add_argument(
        "--emotion-mapping",
        type=str,
        default="/home/rishi/Desktop/Aura-ML/emotion_mapping.json",
        help="Path to emotion mapping JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/rishi/Desktop/Aura-ML/data/processed/ece_generated",
        help="Output directory for generated ECE dataset"
    )
    
    parser.add_argument(
        "--no-bio",
        action="store_true",
        help="Disable BIO annotation (faster, but no sequence labeling data)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    esconv_dir = Path(args.esconv_dir)
    emotion_mapping = Path(args.emotion_mapping)
    
    if not esconv_dir.exists():
        print(f"❌ Error: ESConv directory not found: {esconv_dir}")
        sys.exit(1)
    
    if not emotion_mapping.exists():
        print(f"❌ Error: Emotion mapping file not found: {emotion_mapping}")
        sys.exit(1)
    
    train_path = esconv_dir / "train.jsonl"
    val_path = esconv_dir / "validation.jsonl"
    test_path = esconv_dir / "test.jsonl"
    
    if not train_path.exists():
        print(f"❌ Error: Training file not found: {train_path}")
        sys.exit(1)
    
    print("="*80)
    print("ECE Dataset Generation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  ESConv directory: {esconv_dir}")
    print(f"  Emotion mapping: {emotion_mapping}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  BIO annotation: {'Disabled' if args.no_bio else 'Enabled'}")
    print()
    
    # Initialize generator
    generator = ECEDatasetGenerator(
        emotion_mapping_path=str(emotion_mapping),
        output_dir=args.output_dir,
        use_bio_annotation=not args.no_bio
    )
    
    # Generate complete dataset
    ece_splits = generator.generate_complete_dataset(
        esconv_train_path=str(train_path),
        esconv_val_path=str(val_path) if val_path.exists() else None,
        esconv_test_path=str(test_path) if test_path.exists() else None,
        save_splits=True
    )
    
    print("\n" + "="*80)
    print("✅ ECE Dataset Generation Complete!")
    print("="*80)
    print(f"\nOutput files:")
    output_dir = Path(args.output_dir)
    for file in sorted(output_dir.glob("ece_*.json")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name:20s} ({size_mb:.2f} MB)")
    
    print(f"\nTotal samples: {len(ece_splits['all'])}")
    print(f"  Train: {len(ece_splits.get('train', []))}")
    print(f"  Val: {len(ece_splits.get('val', []))}")
    print(f"  Test: {len(ece_splits.get('test', []))}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
