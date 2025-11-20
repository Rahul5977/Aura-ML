"""
Test script to verify ECE dataset generation pipeline
"""

import sys
from pathlib import Path


def test_keyword_extractor():
    """Test causal keyword extractor"""
    print("\n" + "="*80)
    print("Testing Causal Keyword Extractor")
    print("="*80)
    
    from aura_ml.data.causal_keyword_extractor import CausalKeywordExtractor
    
    extractor = CausalKeywordExtractor()
    
    test_cases = [
        "I am anxious because I might lose my job",
        "I feel sad due to my father's passing",
        "I'm worried about my financial situation",
        "I'm happy for the first time in months"
    ]
    
    for text in test_cases:
        result = extractor.extract_cause(text)
        print(f"\nText: {text}")
        if result:
            print(f"✓ Cause: {result['cause']}")
            print(f"  Category: {result['category']}, Keyword: {result['keyword']}")
        else:
            print("✗ No cause found")
    
    stats = extractor.get_statistics()
    print(f"\nStatistics: {stats['successful_extractions']}/{stats['total_attempts']} "
          f"({stats['coverage_percentage']})")
    
    return True


def test_heuristic_extractor():
    """Test heuristic cause extractor"""
    print("\n" + "="*80)
    print("Testing Heuristic Cause Extractor")
    print("="*80)
    
    from aura_ml.data.heuristic_fallback import HeuristicCauseExtractor
    
    extractor = HeuristicCauseExtractor()
    
    test_cases = [
        "Good idea..",
        "Thank you for your kindness",
        "Online is convenient but its just too much going on"
    ]
    
    for text in test_cases:
        result = extractor.extract_cause(text)
        print(f"\nText: {text}")
        if result:
            print(f"✓ Cause: {result['cause'][:60]}...")
            print(f"  Method: {result['method']}")
        else:
            print("✗ No cause found")
    
    stats = extractor.get_statistics()
    print(f"\nStatistics: {stats['successful_extractions']}/{stats['total_attempts']} "
          f"({stats['coverage_percentage']})")
    
    return True


def test_bio_annotator():
    """Test BIO annotator"""
    print("\n" + "="*80)
    print("Testing BIO Annotator")
    print("="*80)
    
    from aura_ml.data.bio_annotator import BIOAnnotator
    from collections import Counter
    
    annotator = BIOAnnotator()
    
    text = "I am anxious because I might lose my job"
    cause = "I might lose my job"
    
    print(f"\nText: {text}")
    print(f"Cause: {cause}")
    
    result = annotator.annotate(text, cause)
    
    print("\nFirst 15 tokens with BIO tags:")
    for token, bio_tag in list(zip(result["tokens"], result["bio_tags"]))[:15]:
        if token.strip():
            print(f"  {token:20s} -> {bio_tag}")
    
    tag_counts = Counter(result["bio_tags"])
    print(f"\nBIO Tag Distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")
    
    return True


def test_esconv_processor():
    """Test ESConv processor"""
    print("\n" + "="*80)
    print("Testing ESConv Processor")
    print("="*80)
    
    from aura_ml.data.esconv_processor import ESConvProcessor
    
    emotion_mapping_path = "/home/rishi/Desktop/Aura-ML/emotion_mapping.json"
    
    if not Path(emotion_mapping_path).exists():
        print(f"⚠ emotion_mapping.json not found at {emotion_mapping_path}")
        print("Skipping ESConv processor test")
        return False
    
    processor = ESConvProcessor(emotion_mapping_path)
    
    # Just test loading emotion mapping
    print(f"\n✓ Loaded emotion mapping")
    print(f"  Emotion terms: {len(processor.emotion_mapping)}")
    print(f"  Categories: {len(set(processor.emotion_mapping.values()))}")
    
    print("\nEmotion categories:")
    for category in sorted(set(processor.emotion_mapping.values())):
        terms = [k for k, v in processor.emotion_mapping.items() if v == category]
        print(f"  {category}: {len(terms)} terms")
    
    return True


def test_complete_pipeline():
    """Test complete ECE generation pipeline"""
    print("\n" + "="*80)
    print("Testing Complete Pipeline (Dry Run)")
    print("="*80)
    
    # Create a minimal synthetic dataset
    print("\nCreating synthetic utterances...")
    
    synthetic_utterances = [
        {
            "text": "I am anxious because I might lose my job",
            "emotion": "fear",
            "original_emotion": "anxiety"
        },
        {
            "text": "I feel sad due to my father's passing",
            "emotion": "sad",
            "original_emotion": "depression"
        },
        {
            "text": "Good idea..",
            "emotion": "neutral",
            "original_emotion": "neutral"
        }
    ]
    
    # Test Pass 1 + Pass 2 extraction
    from aura_ml.data.causal_keyword_extractor import CausalKeywordExtractor
    from aura_ml.data.heuristic_fallback import HeuristicCauseExtractor
    
    keyword_extractor = CausalKeywordExtractor()
    heuristic_extractor = HeuristicCauseExtractor()
    
    ece_samples = []
    pass1_count = 0
    pass2_count = 0
    
    for utt in synthetic_utterances:
        text = utt["text"]
        
        # Try Pass 1
        keyword_result = keyword_extractor.extract_cause(text)
        if keyword_result:
            ece_samples.append({
                "text": text,
                "emotion": utt["emotion"],
                "cause": keyword_result["cause"],
                "source": keyword_result["source"]
            })
            pass1_count += 1
        else:
            # Try Pass 2
            heuristic_result = heuristic_extractor.extract_cause(text)
            if heuristic_result:
                ece_samples.append({
                    "text": text,
                    "emotion": utt["emotion"],
                    "cause": heuristic_result["cause"],
                    "source": heuristic_result["source"]
                })
                pass2_count += 1
    
    print(f"\n✓ Generated {len(ece_samples)} ECE samples")
    print(f"  Pass 1 (Keyword): {pass1_count}")
    print(f"  Pass 2 (Heuristic): {pass2_count}")
    print(f"  Coverage: {len(ece_samples)}/{len(synthetic_utterances)} "
          f"({len(ece_samples)/len(synthetic_utterances)*100:.1f}%)")
    
    print("\nSample ECE outputs:")
    for i, sample in enumerate(ece_samples, 1):
        print(f"\n{i}. [{sample['emotion']}]")
        print(f"   Text: {sample['text'][:60]}...")
        print(f"   Cause: {sample['cause'][:60]}...")
        print(f"   Source: {sample['source']}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("ECE Dataset Generation Pipeline - Test Suite")
    print("="*80)
    
    tests = [
        ("Keyword Extractor", test_keyword_extractor),
        ("Heuristic Extractor", test_heuristic_extractor),
        ("BIO Annotator", test_bio_annotator),
        ("ESConv Processor", test_esconv_processor),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✓ PASS" if success else "⚠ SKIP"
        except Exception as e:
            results[test_name] = f"✗ FAIL: {str(e)}"
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, result in results.items():
        print(f"{test_name:30s} {result}")
    
    print("\n" + "="*80)
    
    # Return exit code
    failed = sum(1 for r in results.values() if "FAIL" in r)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
