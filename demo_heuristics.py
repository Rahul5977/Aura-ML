"""
Heuristic Rule Demonstration
=============================

This script demonstrates how the linguistic heuristics work for cause extraction.
Run this to see examples of each causal marker category.

Usage: python demo_heuristics.py
"""

# Example sentences demonstrating different causal markers
DEMO_EXAMPLES = {
    "Explicit Causality": [
        {
            "text": "I'm anxious because I might lose my job.",
            "marker": "because",
            "expected_cause": "I might lose my job",
            "confidence": 0.95
        },
        {
            "text": "She's depressed due to her father's illness.",
            "marker": "due to",
            "expected_cause": "her father's illness",
            "confidence": 0.95
        },
        {
            "text": "I'm stressed since I have too much work.",
            "marker": "since",
            "expected_cause": "I have too much work",
            "confidence": 0.90
        }
    ],
    
    "Temporal Causality": [
        {
            "text": "I felt sad when she left me.",
            "marker": "when",
            "expected_cause": "she left me",
            "confidence": 0.75
        },
        {
            "text": "I've been depressed ever since my divorce.",
            "marker": "ever since",
            "expected_cause": "my divorce",
            "confidence": 0.85
        },
        {
            "text": "Anxious after hearing the bad news.",
            "marker": "after",
            "expected_cause": "hearing the bad news",
            "confidence": 0.80
        }
    ],
    
    "Conditional Causality": [
        {
            "text": "I'm worried if I fail this exam.",
            "marker": "if",
            "expected_cause": "I fail this exam",
            "confidence": 0.65
        },
        {
            "text": "Stressed whenever I have a deadline.",
            "marker": "whenever",
            "expected_cause": "I have a deadline",
            "confidence": 0.70
        }
    ],
    
    "Emotional Context": [
        {
            "text": "I'm worried about losing my income.",
            "marker": "worried about",
            "expected_cause": "losing my income",
            "confidence": 0.75
        },
        {
            "text": "She's anxious that she won't get the promotion.",
            "marker": "anxious that",
            "expected_cause": "she won't get the promotion",
            "confidence": 0.75
        },
        {
            "text": "It makes me feel upset when people lie.",
            "marker": "makes me feel",
            "expected_cause": "when people lie",
            "confidence": 0.80
        }
    ],
    
    "Adversative Markers": [
        {
            "text": "I want to work but I'm too sick.",
            "marker": "but",
            "expected_cause": "I'm too sick",
            "confidence": 0.60
        },
        {
            "text": "I'm happy although life is difficult.",
            "marker": "although",
            "expected_cause": "life is difficult",
            "confidence": 0.62
        }
    ],
    
    "Explanatory Markers": [
        {
            "text": "The problem is I don't have enough money.",
            "marker": "the problem is",
            "expected_cause": "I don't have enough money",
            "confidence": 0.85
        },
        {
            "text": "The issue is my manager doesn't listen.",
            "marker": "the issue is",
            "expected_cause": "my manager doesn't listen",
            "confidence": 0.85
        }
    ],
    
    "No Causal Markers (Neutral)": [
        {
            "text": "I'm feeling good today.",
            "marker": "none",
            "expected_cause": None,
            "confidence": 0.0
        },
        {
            "text": "Everything is fine and normal.",
            "marker": "none",
            "expected_cause": None,
            "confidence": 0.0
        }
    ]
}


def print_header(text, char="="):
    """Print formatted header."""
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}\n")


def print_example(example, extracted_cause=None, actual_confidence=None):
    """Print formatted example."""
    print(f"  📝 Text: {example['text']}")
    print(f"  🔍 Marker: '{example['marker']}' (confidence: {example['confidence']})")
    print(f"  ✅ Expected Cause: {example['expected_cause']}")
    
    if extracted_cause is not None:
        if extracted_cause == example['expected_cause'] or \
           (example['expected_cause'] and extracted_cause and 
            all(word in extracted_cause.lower() for word in example['expected_cause'].lower().split()[:3])):
            print(f"  ✅ Extracted: '{extracted_cause}' (MATCH)")
        elif extracted_cause is None and example['expected_cause'] is None:
            print(f"  ✅ Extracted: None (CORRECT - no marker)")
        else:
            print(f"  ⚠️  Extracted: '{extracted_cause}' (DIFFERENT)")
    
    if actual_confidence:
        print(f"  📊 Actual Confidence: {actual_confidence:.2f}")
    
    print()


def demo_without_imports():
    """Demonstrate examples without requiring imports (for documentation)."""
    print_header("🧠 LINGUISTIC HEURISTICS DEMONSTRATION")
    print("This demonstrates the 'Why Engine' - how we extract emotional causes")
    print("using sophisticated pattern matching and linguistic rules.")
    
    for category, examples in DEMO_EXAMPLES.items():
        print_header(category, char="-")
        
        for i, example in enumerate(examples, 1):
            print(f"Example {i}:")
            print_example(example)
    
    print_header("📚 KEY INSIGHTS")
    print("""
1. CONFIDENCE SCORES determine which marker to trust when multiple are found
   - Higher confidence = more reliable causal relationship
   - Explicit markers (because, due to) have highest confidence (0.90-0.95)
   - Adversative markers (but, however) have lower confidence (0.60-0.65)

2. CAPTURE DIRECTION varies by marker type:
   - Most markers capture AFTER: "anxious BECAUSE [this thing]"
   - Some capture BEFORE: "therefore" looks at preceding context
   
3. BOUNDARY DETECTION is intelligent:
   - Stops at sentence endings (periods, exclamation marks)
   - Respects clause boundaries (commas, semicolons, conjunctions)
   - Avoids extracting too much or too little text
   
4. VALIDATION ensures quality:
   - Minimum 2 words, 10 characters
   - Must contain verbs or nouns (semantic validation)
   - Excludes questions, pure punctuation, filler words
   
5. NEUTRAL EMOTION handling:
   - If NO causal markers found → emotion = "neutral"
   - Sample is filtered out (per requirements)
   - Ensures only samples with identifiable causes are included
    """)


def demo_with_extractor():
    """Demonstrate with actual CauseExtractor (requires imports)."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from prepare_data import CauseExtractor
        
        print_header("🔬 LIVE EXTRACTION DEMONSTRATION")
        print("Running actual cause extraction on example sentences...\n")
        
        extractor = CauseExtractor()
        
        for category, examples in DEMO_EXAMPLES.items():
            print_header(category, char="-")
            
            for i, example in enumerate(examples, 1):
                text = example['text']
                result = extractor.extract_cause(text)
                
                extracted_cause = result[0] if result else None
                confidence = result[1] if result else 0.0
                
                print(f"Example {i}:")
                print_example(example, extracted_cause, confidence)
        
        # Additional test: Show BIO tagging
        print_header("🏷️  BIO TAGGING EXAMPLE")
        
        test_text = "I'm anxious because I might lose my job"
        result = extractor.extract_cause(test_text)
        
        if result:
            cause_text, confidence, cause_start, cause_end = result
            bio_tags = extractor.generate_bio_tags(test_text, cause_start, cause_end)
            
            doc = extractor.nlp(test_text)
            tokens = [token.text for token in doc]
            
            print(f"Text: {test_text}")
            print(f"Cause: {cause_text}")
            print(f"\nTokenization and BIO Tags:")
            print(f"{'Token':<15} {'BIO Tag':<10}")
            print("-" * 30)
            
            for token, tag in zip(tokens, bio_tags):
                marker = "👉" if tag != "O" else "  "
                print(f"{marker} {token:<15} {tag:<10}")
        
        print_header("✅ LIVE EXTRACTION COMPLETE")
        
    except ImportError as e:
        print("\n⚠️  Could not import CauseExtractor (dependencies missing)")
        print(f"Error: {e}")
        print("\nTo see live extraction, install dependencies:")
        print("  pip install -r requirements_data_prep.txt")
        print("  python -m spacy download en_core_web_sm")


def main():
    """Main execution."""
    # Always show documentation examples
    demo_without_imports()
    
    # Try to show live extraction if dependencies available
    print("\n" + "=" * 80)
    response = input("Would you like to see LIVE extraction? (requires dependencies) [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        demo_with_extractor()
    else:
        print("\n✅ Demonstration complete!")
        print("\nTo see live extraction later, run:")
        print("  python demo_heuristics.py")
        print("  (then answer 'y' to the prompt)")


if __name__ == "__main__":
    main()
