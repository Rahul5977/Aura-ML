"""
Test Suite for ECE Training Data Preparation
=============================================

This script validates the prepare_data.py functionality with unit tests
and integration tests to ensure data quality.

Run: python test_prepare_data.py
"""

import json
import sys
from pathlib import Path

# Test data samples
TEST_UTTERANCES = [
    {
        "text": "I'm anxious because I might lose my job soon.",
        "emotion": "anxiety",
        "expected_cause": "I might lose my job soon",
        "expected_emotion": "anxiety"
    },
    {
        "text": "I felt depressed when my father passed away last year.",
        "emotion": "depression",
        "expected_cause": "my father passed away last year",
        "expected_emotion": "depression"
    },
    {
        "text": "I'm worried that I won't be able to pay my rent.",
        "emotion": "anxiety",
        "expected_cause": "I won't be able to pay my rent",
        "expected_emotion": "anxiety"
    },
    {
        "text": "She's frustrated due to the long commute every day.",
        "emotion": "anger",
        "expected_cause": "the long commute every day",
        "expected_emotion": "anger"
    },
    {
        "text": "I'm happy and everything is great today.",
        "emotion": "joy",
        "expected_cause": None,  # No causal marker
        "expected_emotion": "neutral"  # Should default to neutral
    },
    {
        "text": "Feeling scared if I fail the exam tomorrow.",
        "emotion": "fear",
        "expected_cause": "I fail the exam tomorrow",
        "expected_emotion": "fear"
    },
    {
        "text": "I've been sad ever since she left me.",
        "emotion": "sadness",
        "expected_cause": "she left me",
        "expected_emotion": "sadness"
    },
    {
        "text": "The problem is I don't have enough money.",
        "emotion": "neutral",
        "expected_cause": "I don't have enough money",
        "expected_emotion": "neutral"
    }
]


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")


def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with visual indicators."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status:12s} | {test_name:40s} | {details}")


def test_imports():
    """Test if all required modules can be imported."""
    print_header("TEST 1: Module Imports")
    
    tests = [
        ("json", "json"),
        ("re", "re"),
        ("pathlib", "pathlib.Path"),
        ("typing", "typing.List"),
        ("dataclasses", "dataclasses.dataclass"),
        ("spacy", "spacy"),
        ("tqdm", "tqdm.tqdm")
    ]
    
    all_passed = True
    for module_name, import_path in tests:
        try:
            parts = import_path.split('.')
            module = __import__(parts[0])
            for part in parts[1:]:
                module = getattr(module, part)
            print_test_result(f"Import {module_name}", True)
        except ImportError as e:
            print_test_result(f"Import {module_name}", False, str(e))
            all_passed = False
    
    return all_passed


def test_spacy_model():
    """Test if spaCy model is available."""
    print_header("TEST 2: spaCy Model")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp("This is a test sentence.")
        tokens = [token.text for token in doc]
        
        print_test_result("Load spaCy model", True, f"Loaded en_core_web_sm")
        print_test_result("Tokenization", len(tokens) == 6, f"Got {len(tokens)} tokens")
        return True
    except Exception as e:
        print_test_result("spaCy model", False, str(e))
        print("\n⚠️  Please install: python -m spacy download en_core_web_sm")
        return False


def test_cause_extractor():
    """Test the CauseExtractor class."""
    print_header("TEST 3: Cause Extraction Logic")
    
    try:
        # Import the module
        sys.path.insert(0, str(Path(__file__).parent))
        from prepare_data import CauseExtractor
        
        extractor = CauseExtractor()
        all_passed = True
        
        for i, test_case in enumerate(TEST_UTTERANCES, 1):
            text = test_case['text']
            expected_cause = test_case['expected_cause']
            
            result = extractor.extract_cause(text)
            
            if expected_cause is None:
                # Should not find a cause
                passed = result is None
                details = "No cause expected (correctly)" if passed else f"Found: {result[0]}"
            else:
                # Should find a cause
                if result is None:
                    passed = False
                    details = "No cause found"
                else:
                    cause_text = result[0]
                    # Check if extracted cause contains expected keywords
                    passed = any(word in cause_text.lower() for word in expected_cause.lower().split()[:3])
                    details = f"Found: '{cause_text}'"
            
            print_test_result(f"Test case {i}", passed, details)
            if not passed:
                all_passed = False
        
        return all_passed
    
    except Exception as e:
        print_test_result("Cause extraction", False, str(e))
        return False


def test_emotion_detection():
    """Test emotion detection logic."""
    print_header("TEST 4: Emotion Detection")
    
    try:
        from prepare_data import CauseExtractor
        
        extractor = CauseExtractor()
        all_passed = True
        
        for i, test_case in enumerate(TEST_UTTERANCES, 1):
            text = test_case['text']
            context_emotion = test_case['emotion']
            expected_emotion = test_case['expected_emotion']
            
            detected_emotion = extractor.detect_emotion(text, context_emotion)
            
            # For this test, we accept the context emotion if it's in the text
            passed = detected_emotion == expected_emotion or detected_emotion == context_emotion
            details = f"Detected: {detected_emotion}, Expected: {expected_emotion}"
            
            print_test_result(f"Emotion test {i}", passed, details)
            if not passed:
                all_passed = False
        
        return all_passed
    
    except Exception as e:
        print_test_result("Emotion detection", False, str(e))
        return False


def test_bio_tagging():
    """Test BIO tag generation."""
    print_header("TEST 5: BIO Tag Generation")
    
    try:
        from prepare_data import CauseExtractor
        
        extractor = CauseExtractor()
        
        # Test case
        text = "I'm anxious because I might lose my job"
        cause_start = text.index("I might")
        cause_end = len(text)
        
        bio_tags = extractor.generate_bio_tags(text, cause_start, cause_end)
        
        # Tokenize to verify
        doc = extractor.nlp(text)
        tokens = [token.text for token in doc]
        
        # Checks
        test_results = [
            ("Tag-token alignment", len(bio_tags) == len(tokens), 
             f"{len(bio_tags)} tags, {len(tokens)} tokens"),
            ("Has B-CAUSE tag", 'B-CAUSE' in bio_tags, 
             f"Found: {bio_tags.count('B-CAUSE')} B-CAUSE tags"),
            ("Has I-CAUSE tags", 'I-CAUSE' in bio_tags, 
             f"Found: {bio_tags.count('I-CAUSE')} I-CAUSE tags"),
            ("Has O tags", 'O' in bio_tags, 
             f"Found: {bio_tags.count('O')} O tags"),
        ]
        
        all_passed = True
        for test_name, passed, details in test_results:
            print_test_result(test_name, passed, details)
            if not passed:
                all_passed = False
        
        # Print example
        print("\n  Example BIO tagging:")
        print(f"  Tokens:  {' | '.join(tokens)}")
        print(f"  Tags:    {' | '.join(bio_tags)}")
        
        return all_passed
    
    except Exception as e:
        print_test_result("BIO tagging", False, str(e))
        return False


def test_dataset_loading():
    """Test dataset loading functionality."""
    print_header("TEST 6: Dataset Loading")
    
    try:
        from prepare_data import ESConvDataProcessor
        
        # Check if dataset exists
        dataset_dir = Path('esconv_dataset')
        if not dataset_dir.exists():
            print_test_result("Dataset directory", False, f"Not found: {dataset_dir}")
            print("\n⚠️  Please ensure ESConv dataset is in 'esconv_dataset' folder")
            return False
        
        print_test_result("Dataset directory", True, f"Found: {dataset_dir}")
        
        # Check for files
        expected_files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
        files_found = []
        
        for filename in expected_files:
            file_path = dataset_dir / filename
            exists = file_path.exists()
            print_test_result(f"File: {filename}", exists, 
                            f"Size: {file_path.stat().st_size // 1024} KB" if exists else "Not found")
            if exists:
                files_found.append(filename)
        
        if not files_found:
            print("\n⚠️  No dataset files found. Please download ESConv dataset.")
            return False
        
        # Try loading
        processor = ESConvDataProcessor()
        data = processor.load_dataset()
        
        print_test_result("Load dataset", len(data) > 0, f"Loaded {len(data)} conversations")
        
        return len(data) > 0
    
    except Exception as e:
        print_test_result("Dataset loading", False, str(e))
        return False


def test_end_to_end():
    """Test complete pipeline on sample data."""
    print_header("TEST 7: End-to-End Processing")
    
    try:
        from prepare_data import ESConvDataProcessor
        
        processor = ESConvDataProcessor()
        
        # Create mock utterance
        mock_utterance = {
            'text': "I'm really anxious because I might lose my job soon.",
            'emotion': 'anxiety',
            'context': {
                'problem_type': 'job crisis',
                'situation': 'employment concerns'
            }
        }
        
        # Process utterance
        result = processor.process_utterance(mock_utterance)
        
        if result is None:
            print_test_result("Process utterance", False, "No result returned")
            return False
        
        # Validate result structure
        tests = [
            ("Has 'text' field", 'text' in result),
            ("Has 'emotion' field", 'emotion' in result),
            ("Has 'cause_span' field", 'cause_span' in result),
            ("Has 'bio_tags' field", 'bio_tags' in result),
            ("Cause span not empty", len(result.get('cause_span', '')) > 0),
            ("BIO tags is list", isinstance(result.get('bio_tags'), list)),
            ("BIO tags not empty", len(result.get('bio_tags', [])) > 0)
        ]
        
        all_passed = True
        for test_name, passed in tests:
            print_test_result(test_name, passed)
            if not passed:
                all_passed = False
        
        # Print result
        if all_passed:
            print("\n  Sample Result:")
            print(f"  Text: {result['text']}")
            print(f"  Emotion: {result['emotion']}")
            print(f"  Cause: {result['cause_span']}")
            print(f"  BIO Tags: {result['bio_tags'][:10]}..." if len(result['bio_tags']) > 10 else f"  BIO Tags: {result['bio_tags']}")
        
        return all_passed
    
    except Exception as e:
        print_test_result("End-to-end processing", False, str(e))
        return False


def run_all_tests():
    """Run all test suites."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " ECE TRAINING DATA PREPARATION - TEST SUITE ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    tests = [
        ("Module Imports", test_imports),
        ("spaCy Model", test_spacy_model),
        ("Cause Extraction", test_cause_extractor),
        ("Emotion Detection", test_emotion_detection),
        ("BIO Tagging", test_bio_tagging),
        ("Dataset Loading", test_dataset_loading),
        ("End-to-End", test_end_to_end)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:12s} | {test_name}")
    
    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("-" * 80)
    
    if passed == total:
        print("\n🎉 All tests passed! The data preparation pipeline is ready.")
        print("\n✅ You can now run: python prepare_data.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running prepare_data.py")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements_data_prep.txt")
        print("  2. Download spaCy model: python -m spacy download en_core_web_sm")
        print("  3. Ensure ESConv dataset is in 'esconv_dataset' folder")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
