"""
Quick import test for audio pipeline
Tests that the code is syntactically correct and importable
"""

import sys

def test_imports():
    """Test that audio processor can be imported"""
    print("Testing audio processor imports...")
    
    try:
        from aura_ml.models.audio_processor import (
            AudioPipeline,
            WhisperSTT,
            SpeechEmotionRecognizer,
            AudioAnalysisResult
        )
        print("✅ All audio classes imported successfully")
        print(f"   - AudioPipeline: {AudioPipeline}")
        print(f"   - WhisperSTT: {WhisperSTT}")
        print(f"   - SpeechEmotionRecognizer: {SpeechEmotionRecognizer}")
        print(f"   - AudioAnalysisResult: {AudioAnalysisResult}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_attributes():
    """Test that classes have expected attributes"""
    print("\nTesting class attributes...")
    
    try:
        from aura_ml.models.audio_processor import SpeechEmotionRecognizer
        
        # Check emotion labels
        emotions = SpeechEmotionRecognizer.EMOTION_LABELS
        expected_emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        
        if emotions == expected_emotions:
            print(f"✅ Emotion labels correct: {emotions}")
            return True
        else:
            print(f"❌ Emotion labels mismatch")
            print(f"   Expected: {expected_emotions}")
            print(f"   Got: {emotions}")
            return False
    except Exception as e:
        print(f"❌ Attribute test failed: {e}")
        return False


def test_dataclass():
    """Test AudioAnalysisResult dataclass"""
    print("\nTesting AudioAnalysisResult dataclass...")
    
    try:
        from aura_ml.models.audio_processor import AudioAnalysisResult
        
        # Create instance
        result = AudioAnalysisResult(
            transcription="Test transcription",
            emotion="happy",
            emotion_confidence=0.85,
            emotion_scores={"happy": 0.85, "sad": 0.15},
            duration=5.0
        )
        
        print(f"✅ AudioAnalysisResult created successfully")
        print(f"   Transcription: {result.transcription}")
        print(f"   Emotion: {result.emotion}")
        print(f"   Confidence: {result.emotion_confidence:.2%}")
        return True
    except Exception as e:
        print(f"❌ Dataclass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("AUDIO PIPELINE IMPORT TEST")
    print("="*60)
    print("This test validates code syntax and imports WITHOUT")
    print("downloading models or running inference.")
    print("="*60 + "\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Class Attributes", test_class_attributes),
        ("Dataclass Test", test_dataclass),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All import tests passed!")
        print("✅ Audio pipeline code is syntactically correct")
        print("✅ Ready for model download and inference testing")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
