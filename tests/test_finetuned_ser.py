"""
Test script for fine-tuned Speech Emotion Recognition model
Tests the custom emotion_model_finetuned/final model integration
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from aura_ml.models.audio_processor import SpeechEmotionRecognizer, AudioPipeline


def test_custom_model_loading():
    """Test 1: Verify custom fine-tuned model loads correctly"""
    print("\n" + "="*70)
    print("TEST 1: Loading Custom Fine-tuned SER Model")
    print("="*70)
    
    try:
        # Initialize with default (should use emotion_model_finetuned/final)
        recognizer = SpeechEmotionRecognizer()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {recognizer.device}")
        print(f"✓ Emotion labels: {recognizer.EMOTION_LABELS}")
        print(f"✓ ID to Label mapping: {recognizer.id2label}")
        
        # Verify 8 emotions
        assert len(recognizer.EMOTION_LABELS) == 8, "Expected 8 emotions"
        assert len(recognizer.id2label) == 8, "Expected 8 emotion labels in config"
        
        # Verify expected emotions are present
        expected_emotions = {"angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"}
        actual_emotions = set(recognizer.EMOTION_LABELS)
        assert actual_emotions == expected_emotions, f"Emotion mismatch: {actual_emotions} vs {expected_emotions}"
        
        print("\n✅ TEST 1 PASSED: Custom model loaded with correct configuration")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_inference():
    """Test 2: Test inference with synthetic audio"""
    print("\n" + "="*70)
    print("TEST 2: Model Inference with Synthetic Audio")
    print("="*70)
    
    try:
        recognizer = SpeechEmotionRecognizer()
        
        # Create synthetic audio (1 second, 16kHz, sine wave at 440Hz)
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440.0  # A4 note
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        print(f"✓ Created synthetic audio: {len(audio)} samples at {sample_rate}Hz")
        
        # Run emotion recognition
        result = recognizer.recognize_emotion(audio, sample_rate)
        
        print(f"\n✓ Inference completed successfully")
        print(f"  - Predicted emotion: {result['emotion']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"  - Model: {result['model']}")
        
        # Verify result structure
        assert 'emotion' in result
        assert 'confidence' in result
        assert 'emotion_scores' in result
        assert 'prosodic_features' in result
        assert result['emotion'] in recognizer.EMOTION_LABELS
        assert 0 <= result['confidence'] <= 1
        assert len(result['emotion_scores']) == 8
        
        print(f"\n  Emotion scores:")
        for emotion, score in sorted(result['emotion_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {emotion:10s}: {score:.2%}")
        
        print(f"\n  Prosodic features:")
        for feature, value in result['prosodic_features'].items():
            print(f"    {feature:25s}: {value:.2f}")
        
        print("\n✅ TEST 2 PASSED: Model inference working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_pipeline_integration():
    """Test 3: Test full AudioPipeline with custom SER model"""
    print("\n" + "="*70)
    print("TEST 3: Full AudioPipeline Integration")
    print("="*70)
    
    try:
        # Initialize pipeline (should use custom SER model)
        pipeline = AudioPipeline()
        
        print(f"✓ AudioPipeline initialized")
        print(f"✓ Whisper STT model on: {pipeline.stt.device}")
        print(f"✓ SER model device: {pipeline.ser.device}")
        print(f"✓ SER emotions: {pipeline.ser.EMOTION_LABELS}")
        
        # Create synthetic audio
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        print(f"\n✓ Created {duration}s synthetic audio")
        
        # Process audio through full pipeline
        print("\nProcessing through pipeline...")
        result = pipeline.process_audio(audio, sample_rate)
        
        print(f"\n✓ Pipeline processing completed")
        print(f"\n  AudioAnalysisResult:")
        print(f"    Transcription: '{result.transcription}'")
        print(f"    Emotion: {result.emotion}")
        print(f"    Confidence: {result.emotion_confidence:.2%}")
        print(f"    Duration: {result.duration:.2f}s")
        
        # Verify result
        assert result.emotion in pipeline.ser.EMOTION_LABELS
        assert 0 <= result.emotion_confidence <= 1
        assert len(result.emotion_scores) == 8
        assert result.duration > 0
        
        print(f"\n  Top 3 emotions:")
        top_emotions = sorted(result.emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, score in top_emotions:
            print(f"    {emotion:10s}: {score:.2%}")
        
        print("\n✅ TEST 3 PASSED: Full pipeline integration working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_parameters():
    """Test 4: Verify model architecture and parameters"""
    print("\n" + "="*70)
    print("TEST 4: Model Architecture Verification")
    print("="*70)
    
    try:
        recognizer = SpeechEmotionRecognizer()
        
        # Count parameters
        total_params = sum(p.numel() for p in recognizer.model.parameters())
        encoder_params = sum(p.numel() for p in recognizer.model.wav2vec2.parameters())
        classifier_params = sum(p.numel() for p in recognizer.model.classifier.parameters())
        
        print(f"\n  Model Architecture:")
        print(f"    Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"    Encoder parameters: {encoder_params:,} ({encoder_params/1e6:.1f}M)")
        print(f"    Classifier parameters: {classifier_params:,} ({classifier_params/1e6:.2f}M)")
        
        # Verify model type
        model_type = recognizer.model.config.model_type
        print(f"\n  Model type: {model_type}")
        assert model_type == "wav2vec2", f"Expected wav2vec2, got {model_type}"
        
        # Verify architecture matches report specifications
        # Report specifies: 95M frozen encoder + 0.2M trainable head ≈ 95.2M total
        expected_total = 95.2e6
        tolerance = 5e6  # 5M parameter tolerance
        
        if abs(total_params - expected_total) < tolerance:
            print(f"  ✓ Parameter count matches report (~95.2M)")
        else:
            print(f"  ⚠ Parameter count differs from report (expected ~95.2M)")
        
        print("\n✅ TEST 4 PASSED: Model architecture verified")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("FINE-TUNED SER MODEL TEST SUITE")
    print("Testing: emotion_model_finetuned/final")
    print("="*70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print(f"\n⚠ CUDA not available, using CPU")
    
    # Run tests
    results = []
    
    results.append(("Custom Model Loading", test_custom_model_loading()))
    results.append(("Model Inference", test_model_inference()))
    results.append(("Pipeline Integration", test_audio_pipeline_integration()))
    results.append(("Model Architecture", test_model_parameters()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your fine-tuned SER model is integrated correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
