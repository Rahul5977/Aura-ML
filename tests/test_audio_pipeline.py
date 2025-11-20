"""
Test script for audio pipeline
Generates a simple test to verify installation
"""

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages are installed"""
    logger.info("Testing imports...")
    
    try:
        import librosa
        logger.info("✅ librosa imported")
    except ImportError as e:
        logger.error(f"❌ librosa import failed: {e}")
        return False
    
    try:
        import soundfile
        logger.info("✅ soundfile imported")
    except ImportError as e:
        logger.error(f"❌ soundfile import failed: {e}")
        return False
    
    try:
        from transformers import WhisperProcessor, Wav2Vec2Processor
        logger.info("✅ transformers imported")
    except ImportError as e:
        logger.error(f"❌ transformers import failed: {e}")
        return False
    
    return True


def test_audio_processor():
    """Test audio processor initialization"""
    logger.info("\nTesting audio processor...")
    
    try:
        from aura_ml.models.audio_processor import AudioPipeline
        logger.info("✅ AudioPipeline imported")
        
        # Test initialization (will download models on first run)
        logger.info("Initializing pipeline (may download models ~670MB)...")
        pipeline = AudioPipeline()
        logger.info("✅ AudioPipeline initialized")
        
        return True
    except Exception as e:
        logger.error(f"❌ AudioPipeline test failed: {e}")
        return False


def test_synthetic_audio():
    """Test with synthetic audio"""
    logger.info("\nTesting with synthetic audio...")
    
    try:
        from aura_ml.models.audio_processor import AudioPipeline
        import numpy as np
        
        # Create synthetic audio (1 second of sine wave at 440 Hz)
        duration = 1.0
        sampling_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sampling_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        logger.info("Created synthetic audio (1 sec, 440 Hz)")
        
        # Initialize pipeline
        pipeline = AudioPipeline()
        
        # Process synthetic audio
        logger.info("Processing synthetic audio...")
        result = pipeline.process_audio(
            audio, 
            sampling_rate,
            return_timestamps=False,
            return_prosodic=True
        )
        
        logger.info(f"✅ Processing complete!")
        logger.info(f"   Transcription: '{result.transcription}'")
        logger.info(f"   Emotion: {result.emotion} ({result.emotion_confidence:.2%})")
        logger.info(f"   Duration: {result.duration:.2f}s")
        
        if result.prosodic_features:
            logger.info(f"   Pitch: {result.prosodic_features['pitch_mean_hz']:.1f} Hz")
        
        return True
    except Exception as e:
        logger.error(f"❌ Synthetic audio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("AUDIO PIPELINE TEST SUITE")
    logger.info("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("⚠️  CUDA not available, will use CPU (slower)")
    
    # Run tests
    tests = [
        ("Package Imports", test_imports),
        ("Audio Processor", test_audio_processor),
        ("Synthetic Audio", test_synthetic_audio),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Audio pipeline is ready.")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
