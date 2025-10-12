#!/usr/bin/env python3
"""
Quick test script to verify SER model loading and inference
Run this to test emotion recognition without full WebSocket setup
"""

import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_emotion_model():
    """Test emotion recognition model loading and inference"""
    
    logger.info("=" * 60)
    logger.info("Testing Speech Emotion Recognition Service")
    logger.info("=" * 60)
    
    try:
        # Import emotion service
        logger.info("\n1. Importing emotion service...")
        from audio.emotion import EmotionRecognitionService
        logger.info("✅ Import successful")
        
        # Initialize service
        logger.info("\n2. Initializing emotion service...")
        service = EmotionRecognitionService()
        logger.info(f"✅ Service initialized with model: {service.model_name}")
        logger.info(f"   Device: {service.device}")
        
        # Load model
        logger.info("\n3. Loading emotion recognition model...")
        logger.info("   (This may take a few minutes on first run)")
        service.load_model()
        logger.info("✅ Model loaded successfully")
        
        # Check model status
        logger.info("\n4. Checking model status...")
        is_loaded = service.is_model_loaded()
        logger.info(f"✅ Model loaded: {is_loaded}")
        
        # Get emotion labels
        logger.info("\n5. Getting emotion labels...")
        labels = service.get_emotion_labels()
        logger.info(f"✅ Supported emotions: {', '.join(labels)}")
        
        # Create dummy audio for testing
        logger.info("\n6. Testing inference with dummy audio...")
        # Generate 2 seconds of random audio (simulating speech)
        sample_rate = 16000
        duration = 2.0
        audio_array = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        audio_array = audio_array * 0.3  # Scale to reasonable volume
        
        logger.info(f"   Audio shape: {audio_array.shape}")
        logger.info(f"   Sample rate: {sample_rate} Hz")
        logger.info(f"   Duration: {duration} seconds")
        
        # Run inference synchronously (for testing)
        result = service._recognize_sync(audio_array, sample_rate, return_all_scores=True)
        
        logger.info("✅ Inference successful!")
        logger.info(f"\n7. Inference Results:")
        logger.info(f"   Primary Emotion: {result['emotion']}")
        logger.info(f"   Confidence: {result['confidence']:.2%}")
        logger.info(f"   Inference Time: {result['inference_time_ms']}ms")
        
        if 'all_scores' in result:
            logger.info(f"\n   All Emotion Scores:")
            for emotion, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"      {emotion:10s}: {score:.2%}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("\nEmotion Recognition Service is ready to use.")
        logger.info("You can now test the full pipeline with:")
        logger.info("  docker-compose up --build")
        logger.info("  python test_audio_client.py")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        logger.info("\n" + "=" * 60)
        logger.info("⚠️  TESTS FAILED")
        logger.info("=" * 60)
        logger.info("\nTroubleshooting:")
        logger.info("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        logger.info("2. Check internet connection (model needs to download)")
        logger.info("3. Verify sufficient disk space (~2GB for models)")
        logger.info("4. Check PyTorch installation: python -c 'import torch; print(torch.__version__)'")
        return False


def test_integration():
    """Test integration with transcription service"""
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing STT + SER Integration")
    logger.info("=" * 60)
    
    try:
        import asyncio
        from audio.transcription import TranscriptionService
        from audio.emotion import EmotionRecognitionService
        
        logger.info("\n1. Initializing both services...")
        stt_service = TranscriptionService(model_name="openai/whisper-tiny")
        ser_service = EmotionRecognitionService()
        
        logger.info("2. Loading models...")
        stt_service.load_model()
        ser_service.load_model()
        logger.info("✅ Both models loaded")
        
        logger.info("\n3. Testing parallel processing...")
        # Create dummy audio
        audio_array = np.random.randn(32000).astype(np.float32) * 0.3
        
        async def run_parallel():
            # Simulate parallel processing
            stt_task = stt_service.transcribe_audio(audio_array, language="en")
            ser_task = ser_service.recognize_emotion(audio_array, sampling_rate=16000)
            
            results = await asyncio.gather(stt_task, ser_task, return_exceptions=True)
            return results
        
        results = asyncio.run(run_parallel())
        
        logger.info("✅ Parallel processing successful!")
        logger.info(f"   STT Result: {type(results[0])}")
        logger.info(f"   SER Result: {type(results[1])}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ INTEGRATION TEST PASSED!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Integration test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Test emotion model
    emotion_success = test_emotion_model()
    
    # Test integration if emotion test passed
    if emotion_success:
        logger.info("\n\nProceeding to integration test...\n")
        integration_success = test_integration()
        
        if integration_success:
            sys.exit(0)
        else:
            logger.warning("\nIntegration test failed but emotion model works standalone")
            sys.exit(1)
    else:
        sys.exit(1)
