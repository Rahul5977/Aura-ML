"""
Test Script for Scene Analysis Pipeline
Demonstrates the complete video captioning workflow

This script tests the scene_captioner module with various scenarios
and provides examples of how to use the API.
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video.scene_captioner import (
    extract_keyframes,
    SceneCaptioner,
    analyze_video_scene
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_extract_keyframes_sample():
    """Test keyframe extraction with sample generation"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Keyframe Extraction with Sample Video")
    logger.info("="*80)
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Create a sample video for testing
        logger.info("Creating sample video...")
        
        output_path = "test_sample_video.mp4"
        fps = 30
        duration = 5  # 5 seconds
        width, height = 640, 480
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Generate frames with changing colors
        for i in range(fps * duration):
            # Create a frame with changing color
            hue = (i * 2) % 180
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = hue  # Hue channel
            frame[:, :, 1] = 255  # Saturation
            frame[:, :, 2] = 200  # Value
            
            # Convert HSV to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            
            # Add timestamp text
            text = f"Frame {i} - Time: {i/fps:.2f}s"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        logger.info(f"✅ Sample video created: {output_path}")
        
        # Test extraction
        logger.info("\nExtracting keyframes...")
        keyframes = extract_keyframes(output_path, interval_sec=1.0)
        
        logger.info(f"\n✅ Extracted {len(keyframes)} keyframes")
        
        for i, kf in enumerate(keyframes[:3]):  # Show first 3
            logger.info(f"Keyframe {i+1}:")
            logger.info(f"  - Timestamp: {kf['timestamp']:.2f}s")
            logger.info(f"  - Formatted: {kf['formatted_time']}")
            logger.info(f"  - Frame #: {kf['frame_number']}")
            logger.info(f"  - Image size: {kf['frame'].size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_scene_captioner_basic():
    """Test SceneCaptioner with a simple image"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: SceneCaptioner Basic Functionality")
    logger.info("="*80)
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create a test image
        logger.info("Creating test image...")
        img = Image.new('RGB', (640, 480), color=(73, 109, 137))
        
        # Add some text
        draw = ImageDraw.Draw(img)
        draw.text((50, 200), "Test Scene", fill=(255, 255, 255))
        draw.rectangle([100, 100, 300, 300], outline=(255, 0, 0), width=3)
        
        logger.info("✅ Test image created")
        
        # Initialize captioner
        logger.info("\nInitializing SceneCaptioner...")
        logger.info("⚠️ Note: This will download the LLaVA model (~13GB) on first run")
        logger.info("   You can press Ctrl+C to skip this test if needed")
        
        try:
            captioner = SceneCaptioner()
            
            # Generate caption
            logger.info("\nGenerating caption...")
            caption = captioner.generate_caption(img)
            
            logger.info(f"\n✅ Caption generated:")
            logger.info(f"   {caption}")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Test skipped by user")
            return None
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline_sample():
    """Test the complete pipeline with sample video"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Full Pipeline with Sample Video")
    logger.info("="*80)
    
    try:
        # Check if sample video exists
        sample_video = "test_sample_video.mp4"
        
        if not Path(sample_video).exists():
            logger.warning(f"Sample video not found: {sample_video}")
            logger.info("Run test_extract_keyframes_sample() first")
            return None
        
        logger.info("⚠️ Note: This test requires the LLaVA model (~13GB)")
        logger.info("   You can press Ctrl+C to skip this test")
        
        try:
            # Run pipeline with limited frames
            logger.info("\nRunning full pipeline (max 3 frames)...")
            results = analyze_video_scene(
                video_path=sample_video,
                interval_sec=1.0,
                max_frames=3,
                save_output=True
            )
            
            logger.info(f"\n✅ Pipeline completed successfully")
            logger.info(f"   Generated {len(results)} captions")
            
            # Display results
            logger.info("\n📝 Results:")
            for result in results:
                logger.info(f"\n[{result['formatted_time']}]")
                logger.info(f"{result['caption']}")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Test skipped by user")
            return None
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_caption_generation():
    """Test batch caption generation"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Batch Caption Generation")
    logger.info("="*80)
    
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create multiple test images
        logger.info("Creating test images...")
        images = []
        
        for i in range(3):
            # Create image with different colors
            color = (
                (i * 80) % 256,
                (i * 120) % 256,
                (i * 160) % 256
            )
            img = Image.new('RGB', (320, 240), color=color)
            
            # Add text
            draw = ImageDraw.Draw(img)
            draw.text((50, 100), f"Image {i+1}", fill=(255, 255, 255))
            
            images.append(img)
        
        logger.info(f"✅ Created {len(images)} test images")
        
        # Initialize captioner
        logger.info("\nInitializing SceneCaptioner...")
        
        try:
            captioner = SceneCaptioner()
            
            # Generate batch captions
            logger.info("\nGenerating batch captions...")
            captions = captioner.generate_batch_captions(images, batch_size=2)
            
            logger.info(f"\n✅ Generated {len(captions)} captions:")
            for i, caption in enumerate(captions, 1):
                logger.info(f"\nImage {i}:")
                logger.info(f"  {caption}")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Test skipped by user")
            return None
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("🧪 RUNNING ALL TESTS FOR SCENE ANALYSIS PIPELINE")
    logger.info("="*80)
    
    tests = [
        ("Keyframe Extraction", test_extract_keyframes_sample),
        ("SceneCaptioner Basic", test_scene_captioner_basic),
        ("Full Pipeline", test_full_pipeline_sample),
        ("Batch Captions", test_batch_caption_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except KeyboardInterrupt:
            logger.warning(f"\n⚠️ Test interrupted: {test_name}")
            results[test_name] = None
            break
        except Exception as e:
            logger.error(f"\n❌ Test crashed: {test_name} - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️ SKIPPED"
        
        logger.info(f"{status} - {test_name}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    logger.info("\n" + "="*80)
    logger.info(f"Total: {len(results)} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Scene Analysis Pipeline")
    parser.add_argument(
        '--test',
        type=str,
        choices=['all', 'keyframes', 'captioner', 'pipeline', 'batch'],
        default='all',
        help='Which test to run'
    )
    
    args = parser.parse_args()
    
    try:
        if args.test == 'all':
            run_all_tests()
        elif args.test == 'keyframes':
            test_extract_keyframes_sample()
        elif args.test == 'captioner':
            test_scene_captioner_basic()
        elif args.test == 'pipeline':
            test_full_pipeline_sample()
        elif args.test == 'batch':
            test_batch_caption_generation()
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Tests interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
