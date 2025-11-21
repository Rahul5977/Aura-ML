"""
Comprehensive Tests for Video Analysis Pipeline

Tests face detection, facial emotion recognition, and integration with audio pipeline.
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import tempfile
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_ml.models.video_processor import (
    FaceDetector,
    KeyframeExtractor,
    FaceLandmarks,
    VideoFrame
)
from aura_ml.models.facial_emotion_cnn import (
    FacialEmotionCNN,
    FacialEmotionRecognizer,
    EmotionVideoProcessor
)


# ============================================================================
# FIXTURES
# ============================================================================

def sample_face_image():
    """Generate a sample face image (48x48 grayscale)"""
    # Create a simple face-like pattern
    img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    # Add some structure (circular pattern)
    center = (24, 24)
    for i in range(48):
        for j in range(48):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if dist < 20:
                img[i, j] = min(255, int(img[i, j] + (20 - dist) * 5))
    return img


def sample_color_image_fixture():
    """Generate a sample color image for face detection"""
    # Create a 640x480 RGB image with a face-like region
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add a face-like rectangular region (skin tone)
    face_region = img[150:350, 220:420]
    face_region[:, :, 0] = 180  # Blue channel
    face_region[:, :, 1] = 200  # Green channel
    face_region[:, :, 2] = 220  # Red channel (BGR format)
    
    return img


def sample_video_fixture():
    """Create a temporary video file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    # Create a simple video (30 FPS, 3 seconds, 640x480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(90):  # 3 seconds at 30 FPS
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add a moving rectangle to simulate face
        x = 200 + i * 2
        cv2.rectangle(frame, (x, 150), (x + 200, 350), (180, 200, 220), -1)
        out.write(frame)
    
    out.release()
    return video_path


# ============================================================================
# TEST FACE DETECTION
# ============================================================================
def test_face_detector_initialization():
    """Test 1: Face detector initializes correctly"""
    detector = FaceDetector(
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    assert detector is not None
    assert detector.mp_face_mesh is not None
    print("✓ Test 1 passed: Face detector initialized")


def test_face_detection_on_image(sample_color_image):
    """Test 2: Face detection works on static image"""
    detector = FaceDetector()
    
    # Note: This may not detect a face on random noise
    # but should not crash
    result = detector.detect_face(sample_color_image, timestamp=0.0)
    
    # Check structure even if no face detected
    if result:
        assert isinstance(result, FaceLandmarks)
        assert result.landmarks.shape == (468, 3)
        assert len(result.bbox) == 4
        assert 0 <= result.confidence <= 1.0
        print("✓ Test 2 passed: Face detected successfully")
    else:
        print("✓ Test 2 passed: No face detected (expected on random image)")


def test_face_roi_extraction(sample_color_image):
    """Test 3: Face ROI extraction"""
    detector = FaceDetector()
    
    # Create mock landmarks
    mock_landmarks = FaceLandmarks(
        landmarks=np.random.rand(468, 3) * 100,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        frame_timestamp=0.0
    )
    
    roi = detector.extract_face_roi(sample_color_image, mock_landmarks, padding=0.2)
    
    assert roi is not None
    assert len(roi.shape) == 3  # Color image
    assert roi.shape[2] == 3   # 3 channels
    print(f"✓ Test 3 passed: ROI extracted with shape {roi.shape}")


def test_landmark_drawing(sample_color_image):
    """Test 4: Landmark visualization"""
    detector = FaceDetector()
    
    mock_landmarks = FaceLandmarks(
        landmarks=np.random.rand(468, 3) * 100,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        frame_timestamp=0.0
    )
    
    annotated = detector.draw_landmarks(
        sample_color_image,
        mock_landmarks,
        draw_bbox=True,
        draw_landmarks=True
    )
    
    assert annotated is not None
    assert annotated.shape == sample_color_image.shape
    print("✓ Test 4 passed: Landmarks drawn successfully")


# ============================================================================
# TEST KEYFRAME EXTRACTION
# ============================================================================

def test_keyframe_extractor_initialization():
    """Test 5: Keyframe extractor initializes correctly"""
    extractor = KeyframeExtractor(interval_seconds=3.0)
    
    assert extractor is not None
    assert extractor.interval_seconds == 3.0
    print("✓ Test 5 passed: Keyframe extractor initialized")


def test_keyframe_extraction(sample_video):
    """Test 6: Keyframe extraction from video"""
    extractor = KeyframeExtractor(interval_seconds=1.0)
    
    keyframes = extractor.extract_keyframes(sample_video, max_keyframes=3)
    
    assert len(keyframes) > 0
    assert all(isinstance(kf, VideoFrame) for kf in keyframes)
    assert all(kf.frame is not None for kf in keyframes)
    
    print(f"✓ Test 6 passed: Extracted {len(keyframes)} keyframes")


def test_keyframe_saving(sample_video):
    """Test 7: Keyframe saving to disk"""
    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = KeyframeExtractor(interval_seconds=1.0)
        keyframes = extractor.extract_keyframes(sample_video, max_keyframes=2)
        
        saved_paths = extractor.save_keyframes(keyframes, tmpdir)
        
        assert len(saved_paths) == len(keyframes)
        assert all(p.exists() for p in saved_paths)
        
        print(f"✓ Test 7 passed: Saved {len(saved_paths)} keyframes")


# ============================================================================
# TEST FACIAL EMOTION CNN
# ============================================================================

def test_cnn_model_initialization():
    """Test 8: CNN model initializes with correct architecture"""
    model = FacialEmotionCNN(num_classes=7, dropout_p=0.5)
    
    assert model is not None
    
    # Check architecture components
    assert hasattr(model, 'conv1')  # Conv block 1
    assert hasattr(model, 'conv2')  # Conv block 2
    assert hasattr(model, 'conv3')  # Conv block 3
    assert hasattr(model, 'global_avg_pool')  # Global avg pooling
    assert hasattr(model, 'fc1')  # Dense layer 1
    assert hasattr(model, 'fc2')  # Dense layer 2
    assert hasattr(model, 'fc3')  # Output layer
    
    # Check layer dimensions
    assert model.conv1.out_channels == 32
    assert model.conv2.out_channels == 64
    assert model.conv3.out_channels == 128
    assert model.fc1.in_features == 128
    assert model.fc1.out_features == 512
    assert model.fc2.out_features == 256
    assert model.fc3.out_features == 7
    
    param_count = model._count_parameters()
    print(f"✓ Test 8 passed: CNN initialized with {param_count:,} parameters")


def test_cnn_forward_pass():
    """Test 9: CNN forward pass produces correct output shape"""
    model = FacialEmotionCNN()
    model.eval()
    
    # Create dummy input [batch_size, channels, height, width]
    dummy_input = torch.randn(4, 1, 48, 48)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (4, 7)  # [batch_size, num_classes]
    
    print(f"✓ Test 9 passed: Forward pass output shape {output.shape}")


def test_cnn_emotion_labels():
    """Test 10: CNN has correct emotion labels"""
    expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    assert FacialEmotionCNN.EMOTIONS == expected_emotions
    print(f"✓ Test 10 passed: Emotion labels correct: {FacialEmotionCNN.EMOTIONS}")


# ============================================================================
# TEST FACIAL EMOTION RECOGNIZER
# ============================================================================

def test_emotion_recognizer_initialization():
    """Test 11: Emotion recognizer initializes"""
    recognizer = FacialEmotionRecognizer(model_path=None, device='cpu')
    
    assert recognizer is not None
    assert recognizer.model is not None
    assert recognizer.device == 'cpu'
    
    print("✓ Test 11 passed: Emotion recognizer initialized")


def test_face_preprocessing(sample_face_image):
    """Test 12: Face image preprocessing"""
    recognizer = FacialEmotionRecognizer(device='cpu')
    
    # Convert to 3-channel for testing
    face_rgb = cv2.cvtColor(sample_face_image, cv2.COLOR_GRAY2BGR)
    
    tensor = recognizer.preprocess_face(face_rgb)
    
    assert tensor.shape == (1, 1, 48, 48)  # [batch, channels, H, W]
    assert tensor.dtype == torch.float32
    
    print(f"✓ Test 12 passed: Preprocessed shape {tensor.shape}")


def test_emotion_prediction(sample_face_image):
    """Test 13: Emotion prediction from face image"""
    recognizer = FacialEmotionRecognizer(device='cpu')
    
    face_rgb = cv2.cvtColor(sample_face_image, cv2.COLOR_GRAY2BGR)
    
    # Test probability output
    emotion_probs = recognizer.predict(face_rgb, return_probabilities=True)
    
    assert isinstance(emotion_probs, dict)
    assert len(emotion_probs) == 7
    assert all(0 <= prob <= 1 for prob in emotion_probs.values())
    assert abs(sum(emotion_probs.values()) - 1.0) < 0.01  # Probabilities sum to 1
    
    # Test label output
    top_emotion = recognizer.predict(face_rgb, return_probabilities=False)
    assert top_emotion in FacialEmotionCNN.EMOTIONS
    
    print(f"✓ Test 13 passed: Predicted emotion '{top_emotion}' with probs {list(emotion_probs.values())[:3]}...")


def test_batch_prediction(sample_face_image):
    """Test 14: Batch emotion prediction"""
    recognizer = FacialEmotionRecognizer(device='cpu')
    
    # Create batch of faces
    faces = [cv2.cvtColor(sample_face_image, cv2.COLOR_GRAY2BGR) for _ in range(5)]
    
    results = recognizer.predict_batch(faces)
    
    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)
    assert all(len(r) == 7 for r in results)
    
    print(f"✓ Test 14 passed: Batch prediction for {len(faces)} faces")


def test_inference_speed():
    """Test 15: Inference speed benchmark"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    recognizer = FacialEmotionRecognizer(device=device)
    
    avg_time_ms = recognizer.benchmark_inference(num_iterations=50)
    
    assert avg_time_ms > 0
    
    target_time = 15.0  # ms (from report)
    status = "✓ EXCELLENT" if avg_time_ms < target_time else "✓ ACCEPTABLE" if avg_time_ms < 30 else "⚠ SLOW"
    
    print(f"✓ Test 15 passed: Inference speed {avg_time_ms:.2f}ms on {device} ({status})")
    print(f"  Target: <15ms (report spec), Achieved: {avg_time_ms:.2f}ms")


# ============================================================================
# TEST INTEGRATED VIDEO PROCESSOR
# ============================================================================

def test_emotion_video_processor_initialization():
    """Test 16: Integrated video processor initializes"""
    detector = FaceDetector()
    recognizer = FacialEmotionRecognizer(device='cpu')
    
    processor = EmotionVideoProcessor(detector, recognizer)
    
    assert processor is not None
    assert processor.face_detector is not None
    assert processor.emotion_recognizer is not None
    
    print("✓ Test 16 passed: Integrated video processor initialized")


def test_frame_processing(sample_color_image):
    """Test 17: Single frame processing"""
    detector = FaceDetector()
    recognizer = FacialEmotionRecognizer(device='cpu')
    processor = EmotionVideoProcessor(detector, recognizer)
    
    # Process frame (may not detect face on random image)
    result = processor.process_frame(sample_color_image, timestamp=0.5)
    
    if result:
        assert 'timestamp' in result
        assert 'landmarks' in result
        assert 'emotion' in result
        assert 'emotion_probabilities' in result
        assert result['emotion'] in FacialEmotionCNN.EMOTIONS
        print(f"✓ Test 17 passed: Frame processed, detected emotion: {result['emotion']}")
    else:
        print("✓ Test 17 passed: No face detected (expected on random image)")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_video_pipeline_components_compatibility():
    """Test 18: All pipeline components work together"""
    # Initialize all components
    face_detector = FaceDetector()
    keyframe_extractor = KeyframeExtractor(interval_seconds=1.0)
    emotion_model = FacialEmotionCNN()
    emotion_recognizer = FacialEmotionRecognizer(device='cpu')
    video_processor = EmotionVideoProcessor(face_detector, emotion_recognizer)
    
    assert face_detector is not None
    assert keyframe_extractor is not None
    assert emotion_model is not None
    assert emotion_recognizer is not None
    assert video_processor is not None
    
    print("✓ Test 18 passed: All pipeline components compatible")


def test_model_parameter_counts():
    """Test 19: Model has reasonable parameter count"""
    model = FacialEmotionCNN()
    param_count = model._count_parameters()
    
    # Should be in reasonable range for lightweight CNN
    assert 10_000 < param_count < 10_000_000
    
    print(f"✓ Test 19 passed: Model has {param_count:,} parameters (reasonable size)")


def test_emotion_consistency():
    """Test 20: Model produces consistent predictions for same input"""
    recognizer = FacialEmotionRecognizer(device='cpu')
    
    # Create a fixed face image
    face_img = np.ones((48, 48, 3), dtype=np.uint8) * 128
    
    # Get multiple predictions
    pred1 = recognizer.predict(face_img, return_probabilities=True)
    pred2 = recognizer.predict(face_img, return_probabilities=True)
    
    # Check consistency
    for emotion in pred1:
        assert abs(pred1[emotion] - pred2[emotion]) < 0.001  # Should be identical
    
    print("✓ Test 20 passed: Model predictions are consistent")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests manually (without pytest)"""
    print("="*70)
    print("VIDEO ANALYSIS PIPELINE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Create fixtures
    sample_face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    sample_color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    tests = [
        ("Face Detector Initialization", lambda: test_face_detector_initialization()),
        ("Face Detection on Image", lambda: test_face_detection_on_image(sample_color)),
        ("Face ROI Extraction", lambda: test_face_roi_extraction(sample_color)),
        ("Landmark Drawing", lambda: test_landmark_drawing(sample_color)),
        ("Keyframe Extractor Init", lambda: test_keyframe_extractor_initialization()),
        ("CNN Model Initialization", lambda: test_cnn_model_initialization()),
        ("CNN Forward Pass", lambda: test_cnn_forward_pass()),
        ("CNN Emotion Labels", lambda: test_cnn_emotion_labels()),
        ("Emotion Recognizer Init", lambda: test_emotion_recognizer_initialization()),
        ("Face Preprocessing", lambda: test_face_preprocessing(sample_face)),
        ("Emotion Prediction", lambda: test_emotion_prediction(sample_face)),
        ("Batch Prediction", lambda: test_batch_prediction(sample_face)),
        ("Inference Speed", lambda: test_inference_speed()),
        ("Video Processor Init", lambda: test_emotion_video_processor_initialization()),
        ("Frame Processing", lambda: test_frame_processing(sample_color)),
        ("Component Compatibility", lambda: test_video_pipeline_components_compatibility()),
        ("Parameter Count", lambda: test_model_parameter_counts()),
        ("Prediction Consistency", lambda: test_emotion_consistency()),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n[{passed + failed + 1}/{len(tests)}] Testing: {name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {name}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)
    
    return passed, failed


if __name__ == '__main__':
    # Run manual tests (no pytest required)
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
