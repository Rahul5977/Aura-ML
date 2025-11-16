"""Face Analysis Pipeline - Multi-model face detection and emotion recognition."""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_face_models(device: Optional[str] = None) -> Dict[str, Any]:
    """Load MTCNN, InceptionResnetV1, and emotion models. Returns models dict."""
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("="*80)
    logger.info("🔧 LOADING FACE ANALYSIS MODELS")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    
    models = {
        'device': device,
        'emotion_labels': []
    }
    
    # ========================================================================
    # 1. LOAD MTCNN - FACE DETECTION
    # ========================================================================
    logger.info("\n📸 Loading MTCNN (Face Detection)...")
    
    try:
        from facenet_pytorch import MTCNN
        
        # Initialize MTCNN with optimal settings
        mtcnn = MTCNN(
            image_size=160,          # Standard size for face recognition
            margin=20,               # Margin around detected face
            min_face_size=20,        # Minimum face size to detect
            thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
            factor=0.709,            # Scale factor for image pyramid
            post_process=True,       # Apply post-processing
            device=device,
            keep_all=True,           # Detect all faces in image
            select_largest=False     # Don't just select largest face
        )
        
        models['mtcnn'] = mtcnn
        logger.info("✅ MTCNN loaded successfully")
        logger.info(f"   - Min face size: 20px")
        logger.info(f"   - Multi-face detection: Enabled")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import facenet-pytorch: {e}")
        logger.error("   Please install: pip install facenet-pytorch")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load MTCNN: {e}")
        raise
    
    # ========================================================================
    # 2. LOAD INCEPTIONRESNETV1 - IDENTITY EMBEDDINGS
    # ========================================================================
    logger.info("\n🔍 Loading InceptionResnetV1 (Identity Embeddings)...")
    
    try:
        from facenet_pytorch import InceptionResnetV1
        
        # Load pre-trained model on VGGFace2 dataset
        identity_model = InceptionResnetV1(
            pretrained='vggface2',
            device=device
        ).eval()  # Set to evaluation mode
        
        # Ensure model is on correct device
        identity_model = identity_model.to(device)
        
        models['identity_model'] = identity_model
        logger.info("✅ InceptionResnetV1 loaded successfully")
        logger.info(f"   - Pre-trained on: VGGFace2")
        logger.info(f"   - Embedding size: 512 dimensions")
        logger.info(f"   - Mode: Evaluation (no gradients)")
        
    except Exception as e:
        logger.error(f"❌ Failed to load InceptionResnetV1: {e}")
        raise
    
    # ========================================================================
    # 3. LOAD FACIAL EMOTION RECOGNITION MODEL
    # ========================================================================
    logger.info("\n😊 Loading Facial Emotion Recognition Model...")
    
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        
        # Use a robust emotion recognition model
        # Alternative: "dima806/facial_emotions_image_detection"
        #             "trpakov/vit-face-expression"
        emotion_model_name = "trpakov/vit-face-expression"
        
        logger.info(f"   Model: {emotion_model_name}")
        
        # Load processor (handles image preprocessing)
        emotion_processor = AutoImageProcessor.from_pretrained(emotion_model_name)
        
        # Load model
        emotion_model = AutoModelForImageClassification.from_pretrained(
            emotion_model_name
        )
        
        # Move to device and set to eval mode
        emotion_model = emotion_model.to(device)
        emotion_model.eval()
        
        # Extract emotion labels from model config
        emotion_labels = list(emotion_model.config.id2label.values())
        
        models['emotion_model'] = emotion_model
        models['emotion_processor'] = emotion_processor
        models['emotion_labels'] = emotion_labels
        
        logger.info("✅ Emotion model loaded successfully")
        logger.info(f"   - Model: Vision Transformer (ViT)")
        logger.info(f"   - Emotions: {', '.join(emotion_labels)}")
        logger.info(f"   - Total classes: {len(emotion_labels)}")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import transformers: {e}")
        logger.error("   Please install: pip install transformers")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load emotion model: {e}")
        logger.warning("   Falling back to basic emotion categories")
        
        # Fallback: Create a simple emotion model placeholder
        models['emotion_model'] = None
        models['emotion_processor'] = None
        models['emotion_labels'] = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust']
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ ALL FACE ANALYSIS MODELS LOADED")
    logger.info("="*80)
    logger.info(f"Models ready on device: {device}")
    logger.info(f"Total models: {len([k for k in models.keys() if k not in ['device', 'emotion_labels']])}")
    logger.info("="*80 + "\n")
    
    return models


# ============================================================================
# CORE FACE ANALYSIS FUNCTION
# ============================================================================

def analyze_faces_in_frame(
    frame: Image.Image,
    mtcnn,
    identity_model,
    emotion_model,
    emotion_processor,
    emotion_labels: List[str],
    device: str = 'cpu',
    confidence_threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """Detect faces, extract embeddings, and classify emotions in a frame."""
    faces_data = []
    
    try:
        # ====================================================================
        # STEP 1: FACE DETECTION WITH MTCNN
        # ====================================================================
        
        # Detect faces and get bounding boxes + probabilities
        boxes, probs = mtcnn.detect(frame)
        
        # Check if any faces were detected
        if boxes is None or len(boxes) == 0:
            logger.debug("No faces detected in frame")
            return faces_data
        
        logger.debug(f"Detected {len(boxes)} face(s)")
        
        # ====================================================================
        # STEP 2 & 3: PROCESS EACH DETECTED FACE
        # ====================================================================
        
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            # Skip low-confidence detections
            if prob < confidence_threshold:
                logger.debug(f"Skipping face {i+1} (confidence {prob:.2f} < {confidence_threshold})")
                continue
            
            try:
                # Get bounding box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Ensure coordinates are within image bounds
                img_width, img_height = frame.size
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)
                
                # Skip if box is too small or invalid
                if x2 <= x1 or y2 <= y1:
                    logger.debug(f"Skipping invalid bounding box: {box}")
                    continue
                
                # ============================================================
                # CROP FACE FROM FRAME
                # ============================================================
                face_crop = frame.crop((x1, y1, x2, y2))
                
                # ============================================================
                # EXTRACT IDENTITY EMBEDDING (InceptionResnetV1)
                # ============================================================
                
                # Resize face to 160x160 (required by InceptionResnetV1)
                face_resized = face_crop.resize((160, 160), Image.BILINEAR)
                
                # Convert to tensor and normalize
                # InceptionResnetV1 expects images normalized to [-1, 1]
                face_tensor = torch.tensor(np.array(face_resized)).float()
                face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
                face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
                face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
                face_tensor = face_tensor.to(device)
                
                # Get identity embedding
                with torch.no_grad():
                    identity_embedding = identity_model(face_tensor)
                    identity_embedding = identity_embedding.cpu().numpy()[0]
                
                logger.debug(f"Face {i+1}: Extracted {len(identity_embedding)}-d embedding")
                
                # ============================================================
                # PREDICT FACIAL EMOTION (ViT)
                # ============================================================
                
                emotion = "neutral"
                emotion_confidence = 0.0
                emotion_scores = {}
                
                if emotion_model is not None and emotion_processor is not None:
                    try:
                        # Preprocess face for emotion model
                        inputs = emotion_processor(
                            images=face_crop,
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Get emotion prediction
                        with torch.no_grad():
                            outputs = emotion_model(**inputs)
                            logits = outputs.logits
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            
                            # Get predicted emotion
                            predicted_idx = torch.argmax(probs, dim=-1).item()
                            emotion = emotion_labels[predicted_idx]
                            emotion_confidence = probs[0][predicted_idx].item()
                            
                            # Get all emotion scores
                            emotion_scores = {
                                label: float(probs[0][idx])
                                for idx, label in enumerate(emotion_labels)
                            }
                        
                        logger.debug(f"Face {i+1}: Emotion = {emotion} ({emotion_confidence:.2f})")
                        
                    except Exception as e:
                        logger.warning(f"Failed to predict emotion for face {i+1}: {e}")
                        emotion = "unknown"
                        emotion_confidence = 0.0
                
                # ============================================================
                # STORE RESULTS
                # ============================================================
                
                face_data = {
                    'box': [x1, y1, x2, y2],
                    'confidence': float(prob),
                    'identity_embedding': identity_embedding.tolist(),
                    'embedding_norm': float(np.linalg.norm(identity_embedding)),
                    'emotion': emotion,
                    'emotion_confidence': emotion_confidence,
                    'emotion_scores': emotion_scores
                }
                
                faces_data.append(face_data)
                
            except Exception as e:
                logger.warning(f"Error processing face {i+1}: {e}")
                continue
        
        logger.debug(f"Successfully analyzed {len(faces_data)} face(s)")
        
    except Exception as e:
        logger.error(f"Error in face analysis: {e}")
    
    return faces_data


# ============================================================================
# BATCH PROCESSING FUNCTION
# ============================================================================

def analyze_faces_in_video_frames(
    frames: List[Image.Image],
    models: Dict[str, Any],
    show_progress: bool = True
) -> List[List[Dict[str, Any]]]:
    """Analyze faces in multiple frames. Returns per-frame face results."""
    all_results = []
    
    for i, frame in enumerate(frames):
        if show_progress and (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(frames)} frames...")
        
        faces = analyze_faces_in_frame(
            frame=frame,
            mtcnn=models['mtcnn'],
            identity_model=models['identity_model'],
            emotion_model=models['emotion_model'],
            emotion_processor=models['emotion_processor'],
            emotion_labels=models['emotion_labels'],
            device=models['device']
        )
        
        all_results.append(faces)
    
    if show_progress:
        total_faces = sum(len(faces) for faces in all_results)
        logger.info(f"✅ Analyzed {len(frames)} frames, found {total_faces} total faces")
    
    return all_results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_face_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings (0.0-1.0)."""
    # Normalize embeddings
    emb1_norm = embedding1 / np.linalg.norm(embedding1)
    emb2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Compute cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # Convert to 0-1 range
    similarity = (similarity + 1.0) / 2.0
    
    return float(similarity)


def track_identities_across_frames(
    all_faces: List[List[Dict[str, Any]]],
    similarity_threshold: float = 0.6
) -> List[List[Dict[str, Any]]]:
    """Assign consistent identity IDs to same person across frames."""
    known_identities = []  # List of representative embeddings
    results = []
    
    for frame_faces in all_faces:
        frame_results = []
        
        for face in frame_faces:
            embedding = np.array(face['identity_embedding'])
            
            # Find best matching known identity
            best_match_id = -1
            best_similarity = 0.0
            
            for identity_id, known_embedding in enumerate(known_identities):
                similarity = compute_face_similarity(embedding, known_embedding)
                
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match_id = identity_id
            
            # Assign identity ID
            if best_match_id >= 0:
                # Existing identity
                identity_id = best_match_id
            else:
                # New identity
                identity_id = len(known_identities)
                known_identities.append(embedding)
            
            # Add identity ID to face data
            face_with_id = face.copy()
            face_with_id['identity_id'] = identity_id
            face_with_id['identity_similarity'] = best_similarity if best_match_id >= 0 else 1.0
            
            frame_results.append(face_with_id)
        
        results.append(frame_results)
    
    logger.info(f"Identified {len(known_identities)} unique person(s) across all frames")
    
    return results


# ============================================================================
# MAIN TESTING BLOCK
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logger.info("="*80)
    logger.info("🎭 FACE ANALYSIS PIPELINE - DEMO")
    logger.info("="*80)
    
    # ========================================================================
    # LOAD MODELS
    # ========================================================================
    
    try:
        models = load_face_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)
    
    # ========================================================================
    # LOAD SAMPLE IMAGE
    # ========================================================================
    
    # Try to load a sample image
    sample_image_paths = [
        'sample_face.jpg',
        'test_image.jpg',
        'face.jpg',
        '../sample_face.jpg'
    ]
    
    sample_image = None
    sample_path = None
    
    for path in sample_image_paths:
        if Path(path).exists():
            sample_path = path
            break
    
    if sample_path:
        logger.info(f"\n📷 Loading sample image: {sample_path}")
        sample_image = Image.open(sample_path).convert('RGB')
        logger.info(f"   Image size: {sample_image.size}")
    else:
        # Create a synthetic test image
        logger.info("\n📷 No sample image found, creating synthetic test image...")
        logger.info("   (For best results, provide a real photo with faces)")
        
        # Create a simple colored image
        sample_image = Image.new('RGB', (640, 480), color=(200, 200, 200))
        
        # Note: This won't detect faces, but demonstrates the API
        logger.warning("   ⚠️ Synthetic image won't contain detectable faces")
        logger.warning("   ⚠️ Please provide a real image with faces for testing")
    
    # ========================================================================
    # ANALYZE FACES IN IMAGE
    # ========================================================================
    
    logger.info("\n🔍 Analyzing faces in image...")
    logger.info("-"*80)
    
    faces = analyze_faces_in_frame(
        frame=sample_image,
        mtcnn=models['mtcnn'],
        identity_model=models['identity_model'],
        emotion_model=models['emotion_model'],
        emotion_processor=models['emotion_processor'],
        emotion_labels=models['emotion_labels'],
        device=models['device'],
        confidence_threshold=0.9
    )
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("📊 ANALYSIS RESULTS")
    logger.info("="*80)
    
    if len(faces) == 0:
        logger.info("❌ No faces detected in the image")
        logger.info("\nTo test with real faces:")
        logger.info("  1. Place an image with faces in the current directory")
        logger.info("  2. Name it 'sample_face.jpg' or 'test_image.jpg'")
        logger.info("  3. Run this script again")
    else:
        logger.info(f"✅ Found {len(faces)} face(s)\n")
        
        for i, face in enumerate(faces, 1):
            logger.info(f"Face {i}:")
            logger.info(f"  📍 Bounding Box: {face['box']}")
            logger.info(f"  🎯 Detection Confidence: {face['confidence']:.3f}")
            logger.info(f"  🔢 Embedding Dimension: {len(face['identity_embedding'])}")
            logger.info(f"  📊 Embedding Norm: {face['embedding_norm']:.3f}")
            logger.info(f"  😊 Emotion: {face['emotion']}")
            logger.info(f"  💯 Emotion Confidence: {face['emotion_confidence']:.3f}")
            
            if face['emotion_scores']:
                logger.info(f"  📈 All Emotion Scores:")
                for emotion, score in sorted(face['emotion_scores'].items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"     - {emotion}: {score:.3f}")
            
            logger.info("")
    
    # ========================================================================
    # DEMO: FACE SIMILARITY (if multiple faces detected)
    # ========================================================================
    
    if len(faces) >= 2:
        logger.info("="*80)
        logger.info("🔗 FACE SIMILARITY COMPARISON")
        logger.info("="*80)
        
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                emb1 = np.array(faces[i]['identity_embedding'])
                emb2 = np.array(faces[j]['identity_embedding'])
                
                similarity = compute_face_similarity(emb1, emb2)
                
                logger.info(f"Face {i+1} vs Face {j+1}: {similarity:.3f} similarity")
                
                if similarity > 0.6:
                    logger.info(f"  ✅ Likely the same person!")
                else:
                    logger.info(f"  ❌ Likely different people")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("✅ DEMO COMPLETE")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Integrate with video frame extraction")
    logger.info("  2. Process multiple frames for identity tracking")
    logger.info("  3. Combine with Scene Analysis Pipeline")
    logger.info("="*80)
