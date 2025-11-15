"""
Scene Analysis Pipeline - Week 8
Multi-Modal Vision-Language Model for Video Caption Generation

This module provides a complete pipeline for extracting keyframes from videos
and generating descriptive captions using the LLaVA (Large Language and Vision Assistant) model.

Components:
1. Video Processing: Extract keyframes from video files at specified intervals
2. LLaVA Integration: Generate descriptive captions for each frame
3. Scene Analysis Orchestration: Complete pipeline from video to captions

Author: Aura ML Team
Date: November 2025
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
from datetime import timedelta
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
# VIDEO PROCESSING FUNCTIONS
# ============================================================================

def extract_keyframes(
    video_path: str, 
    interval_sec: float = 1.0,
    max_frames: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract keyframes from a video file at specified time intervals.
    
    This function opens a video file using OpenCV, iterates through frames,
    and extracts one frame every `interval_sec` seconds. Each extracted frame
    is converted to a PIL Image for compatibility with vision-language models.
    
    Args:
        video_path (str): Path to the input video file
        interval_sec (float): Time interval in seconds between keyframes (default: 1.0)
        max_frames (Optional[int]): Maximum number of frames to extract (default: None)
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing:
            - 'frame': PIL.Image object of the keyframe
            - 'timestamp': Float timestamp in seconds
            - 'frame_number': Integer frame index in the video
            - 'formatted_time': Human-readable timestamp (HH:MM:SS)
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or has no frames
        
    Example:
        >>> keyframes = extract_keyframes('video.mp4', interval_sec=2.0)
        >>> print(f"Extracted {len(keyframes)} keyframes")
        >>> print(f"First frame at {keyframes[0]['formatted_time']}")
    """
    # Validate video path
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"📹 Opening video file: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"📊 Video Properties:")
    logger.info(f"   - Resolution: {width}x{height}")
    logger.info(f"   - FPS: {fps:.2f}")
    logger.info(f"   - Total Frames: {total_frames}")
    logger.info(f"   - Duration: {duration:.2f} seconds")
    
    # Calculate frame interval
    frame_interval = int(fps * interval_sec)
    if frame_interval < 1:
        frame_interval = 1
        logger.warning(f"Frame interval too small, using 1 frame")
    
    expected_frames = int(duration / interval_sec)
    logger.info(f"🎯 Extracting keyframes every {interval_sec}s ({frame_interval} frames)")
    logger.info(f"   Expected keyframes: ~{expected_frames}")
    
    # Extract keyframes
    keyframes = []
    frame_count = 0
    extracted_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Calculate timestamp
                timestamp = frame_count / fps if fps > 0 else 0
                
                # Format timestamp as HH:MM:SS
                time_delta = timedelta(seconds=timestamp)
                formatted_time = str(time_delta).split('.')[0]  # Remove microseconds
                
                # Store keyframe data
                keyframe_data = {
                    'frame': pil_image,
                    'timestamp': timestamp,
                    'frame_number': frame_count,
                    'formatted_time': formatted_time
                }
                
                keyframes.append(keyframe_data)
                extracted_count += 1
                
                # Log progress every 10 frames
                if extracted_count % 10 == 0:
                    logger.info(f"   Extracted {extracted_count} keyframes...")
                
                # Check max frames limit
                if max_frames and extracted_count >= max_frames:
                    logger.info(f"✅ Reached maximum frame limit: {max_frames}")
                    break
            
            frame_count += 1
    
    finally:
        cap.release()
    
    logger.info(f"✅ Successfully extracted {len(keyframes)} keyframes")
    
    if len(keyframes) == 0:
        raise ValueError("No frames could be extracted from the video")
    
    return keyframes


# ============================================================================
# LLaVA INTEGRATION CLASS
# ============================================================================

class SceneCaptioner:
    """
    LLaVA-based Scene Captioner for generating descriptive captions from images.
    
    This class integrates the LLaVA (Large Language and Vision Assistant) model
    from Hugging Face to generate natural language descriptions of video frames.
    
    LLaVA is a multimodal model that combines a vision encoder (CLIP) with a
    large language model to understand images and generate detailed descriptions.
    
    Attributes:
        model_name (str): Hugging Face model identifier
        device (str): Device for inference ('cuda' or 'cpu')
        model: The loaded LLaVA model
        processor: The loaded LLaVA processor
        is_loaded (bool): Whether the model is successfully loaded
    
    Example:
        >>> captioner = SceneCaptioner()
        >>> image = Image.open('frame.jpg')
        >>> caption = captioner.generate_caption(image)
        >>> print(caption)
    """
    
    def __init__(
        self, 
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize the Scene Captioner with LLaVA model.
        
        Args:
            model_name (str): Hugging Face model name (default: llava-1.5-7b-hf)
            device (Optional[str]): Device to use ('cuda' or 'cpu'). Auto-detected if None
            load_in_8bit (bool): Whether to load model in 8-bit for memory efficiency
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        logger.info(f"🚀 Initializing SceneCaptioner")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   8-bit Loading: {load_in_8bit}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """
        Load the LLaVA model and processor from Hugging Face.
        
        This method handles the actual model loading with proper error handling
        and memory optimization options.
        """
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            logger.info(f"📥 Loading LLaVA model from Hugging Face...")
            logger.info(f"   This may take a few minutes on first run...")
            
            # Load processor (handles image preprocessing and tokenization)
            logger.info("   Loading processor...")
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate settings
            logger.info("   Loading model...")
            
            if self.load_in_8bit and self.device == 'cuda':
                # 8-bit quantization for memory efficiency (requires bitsandbytes)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                logger.info("   ✅ Model loaded in 8-bit mode")
            else:
                # Standard loading
                torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
                
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                )
                
                # Move to device
                self.model = self.model.to(self.device)
                logger.info(f"   ✅ Model loaded in {torch_dtype} precision")
            
            # Set to evaluation mode (no gradient computation)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"✅ SceneCaptioner ready for inference on {self.device}")
            
        except ImportError as e:
            logger.error(f"❌ Missing required library: {e}")
            logger.error("   Please install: pip install transformers accelerate")
            raise
        
        except Exception as e:
            logger.error(f"❌ Failed to load LLaVA model: {e}")
            logger.error("   Please check your internet connection and model name")
            raise
    
    def generate_caption(
        self, 
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate a descriptive caption for a single image.
        
        This method uses the LLaVA model to analyze an image and generate
        a natural language description of what's happening in the scene.
        
        Args:
            image (Image.Image): PIL Image to caption
            prompt (Optional[str]): Custom prompt for caption generation
            max_new_tokens (int): Maximum length of generated caption
            temperature (float): Sampling temperature (0.0 = deterministic, 1.0 = creative)
            do_sample (bool): Whether to use sampling (vs greedy decoding)
        
        Returns:
            str: Generated caption describing the image
        
        Raises:
            RuntimeError: If model is not loaded
            
        Example:
            >>> image = Image.open('scene.jpg')
            >>> caption = captioner.generate_caption(image)
            >>> print(caption)
            "A person is sitting at a desk working on a laptop computer..."
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please initialize SceneCaptioner first.")
        
        # Default prompt if none provided
        if prompt is None:
            prompt = "USER: <image>\nWhat is happening in this scene? Provide a detailed description.\nASSISTANT:"
        
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=0.9,
                    num_beams=1  # Beam search disabled for speed
                )
            
            # Decode generated tokens
            generated_text = self.processor.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            if "ASSISTANT:" in generated_text:
                caption = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                caption = generated_text.strip()
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"[Error: Could not generate caption - {str(e)}]"
    
    def generate_batch_captions(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images efficiently.
        
        Args:
            images (List[Image.Image]): List of PIL Images
            batch_size (int): Number of images to process at once
            **kwargs: Additional arguments passed to generate_caption
        
        Returns:
            List[str]: List of generated captions
        """
        captions = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for image in batch:
                caption = self.generate_caption(image, **kwargs)
                captions.append(caption)
            
            logger.info(f"   Generated captions for {min(i + batch_size, len(images))}/{len(images)} images")
        
        return captions
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================

def analyze_video_scene(
    video_path: str,
    interval_sec: float = 1.0,
    max_frames: Optional[int] = None,
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    save_output: bool = True,
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Complete pipeline: Extract keyframes from video and generate captions.
    
    This is the main orchestration function that combines video processing
    and caption generation into a single pipeline. It extracts keyframes
    from a video file and generates descriptive captions for each frame.
    
    Args:
        video_path (str): Path to input video file
        interval_sec (float): Time interval between keyframes in seconds
        max_frames (Optional[int]): Maximum number of frames to process
        model_name (str): Hugging Face model name for LLaVA
        save_output (bool): Whether to save results to JSON file
        output_path (Optional[str]): Path for output JSON file
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing:
            - 'timestamp': Float timestamp in seconds
            - 'formatted_time': Human-readable timestamp
            - 'frame_number': Frame index
            - 'caption': Generated caption text
    
    Example:
        >>> results = analyze_video_scene('interview.mp4', interval_sec=2.0)
        >>> for result in results:
        ...     print(f"[{result['formatted_time']}] {result['caption']}")
        [00:00:00] A man is sitting at a desk...
        [00:00:02] The man is typing on a laptop...
    """
    logger.info("="*80)
    logger.info("🎬 STARTING SCENE ANALYSIS PIPELINE")
    logger.info("="*80)
    
    # Step 1: Extract keyframes
    logger.info("\n📹 STEP 1: EXTRACTING KEYFRAMES")
    logger.info("-"*80)
    
    try:
        keyframes = extract_keyframes(
            video_path=video_path,
            interval_sec=interval_sec,
            max_frames=max_frames
        )
    except Exception as e:
        logger.error(f"❌ Failed to extract keyframes: {e}")
        raise
    
    # Step 2: Initialize captioner
    logger.info("\n🤖 STEP 2: INITIALIZING LLaVA MODEL")
    logger.info("-"*80)
    
    try:
        captioner = SceneCaptioner(model_name=model_name)
    except Exception as e:
        logger.error(f"❌ Failed to initialize captioner: {e}")
        raise
    
    # Step 3: Generate captions
    logger.info("\n💬 STEP 3: GENERATING CAPTIONS")
    logger.info("-"*80)
    
    results = []
    
    for i, keyframe_data in enumerate(keyframes, 1):
        logger.info(f"\n📸 Processing frame {i}/{len(keyframes)}")
        logger.info(f"   Timestamp: {keyframe_data['formatted_time']}")
        
        # Generate caption
        caption = captioner.generate_caption(keyframe_data['frame'])
        
        # Store result
        result = {
            'timestamp': keyframe_data['timestamp'],
            'formatted_time': keyframe_data['formatted_time'],
            'frame_number': keyframe_data['frame_number'],
            'caption': caption
        }
        
        results.append(result)
        
        # Print caption
        logger.info(f"   Caption: {caption}")
    
    # Step 4: Save results (optional)
    if save_output:
        logger.info("\n💾 STEP 4: SAVING RESULTS")
        logger.info("-"*80)
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_captions.json"
        
        try:
            import json
            
            # Prepare output data
            output_data = {
                'video_path': str(video_path),
                'interval_sec': interval_sec,
                'total_frames': len(results),
                'model_name': model_name,
                'results': results
            }
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Results saved to: {output_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save results: {e}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✅ SCENE ANALYSIS PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"📊 Summary:")
    logger.info(f"   - Total frames analyzed: {len(results)}")
    logger.info(f"   - Time interval: {interval_sec}s")
    logger.info(f"   - Video duration: {keyframes[-1]['timestamp']:.2f}s")
    logger.info("="*80)
    
    return results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scene Analysis Pipeline - Generate captions from video using LLaVA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze video with default settings (1 frame per second)
  python scene_captioner.py video.mp4
  
  # Extract frames every 2 seconds
  python scene_captioner.py video.mp4 --interval 2.0
  
  # Limit to first 10 frames
  python scene_captioner.py video.mp4 --max-frames 10
  
  # Use custom model
  python scene_captioner.py video.mp4 --model llava-hf/llava-1.5-13b-hf
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Time interval between keyframes in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (default: unlimited)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help='Hugging Face model name (default: llava-1.5-7b-hf)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: <video_name>_captions.json)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    try:
        results = analyze_video_scene(
            video_path=args.video_path,
            interval_sec=args.interval,
            max_frames=args.max_frames,
            model_name=args.model,
            save_output=not args.no_save,
            output_path=args.output
        )
        
        print("\n" + "="*80)
        print("📝 GENERATED CAPTIONS:")
        print("="*80)
        
        for result in results:
            print(f"\n[{result['formatted_time']}]")
            print(f"{result['caption']}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        raise
