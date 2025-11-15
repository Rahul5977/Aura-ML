"""
Integrated Video Analysis - Week 8 Complete
Combines Scene Analysis + Face Analysis for comprehensive video understanding

This module provides a unified interface for multi-modal video analysis:
1. Scene Understanding (LLaVA-based captioning)
2. Face Detection & Tracking (MTCNN + InceptionResnetV1)
3. Emotion Recognition (Vision Transformer)

Author: Aura ML Team
Date: November 2025
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import json
from collections import Counter

from .scene_captioner import extract_keyframes, SceneCaptioner
from .face_analysis import (
    load_face_models,
    analyze_faces_in_video_frames,
    track_identities_across_frames
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_video_complete(
    video_path: str,
    keyframe_interval: float = 2.0,
    scene_model: str = 'llava-hf/llava-1.5-7b-hf',
    face_confidence: float = 0.9,
    identity_threshold: float = 0.6,
    analyze_scenes: bool = True,
    analyze_faces: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform complete multi-modal video analysis.
    
    This function extracts keyframes and analyzes both:
    1. Scene content (what's happening in the video)
    2. Face attributes (who's there, their emotions)
    
    Args:
        video_path (str): Path to video file
        keyframe_interval (float): Seconds between extracted keyframes
        scene_model (str): LLaVA model ID for scene captioning
        face_confidence (float): Minimum confidence for face detection (0-1)
        identity_threshold (float): Minimum similarity for identity tracking (0-1)
        analyze_scenes (bool): Whether to perform scene analysis
        analyze_faces (bool): Whether to perform face analysis
    
    Returns:
        List[Dict[str, Any]]: Analysis results for each keyframe:
            - timestamp: Time in video (seconds)
            - frame_number: Frame index in video
            - scene_caption: AI-generated description of scene (if enabled)
            - faces: List of detected faces with emotions and embeddings
            - num_faces: Count of detected faces in frame
            - emotions: List of emotions present in frame
            - unique_people: Count of unique identities in frame
    
    Example:
        >>> # Full analysis
        >>> results = analyze_video_complete('meeting.mp4')
        >>> print(f"Analyzed {len(results)} frames")
        >>> 
        >>> # Scene only
        >>> results = analyze_video_complete('video.mp4', analyze_faces=False)
        >>> 
        >>> # Faces only
        >>> results = analyze_video_complete('video.mp4', analyze_scenes=False)
    """
    
    logger.info("="*80)
    logger.info("🎬 COMPLETE VIDEO ANALYSIS")
    logger.info("="*80)
    logger.info(f"📁 Video: {video_path}")
    logger.info(f"⏱️  Keyframe interval: {keyframe_interval}s")
    logger.info(f"🎨 Scene analysis: {'ENABLED' if analyze_scenes else 'DISABLED'}")
    logger.info(f"🎭 Face analysis: {'ENABLED' if analyze_faces else 'DISABLED'}")
    logger.info("="*80)
    logger.info("")
    
    # Validate inputs
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not analyze_scenes and not analyze_faces:
        raise ValueError("At least one analysis type must be enabled")
    
    # ========================================================================
    # STEP 1: EXTRACT KEYFRAMES FROM VIDEO
    # ========================================================================
    
    logger.info("📹 STEP 1: Extracting keyframes from video...")
    logger.info("-"*80)
    
    try:
        keyframes = extract_keyframes(video_path, interval_sec=keyframe_interval)
        
        logger.info(f"✅ Extracted {len(keyframes)} keyframes")
        logger.info(f"   Duration: {keyframes[-1]['timestamp']:.2f}s")
        logger.info(f"   FPS: {keyframes[0].get('fps', 'unknown')}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Failed to extract keyframes: {e}")
        raise
    
    # Extract frame data
    frames = [kf['frame'] for kf in keyframes]
    timestamps = [kf['timestamp'] for kf in keyframes]
    frame_numbers = [kf['frame_number'] for kf in keyframes]
    
    # ========================================================================
    # STEP 2: SCENE ANALYSIS (OPTIONAL)
    # ========================================================================
    
    scene_captions = []
    
    if analyze_scenes:
        logger.info("🎨 STEP 2: Analyzing scene content...")
        logger.info("-"*80)
        
        try:
            # Initialize scene captioner
            scene_captioner = SceneCaptioner(model_name=scene_model)
            logger.info(f"   Model: {scene_model}")
            logger.info("")
            
            # Caption each frame
            for i, (frame, ts) in enumerate(zip(frames, timestamps), 1):
                logger.info(f"   Frame {i}/{len(frames)} ({ts:.1f}s)")
                
                caption = scene_captioner.caption_frame(frame)
                scene_captions.append(caption)
                
                logger.info(f"   → {caption}")
            
            logger.info(f"\n✅ Scene analysis complete")
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Scene analysis failed: {e}")
            logger.warning("   Continuing with face analysis...")
            scene_captions = [None] * len(frames)
    else:
        logger.info("⏭️  STEP 2: Scene analysis skipped")
        logger.info("")
        scene_captions = [None] * len(frames)
    
    # ========================================================================
    # STEP 3: FACE ANALYSIS (OPTIONAL)
    # ========================================================================
    
    all_faces = []
    
    if analyze_faces:
        logger.info("🎭 STEP 3: Analyzing faces and emotions...")
        logger.info("-"*80)
        
        try:
            # Load face analysis models
            logger.info("   Loading models...")
            face_models = load_face_models()
            logger.info("")
            
            # Analyze faces in all frames
            logger.info("   Processing frames...")
            all_faces = analyze_faces_in_video_frames(
                frames, 
                face_models, 
                show_progress=True
            )
            
            # Track identities across frames
            logger.info("\n   Tracking identities...")
            all_faces = track_identities_across_frames(
                all_faces, 
                similarity_threshold=identity_threshold
            )
            
            # Calculate statistics
            total_detections = sum(len(faces) for faces in all_faces)
            frames_with_faces = sum(1 for faces in all_faces if len(faces) > 0)
            unique_people = len(set(
                face['identity_id'] 
                for frame_faces in all_faces 
                for face in frame_faces
            ))
            
            logger.info(f"\n✅ Face analysis complete")
            logger.info(f"   Total detections: {total_detections}")
            logger.info(f"   Frames with faces: {frames_with_faces}/{len(frames)}")
            logger.info(f"   Unique people: {unique_people}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Face analysis failed: {e}")
            import traceback
            traceback.print_exc()
            all_faces = [[]] * len(frames)
    else:
        logger.info("⏭️  STEP 3: Face analysis skipped")
        logger.info("")
        all_faces = [[]] * len(frames)
    
    # ========================================================================
    # STEP 4: COMBINE AND STRUCTURE RESULTS
    # ========================================================================
    
    logger.info("🔗 STEP 4: Combining results...")
    logger.info("-"*80)
    
    combined_results = []
    
    for i in range(len(keyframes)):
        # Extract frame metadata
        timestamp = timestamps[i]
        frame_number = frame_numbers[i]
        
        # Extract face data
        faces_in_frame = all_faces[i]
        
        # Aggregate emotions
        emotions = [face['emotion'] for face in faces_in_frame]
        
        # Count unique people in this frame
        unique_people_in_frame = len(set(
            face['identity_id'] 
            for face in faces_in_frame
        ))
        
        # Build result object
        result = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'scene_caption': scene_captions[i],
            'faces': faces_in_frame,
            'num_faces': len(faces_in_frame),
            'emotions': emotions,
            'unique_people': unique_people_in_frame
        }
        
        combined_results.append(result)
    
    logger.info(f"✅ Combined {len(combined_results)} frame results")
    logger.info("")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    logger.info("="*80)
    logger.info("✅ VIDEO ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"📊 Summary:")
    logger.info(f"   Frames analyzed: {len(combined_results)}")
    logger.info(f"   Total duration: {timestamps[-1]:.2f}s")
    
    if analyze_faces:
        total_faces = sum(r['num_faces'] for r in combined_results)
        all_emotions = [e for r in combined_results for e in r['emotions']]
        
        logger.info(f"   Total faces detected: {total_faces}")
        
        if all_emotions:
            emotion_dist = Counter(all_emotions)
            top_emotion = emotion_dist.most_common(1)[0]
            logger.info(f"   Most common emotion: {top_emotion[0]} ({top_emotion[1]} occurrences)")
    
    logger.info("="*80)
    logger.info("")
    
    return combined_results


def save_analysis_results(
    results: List[Dict[str, Any]],
    output_path: str,
    include_embeddings: bool = False
) -> None:
    """
    Save analysis results to JSON file.
    
    Args:
        results (List[Dict[str, Any]]): Results from analyze_video_complete()
        output_path (str): Path to save JSON file
        include_embeddings (bool): Whether to include identity embeddings
                                   (Warning: embeddings are large ~2KB per face)
    
    Example:
        >>> results = analyze_video_complete('video.mp4')
        >>> save_analysis_results(results, 'output.json')
        >>> 
        >>> # With embeddings for downstream identity tasks
        >>> save_analysis_results(results, 'output.json', include_embeddings=True)
    """
    
    logger.info(f"💾 Saving results to: {output_path}")
    
    # Prepare data for JSON serialization
    json_data = []
    
    for result in results:
        result_copy = {
            'timestamp': result['timestamp'],
            'frame_number': result['frame_number'],
            'scene_caption': result['scene_caption'],
            'num_faces': result['num_faces'],
            'emotions': result['emotions'],
            'unique_people': result['unique_people'],
            'faces': []
        }
        
        # Add face data (optionally exclude embeddings to reduce file size)
        for face in result['faces']:
            face_copy = {
                'box': face['box'],
                'confidence': face['confidence'],
                'emotion': face['emotion'],
                'emotion_confidence': face['emotion_confidence'],
                'emotion_scores': face.get('emotion_scores', {}),
                'identity_id': face.get('identity_id', -1),
                'identity_similarity': face.get('identity_similarity', 0.0)
            }
            
            # Optionally include embeddings (makes file much larger)
            if include_embeddings:
                face_copy['identity_embedding'] = face['identity_embedding']
                face_copy['embedding_norm'] = face.get('embedding_norm', 0.0)
            
            result_copy['faces'].append(face_copy)
        
        json_data.append(result_copy)
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    file_size = Path(output_path).stat().st_size / 1024  # KB
    logger.info(f"✅ Saved {len(json_data)} frame results ({file_size:.1f} KB)")


def generate_text_summary(results: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable text summary of video analysis.
    
    Args:
        results (List[Dict[str, Any]]): Results from analyze_video_complete()
    
    Returns:
        str: Formatted text summary
    
    Example:
        >>> results = analyze_video_complete('video.mp4')
        >>> summary = generate_text_summary(results)
        >>> print(summary)
        >>> 
        >>> # Save to file
        >>> with open('summary.txt', 'w') as f:
        ...     f.write(summary)
    """
    
    lines = []
    lines.append("="*80)
    lines.append("VIDEO ANALYSIS SUMMARY")
    lines.append("="*80)
    lines.append("")
    
    # ========================================================================
    # OVERALL STATISTICS
    # ========================================================================
    
    total_frames = len(results)
    total_faces = sum(r['num_faces'] for r in results)
    total_duration = results[-1]['timestamp'] if results else 0
    
    # Collect all unique people
    all_people = set()
    all_emotions = []
    
    for r in results:
        for face in r['faces']:
            all_people.add(face.get('identity_id', -1))
            all_emotions.append(face['emotion'])
    
    # Filter out invalid IDs
    all_people = {p for p in all_people if p >= 0}
    
    lines.append("📊 OVERALL STATISTICS")
    lines.append("-"*80)
    lines.append(f"Video Duration:        {total_duration:.2f}s")
    lines.append(f"Frames Analyzed:       {total_frames}")
    lines.append(f"Total Face Detections: {total_faces}")
    lines.append(f"Unique People:         {len(all_people)}")
    lines.append("")
    
    # ========================================================================
    # EMOTION DISTRIBUTION
    # ========================================================================
    
    if all_emotions:
        lines.append("😊 EMOTION DISTRIBUTION")
        lines.append("-"*80)
        
        emotion_counts = Counter(all_emotions)
        
        for emotion, count in emotion_counts.most_common():
            pct = (count / len(all_emotions)) * 100
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            
            lines.append(f"{emotion:12} | {bar} {count:4} ({pct:5.1f}%)")
        
        lines.append("")
    
    # ========================================================================
    # PERSON TIMELINE
    # ========================================================================
    
    if len(all_people) > 0:
        lines.append("👥 PERSON APPEARANCES")
        lines.append("-"*80)
        
        # Count appearances per person
        person_frames = {pid: [] for pid in all_people}
        
        for i, r in enumerate(results):
            for face in r['faces']:
                pid = face.get('identity_id', -1)
                if pid >= 0:
                    person_frames[pid].append(i)
        
        for pid in sorted(all_people):
            frames = person_frames[pid]
            appearances = len(frames)
            first_ts = results[frames[0]]['timestamp'] if frames else 0
            last_ts = results[frames[-1]]['timestamp'] if frames else 0
            
            lines.append(f"Person {pid}: {appearances} frames ({first_ts:.1f}s - {last_ts:.1f}s)")
        
        lines.append("")
    
    # ========================================================================
    # FRAME-BY-FRAME TIMELINE
    # ========================================================================
    
    lines.append("🎬 FRAME-BY-FRAME TIMELINE")
    lines.append("="*80)
    lines.append("")
    
    for r in results:
        ts = r['timestamp']
        frame_num = r['frame_number']
        caption = r['scene_caption'] or "No caption available"
        num_faces = r['num_faces']
        emotions_str = ', '.join(r['emotions']) if r['emotions'] else "None"
        
        # Truncate caption if too long
        if len(caption) > 60:
            caption = caption[:57] + "..."
        
        lines.append(f"⏱️  {ts:7.1f}s | Frame {frame_num:5d}")
        lines.append(f"   Scene:    {caption}")
        lines.append(f"   People:   {num_faces} | Emotions: {emotions_str}")
        lines.append("")
    
    lines.append("="*80)
    lines.append("END OF SUMMARY")
    lines.append("="*80)
    
    return '\n'.join(lines)


def get_analysis_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from analysis results.
    
    Args:
        results (List[Dict[str, Any]]): Results from analyze_video_complete()
    
    Returns:
        Dict[str, Any]: Statistics including:
            - total_frames: Number of frames analyzed
            - total_duration: Video duration in seconds
            - total_faces: Total face detections
            - unique_people: Number of unique identities
            - emotion_distribution: Dict of emotion counts
            - avg_faces_per_frame: Average faces per frame
            - frames_with_faces: Number of frames containing faces
    
    Example:
        >>> results = analyze_video_complete('video.mp4')
        >>> stats = get_analysis_statistics(results)
        >>> print(f"Average faces per frame: {stats['avg_faces_per_frame']:.2f}")
    """
    
    total_frames = len(results)
    total_duration = results[-1]['timestamp'] if results else 0
    total_faces = sum(r['num_faces'] for r in results)
    frames_with_faces = sum(1 for r in results if r['num_faces'] > 0)
    
    # Collect all people and emotions
    all_people = set()
    all_emotions = []
    
    for r in results:
        for face in r['faces']:
            pid = face.get('identity_id', -1)
            if pid >= 0:
                all_people.add(pid)
            all_emotions.append(face['emotion'])
    
    # Calculate averages
    avg_faces_per_frame = total_faces / total_frames if total_frames > 0 else 0
    
    # Emotion distribution
    emotion_counts = dict(Counter(all_emotions))
    
    return {
        'total_frames': total_frames,
        'total_duration': total_duration,
        'total_faces': total_faces,
        'unique_people': len(all_people),
        'emotion_distribution': emotion_counts,
        'avg_faces_per_frame': avg_faces_per_frame,
        'frames_with_faces': frames_with_faces,
        'face_presence_ratio': frames_with_faces / total_frames if total_frames > 0 else 0
    }


# ============================================================================
# MAIN DEMO/TEST BLOCK
# ============================================================================

if __name__ == "__main__":
    """
    Demo script showing complete video analysis workflow.
    """
    
    import sys
    
    print("\n" + "="*80)
    print("🎬 INTEGRATED VIDEO ANALYSIS - DEMO")
    print("="*80)
    print("")
    
    # Check for video path argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("ℹ️  No video path provided")
        print("\nUsage:")
        print("  python integrated_analysis.py <video_path>")
        print("\nExample:")
        print("  python integrated_analysis.py sample_video.mp4")
        print("")
        
        # Try to find a sample video
        sample_paths = [
            'sample_video.mp4',
            '../test/sample_video.mp4',
            'test_video.mp4'
        ]
        
        video_path = None
        for path in sample_paths:
            if Path(path).exists():
                video_path = path
                print(f"✅ Found sample video: {path}")
                break
        
        if not video_path:
            print("❌ No video file found")
            print("\n💡 To test this module:")
            print("   1. Provide a video file path as argument")
            print("   2. Or place a video named 'sample_video.mp4' in current directory")
            sys.exit(1)
    
    # Run complete analysis
    try:
        results = analyze_video_complete(
            video_path=video_path,
            keyframe_interval=2.0,
            analyze_scenes=True,
            analyze_faces=True
        )
        
        # Generate and print summary
        summary = generate_text_summary(results)
        print(summary)
        
        # Save results
        output_json = 'video_analysis_results.json'
        save_analysis_results(results, output_json)
        
        # Save summary
        output_txt = 'video_analysis_summary.txt'
        with open(output_txt, 'w') as f:
            f.write(summary)
        
        print(f"\n💾 Results saved:")
        print(f"   JSON: {output_json}")
        print(f"   Text: {output_txt}")
        
        # Print statistics
        stats = get_analysis_statistics(results)
        print(f"\n📊 Quick Stats:")
        print(f"   Duration: {stats['total_duration']:.1f}s")
        print(f"   Avg faces/frame: {stats['avg_faces_per_frame']:.1f}")
        print(f"   Face presence: {stats['face_presence_ratio']:.1%}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Demo complete!")
    print("="*80)
