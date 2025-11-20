#!/usr/bin/env python3
"""
Audio Analysis CLI
Test audio pipeline with local audio files
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from aura_ml.models.audio_processor import AudioPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )


def print_result(result, show_prosodic: bool = True):
    """Pretty print analysis result"""
    print("\n" + "="*60)
    print("📝 AUDIO ANALYSIS RESULTS")
    print("="*60)
    
    # Transcription
    print(f"\n🗣️  Transcription:")
    print(f"    {result.transcription}")
    
    # Emotion
    print(f"\n😊 Emotion: {result.emotion.upper()}")
    print(f"   Confidence: {result.emotion_confidence:.1%}")
    
    # All emotion scores
    print(f"\n📊 Emotion Scores:")
    sorted_emotions = sorted(
        result.emotion_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    for emotion, score in sorted_emotions:
        bar = "█" * int(score * 50)
        print(f"   {emotion:12s}: {score:.1%} {bar}")
    
    # Duration
    print(f"\n⏱️  Duration: {result.duration:.2f} seconds")
    
    # Prosodic features
    if show_prosodic and result.prosodic_features:
        print(f"\n🎵 Prosodic Features:")
        features = result.prosodic_features
        print(f"   Pitch (mean):    {features['pitch_mean_hz']:.1f} Hz")
        print(f"   Pitch (std):     {features['pitch_std_hz']:.1f} Hz")
        print(f"   Energy (mean):   {features['energy_mean']:.4f}")
        print(f"   Energy (std):    {features['energy_std']:.4f}")
        print(f"   Zero-cross rate: {features['zero_crossing_rate']:.4f}")
        print(f"   Spectral center: {features['spectral_centroid_hz']:.1f} Hz")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Analyze audio files for transcription and emotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single audio file
  python cli/audio.py path/to/audio.wav
  
  # Analyze without prosodic features
  python cli/audio.py audio.wav --no-prosodic
  
  # Use different models
  python cli/audio.py audio.wav --whisper-model openai/whisper-small
  
  # Verbose output
  python cli/audio.py audio.wav -v

Supported formats: WAV, MP3, FLAC, OGG, M4A
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file to analyze"
    )
    
    # Optional arguments
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-base",
        help="Whisper model to use (default: openai/whisper-base)"
    )
    
    parser.add_argument(
        "--ser-model",
        type=str,
        default="superb/wav2vec2-base-superb-er",
        help="Speech Emotion Recognition model (default: superb/wav2vec2-base-superb-er)"
    )
    
    parser.add_argument(
        "--no-prosodic",
        action="store_true",
        help="Don't extract prosodic features"
    )
    
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Return word-level timestamps (experimental)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate audio file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Check file extension
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm']
    if audio_path.suffix.lower() not in valid_extensions:
        logger.error(
            f"Unsupported audio format: {audio_path.suffix}\n"
            f"Supported formats: {', '.join(valid_extensions)}"
        )
        sys.exit(1)
    
    try:
        # Initialize pipeline
        logger.info("Initializing audio pipeline...")
        pipeline = AudioPipeline(
            whisper_model=args.whisper_model,
            ser_model=args.ser_model
        )
        
        # Process audio
        logger.info(f"Processing audio file: {audio_path}")
        result = pipeline.process_file(
            str(audio_path),
            return_timestamps=args.timestamps,
            return_prosodic=not args.no_prosodic
        )
        
        # Print results
        print_result(result, show_prosodic=not args.no_prosodic)
        
        logger.info("✅ Analysis complete!")
        
    except Exception as e:
        logger.error(f"❌ Error processing audio: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
