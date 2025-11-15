"""
Test Suite for Integrated Video Analysis - Week 8
Tests the complete video analysis pipeline (Scene + Face)

Run with: pytest test_integrated_analysis.py -v
"""

import pytest
from pathlib import Path
import json
import numpy as np
from PIL import Image

from video.integrated_analysis import (
    analyze_video_complete,
    save_analysis_results,
    generate_text_summary,
    get_analysis_statistics
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_video_path():
    """Path to sample video file."""
    paths = [
        'test_data/sample.mp4',
        '../test_data/sample.mp4',
        'sample_video.mp4'
    ]
    
    for path in paths:
        if Path(path).exists():
            return path
    
    pytest.skip("No sample video available for testing")


@pytest.fixture
def sample_results():
    """Sample analysis results for testing export functions."""
    return [
        {
            'timestamp': 0.0,
            'frame_number': 0,
            'scene_caption': 'A person sitting at a desk',
            'num_faces': 1,
            'unique_people': 1,
            'emotions': ['neutral'],
            'faces': [
                {
                    'box': [100, 100, 200, 200],
                    'confidence': 0.99,
                    'identity_embedding': np.random.randn(512).tolist(),
                    'embedding_norm': 23.5,
                    'emotion': 'neutral',
                    'emotion_confidence': 0.85,
                    'emotion_scores': {'neutral': 0.85, 'happy': 0.10, 'sad': 0.05},
                    'identity_id': 0,
                    'identity_similarity': 1.0
                }
            ]
        },
        {
            'timestamp': 2.0,
            'frame_number': 60,
            'scene_caption': 'The person is smiling',
            'num_faces': 1,
            'unique_people': 1,
            'emotions': ['happy'],
            'faces': [
                {
                    'box': [105, 105, 205, 205],
                    'confidence': 0.98,
                    'identity_embedding': np.random.randn(512).tolist(),
                    'embedding_norm': 23.7,
                    'emotion': 'happy',
                    'emotion_confidence': 0.92,
                    'emotion_scores': {'happy': 0.92, 'neutral': 0.05, 'sad': 0.03},
                    'identity_id': 0,
                    'identity_similarity': 0.85
                }
            ]
        }
    ]


# ============================================================================
# TEST COMPLETE ANALYSIS
# ============================================================================

def test_analyze_video_complete_structure(sample_video_path):
    """Test that analyze_video_complete returns proper structure."""
    
    # Run analysis with minimal frames for speed
    results = analyze_video_complete(
        video_path=sample_video_path,
        keyframe_interval=5.0,  # Large interval for speed
        analyze_scenes=False,   # Disable slow scene analysis
        analyze_faces=True
    )
    
    # Verify results structure
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check first result structure
    result = results[0]
    assert 'timestamp' in result
    assert 'frame_number' in result
    assert 'scene_caption' in result
    assert 'faces' in result
    assert 'num_faces' in result
    assert 'emotions' in result
    assert 'unique_people' in result
    
    # Check types
    assert isinstance(result['timestamp'], float)
    assert isinstance(result['frame_number'], int)
    assert isinstance(result['faces'], list)
    assert isinstance(result['num_faces'], int)
    assert isinstance(result['emotions'], list)
    assert isinstance(result['unique_people'], int)


def test_analyze_video_scene_only(sample_video_path):
    """Test scene-only analysis."""
    
    results = analyze_video_complete(
        video_path=sample_video_path,
        keyframe_interval=10.0,
        analyze_scenes=True,
        analyze_faces=False
    )
    
    assert len(results) > 0
    
    # Scene captions should exist
    assert results[0]['scene_caption'] is not None
    
    # No face data should be present
    assert results[0]['num_faces'] == 0
    assert len(results[0]['faces']) == 0
    assert len(results[0]['emotions']) == 0


def test_analyze_video_face_only(sample_video_path):
    """Test face-only analysis."""
    
    results = analyze_video_complete(
        video_path=sample_video_path,
        keyframe_interval=5.0,
        analyze_scenes=False,
        analyze_faces=True
    )
    
    assert len(results) > 0
    
    # Scene captions should be None
    assert results[0]['scene_caption'] is None
    
    # Face data may or may not be present (depends on video content)
    assert isinstance(results[0]['faces'], list)


def test_analyze_video_invalid_path():
    """Test error handling for invalid video path."""
    
    with pytest.raises(FileNotFoundError):
        analyze_video_complete('nonexistent_video.mp4')


def test_analyze_video_invalid_settings():
    """Test error handling for invalid settings."""
    
    with pytest.raises(ValueError):
        analyze_video_complete(
            'dummy.mp4',
            analyze_scenes=False,
            analyze_faces=False
        )


# ============================================================================
# TEST EXPORT FUNCTIONS
# ============================================================================

def test_save_analysis_results(sample_results, tmp_path):
    """Test JSON export functionality."""
    
    output_path = tmp_path / "results.json"
    
    # Save without embeddings
    save_analysis_results(sample_results, str(output_path), include_embeddings=False)
    
    # Verify file exists
    assert output_path.exists()
    
    # Load and verify content
    with open(output_path) as f:
        data = json.load(f)
    
    assert len(data) == 2
    assert data[0]['timestamp'] == 0.0
    assert data[0]['num_faces'] == 1
    
    # Embeddings should not be included
    assert 'identity_embedding' not in data[0]['faces'][0]


def test_save_analysis_results_with_embeddings(sample_results, tmp_path):
    """Test JSON export with embeddings."""
    
    output_path = tmp_path / "results_with_embeddings.json"
    
    # Save with embeddings
    save_analysis_results(sample_results, str(output_path), include_embeddings=True)
    
    # Verify file exists
    assert output_path.exists()
    
    # Load and verify content
    with open(output_path) as f:
        data = json.load(f)
    
    # Embeddings should be included
    assert 'identity_embedding' in data[0]['faces'][0]
    assert len(data[0]['faces'][0]['identity_embedding']) == 512


def test_generate_text_summary(sample_results):
    """Test text summary generation."""
    
    summary = generate_text_summary(sample_results)
    
    # Verify summary structure
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Check for key sections
    assert 'VIDEO ANALYSIS SUMMARY' in summary
    assert 'OVERALL STATISTICS' in summary
    assert 'EMOTION DISTRIBUTION' in summary
    assert 'FRAME-BY-FRAME TIMELINE' in summary
    
    # Check for data presence
    assert '0.0s' in summary  # First timestamp
    assert '2.0s' in summary  # Second timestamp
    assert 'neutral' in summary
    assert 'happy' in summary


def test_get_analysis_statistics(sample_results):
    """Test statistics calculation."""
    
    stats = get_analysis_statistics(sample_results)
    
    # Verify stats structure
    assert isinstance(stats, dict)
    
    # Check required fields
    assert 'total_frames' in stats
    assert 'total_duration' in stats
    assert 'total_faces' in stats
    assert 'unique_people' in stats
    assert 'emotion_distribution' in stats
    assert 'avg_faces_per_frame' in stats
    assert 'frames_with_faces' in stats
    assert 'face_presence_ratio' in stats
    
    # Verify values
    assert stats['total_frames'] == 2
    assert stats['total_duration'] == 2.0
    assert stats['total_faces'] == 2
    assert stats['unique_people'] == 1
    assert stats['avg_faces_per_frame'] == 1.0
    assert stats['frames_with_faces'] == 2
    assert stats['face_presence_ratio'] == 1.0
    
    # Check emotion distribution
    assert 'neutral' in stats['emotion_distribution']
    assert 'happy' in stats['emotion_distribution']
    assert stats['emotion_distribution']['neutral'] == 1
    assert stats['emotion_distribution']['happy'] == 1


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_empty_results():
    """Test handling of empty results."""
    
    empty_results = []
    
    # Should not crash
    stats = get_analysis_statistics(empty_results)
    assert stats['total_frames'] == 0
    
    summary = generate_text_summary(empty_results)
    assert isinstance(summary, str)


def test_no_faces_results():
    """Test results with no faces detected."""
    
    no_faces_results = [
        {
            'timestamp': 0.0,
            'frame_number': 0,
            'scene_caption': 'Empty room',
            'num_faces': 0,
            'unique_people': 0,
            'emotions': [],
            'faces': []
        }
    ]
    
    stats = get_analysis_statistics(no_faces_results)
    assert stats['total_faces'] == 0
    assert stats['unique_people'] == 0
    assert stats['avg_faces_per_frame'] == 0.0
    assert stats['face_presence_ratio'] == 0.0
    
    summary = generate_text_summary(no_faces_results)
    assert 'Empty room' in summary


def test_multiple_people():
    """Test results with multiple people."""
    
    multi_person_results = [
        {
            'timestamp': 0.0,
            'frame_number': 0,
            'scene_caption': 'Group meeting',
            'num_faces': 3,
            'unique_people': 3,
            'emotions': ['happy', 'neutral', 'happy'],
            'faces': [
                {
                    'box': [100, 100, 200, 200],
                    'confidence': 0.99,
                    'identity_embedding': np.random.randn(512).tolist(),
                    'emotion': 'happy',
                    'emotion_confidence': 0.9,
                    'identity_id': 0
                },
                {
                    'box': [300, 100, 400, 200],
                    'confidence': 0.98,
                    'identity_embedding': np.random.randn(512).tolist(),
                    'emotion': 'neutral',
                    'emotion_confidence': 0.85,
                    'identity_id': 1
                },
                {
                    'box': [500, 100, 600, 200],
                    'confidence': 0.97,
                    'identity_embedding': np.random.randn(512).tolist(),
                    'emotion': 'happy',
                    'emotion_confidence': 0.88,
                    'identity_id': 2
                }
            ]
        }
    ]
    
    stats = get_analysis_statistics(multi_person_results)
    assert stats['total_faces'] == 3
    assert stats['unique_people'] == 3
    assert stats['emotion_distribution']['happy'] == 2
    assert stats['emotion_distribution']['neutral'] == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_pipeline(sample_video_path, tmp_path):
    """Test complete pipeline from video to exported results."""
    
    # Run analysis
    results = analyze_video_complete(
        video_path=sample_video_path,
        keyframe_interval=10.0,  # Large interval for speed
        analyze_scenes=False,     # Disable for speed
        analyze_faces=True
    )
    
    # Export JSON
    json_path = tmp_path / "results.json"
    save_analysis_results(results, str(json_path))
    assert json_path.exists()
    
    # Generate summary
    summary = generate_text_summary(results)
    summary_path = tmp_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    assert summary_path.exists()
    
    # Calculate stats
    stats = get_analysis_statistics(results)
    assert stats['total_frames'] > 0
    
    print("\n" + "="*80)
    print("FULL PIPELINE TEST RESULTS:")
    print("="*80)
    print(f"Frames analyzed: {stats['total_frames']}")
    print(f"Total faces: {stats['total_faces']}")
    print(f"JSON saved: {json_path}")
    print(f"Summary saved: {summary_path}")
    print("="*80)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.slow
def test_performance_benchmark(sample_video_path):
    """Benchmark analysis performance."""
    import time
    
    start_time = time.time()
    
    results = analyze_video_complete(
        video_path=sample_video_path,
        keyframe_interval=2.0,
        analyze_scenes=False,  # Disable for face-only benchmark
        analyze_faces=True
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    num_frames = len(results)
    fps = num_frames / elapsed if elapsed > 0 else 0
    
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK:")
    print("="*80)
    print(f"Frames processed: {num_frames}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Processing speed: {fps:.2f} FPS")
    print("="*80)
    
    # Should process at least 1 FPS (very conservative)
    assert fps >= 1.0


if __name__ == "__main__":
    """
    Run tests manually without pytest.
    """
    print("\n" + "="*80)
    print("INTEGRATED VIDEO ANALYSIS - TEST SUITE")
    print("="*80)
    print("\nTo run all tests:")
    print("  pytest test_integrated_analysis.py -v")
    print("\nTo run slow tests:")
    print("  pytest test_integrated_analysis.py -v -m slow")
    print("\nTo run without slow tests:")
    print("  pytest test_integrated_analysis.py -v -m 'not slow'")
    print("="*80)
