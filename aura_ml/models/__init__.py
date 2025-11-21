"""Models package"""

# Import models with optional dependencies
try:
    from aura_ml.models.ece_classifier import RoBERTaForECE, ECEModelOutput
    _has_ece = True
except ImportError:
    _has_ece = False

try:
    from aura_ml.models.llm_wrapper import AuraLLM
    _has_llm = True
except ImportError:
    _has_llm = False

try:
    from aura_ml.models.audio_processor import (
        AudioPipeline,
        WhisperSTT,
        SpeechEmotionRecognizer,
        AudioAnalysisResult
    )
    _has_audio = True
except ImportError:
    _has_audio = False

try:
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
    _has_video = True
except ImportError:
    _has_video = False

# Build __all__ dynamically based on available imports
__all__ = []

if _has_ece:
    __all__.extend(["RoBERTaForECE", "ECEModelOutput"])

if _has_llm:
    __all__.append("AuraLLM")

if _has_audio:
    __all__.extend([
        "AudioPipeline",
        "WhisperSTT", 
        "SpeechEmotionRecognizer",
        "AudioAnalysisResult"
    ])

if _has_video:
    __all__.extend([
        "FaceDetector",
        "KeyframeExtractor",
        "FaceLandmarks",
        "VideoFrame",
        "FacialEmotionCNN",
        "FacialEmotionRecognizer",
        "EmotionVideoProcessor"
    ])

