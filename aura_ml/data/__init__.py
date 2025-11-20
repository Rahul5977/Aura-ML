"""
Data Processing Module

This module contains data processing pipelines for Aura-ML:

ECE Dataset Generation:
- CausalKeywordExtractor: Pass 1 - Rule-based extraction using 50+ causal keywords
- HeuristicCauseExtractor: Pass 2 - Heuristic-based extraction for implicit causes
- BIOAnnotator: Token-level BIO annotation using RoBERTa tokenizer
- ESConvProcessor: ESConv dataset loader and preprocessor
- ECEDatasetGenerator: Main pipeline orchestrator for ECE training data

Hypercontextual Dataset Generation:
- HypercontextualDatasetGenerator: Enriches ESConv with multi-modal analysis
  to create instruction-completion pairs for LLM fine-tuning
"""

from aura_ml.data.causal_keyword_extractor import CausalKeywordExtractor
from aura_ml.data.heuristic_fallback import HeuristicCauseExtractor
from aura_ml.data.bio_annotator import BIOAnnotator
from aura_ml.data.esconv_processor import ESConvProcessor
from aura_ml.data.ece_dataset_generator import ECEDatasetGenerator
from aura_ml.data.hypercontextual_dataset_generator import HypercontextualDatasetGenerator

__all__ = [
    'CausalKeywordExtractor',
    'HeuristicCauseExtractor',
    'BIOAnnotator',
    'ESConvProcessor',
    'ECEDatasetGenerator',
    'HypercontextualDatasetGenerator',
]
