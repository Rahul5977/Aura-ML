"""
Contextual Analysis Module

Provides advanced NLP capabilities for understanding conversational dynamics:
- Named Entity Recognition (NER)
- Commonsense Emotional Reasoning (COMET)
- Dynamic Knowledge Graph Building
"""

from .ner_service import ner_service, NERService
from .comet_service import comet_service, COMETService
from .knowledge_graph_service import knowledge_graph_service, KnowledgeGraphService
from .contextual_analyzer import contextual_analyzer, ContextualAnalyzer, initialize_contextual_services

__all__ = [
    'ner_service',
    'NERService',
    'comet_service',
    'COMETService',
    'knowledge_graph_service',
    'KnowledgeGraphService',
    'contextual_analyzer',
    'ContextualAnalyzer',
    'initialize_contextual_services',
]
