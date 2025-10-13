# Contextual Analysis Module

This module provides advanced NLP capabilities for understanding conversational dynamics:

## Features

1. **Named Entity Recognition (NER)**

   - Extracts people, places, organizations, concepts from text
   - Uses spaCy's pre-trained models for accurate entity detection

2. **Commonsense Emotional Reasoning**

   - Integrates COMET for inferring emotional effects
   - Understands emotional impact of statements on participants

3. **Dynamic Knowledge Graph Service**
   - Structures entities and relationships
   - Provides foundation for semantic queries and reasoning

## Components

- `ner_service.py`: Named Entity Recognition using spaCy
- `comet_service.py`: COMET integration for emotional inference
- `knowledge_graph_service.py`: Dynamic Knowledge Graph builder
- `contextual_analyzer.py`: Main analyzer orchestrating all services

## Usage

```python
from contextual.contextual_analyzer import contextual_analyzer

# Analyze text
result = await contextual_analyzer.analyze(
    text="John met Sarah at the coffee shop in Seattle.",
    conversation_id="conv_123"
)

# Result contains:
# - entities: {people: [...], places: [...], organizations: [...]}
# - emotional_effects: {happiness: 0.8, ...}
# - graph_updates: [nodes, relationships]
```

## Models Used

- **spaCy**: `en_core_web_sm` or `en_core_web_lg` for NER
- **COMET**: `comet-atomic_2020_BART` for commonsense reasoning
