# Week 5: Contextual Analysis - Implementation Guide

## 🎯 Overview

Week 5 introduces advanced contextual analysis capabilities to the Aura backend, transforming raw conversational text into structured, insightful information. The system now understands not just _what_ was said, but also _who_, _what_, _where_, and the underlying _emotional dynamics_.

## 📋 Features Implemented

### 1. Named Entity Recognition (NER) 🏷️

**Technology:** spaCy (`en_core_web_sm`)

**Capabilities:**

- Identifies and categorizes entities in conversational text
- Extracts people, places, organizations, concepts, and dates
- Provides position information (start/end character indices)
- Supports batch processing for multiple texts

**Entity Categories:**

- **People (PERSON):** Names of individuals
- **Places (GPE/LOC/FAC):** Geographic locations, facilities
- **Organizations (ORG):** Companies, institutions
- **Concepts (PRODUCT/EVENT/WORK_OF_ART):** Products, events, topics
- **Dates (DATE/TIME):** Temporal references

**Example:**

```json
{
  "entities": {
    "people": [{ "text": "John", "start": 0, "end": 4, "label": "PERSON" }],
    "places": [{ "text": "Seattle", "start": 25, "end": 32, "label": "GPE" }],
    "organizations": [
      { "text": "Microsoft", "start": 50, "end": 59, "label": "ORG" }
    ]
  }
}
```

### 2. Commonsense Emotional Reasoning (COMET) 💭

**Technology:** AllenAI COMET (`comet-atomic_2020_BART`)

**Capabilities:**

- Infers emotional effects of statements on participants
- Understands wants, needs, and reactions
- Provides commonsense reasoning about social dynamics
- Detects emotional states: happiness, sadness, anger, fear, etc.

**Inference Types:**

- **xReact:** How does the subject feel?
- **oReact:** How do others feel?
- **xWant:** What does the subject want?
- **oWant:** What do others want?
- **xEffect:** What effects occur to the subject?
- **oEffect:** What effects occur to others?

**Example:**

```json
{
  "emotional_context": {
    "subject_emotions": ["happy", "excited", "hopeful"],
    "other_emotions": ["interested", "engaged"],
    "subject_wants": ["to collaborate", "to make progress"],
    "other_wants": ["to hear more", "to participate"]
  },
  "emotions_detected": ["happy", "excited"]
}
```

### 3. Dynamic Knowledge Graph 🕸️

**Technology:** Custom graph service (in-memory, Neo4j-ready)

**Capabilities:**

- Structures entities and relationships in graph format
- Tracks entity occurrences across conversations
- Models emotional relationships (FEELS, WANTS)
- Supports semantic queries and traversal
- Exports graph data in JSON format

**Node Types:**

- **PERSON:** People mentioned
- **PLACE:** Locations
- **ORGANIZATION:** Organizations
- **CONCEPT:** Topics and ideas
- **EMOTION:** Emotional states
- **DESIRE:** Wants and needs
- **CONVERSATION:** Conversation contexts

**Relationship Types:**

- **FEELS:** Person → Emotion
- **WANTS:** Person → Desire
- **MENTIONED_IN:** Entity → Conversation
- **RELATED_TO:** Entity → Entity

**Example Graph Structure:**

```
(John:PERSON)-[:FEELS]->(happy:EMOTION)
(John:PERSON)-[:WANTS]->(collaborate:DESIRE)
(John:PERSON)-[:MENTIONED_IN]->(Conv1:CONVERSATION)
(Seattle:PLACE)-[:MENTIONED_IN]->(Conv1:CONVERSATION)
```

## 🛠️ Architecture

### Module Structure

```
contextual/
├── __init__.py                  # Module exports
├── README.md                    # Module documentation
├── ner_service.py              # Named Entity Recognition
├── comet_service.py            # Emotional reasoning
├── knowledge_graph_service.py  # Graph building
└── contextual_analyzer.py      # Main orchestrator
```

### Service Integration

```python
┌──────────────────────────────────────────┐
│      Contextual Analyzer                 │
│      (Main Orchestrator)                 │
└─────────┬──────────┬──────────┬──────────┘
          │          │          │
    ┌─────▼────┐ ┌──▼─────┐ ┌──▼──────────┐
    │   NER    │ │ COMET  │ │   KG        │
    │ Service  │ │Service │ │  Service    │
    └─────┬────┘ └──┬─────┘ └──┬──────────┘
          │          │          │
    ┌─────▼────┐ ┌──▼─────┐ ┌──▼──────────┐
    │  spaCy   │ │Transformers│ In-Memory │
    │ en_core  │ │  BART  │ │   Graph   │
    └──────────┘ └────────┘ └────────────┘
```

## 🚀 API Endpoints

### 1. Analyze Text

**Endpoint:** `POST /analyze/text`

**Description:** Perform comprehensive contextual analysis on text

**Parameters:**

- `text` (string): Text to analyze
- `conversation_id` (string): Conversation identifier
- `speaker_id` (string, optional): Speaker identifier
- `include_graph` (bool, default: true): Update knowledge graph

**Authentication:** Required (Bearer token)

**Response:**

```json
{
  "entities": {
    "people": [...],
    "places": [...],
    "organizations": [...],
    "concepts": [...]
  },
  "emotional_context": {
    "subject_emotions": [...],
    "other_emotions": [...],
    "subject_wants": [...],
    "other_wants": [...]
  },
  "emotions_detected": ["happy", "excited"],
  "graph_updates": {
    "entity_nodes_count": 5,
    "emotional_relationships_count": 3
  },
  "metadata": {
    "conversation_id": "conv_123",
    "processing_time_ms": 450,
    "timestamp": "2025-10-13T10:30:00"
  }
}
```

### 2. Get Conversation Context

**Endpoint:** `GET /analyze/conversation/{conversation_id}`

**Description:** Retrieve accumulated context from knowledge graph

**Authentication:** Required (Bearer token)

**Response:**

```json
{
  "conversation_id": "conv_123",
  "related_entities": [
    {
      "id": "PERSON_john",
      "type": "PERSON",
      "label": "John",
      "properties": {...}
    }
  ],
  "relationships": [
    {
      "id": "rel_1",
      "type": "FEELS",
      "from": "PERSON_john",
      "to": "EMOTION_happy"
    }
  ],
  "summary": {
    "total_nodes": 15,
    "total_relationships": 8
  }
}
```

### 3. Knowledge Graph Summary

**Endpoint:** `GET /knowledge-graph/summary`

**Description:** Get statistics of the knowledge graph

**Authentication:** Required (Bearer token)

**Response:**

```json
{
  "total_nodes": 45,
  "total_relationships": 62,
  "nodes_by_type": {
    "PERSON": 12,
    "PLACE": 8,
    "ORGANIZATION": 5,
    "EMOTION": 15,
    "CONCEPT": 5
  },
  "relationships_by_type": {
    "FEELS": 25,
    "WANTS": 18,
    "MENTIONED_IN": 19
  }
}
```

### 4. Export Knowledge Graph

**Endpoint:** `GET /knowledge-graph/export`

**Description:** Export graph data in JSON format

**Parameters:**

- `format` (string, default: "json"): Export format

**Authentication:** Required (Bearer token)

**Response:**

```json
{
  "format": "json",
  "data": "{\"nodes\": [...], \"relationships\": [...]}"
}
```

## 💻 Usage Examples

### Python Client Example

```python
import requests

# Authenticate
response = requests.post("http://localhost:8000/auth/login", json={
    "username": "user@example.com",
    "password": "password"
})
token = response.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# Analyze text
text = "Sarah met John at the coffee shop to discuss their startup idea."
response = requests.post(
    "http://localhost:8000/analyze/text",
    params={
        "text": text,
        "conversation_id": "conv_001",
        "include_graph": True
    },
    headers=headers
)

analysis = response.json()

# Display entities
for entity in analysis["entities"]["people"]:
    print(f"Person: {entity['text']}")

# Display emotions
for emotion in analysis["emotions_detected"]:
    print(f"Emotion: {emotion}")

# Get conversation context
response = requests.get(
    "http://localhost:8000/analyze/conversation/conv_001",
    headers=headers
)

context = response.json()
print(f"Related entities: {len(context['related_entities'])}")
```

### JavaScript/TypeScript Example

```typescript
// Authenticate
const loginResponse = await fetch("http://localhost:8000/auth/login", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    username: "user@example.com",
    password: "password",
  }),
});

const { access_token } = await loginResponse.json();

// Analyze text
const text = "Sarah met John at the coffee shop to discuss their startup idea.";
const analyzeResponse = await fetch(
  `http://localhost:8000/analyze/text?text=${encodeURIComponent(
    text
  )}&conversation_id=conv_001`,
  {
    method: "POST",
    headers: { Authorization: `Bearer ${access_token}` },
  }
);

const analysis = await analyzeResponse.json();

// Display results
console.log("Entities:", analysis.entities);
console.log("Emotions:", analysis.emotions_detected);
console.log("Processing time:", analysis.metadata.processing_time_ms, "ms");
```

## 🧪 Testing

### Running Week 5 Tests

```bash
# Ensure backend is running
docker-compose up -d

# Run Week 5 test suite
python test_week5.py
```

### Expected Test Output

```
============================================================
  WEEK 5: CONTEXTUAL ANALYSIS TEST SUITE
  Testing NER, COMET, and Knowledge Graph
============================================================

✅ Health check passed!
✅ Authentication successful!

--- Test 1/4 ---
Text: John met Sarah at Starbucks in Seattle...

📍 Entities Found:
  People: ['John', 'Sarah']
  Places: ['Starbucks', 'Seattle']
  Concepts: ['AI project']

😊 Emotions Detected: ['happy', 'excited']

💭 Emotional Context:
  Subject feels: ['interested', 'engaged']
  Subject wants: ['to collaborate', 'to discuss']

⏱️  Processing time: 450ms
✅ Analysis successful!

... (more tests)

🎉 All Week 5 features are operational!
```

## 📊 Performance Metrics

### Processing Times (Approximate)

| Operation          | Time (ms)   | Notes                |
| ------------------ | ----------- | -------------------- |
| NER Extraction     | 50-100      | spaCy is very fast   |
| COMET Inference    | 300-500     | Transformer-based    |
| Graph Update       | 10-20       | In-memory operations |
| **Total Analysis** | **400-650** | Parallel execution   |

### Resource Usage

- **Memory:** ~2GB (includes COMET and spaCy models)
- **CPU:** Multi-threaded (async processing)
- **GPU:** Optional (speeds up COMET by 2-3x)

## 🔧 Configuration

### Environment Variables

Add to `.env`:

```bash
# Contextual Analysis Settings
NER_MODEL=en_core_web_sm  # or en_core_web_md, en_core_web_lg
COMET_MODEL=comet-atomic_2020_BART
ENABLE_CONTEXTUAL_ANALYSIS=true
```

### Model Selection

**spaCy NER Models:**

- `en_core_web_sm`: Fast, good accuracy (~12MB)
- `en_core_web_md`: Better accuracy (~40MB)
- `en_core_web_lg`: Best accuracy (~560MB)

**COMET Models:**

- `comet-atomic_2020_BART`: Recommended (~1.6GB)
- `comet-commonsense`: Alternative (~1.2GB)

## 🚧 Limitations & Future Work

### Current Limitations

1. **In-Memory Graph Storage**

   - Graph data lost on restart
   - Doesn't scale across multiple servers
   - No persistence layer

2. **Sequential COMET Processing**

   - One inference at a time per request
   - Could benefit from batching

3. **Basic Entity Linking**
   - No coreference resolution
   - Doesn't link entities across conversations

### Planned Improvements

1. **Neo4j Integration**

   - Persistent graph storage
   - Advanced graph queries (Cypher)
   - Distributed graph processing

2. **Advanced NLP**

   - Coreference resolution
   - Entity linking and disambiguation
   - Relation extraction

3. **Real-time Analysis**

   - Stream processing for WebSocket messages
   - Incremental graph updates
   - Live entity tracking

4. **Visualization**
   - Interactive graph visualization
   - Emotion timeline charts
   - Entity relationship diagrams

## 📚 References

### Academic Papers

1. **COMET:** Bosselut et al. (2019) "COMET: Commonsense Transformers for Automatic Knowledge Graph Construction"
2. **spaCy:** Honnibal & Montani (2017) "spaCy 2: Natural language understanding with Bloom embeddings"

### Libraries & Models

- **spaCy:** https://spacy.io/
- **COMET (AllenAI):** https://huggingface.co/allenai/comet-atomic_2020_BART
- **Transformers:** https://huggingface.co/docs/transformers

## 🎓 Key Concepts

### Named Entity Recognition (NER)

NER is the task of identifying and classifying named entities in text. It's a fundamental NLP task that helps structure unstructured text.

**Use Cases:**

- Information extraction
- Content recommendation
- Search and indexing
- Question answering

### Commonsense Reasoning

COMET uses large-scale transformer models trained on ATOMIC (Atlas of Machine Commonsense) to infer unstated knowledge about social situations, emotional reactions, and causal effects.

**Use Cases:**

- Dialogue systems
- Story understanding
- Empathetic AI
- Social intelligence

### Knowledge Graphs

Knowledge graphs represent entities and their relationships in a structured format, enabling semantic queries and inferential reasoning.

**Use Cases:**

- Semantic search
- Recommendation systems
- Question answering
- Context understanding

---

**Week 5 Status:** ✅ Implementation Complete  
**Next:** Week 6 - Advanced Knowledge Graph Features & LLM Integration  
**Updated:** October 13, 2025
