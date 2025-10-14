# Week 7: Advanced AI Orchestration - Neo4j & LLM Integration

## Overview

Week 7 builds upon the solid Week 6 foundation by adding:

1. **Neo4j Integration** - Persistent graph database for knowledge storage
2. **LLM Integration** - Intelligent response generation using graph context
3. **Enhanced Orchestration** - Smarter pipeline with context-aware processing

## Current State (Week 6 Complete ✅)

Already implemented:

- ✅ Chat orchestrator service (`chat_orchestrator.py`)
- ✅ Sequential AI model invocation (STT → SER → NER → COMET)
- ✅ Parallel optimization (STT + SER run together)
- ✅ Aggregated analysis packet (JSON response)
- ✅ In-memory knowledge graph
- ✅ FastAPI endpoint (`POST /orchestrate/analyze-audio`)
- ✅ Comprehensive error handling
- ✅ Processing metrics

## Week 7 Enhancements

### 1. Neo4j Integration for Persistent Graphs

**Why Neo4j?**

- Native graph database optimized for relationships
- Cypher query language for powerful graph traversal
- Scalable for millions of nodes and relationships
- Built-in graph algorithms
- Excellent visualization tools

**Implementation:**

```python
# File: aura-backend/contextual/neo4j_graph_service.py

from neo4j import AsyncGraphDatabase
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Neo4jGraphService:
    """
    Neo4j-backed knowledge graph service.
    Provides persistent storage and advanced graph operations.
    """

    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j at {uri}")

    async def close(self):
        """Close Neo4j connection."""
        await self.driver.close()

    async def add_entity_node(
        self,
        node_type: str,
        label: str,
        properties: Dict[str, Any]
    ) -> str:
        """
        Add or update an entity node in Neo4j.

        Args:
            node_type: Type of entity (PERSON, PLACE, etc.)
            label: Entity label/name
            properties: Additional properties

        Returns:
            Node ID
        """
        async with self.driver.session() as session:
            query = f"""
            MERGE (n:{node_type} {{label: $label}})
            ON CREATE SET n += $properties, n.created_at = timestamp()
            ON MATCH SET n.last_seen = timestamp(),
                        n.occurrence_count = coalesce(n.occurrence_count, 0) + 1
            RETURN id(n) as node_id
            """
            result = await session.run(
                query,
                label=label,
                properties=properties
            )
            record = await result.single()
            return str(record["node_id"])

    async def add_relationship(
        self,
        from_node_label: str,
        from_node_type: str,
        to_node_label: str,
        to_node_type: str,
        rel_type: str,
        properties: Dict[str, Any]
    ) -> str:
        """
        Create relationship between nodes.

        Args:
            from_node_label: Source node label
            from_node_type: Source node type
            to_node_label: Target node label
            to_node_type: Target node type
            rel_type: Relationship type
            properties: Relationship properties

        Returns:
            Relationship ID
        """
        async with self.driver.session() as session:
            query = f"""
            MATCH (a:{from_node_type} {{label: $from_label}})
            MATCH (b:{to_node_type} {{label: $to_label}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r += $properties, r.created_at = timestamp()
            ON MATCH SET r.last_updated = timestamp(),
                        r.strength = coalesce(r.strength, 0) + 1
            RETURN id(r) as rel_id
            """
            result = await session.run(
                query,
                from_label=from_node_label,
                to_label=to_node_label,
                properties=properties
            )
            record = await result.single()
            return str(record["rel_id"])

    async def get_conversation_context(
        self,
        conversation_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve graph context for a conversation.

        Args:
            conversation_id: Conversation ID
            depth: Graph traversal depth

        Returns:
            Context dictionary with nodes and relationships
        """
        async with self.driver.session() as session:
            query = """
            MATCH (n)-[r*1..%d]-(m)
            WHERE n.conversation_id = $conv_id OR m.conversation_id = $conv_id
            RETURN collect(DISTINCT n) as nodes,
                   collect(DISTINCT r) as relationships
            """ % depth

            result = await session.run(query, conv_id=conversation_id)
            record = await result.single()

            return {
                "nodes": [dict(n) for n in record["nodes"]],
                "relationships": [dict(r) for r in record["relationships"]]
            }

    async def get_entity_connections(
        self,
        entity_label: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all connections for an entity.

        Args:
            entity_label: Entity to query
            limit: Maximum results

        Returns:
            List of connected entities
        """
        async with self.driver.session() as session:
            query = """
            MATCH (a {label: $label})-[r]-(b)
            RETURN type(r) as relationship,
                   b.label as connected_entity,
                   labels(b)[0] as entity_type,
                   r.strength as strength
            ORDER BY r.strength DESC
            LIMIT $limit
            """
            result = await session.run(query, label=entity_label, limit=limit)
            return [dict(record) async for record in result]
```

### 2. LLM Integration with Graph Context

**Why LLM?**

- Generate intelligent, context-aware responses
- Understand user intent beyond keywords
- Provide conversational, natural interactions
- Leverage graph knowledge for enriched responses

**Implementation:**

```python
# File: aura-backend/llm/llm_service.py

from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """
    LLM service for generating intelligent responses.
    Uses graph context to provide enriched, context-aware answers.
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize LLM service."""
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized LLM service with model: {model}")

    async def generate_response(
        self,
        user_message: str,
        analysis_packet: Dict[str, Any],
        graph_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate intelligent response using analysis and context.

        Args:
            user_message: Transcribed user message
            analysis_packet: Complete analysis from orchestrator
            graph_context: Knowledge graph context
            conversation_history: Previous messages

        Returns:
            Response dictionary with text and metadata
        """
        # Build context prompt
        context_prompt = self._build_context_prompt(
            user_message,
            analysis_packet,
            graph_context,
            conversation_history
        )

        # Generate response
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Aura, an empathetic AI assistant with deep contextual understanding. "
                                   "You have access to knowledge graphs and emotional intelligence to provide "
                                   "personalized, context-aware responses."
                    },
                    {
                        "role": "user",
                        "content": context_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )

            response_text = response.choices[0].message.content

            return {
                "text": response_text,
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "text": "I'm having trouble processing your request right now.",
                "error": str(e)
            }

    def _build_context_prompt(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        graph_context: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build enriched prompt with all context."""

        prompt_parts = []

        # User message
        prompt_parts.append(f"User said: \"{user_message}\"")

        # Emotional context
        emotion_audio = analysis.get("emotion", {}).get("from_audio", {})
        emotion_text = analysis.get("emotion", {}).get("from_text", {})

        if emotion_audio.get("primary"):
            prompt_parts.append(
                f"\nUser's vocal emotion: {emotion_audio['primary']} "
                f"(confidence: {emotion_audio.get('confidence', 0):.2f})"
            )

        if emotion_text.get("detected"):
            prompt_parts.append(
                f"Detected feelings: {', '.join(emotion_text['detected'])}"
            )

        # Entities and context
        entities = analysis.get("entities", {})
        if entities:
            prompt_parts.append("\nMentioned entities:")
            for category, items in entities.items():
                if items:
                    names = [e["text"] for e in items]
                    prompt_parts.append(f"- {category}: {', '.join(names)}")

        # Commonsense understanding
        commonsense = analysis.get("commonsense", {}).get("inferences", {})
        if commonsense:
            subject = commonsense.get("subject", {})
            if subject.get("wants"):
                prompt_parts.append(
                    f"\nUser likely wants: {', '.join(subject['wants'][:3])}"
                )
            if subject.get("feelings"):
                prompt_parts.append(
                    f"User might feel: {', '.join(subject['feelings'][:3])}"
                )

        # Graph context
        if graph_context and graph_context.get("nodes"):
            prompt_parts.append(
                f"\nRelated context from knowledge graph: "
                f"{len(graph_context['nodes'])} connected concepts"
            )

        # Conversation history
        if history:
            prompt_parts.append("\nRecent conversation:")
            for msg in history[-3:]:  # Last 3 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:100]  # Truncate
                prompt_parts.append(f"- {role}: {content}")

        prompt_parts.append(
            "\nPlease respond in a helpful, empathetic way that acknowledges "
            "the user's emotions and context. Be conversational and natural."
        )

        return "\n".join(prompt_parts)
```

### 3. Enhanced Orchestrator with LLM

```python
# Update to: aura-backend/chat_orchestrator.py

class ChatOrchestrator:
    """Enhanced orchestrator with Neo4j and LLM integration."""

    def __init__(
        self,
        transcription_service,
        emotion_service,
        contextual_analyzer,
        neo4j_service: Optional['Neo4jGraphService'] = None,
        llm_service: Optional['LLMService'] = None
    ):
        self.transcription_service = transcription_service
        self.emotion_service = emotion_service
        self.contextual_analyzer = contextual_analyzer
        self.neo4j_service = neo4j_service
        self.llm_service = llm_service

        logger.info("Initialized Enhanced ChatOrchestrator")

    async def process_audio_with_response(
        self,
        audio_bytes: bytes,
        conversation_id: str,
        speaker_id: Optional[str] = None,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio and optionally generate LLM response.

        NEW in Week 7: Adds LLM response generation
        """
        # Step 1-4: Run existing pipeline (STT, SER, NER, COMET)
        analysis_packet = await self.process_audio(
            audio_bytes=audio_bytes,
            conversation_id=conversation_id,
            speaker_id=speaker_id
        )

        # Step 5: Update Neo4j (if available)
        if self.neo4j_service:
            await self._update_neo4j_graph(
                analysis_packet,
                conversation_id
            )

        # Step 6: Generate LLM response (if requested and available)
        if generate_response and self.llm_service:
            # Get graph context
            graph_context = None
            if self.neo4j_service:
                graph_context = await self.neo4j_service.get_conversation_context(
                    conversation_id
                )

            # Generate response
            user_message = analysis_packet["transcript"]["text"]
            llm_response = await self.llm_service.generate_response(
                user_message=user_message,
                analysis_packet=analysis_packet,
                graph_context=graph_context
            )

            analysis_packet["ai_response"] = llm_response

        return analysis_packet
```

## Docker Compose Configuration

```yaml
# Add to docker-compose.yml

services:
  # ...existing services...

  neo4j:
    image: neo4j:5.13
    ports:
      - "7474:7474" # HTTP
      - "7687:7687" # Bolt
    environment:
      - NEO4J_AUTH=neo4j/your_password
      - NEO4J_PLUGINS=["graph-data-science", "apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  # ...existing volumes...
  neo4j_data:
  neo4j_logs:
```

## Environment Variables

```bash
# Add to .env

# Neo4j Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI Configuration (for LLM)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Or use local LLM
LOCAL_LLM_ENDPOINT=http://localhost:8080
LOCAL_LLM_MODEL=llama2
```

## Installation & Testing

### 1. Install Dependencies

```bash
cd aura-backend
source venv/bin/activate
pip install neo4j openai
```

### 2. Start Neo4j

```bash
docker-compose up neo4j -d

# Wait for Neo4j to start
docker-compose logs -f neo4j

# Access Neo4j Browser at http://localhost:7474
```

### 3. Test Neo4j Connection

```python
# test_neo4j_connection.py

from neo4j import AsyncGraphDatabase

async def test_connection():
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    )

    async with driver.session() as session:
        result = await session.run("RETURN 'Neo4j Connected!' as message")
        record = await result.single()
        print(record["message"])

    await driver.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())
```

### 4. Test Enhanced Orchestrator

```python
# test_week7_orchestrator.py

import asyncio
from chat_orchestrator import ChatOrchestrator
from contextual.neo4j_graph_service import Neo4jGraphService
from llm.llm_service import LLMService

async def test_enhanced_orchestrator():
    # Initialize services
    neo4j = Neo4jGraphService(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your_password"
    )

    llm = LLMService(
        api_key="your_openai_key",
        model="gpt-4"
    )

    orchestrator = ChatOrchestrator(
        transcription_service=...,
        emotion_service=...,
        contextual_analyzer=...,
        neo4j_service=neo4j,
        llm_service=llm
    )

    # Test with audio
    result = await orchestrator.process_audio_with_response(
        audio_bytes=audio_data,
        conversation_id="test_001",
        generate_response=True
    )

    print("Analysis:", result["transcript"]["text"])
    print("AI Response:", result["ai_response"]["text"])

    await neo4j.close()

if __name__ == "__main__":
    asyncio.run(test_enhanced_orchestrator())
```

## API Endpoints (Week 7)

### Enhanced Orchestrator Endpoint

```python
@app.post("/orchestrate/analyze-audio-v2")
async def orchestrate_audio_analysis_v2(
    file: UploadFile = File(...),
    conversation_id: str = Query(...),
    speaker_id: Optional[str] = Query(None),
    generate_response: bool = Query(True),
    include_graph: bool = Query(True),
    current_user = Depends(get_current_user)
):
    """
    Week 7: Enhanced orchestrator with Neo4j and LLM.

    Returns analysis + AI-generated response.
    """
    audio_bytes = await file.read()

    result = await chat_orchestrator.process_audio_with_response(
        audio_bytes=audio_bytes,
        conversation_id=conversation_id,
        speaker_id=speaker_id or str(current_user.id),
        generate_response=generate_response
    )

    return result
```

## Week 7 Milestone Achievement

✅ **Single Function Call Triggers Everything:**

```python
result = await chat_orchestrator.process_audio_with_response(
    audio_bytes=audio_data,
    conversation_id="conv_001"
)

# Returns complete analysis packet:
{
  "transcript": {...},
  "emotion": {...},
  "entities": {...},
  "commonsense": {...},
  "graph_updates": {...},
  "processing": {...},
  "ai_response": {  # NEW in Week 7
    "text": "I understand you're meeting Sarah...",
    "model": "gpt-4",
    "tokens_used": 245
  }
}
```

## Benefits of Week 7 Enhancements

1. **Persistent Knowledge** - Neo4j stores all graph data permanently
2. **Advanced Queries** - Cypher enables complex graph traversal
3. **Intelligent Responses** - LLM generates context-aware replies
4. **Scalability** - Neo4j handles millions of nodes efficiently
5. **Rich Context** - Graph relationships enhance LLM prompts
6. **Production Ready** - Enterprise-grade database and AI

## Next Steps

1. Implement Neo4j service
2. Add LLM service
3. Enhance orchestrator
4. Update Docker Compose
5. Test integration
6. Deploy to production

---

**Status**: Ready for Week 7 Implementation  
**Foundation**: Week 6 Complete ✅  
**Enhancements**: Neo4j + LLM Integration 🚀
