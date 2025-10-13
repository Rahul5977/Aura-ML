# Week 7 Quick Start Guide

## What's New in Week 7

Week 7 enhances the Week 6 orchestrator with:

- **Neo4j Integration** - Persistent graph database
- **LLM Integration** - Intelligent response generation
- **Enhanced Context** - Graph-powered AI responses

## Prerequisites

-Week 6 complete ✅

- Docker and Docker Compose
- OpenAI API key (optional, for LLM)
- 4GB+ RAM (for Neo4j)

## Quick Start

### 1. Start Services

```bash
cd /Users/rahulraj/Desktop/ML_Proj

# Start all services including Neo4j
docker-compose up -d

# Check Neo4j is running
docker-compose logs neo4j

# Access Neo4j Browser
open http://localhost:7474
# Login: neo4j / aura_neo4j_pass
```

### 2. Install Dependencies

```bash
cd aura-backend
source venv/bin/activate

# Install Neo4j and OpenAI drivers
pip install neo4j openai

# Update requirements.txt
echo "neo4j>=5.13.0" >> requirements.txt
echo "openai>=1.3.0" >> requirements.txt
```

### 3. Configure Environment

```bash
# Edit .env file
cat >> .env << EOF

# Week 7: Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=aura_neo4j_pass

# Week 7: OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

EOF
```

### 4. Test Neo4j Connection

```bash
# Run Week 7 tests
python3 test_week7.py
```

Expected output:

```
======================================================================
  TEST 1: Neo4j Connection
======================================================================
✅ Connected to Neo4j
✅ Schema initialized
✅ Created test node: ...
✅ Graph summary: 1 nodes, 0 relationships
✅ Neo4j test passed!
```

### 5. Test LLM Service

```bash
# Set API key
export OPENAI_API_KEY="your_key_here"

# Run LLM test
python3 -c "
import asyncio
from llm.llm_service import LLMService

async def test():
    llm = LLMService(api_key='your_key', model='gpt-3.5-turbo')
    response = await llm.generate_simple_response('Hello, how are you?')
    print(response)

asyncio.run(test())
"
```

## API Usage

### Week 7 Enhanced Endpoint

```bash
# Upload audio and get AI response
curl -X POST http://localhost:8000/orchestrate/analyze-audio-v2 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@audio.wav" \
  -F "conversation_id=conv_001" \
  -F "generate_response=true"
```

Response includes:

```json
{
  "transcript": {...},
  "emotion": {...},
  "entities": {...},
  "commonsense": {...},
  "graph_updates": {...},
  "ai_response": {
    "text": "I understand you're feeling stressed...",
    "model": "gpt-4",
    "tokens_used": 245
  }
}
```

## Verification Steps

### ✅ Check 1: Neo4j Running

```bash
# Should return "Neo4j Connected!"
docker exec -it ml_proj-neo4j-1 cypher-shell -u neo4j -p aura_neo4j_pass \
  "RETURN 'Neo4j Connected!' as message"
```

### ✅ Check 2: Graph Data

```bash
# View nodes in Neo4j Browser
open http://localhost:7474

# Run query:
MATCH (n) RETURN n LIMIT 25
```

### ✅ Check 3: LLM Service

```python
# test_llm_quick.py
import asyncio
from llm.llm_service import LLMService

async def main():
    llm = LLMService(api_key="your_key", model="gpt-3.5-turbo")
    if llm.is_ready():
        print("✅ LLM service ready")
    else:
        print("❌ LLM service not available")

asyncio.run(main())
```

### ✅ Check 4: End-to-End

```bash
# Test complete pipeline
python3 test_week7.py

# All tests should pass
```

## Troubleshooting

### Issue: Neo4j Won't Start

```bash
# Check logs
docker-compose logs neo4j

# Restart service
docker-compose restart neo4j

# Check memory
docker stats ml_proj-neo4j-1
```

### Issue: Connection Refused

```bash
# Check Neo4j is listening
docker exec ml_proj-neo4j-1 netstat -tulpn | grep 7687

# Test from host
telnet localhost 7687
```

### Issue: LLM Not Working

```bash
# Check API key
echo $OPENAI_API_KEY

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Issue: Graph Empty

```cypher
// In Neo4j Browser (http://localhost:7474)

// Check constraints
SHOW CONSTRAINTS

// Check indexes
SHOW INDEXES

// Count nodes
MATCH (n) RETURN count(n)

// View all nodes
MATCH (n) RETURN n LIMIT 100
```

## Usage Examples

### Example 1: Simple Conversation

```python
import asyncio
from chat_orchestrator import ChatOrchestrator

async def example():
    orchestrator = ChatOrchestrator(
        transcription_service=...,
        emotion_service=...,
        contextual_analyzer=...,
        neo4j_service=neo4j,
        llm_service=llm
    )

    result = await orchestrator.process_audio_with_response(
        audio_bytes=audio_data,
        conversation_id="user_123",
        generate_response=True
    )

    print("User:", result["transcript"]["text"])
    print("Aura:", result["ai_response"]["text"])
```

### Example 2: Graph Query

```python
async def example():
    from contextual.neo4j_graph_service import Neo4jGraphService

    neo4j = Neo4jGraphService(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="aura_neo4j_pass"
    )
    await neo4j.connect()

    # Get connections for an entity
    connections = await neo4j.get_entity_connections("Sarah")
    for conn in connections:
        print(f"{conn['relationship']}: {conn['connected_entity']}")

    await neo4j.close()
```

### Example 3: Context-Aware Response

```python
async def example():
    # Get graph context
    context = await neo4j.get_conversation_context("conv_001")

    # Generate response with context
    response = await llm.generate_response(
        user_message="Tell me about Sarah",
        analysis_packet=analysis,
        graph_context=context
    )

    print(response["text"])
```

## Performance Tips

### Neo4j Optimization

```cypher
// Create indexes for faster lookups
CREATE INDEX entity_label IF NOT EXISTS FOR (n) ON (n.label);
CREATE INDEX conversation_idx IF NOT EXISTS FOR (n) ON (n.conversation_id);

// View query performance
PROFILE MATCH (n:PERSON {label: 'Sarah'}) RETURN n;
```

### LLM Optimization

```python
# Use cheaper model for simple queries
llm = LLMService(model="gpt-3.5-turbo")  # Faster, cheaper

# Cache responses (implement your own caching)
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_response(message):
    return asyncio.run(llm.generate_simple_response(message))
```

### Graph Query Optimization

```python
# Limit traversal depth
context = await neo4j.get_conversation_context(
    conversation_id="conv_001",
    depth=1,  # Shallow traversal
    limit=20  # Fewer nodes
)
```

## Next Steps

After Week 7 is working:

1. **Test with Real Audio**

   ```bash
   # Record audio
   # Upload via API
   # View graph in Neo4j Browser
   ```

2. **Visualize Knowledge Graph**

   - Open Neo4j Browser
   - Run: `MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50`
   - Explore relationships

3. **Monitor Performance**

   - Track LLM token usage
   - Monitor Neo4j query times
   - Check graph growth

4. **Deploy to Production**
   - Use managed Neo4j (AuraDB)
   - Set up OpenAI organization
   - Configure rate limits

## Resources

- **Neo4j Browser**: http://localhost:7474
- **Neo4j Documentation**: https://neo4j.com/docs/
- **OpenAI API**: https://platform.openai.com/docs/
- **Cypher Query Language**: https://neo4j.com/docs/cypher-manual/

---

**Week 7 Status**: Implementation Complete  
**Ready for**: Production Testing  
**Next**: Week 8 - Frontend Integration & UX
