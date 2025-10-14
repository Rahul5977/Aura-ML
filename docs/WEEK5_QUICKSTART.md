# Week 5 Quick Start Guide

## 🚀 5-Minute Guide to Contextual Analysis

This guide shows you how to use the new Week 5 features in 5 minutes.

## Prerequisites

```bash
# Ensure Docker is running
docker-compose up -d --build

# Wait for services to start (~30 seconds)
sleep 30
```

## Step 1: Authenticate (30 seconds)

```bash
# Register a user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "full_name": "Test User",
    "password": "password123"
  }'

# Login and get token
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }' | jq -r '.access_token')

echo "Token: $TOKEN"
```

## Step 2: Analyze Text (1 minute)

```bash
# Analyze a conversation
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -G \
  --data-urlencode "text=John met Sarah at Starbucks in Seattle to discuss their new AI startup idea." \
  --data-urlencode "conversation_id=quick_start_001" \
  --data-urlencode "include_graph=true" | jq
```

**Expected Response:**

```json
{
  "entities": {
    "people": [
      { "text": "John", "start": 0, "end": 4 },
      { "text": "Sarah", "start": 9, "end": 14 }
    ],
    "places": [
      { "text": "Starbucks", "start": 18, "end": 27 },
      { "text": "Seattle", "start": 31, "end": 38 }
    ],
    "concepts": [{ "text": "AI startup idea", "start": 60, "end": 75 }]
  },
  "emotional_context": {
    "subject_emotions": ["interested", "excited", "hopeful"],
    "subject_wants": ["to collaborate", "to discuss ideas"]
  },
  "emotions_detected": ["excited", "hopeful"],
  "metadata": {
    "processing_time_ms": 450
  }
}
```

## Step 3: Check Knowledge Graph (1 minute)

```bash
# Get graph summary
curl -X GET "http://localhost:8000/knowledge-graph/summary" \
  -H "Authorization: Bearer $TOKEN" | jq

# Get conversation context
curl -X GET "http://localhost:8000/analyze/conversation/quick_start_001" \
  -H "Authorization: Bearer $TOKEN" | jq
```

## Step 4: Analyze More Conversations (2 minutes)

```bash
# Add more context
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Authorization: Bearer $TOKEN" \
  -G \
  --data-urlencode "text=Sarah was excited about the opportunity and wanted to work with John." \
  --data-urlencode "conversation_id=quick_start_001" | jq

# Check updated graph
curl -X GET "http://localhost:8000/knowledge-graph/summary" \
  -H "Authorization: Bearer $TOKEN" | jq
```

## Python Quick Start

```python
import requests

# Login
response = requests.post('http://localhost:8000/auth/login', json={
    'username': 'testuser',
    'password': 'password123'
})
token = response.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

# Analyze text
text = "John met Sarah at Starbucks in Seattle to discuss their startup."
response = requests.post(
    'http://localhost:8000/analyze/text',
    params={
        'text': text,
        'conversation_id': 'quick_001',
        'include_graph': True
    },
    headers=headers
)

result = response.json()

# Print entities
print("People:", [e['text'] for e in result['entities']['people']])
print("Places:", [e['text'] for e in result['entities']['places']])
print("Emotions:", result['emotions_detected'])
```

## JavaScript Quick Start

```javascript
// Login
const loginResponse = await fetch("http://localhost:8000/auth/login", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    username: "testuser",
    password: "password123",
  }),
});

const { access_token } = await loginResponse.json();

// Analyze text
const text = "John met Sarah at Starbucks in Seattle to discuss their startup.";
const analyzeResponse = await fetch(
  `http://localhost:8000/analyze/text?text=${encodeURIComponent(
    text
  )}&conversation_id=quick_001`,
  {
    method: "POST",
    headers: { Authorization: `Bearer ${access_token}` },
  }
);

const result = await analyzeResponse.json();

// Display results
console.log(
  "People:",
  result.entities.people.map((e) => e.text)
);
console.log(
  "Places:",
  result.entities.places.map((e) => e.text)
);
console.log("Emotions:", result.emotions_detected);
```

## Testing Everything

```bash
# Run the comprehensive test suite
python test_week5.py
```

## What You Get

### 1. Named Entities

- **People:** Names of individuals
- **Places:** Locations (cities, venues)
- **Organizations:** Companies, institutions
- **Concepts:** Topics, products, ideas

### 2. Emotional Intelligence

- **Feelings:** How people feel (happy, sad, excited, etc.)
- **Wants:** What people desire or need
- **Effects:** What happens as a result

### 3. Knowledge Graph

- **Structured Knowledge:** Entities and relationships
- **Context Accumulation:** Builds over conversations
- **Queryable:** Retrieve related information

## Common Use Cases

### 1. Meeting Summarization

```python
# Analyze meeting notes
text = "The team discussed the Q4 roadmap with Sarah and decided to focus on AI features."
response = requests.post(url, params={'text': text, ...})

# Extract key entities
people = response.json()['entities']['people']
topics = response.json()['entities']['concepts']
```

### 2. Customer Support Analysis

```python
# Analyze support ticket
text = "Customer John Smith reported an issue with the payment system in New York."
response = requests.post(url, params={'text': text, ...})

# Extract customer info and location
customer = response.json()['entities']['people'][0]
location = response.json()['entities']['places'][0]
```

### 3. Conversation Context

```python
# Get accumulated knowledge
response = requests.get(f'http://localhost:8000/analyze/conversation/{conv_id}')
context = response.json()

# Use context for informed responses
related_people = context['related_entities']
```

## Troubleshooting

### Services Not Loading?

```bash
# Check Docker logs
docker logs ml_proj-backend-1

# Look for:
# ✅ Contextual analysis services loaded successfully
```

### Models Not Downloaded?

```bash
# Manually download spaCy model
docker exec ml_proj-backend-1 python -m spacy download en_core_web_sm
```

### Slow Response Times?

- First request is slower (model loading)
- Subsequent requests: ~400-650ms
- Consider using GPU for COMET (2-3x faster)

## Next Steps

1. **Read full documentation:** `WEEK5_CONTEXTUAL_ANALYSIS.md`
2. **Explore API:** http://localhost:8000/docs
3. **Check examples:** `test_week5.py`
4. **Build features:** Use graph context in your app

---

**Happy analyzing! 🎉**

For more details, see:

- [Full Documentation](WEEK5_CONTEXTUAL_ANALYSIS.md)
- [Completion Summary](WEEK5_COMPLETION_SUMMARY.md)
- [API Reference](http://localhost:8000/docs)
