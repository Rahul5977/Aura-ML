# Week 6 Quick Start Guide

## Prerequisites

- Python 3.9+
- Virtual environment
- Audio files for testing (or use generated test audio)

## Installation

### 1. Set Up Virtual Environment

```bash
cd /Users/rahulraj/Desktop/ML_Proj/aura-backend
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

For quick testing (without full ML models):

```bash
pip install fastapi uvicorn python-multipart python-dotenv requests scipy numpy
```

For full production setup:

```bash
pip install -r requirements.txt
```

## Testing Options

### Option 1: Run the Demo (No ML Models Required) ⚡

This demonstrates the orchestrator architecture without needing ML models:

```bash
cd /Users/rahulraj/Desktop/ML_Proj
python3 chat_orchestrator_demo.py
```

**What you'll see**:

- Complete pipeline execution
- Mock STT, SER, NER, and COMET results
- Aggregated JSON response
- Processing metrics
- ~250ms execution time

**Output**:

```
======================================================================
  WEEK 6: CHAT ORCHESTRATOR DEMONSTRATION
======================================================================

Phase 1: Running audio-based models in parallel...
🎤 Running STT (Speech-to-Text)...
😊 Running SER (Speech Emotion Recognition)...

✓ Audio processing complete
  Transcribed: 'I'm meeting Sarah at the new coffee shop...'
  Emotion: neutral (0.85)

Phase 2: Running text-based models...
🏷️  Running NER (Named Entity Recognition)...
🧠 Running COMET (Commonsense Reasoning)...

✓ Text processing complete
  Entities found: 4
  Emotions detected: ['hopeful', 'excited', 'interested']

[Full JSON Response]
```

### Option 2: Test with Backend Server (Full ML Models) 🚀

**Terminal 1 - Start Backend**:

```bash
cd /Users/rahulraj/Desktop/ML_Proj/aura-backend
source venv/bin/activate

# Make sure all dependencies are installed
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials

# Start the server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Run Tests**:

```bash
cd /Users/rahulraj/Desktop/ML_Proj
source aura-backend/venv/bin/activate

# Run the Week 6 test suite
python3 test_week6.py
```

**What the tests do**:

1. ✅ Health check
2. ✅ User registration and authentication
3. ✅ Single audio analysis
4. ✅ Multiple audio samples
5. ✅ Knowledge graph accumulation

### Option 3: Manual API Testing with curl 🔧

```bash
# 1. Register a user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "full_name": "Test User",
    "password": "testpass123"
  }'

# 2. Login to get token
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }' | jq -r '.access_token')

echo "Token: $TOKEN"

# 3. Create test audio file
python3 -c "
import numpy as np
from scipy.io import wavfile
t = np.linspace(0, 2, 32000)
audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
wavfile.write('test_audio.wav', 16000, audio)
"

# 4. Test the orchestrator endpoint
curl -X POST http://localhost:8000/orchestrate/analyze-audio \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_audio.wav" \
  -F "conversation_id=test_conv_001" \
  -F "speaker_id=test_user" \
  | jq '.'
```

## Response Structure

The orchestrator returns a comprehensive JSON response:

```json
{
  "transcript": {
    "text": "Transcribed text",
    "language": "en",
    "confidence": 0.95
  },
  "emotion": {
    "from_audio": {
      "primary": "neutral",
      "confidence": 0.85,
      "all_scores": {...}
    },
    "from_text": {
      "detected": ["hopeful", "excited"],
      "context": {...}
    }
  },
  "entities": {
    "people": [...],
    "places": [...],
    "dates": [...]
  },
  "commonsense": {
    "inferences": {
      "subject": {
        "feelings": [...],
        "wants": [...],
        "effects": [...]
      },
      "others": {...}
    }
  },
  "graph_updates": {
    "entity_nodes_count": 4,
    "emotional_relationships_count": 3,
    "updated": true
  },
  "processing": {
    "total_time_ms": 650,
    "stt_completed": true,
    "ser_completed": true,
    "ner_completed": true,
    "comet_completed": true
  },
  "metadata": {
    "conversation_id": "test_conv_001",
    "speaker_id": "test_user",
    "timestamp": "2025-10-13T...",
    "text_length": 60,
    "entity_count": 4
  }
}
```

## Verifying the Implementation

### Check 1: Code Structure ✅

```bash
# Verify all files exist
ls -la aura-backend/chat_orchestrator.py
ls -la aura-backend/main.py
ls -la test_week6.py
ls -la chat_orchestrator_demo.py
```

### Check 2: Imports and Integration ✅

```bash
# Check that orchestrator is imported in main.py
grep "chat_orchestrator" aura-backend/main.py

# Should show:
# - from chat_orchestrator import ...
# - initialize_chat_orchestrator(...)
# - @app.post("/orchestrate/analyze-audio")
```

### Check 3: Endpoint Availability ✅

```bash
# Start server and check health
curl http://localhost:8000/health

# Check OpenAPI docs
open http://localhost:8000/docs
# Look for POST /orchestrate/analyze-audio
```

## Troubleshooting

### Issue: Dependencies Not Installed

```bash
cd aura-backend
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Models Not Loading

Check the startup logs:

```bash
python3 -m uvicorn main:app --reload

# Look for:
# ✅ Transcription service loaded successfully
# ✅ Emotion recognition service loaded successfully
# ✅ Contextual analysis services loaded successfully
# ✅ Chat orchestrator initialized successfully
```

### Issue: Database Connection Error

```bash
# Check .env file
cat aura-backend/.env

# Make sure DATABASE_URL is set
# DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
```

### Issue: Import Errors

```bash
# Make sure you're in the right directory
cd /Users/rahulraj/Desktop/ML_Proj/aura-backend

# Make sure virtual environment is activated
source venv/bin/activate

# Check Python path
python3 -c "import sys; print(sys.path)"
```

## Performance Expectations

### Demo Mode (Mock Models)

- Processing time: ~250ms
- Memory usage: <100MB
- CPU usage: Low

### Production Mode (Real ML Models)

- Processing time: 600-900ms
- Memory usage: ~3.5GB
- CPU usage: High during inference

## Next Steps

After verifying the orchestrator works:

1. **Frontend Integration**: Update client to use `/orchestrate/analyze-audio`
2. **Performance Monitoring**: Track processing times and error rates
3. **Production Deployment**: Deploy with proper resource allocation
4. **Scaling**: Consider model caching and request batching
5. **Enhancement**: Add additional features (language detection, speaker diarization)

## Key Files Reference

- `aura-backend/chat_orchestrator.py` - Main orchestrator implementation
- `aura-backend/main.py` - FastAPI endpoint definition
- `test_week6.py` - Comprehensive test suite
- `chat_orchestrator_demo.py` - Standalone demonstration
- `WEEK6_IMPLEMENTATION_COMPLETE.md` - Full documentation

## Support

If you encounter issues:

1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure models are properly loaded
4. Test with the demo script first
5. Review the documentation

## Success Criteria ✅

Your implementation is complete when:

- ✅ Demo script runs successfully
- ✅ Backend starts without errors
- ✅ All models initialize properly
- ✅ Endpoint returns aggregated JSON
- ✅ Test suite passes
- ✅ Processing time is reasonable
- ✅ Error handling works correctly

---

**Ready to test?** Start with the demo:

```bash
python3 chat_orchestrator_demo.py
```
