# File Migration Map

## From `fine-tuining/` → To New Structure

### Core Models
```
fine-tuining/ece_model.py
  → aura_ml/models/ece_classifier.py

fine-tuining/chat_aura.py (logic)
  → aura_ml/models/llm_wrapper.py
  → aura_ml/inference/chatbot.py

fine-tuining/llama3_finetuned_final/
  → data/models/llm/llama3_finetuned_final/
```

### CLI
```
fine-tuining/chat_aura.py
  → cli/chat.py (cleaner, better organized)

fine-tuining/start_chat.sh
  → cli/chat.py (integrated into Python script)
```

### Training (Future Migration)
```
fine-tuining/train_llama3.py
  → scripts/train_llm.py (TODO: refactor)

fine-tuining/train_aura.py
  → scripts/train_llm.py (TODO: refactor)

fine-tuining/setup_unsloth.py
  → aura_ml/training/trainer.py (TODO: refactor)
```

### Configuration
```
fine-tuining/*.py (hardcoded configs)
  → aura_ml/config/settings.py (environment-based)
  → aura_ml/config/model_config.py (dataclass configs)
  → .env (user configuration)
```

### API (NEW!)
```
Nothing in fine-tuining/
  → api/ (completely new)
    ├── main.py
    ├── routers/
    ├── models/
    └── services/
```

### Documentation
```
fine-tuining/INFERENCE_GUIDE.md
fine-tuining/SETUP_COMPLETE.md
fine-tuining/TRAINING_GUIDE.md
  → docs/ (organized)
  → README.md (consolidated)
```

## What Stays in `fine-tuining/` (for now)

These will be migrated later as needed:
- Training scripts (will refactor to `scripts/`)
- Experiment notebooks (move to `notebooks/`)
- Test scripts (refactor to `tests/`)
- Setup utilities (integrate into package)

## What Gets Organized

### Models
```
OLD: fine-tuining/llama3_finetuned_final/
NEW: data/models/llm/llama3_finetuned_final/

OLD: fine-tuining/ece_model_output/
NEW: data/models/ece/ece_roberta_model/
```

### Datasets
```
OLD: datasets/ (scattered)
NEW: data/processed/ (organized)
     data/raw/ (original data)
```

### Logs & Outputs
```
OLD: fine-tuining/*.log
NEW: data/outputs/*.log

OLD: fine-tuining/training_outputs/
NEW: data/outputs/training/
```

## Import Changes

### Before (Cluttered)
```python
# Can't import! Must run script directly
# python fine-tuining/chat_aura.py

# Or hacky path manipulation
import sys
sys.path.insert(0, "fine-tuining")
from chat_aura import AuraChat
```

### After (Clean)
```python
# Install once
# pip install -e .

# Then import anywhere
from aura_ml.models import AuraLLM
from aura_ml.inference import AuraChatbot
from aura_ml.config import settings
```

## Usage Changes

### Before
```bash
# Multiple scattered entry points
cd fine-tuining
./start_chat.sh

# Or
python chat_aura.py

# Or
python inference_llama3.py

# Or... (confusion!)
```

### After
```bash
# Single clear entry points

# CLI
python cli/chat.py
# or
aura-chat

# API
uvicorn api.main:app

# Python
python -c "from aura_ml.inference import AuraChatbot; ..."
```

## Configuration Changes

### Before (Hardcoded)
```python
# In chat_aura.py
MODEL_PATH = "./llama3_finetuned_final"  # Hardcoded!
MAX_TOKENS = 128                          # Hardcoded!
TEMPERATURE = 0.7                         # Hardcoded!
```

### After (Configurable)
```python
# aura_ml/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LLM_MODEL_PATH: Path = Path("data/models/llm/...")
    
    class Config:
        env_file = ".env"  # User can override!

# .env file
LLM_MODEL_PATH=data/models/llm/my_custom_model
```

## API Patterns

### Before (No API)
```
No way to use via HTTP!
Must run Python script locally
```

### After (REST API)
```bash
# Health check
GET /api/v1/health

# Chat
POST /api/v1/chat
{
  "message": "I'm anxious",
  "emotion": "anxious"
}

# Emotion detection
POST /api/v1/emotion/detect
{
  "text": "I'm so worried"
}
```

## Package Structure

### Before (No Package)
```
fine-tuining/
├── 50+ loose Python files
├── No __init__.py
├── No setup.py
└── Can't pip install
```

### After (Proper Package)
```
aura_ml/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── ...
├── models/
│   ├── __init__.py
│   └── ...
└── ...

setup.py
requirements/

$ pip install -e .
$ python -c "import aura_ml; print(aura_ml.__version__)"
1.0.0
```

## Development Workflow

### Before
```bash
# Edit files directly in fine-tuining/
cd fine-tuining
vim chat_aura.py

# Test by running
python chat_aura.py

# Deploy... somehow?
```

### After
```bash
# Develop in clean package
cd aura_ml
vim inference/chatbot.py

# Install in dev mode
pip install -e ".[dev]"

# Test with pytest
pytest tests/

# Deploy with Docker
docker build -t aura-api .
docker run -p 8000:8000 aura-api
```

## Summary: What Changed

| Aspect | Before | After |
|--------|--------|-------|
| **Structure** | Flat, cluttered | Hierarchical, organized |
| **Imports** | ❌ Not possible | ✅ `from aura_ml import ...` |
| **Entry Points** | Scattered scripts | CLI, API, Package |
| **Configuration** | Hardcoded | Environment variables |
| **API** | ❌ None | ✅ FastAPI REST |
| **Testing** | Ad-hoc scripts | Proper test suite |
| **Documentation** | Scattered READMEs | Organized docs/ |
| **Deployment** | Manual | Docker-ready |
| **Maintainability** | ⚠️ Difficult | ✅ Easy |
| **Scalability** | ❌ Hard to scale | ✅ Microservice-ready |

---

**Bottom Line:** Same powerful Aura AI, now with professional structure! 🚀
