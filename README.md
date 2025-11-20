# Aura ML - Production README

<div align="center">

# 🌟 Aura ML - Emotional Support AI System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready emotional support chatbot powered by fine-tuned Llama 3.2 3B**

[Features](#features) • [Quick Start](#quick-start) • [API](#api-usage) • [Training](#training) • [Documentation](#documentation)

</div>

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI Chat](#cli-chat)
  - [API](#api-usage)
  - [Python API](#python-api)
- [Training](#training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 🤖 AI Capabilities
- **Empathetic Conversations**: Fine-tuned Llama 3.2 3B for emotional support
- **Emotion Context Awareness**: Maintains emotional state throughout conversation
- **Fast Inference**: 2x faster with Unsloth optimizations
- **Streaming Responses**: ChatGPT-style token-by-token output

### 🛠️ Technical Features
- **Production-Ready API**: FastAPI backend with OpenAPI documentation
- **CLI Interface**: Interactive terminal chatbot
- **Modular Architecture**: Clean, maintainable codebase
- **GPU Optimized**: 4-bit quantization for 6GB VRAM
- **Type Safety**: Full type hints with Pydantic validation

### 📊 Model Performance
- **Base Model**: Llama 3.2 3B Instruct
- **Fine-tuning**: 3,510 emotional support examples
- **Training Loss**: 0.5777 (3 epochs)
- **Inference Speed**: ~25-30 tokens/second on RTX 4050
- **VRAM Usage**: ~4GB

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
├──────────────┬──────────────────┬────────────────────────────┤
│   CLI Chat   │    FastAPI       │   Python API               │
│              │    REST API      │                            │
└──────────────┴──────────────────┴────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼──────────┐          ┌────────▼──────────┐
│   Chat Service   │          │  Emotion Service  │
│                  │          │   (ECE Model)     │
└───────┬──────────┘          └────────┬──────────┘
        │                               │
┌───────▼──────────┐          ┌────────▼──────────┐
│   LLM Wrapper    │          │  ECE Classifier   │
│  (Llama 3.2 3B)  │          │  (RoBERTa-based)  │
└──────────────────┘          └───────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (6GB+ VRAM recommended)
- CUDA Toolkit 11.8+ or 12.x

### Option 1: Quick Install (User)

```bash
# Clone the repository
git clone https://github.com/yourusername/aura-ml.git
cd aura-ml

# Install package
pip install -e .

# Install training dependencies (optional)
pip install -e ".[training]"

# Install API dependencies (optional)
pip install -e ".[api]"
```

### Option 2: Development Install

```bash
# Clone and setup
git clone https://github.com/yourusername/aura-ml.git
cd aura-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

---

## 🎯 Quick Start

### 1. Download or Train Model

**Option A: Use Pre-trained Model**
```bash
# Place your fine-tuned model in:
# data/models/llm/llama3_finetuned_final/
```

**Option B: Train from Scratch**
```bash
# Generate training data
python scripts/generate_prompts.py

# Train the model
python scripts/train_llm.py

# Model will be saved to data/models/llm/
```

### 2. Start Chatting

**CLI Mode:**
```bash
aura-chat

# Or with custom model path
aura-chat --model ./path/to/your/model
```

**API Mode:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Access API docs at: http://localhost:8000/docs
```

---

## 💬 Usage

### CLI Chat

```bash
# Start interactive chat
aura-chat

# With custom settings
aura-chat --model ./my_model --temperature 0.8 --max-tokens 256

# Quick test
aura-chat --test
```

**Commands:**
- `/emotion <emotion> <cause>` - Set emotional context
- `/clear` - Clear emotion context
- `/history` - View conversation history
- `/reset` - Reset conversation
- `/help` - Show help
- `/quit` - Exit

**Example Session:**
```
🌟 AURA - Emotional Support AI Assistant

😊 You: /emotion anxious I have an exam tomorrow

✅ Emotion set: anxious
   Cause: I have an exam tomorrow

[Context: anxious - I have an exam tomorrow]
😊 You: I can't stop worrying about it

🌟 Aura: I understand how you're feeling. Exam anxiety is very common...
```

### API Usage

**Start API Server:**
```bash
uvicorn api.main:app --reload
```

**Send Chat Request:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "I'm feeling anxious",
        "emotion": "anxious",
        "cause": "upcoming exam",
        "max_tokens": 128,
        "temperature": 0.7
    }
)

print(response.json())
# {
#   "response": "I understand you're feeling anxious...",
#   "emotion_context": {"emotion": "anxious", "cause": "upcoming exam"}
# }
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need help with my anxiety",
    "emotion": "anxious",
    "max_tokens": 128
  }'
```

### Python API

```python
from aura_ml.models.llm_wrapper import AuraLLM
from aura_ml.inference.chatbot import AuraChatbot

# Initialize
llm = AuraLLM(model_path="data/models/llm/llama3_finetuned_final")
llm.load_model()

chatbot = AuraChatbot(llm)

# Set emotion context
chatbot.set_emotion_context("anxious", "exam tomorrow")

# Chat
response = chatbot.chat("How can I calm down?", stream=False)
print(response)
```

---

## 🎓 Training

### Train ECE Model (Emotion-Cause Extraction)

```bash
python scripts/train_ece.py \
  --data-path ./datasets/esconv_dataset \
  --output-dir ./data/models/ece \
  --epochs 3 \
  --batch-size 16
```

### Generate Training Prompts

```bash
python scripts/generate_prompts.py \
  --input ./datasets/esconv_dataset \
  --output ./datasets/llama3_training_data \
  --ece-model ./data/models/ece/ece_roberta_model
```

### Fine-tune LLM

```bash
python scripts/train_llm.py \
  --dataset ./datasets/llama3_training_data \
  --output-dir ./data/models/llm/llama3_finetuned \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 8
```

**Training Configuration (6GB VRAM):**
- Batch size: 2
- Gradient accumulation: 8 steps (effective batch = 16)
- LoRA: r=16, alpha=16
- Optimizer: adamw_8bit
- Precision: BF16 (RTX 40 series)
- Expected time: ~1-2 hours for 3,500 examples

---

## 📁 Project Structure

```
aura-ml/
├── aura_ml/                    # Main Python package
│   ├── config/                 # Configuration management
│   ├── models/                 # Model implementations
│   ├── training/               # Training modules
│   ├── inference/              # Inference modules
│   ├── data/                   # Data processing
│   └── utils/                  # Utilities
│
├── api/                        # FastAPI backend
│   ├── routers/                # API routes
│   ├── models/                 # Request/response schemas
│   └── services/               # Business logic
│
├── cli/                        # Command-line interface
│   └── chat.py                 # Interactive chat CLI
│
├── scripts/                    # Executable scripts
│   ├── train_ece.py
│   ├── train_llm.py
│   └── generate_prompts.py
│
├── tests/                      # Test suite
│   ├── unit/
│   └── integration/
│
├── data/                       # Data directory
│   ├── models/                 # Saved models
│   ├── processed/              # Processed datasets
│   └── outputs/                # Training outputs
│
├── docs/                       # Documentation
├── configs/                    # Configuration files
├── requirements/               # Requirements files
└── setup.py                    # Package setup
```

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Environment
ENV=development

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model Paths
LLM_MODEL_PATH=data/models/llm/llama3_finetuned_final

# GPU
USE_GPU=true
MAX_GPU_MEMORY=6GB

# Logging
LOG_LEVEL=INFO
```

### Model Configuration

Edit `aura_ml/config/model_config.py`:

```python
@dataclass
class LLMConfig:
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    # ... more settings
```

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=aura_ml --cov-report=html

# Specific test
pytest tests/unit/test_llm_wrapper.py
```

### Code Quality

```bash
# Format code
black aura_ml api cli scripts

# Lint
flake8 aura_ml api cli scripts

# Type check
mypy aura_ml
```

---

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -f docker/Dockerfile.api -t aura-api .

# Run container
docker run -p 8000:8000 --gpus all aura-api
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Considerations

1. **Model Loading**: Pre-load models during startup
2. **Caching**: Implement response caching for common queries
3. **Rate Limiting**: Add rate limiting to API endpoints
4. **Monitoring**: Set up logging and metrics collection
5. **Scaling**: Use multiple workers with load balancer
6. **Security**: Add authentication and HTTPS

---

## 📚 Documentation

- [API Documentation](docs/API.md) - REST API reference
- [Training Guide](docs/TRAINING.md) - Model training instructions
- [Inference Guide](docs/INFERENCE.md) - Using the chatbot
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guide
- Tests pass (`pytest`)
- Documentation is updated

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Llama 3.2**: Meta's open-source language model
- **Unsloth**: Efficient fine-tuning library
- **HuggingFace**: Transformers and datasets libraries
- **FastAPI**: Modern web framework
- **ESConv Dataset**: Emotional support conversation data

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/aura-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/aura-ml/discussions)
- **Email**: support@aura-ml.example.com

---

<div align="center">

**Made with ❤️ by the Aura ML Team**

⭐ Star us on GitHub if you find this project useful!

</div>
