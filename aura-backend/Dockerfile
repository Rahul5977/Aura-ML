# Dockerfile for FastAPI backend with Prisma and ML models
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Hugging Face cache
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models to speed up container startup
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('openai/whisper-tiny'); \
    WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')" && \
    python -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification; \
    Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'); \
    Wav2Vec2ForSequenceClassification.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')"

# Copy application code
COPY . .

# Generate Prisma client
RUN prisma generate

EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
