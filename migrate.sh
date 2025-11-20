#!/bin/bash
# Migration script to organize trained models into new structure

set -e

echo "🔄 Migrating Aura ML to production structure..."
echo ""

PROJECT_ROOT="/home/rishi/Desktop/Aura-ML"
cd "$PROJECT_ROOT"

# Create data directories if they don't exist
mkdir -p data/models/{ece,llm}
mkdir -p data/processed
mkdir -p data/outputs

echo "📦 Step 1: Migrating trained models..."

# Migrate ECE model if it exists
if [ -d "fine-tuining/ece_model_output" ]; then
    echo "  • Copying ECE model..."
    cp -r fine-tuining/ece_model_output data/models/ece/ece_roberta_model
    echo "    ✅ ECE model migrated"
fi

# Migrate LLM model if it exists
if [ -d "fine-tuining/llama3_finetuned_final" ]; then
    echo "  • Copying LLM model..."
    cp -r fine-tuining/llama3_finetuned_final data/models/llm/
    echo "    ✅ LLM model migrated"
fi

echo ""
echo "📊 Step 2: Migrating datasets..."

# Migrate processed datasets
if [ -d "datasets/llama3_training_data" ]; then
    echo "  • Copying training data..."
    cp -r datasets/llama3_training_data data/processed/
    echo "    ✅ Training data migrated"
fi

echo ""
echo "📝 Step 3: Creating environment file..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✅ Created .env file (please configure it)"
else
    echo "  ⚠️  .env already exists, skipping"
fi

echo ""
echo "🎨 Step 4: Setting up package..."

# Install package in development mode
if [ -d ".venv" ]; then
    echo "  • Using existing virtual environment"
    source .venv/bin/activate
elif [ -d "fine-tuining/.venv" ]; then
    echo "  • Using fine-tuining virtual environment"
    source fine-tuining/.venv/bin/activate
else
    echo "  • Creating new virtual environment"
    python -m venv venv
    source venv/bin/activate
fi

# Install package
echo "  • Installing Aura ML package..."
pip install -e . > /dev/null 2>&1
echo "    ✅ Package installed"

echo ""
echo "✅ Migration complete!"
echo ""
echo "📁 New structure:"
echo "  • Models: data/models/"
echo "  • Datasets: data/processed/"
echo "  • Source code: aura_ml/"
echo "  • API: api/"
echo "  • CLI: cli/"
echo ""
echo "🚀 Quick start:"
echo "  • Chat CLI: python cli/chat.py"
echo "  • Start API: uvicorn api.main:app --reload"
echo "  • Run tests: pytest tests/"
echo ""
echo "📚 Documentation: README.md"
echo ""
