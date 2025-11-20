#!/bin/bash
# Cleanup script - Keep only new production structure

set -e

echo "🧹 Cleaning up - Keeping only production structure..."
echo ""

PROJECT_ROOT="/home/rishi/Desktop/Aura-ML"
cd "$PROJECT_ROOT"

# Create archive directory for old files
ARCHIVE_DIR="archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "📦 Step 1: Archiving old structure..."

# Archive fine-tuining directory (but keep models for migration if needed)
if [ -d "fine-tuining" ]; then
    echo "  • Archiving fine-tuining/ directory..."
    mv fine-tuining "$ARCHIVE_DIR/"
    echo "    ✅ Archived to $ARCHIVE_DIR/fine-tuining/"
fi

# Archive old datasets directory (raw data preserved)
if [ -d "datasets" ]; then
    echo "  • Archiving datasets/ directory..."
    mv datasets "$ARCHIVE_DIR/"
    echo "    ✅ Archived to $ARCHIVE_DIR/datasets/"
fi

# Archive old aura-backend if exists
if [ -d "aura-backend" ]; then
    echo "  • Archiving aura-backend/ directory..."
    mv aura-backend "$ARCHIVE_DIR/"
    echo "    ✅ Archived to $ARCHIVE_DIR/aura-backend/"
fi

# Archive scattered files in root
echo ""
echo "📄 Step 2: Archiving scattered root files..."

FILES_TO_ARCHIVE=(
    "demo_heuristics.py"
    "prepare_data.py"
    "test_prepare_data.py"
    "req.txt"
    "requirements_data_prep.txt"
    "requirements_ece.txt"
)

for file in "${FILES_TO_ARCHIVE[@]}"; do
    if [ -f "$file" ]; then
        echo "  • Archiving $file..."
        mv "$file" "$ARCHIVE_DIR/"
    fi
done

# Archive old notebooks if scattered in root
if [ -f "Video_Pipeline_Testing.ipynb" ]; then
    mv Video_Pipeline_Testing.ipynb "$ARCHIVE_DIR/"
fi
if [ -f "Week10_Hyper_Contextual_Dataset.ipynb" ]; then
    mv Week10_Hyper_Contextual_Dataset.ipynb "$ARCHIVE_DIR/"
fi
if [ -f "Week9_ECE_Data_Pipeline.ipynb" ]; then
    mv Week9_ECE_Data_Pipeline.ipynb "$ARCHIVE_DIR/"
fi
if [ -f "Week9_ECE_Model_Training.ipynb" ]; then
    mv Week9_ECE_Model_Training.ipynb "$ARCHIVE_DIR/"
fi

echo "    ✅ Root files archived"

echo ""
echo "🔄 Step 3: Running migration to new structure..."

# Run the migration script to copy models to new structure
if [ -d "$ARCHIVE_DIR/fine-tuining" ]; then
    # Copy models from archive to new structure
    mkdir -p data/models/{ece,llm}
    
    if [ -d "$ARCHIVE_DIR/fine-tuining/ece_model_output" ]; then
        echo "  • Copying ECE model to data/models/ece/..."
        cp -r "$ARCHIVE_DIR/fine-tuining/ece_model_output" data/models/ece/ece_roberta_model
        echo "    ✅ ECE model migrated"
    fi
    
    if [ -d "$ARCHIVE_DIR/fine-tuining/llama3_finetuned_final" ]; then
        echo "  • Copying LLM model to data/models/llm/..."
        cp -r "$ARCHIVE_DIR/fine-tuining/llama3_finetuned_final" data/models/llm/
        echo "    ✅ LLM model migrated"
    fi
    
    # Copy datasets
    if [ -d "$ARCHIVE_DIR/datasets/llama3_training_data" ]; then
        echo "  • Copying training data to data/processed/..."
        mkdir -p data/processed
        cp -r "$ARCHIVE_DIR/datasets/llama3_training_data" data/processed/
        echo "    ✅ Training data migrated"
    fi
fi

echo ""
echo "📝 Step 4: Creating .env file..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    
    # Update paths in .env to point to new structure
    if [ -d "data/models/llm/llama3_finetuned_final" ]; then
        sed -i 's|LLM_MODEL_PATH=.*|LLM_MODEL_PATH=data/models/llm/llama3_finetuned_final|' .env
    fi
    if [ -d "data/models/ece/ece_roberta_model" ]; then
        sed -i 's|ECE_MODEL_PATH=.*|ECE_MODEL_PATH=data/models/ece/ece_roberta_model|' .env
    fi
    
    echo "  ✅ Created .env with correct paths"
else
    echo "  ⚠️  .env already exists, skipping"
fi

echo ""
echo "🎨 Step 5: Final structure verification..."

echo ""
echo "📁 Production structure:"
ls -d */ 2>/dev/null | grep -v "$ARCHIVE_DIR" | sort

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📦 Old files archived in: $ARCHIVE_DIR/"
echo "   (You can safely delete this directory once you verify everything works)"
echo ""
echo "🎯 Your new clean structure:"
echo "   • aura_ml/     - Core Python package"
echo "   • api/         - FastAPI backend"
echo "   • cli/         - Command-line interface"
echo "   • data/        - Models and datasets"
echo "   • docs/        - Documentation"
echo "   • scripts/     - Executable scripts"
echo "   • tests/       - Test suite"
echo ""
echo "🚀 Quick start:"
echo "   1. python cli/chat.py --test"
echo "   2. uvicorn api.main:app --reload"
echo ""
echo "📚 See README.md for full documentation"
echo ""
