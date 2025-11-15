#!/bin/bash

# ============================================================================
# Week 8 - Scene Analysis Pipeline Quick Start
# Installation and Setup Script
# ============================================================================

echo "=========================================================================="
echo "🎬 AURA WEEK 8 - SCENE ANALYSIS PIPELINE SETUP"
echo "=========================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Not running in a virtual environment${NC}"
    echo "   It's recommended to use a virtual environment"
    echo ""
    echo "   Create one with:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate  # On Unix/macOS"
    echo "   .\\venv\\Scripts\\activate  # On Windows"
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ Running in virtual environment: $VIRTUAL_ENV${NC}"
fi

echo ""
echo "=========================================================================="
echo "📦 STEP 1: INSTALLING CORE DEPENDENCIES"
echo "=========================================================================="
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install numpy first (to avoid conflicts)
echo ""
echo "Installing numpy (compatible version)..."
pip install "numpy>=1.24.0,<2.0.0"

# Check for CUDA availability
echo ""
echo "🔍 Checking for CUDA/GPU support..."
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}✅ CUDA detected: Version $cuda_version${NC}"
    echo ""
    echo "Select PyTorch installation:"
    echo "1) CUDA 11.8"
    echo "2) CUDA 12.1"
    echo "3) CPU only"
    read -p "Enter choice (1-3): " cuda_choice
    
    case $cuda_choice in
        1)
            echo "Installing PyTorch for CUDA 11.8..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
            ;;
        2)
            echo "Installing PyTorch for CUDA 12.1..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            ;;
        *)
            echo "Installing PyTorch for CPU..."
            pip install torch torchvision
            ;;
    esac
else
    echo -e "${YELLOW}⚠️  No CUDA detected, installing CPU version${NC}"
    pip install torch torchvision
fi

echo ""
echo "=========================================================================="
echo "📦 STEP 2: INSTALLING VIDEO ANALYSIS DEPENDENCIES"
echo "=========================================================================="
echo ""

# Install requirements
if [ -f "requirements_video.txt" ]; then
    echo "Installing from requirements_video.txt..."
    pip install -r requirements_video.txt
else
    echo "requirements_video.txt not found, installing manually..."
    
    # Core dependencies
    pip install transformers>=4.35.0
    pip install accelerate>=0.24.0
    pip install opencv-python>=4.8.0
    pip install Pillow>=10.0.0
    pip install scipy>=1.11.0
    pip install tqdm>=4.66.0
    
    # Optional: Face analysis
    pip install facenet-pytorch>=2.5.3
    
    # Optional: 8-bit quantization
    echo ""
    read -p "Install bitsandbytes for 8-bit model loading? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install bitsandbytes>=0.41.0
    fi
fi

echo ""
echo "=========================================================================="
echo "📦 STEP 3: VERIFYING INSTALLATION"
echo "=========================================================================="
echo ""

# Test imports
python3 << EOF
import sys

print("Testing imports...")
errors = []

try:
    import cv2
    print("✅ opencv-python: ", cv2.__version__)
except Exception as e:
    print("❌ opencv-python failed:", e)
    errors.append("opencv-python")

try:
    import torch
    print("✅ torch: ", torch.__version__)
    print("   CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("   CUDA device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("❌ torch failed:", e)
    errors.append("torch")

try:
    import transformers
    print("✅ transformers: ", transformers.__version__)
except Exception as e:
    print("❌ transformers failed:", e)
    errors.append("transformers")

try:
    from PIL import Image
    print("✅ Pillow: OK")
except Exception as e:
    print("❌ Pillow failed:", e)
    errors.append("Pillow")

try:
    import numpy as np
    print("✅ numpy: ", np.__version__)
except Exception as e:
    print("❌ numpy failed:", e)
    errors.append("numpy")

if errors:
    print("\n❌ Some imports failed:", ", ".join(errors))
    sys.exit(1)
else:
    print("\n✅ All core dependencies installed successfully!")
    sys.exit(0)
EOF

install_status=$?

if [ $install_status -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Installation verification failed${NC}"
    echo "   Please check the errors above and reinstall failed packages"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "📥 STEP 4: DOWNLOADING LLAVA MODEL (OPTIONAL)"
echo "=========================================================================="
echo ""

echo "The LLaVA model (~13GB) will be downloaded on first use."
echo "You can pre-download it now to save time later."
echo ""
read -p "Download LLaVA model now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Downloading LLaVA model..."
    echo "This may take 10-30 minutes depending on your internet speed..."
    echo ""
    
    python3 << EOF
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

print("📥 Downloading LLaVA model...")
model_name = "llava-hf/llava-1.5-7b-hf"

try:
    processor = LlavaNextProcessor.from_pretrained(model_name)
    print("✅ Processor downloaded")
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    print("✅ Model downloaded")
    
    print("\n✅ LLaVA model ready for use!")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("   The model will be downloaded automatically on first use.")
EOF
else
    echo "Skipping model download. Model will be downloaded on first use."
fi

echo ""
echo "=========================================================================="
echo "✅ INSTALLATION COMPLETE!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test the installation:"
echo "   python test_scene_captioner.py --test keyframes"
echo ""
echo "2. Run the scene captioner:"
echo "   python video/scene_captioner.py your_video.mp4"
echo ""
echo "3. Or use it in Python:"
cat << 'EOF'
   from video.scene_captioner import analyze_video_scene
   
   results = analyze_video_scene('video.mp4', interval_sec=1.0)
   for r in results:
       print(f"[{r['formatted_time']}] {r['caption']}")
EOF

echo ""
echo "📚 For more information, see:"
echo "   - README_WEEK8.md"
echo "   - video/scene_captioner.py (documentation)"
echo ""
echo "=========================================================================="
