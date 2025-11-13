#!/bin/bash

# Aura ML Pipeline - Quick Start Script
# This script sets up and runs both the backend and Streamlit UI

set -e  # Exit on error

echo "======================================================================"
echo "  AURA ML PIPELINE - QUICK START"
echo "======================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install Streamlit requirements (minimal for UI to work)
echo -e "${BLUE}Installing Streamlit UI requirements...${NC}"
pip install -q streamlit requests pandas plotly soundfile numpy python-dateutil
echo -e "${GREEN}✅ Streamlit requirements installed${NC}"

# Check if we should install full backend requirements
read -p "$(echo -e ${YELLOW}Install full backend ML requirements? This will take several minutes. [y/N]: ${NC})" install_backend

if [[ $install_backend =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Installing backend requirements...${NC}"
    cd aura-backend
    pip install -q -r requirements.txt || {
        echo -e "${YELLOW}⚠️  Some backend packages may have failed to install${NC}"
        echo -e "${YELLOW}   The UI will still work if you have a running backend${NC}"
    }
    cd ..
    echo -e "${GREEN}✅ Backend requirements installed${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping backend installation${NC}"
    echo -e "${YELLOW}   You'll need to install backend requirements manually or use a pre-configured backend${NC}"
fi

echo ""
echo "======================================================================"
echo "  STARTING SERVICES"
echo "======================================================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# Start backend if requested
if [[ $install_backend =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Starting backend server...${NC}"
    cd aura-backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    cd ..
    echo -e "${GREEN}✅ Backend starting on http://localhost:8000${NC}"
    echo -e "${BLUE}Waiting for backend to be ready...${NC}"
    sleep 5
else
    echo -e "${YELLOW}⚠️  Backend not started. Make sure you have a backend running on http://localhost:8000${NC}"
    echo -e "${YELLOW}   Or start it manually: cd aura-backend && uvicorn main:app --reload${NC}"
    sleep 2
fi

# Start Streamlit
echo -e "${BLUE}Starting Streamlit UI...${NC}"
streamlit run streamlit_app.py &
STREAMLIT_PID=$!

echo ""
echo "======================================================================"
echo -e "${GREEN}  ✅ AURA ML PIPELINE IS RUNNING${NC}"
echo "======================================================================"
echo ""
echo -e "${GREEN}Streamlit UI:${NC}  http://localhost:8501"
if [[ $install_backend =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Backend API:${NC}   http://localhost:8000"
    echo -e "${GREEN}API Docs:${NC}      http://localhost:8000/docs"
fi
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for processes
wait
