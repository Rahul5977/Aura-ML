#!/bin/bash

# Quick Start Script for Aura Backend
# This script sets up and starts the Aura backend server

set -e  # Exit on error

echo "🚀 Starting Aura Backend..."
echo ""

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Check if Prisma client is generated
if ! python -c "from prisma import Prisma" 2>/dev/null; then
    echo "🔧 Generating Prisma client..."
    prisma generate
    echo "✅ Prisma client generated"
    echo ""
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Copying from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file - please update it with your credentials"
    else
        echo "❌ .env.example not found. Please create .env manually."
    fi
    echo ""
fi

echo "📊 Checking database connection..."
echo "Make sure Docker is running and databases are started:"
echo "  docker-compose up -d db neo4j"
echo ""

# Start the server
echo "🎯 Starting Uvicorn server..."
echo "Backend will be available at: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
