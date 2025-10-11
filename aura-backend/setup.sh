#!/bin/bash

# Aura Backend Setup Script
# This script sets up the development environment and database

echo "🚀 Setting up Aura Backend..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "📝 Please edit .env file with your database credentials before proceeding."
    echo "   Default DATABASE_URL: postgresql://username:password@localhost:5432/aura_db"
    read -p "Press Enter to continue after updating .env file..."
fi

# Generate Prisma client
echo "🗄️ Generating Prisma client..."
prisma generate

# Check if database exists and run migrations
echo "🔄 Setting up database..."
prisma db push

echo "✅ Setup complete!"
echo ""
echo "To start the development server:"
echo "  source .venv/bin/activate"
echo "  uvicorn main:app --reload"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To view API documentation:"
echo "  Open http://localhost:8000/docs in your browser"
