#!/bin/bash

# NEPSE Price Prediction - Startup Script
# This script sets up and runs the application

set -e

echo "🚀 NEPSE Price Prediction System Launcher"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models/trained data

# Setup .env file
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "   ℹ️  Please edit .env with your configuration"
fi

# Run tests
echo ""
echo "🧪 Running tests..."
pytest tests/ -v --tb=short 2>&1 | head -50

# Success message
echo ""
echo "✅ Setup complete!"
echo ""
echo "Available commands:"
echo "  python train.py                    - Train models"
echo "  python -m data_collection.collector --mode once  - Collect data"
echo "  pytest tests/ -v                   - Run tests"
echo ""
echo "For more info, see README.md"
