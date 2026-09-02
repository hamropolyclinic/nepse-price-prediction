.PHONY: help install test run-train run-collect clean setup

help:
	@echo "NEPSE Price Prediction - Available Commands"
	@echo ""
	@echo "  make setup         - Complete setup (venv, install, .env, db)"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make run-train     - Train ML models"
	@echo "  make run-collect   - Collect data once"
	@echo "  make collect-continuous - Continuous data collection"
	@echo "  make clean         - Remove generated files and cache"
	@echo "  make lint          - Run code linting"
	@echo ""

setup:
	@echo "🔧 Setting up NEPSE Price Prediction System..."
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env && echo "✅ Created .env file"; fi
	@mkdir -p logs models/trained data
	@echo "✅ Setup complete! Run: source venv/bin/activate"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

run-train:
	python train.py

run-collect:
	python -m data_collection.collector --mode once

collect-continuous:
	python -m data_collection.collector --mode continuous --interval 300

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build dist *.egg-info
	@echo "✅ Cleaned up cache and build files"

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
