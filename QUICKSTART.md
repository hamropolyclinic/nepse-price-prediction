# NEPSE Price Prediction - Quick Start Guide

## 🚀 Getting Started (5 minutes)

### Option 1: Automated Setup (Recommended)
```bash
chmod +x run.sh
./run.sh
```
This will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Set up .env file
- ✅ Run tests to verify installation

### Option 2: Manual Setup Using Make
```bash
make setup
make test
```

### Option 3: Manual Step-by-Step
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env

# 4. Create directories
mkdir -p logs models/trained data

# 5. Run tests
pytest tests/ -v
```

---

## 📊 Available Commands

### Training Models
```bash
python train.py
```
Trains LSTM, GRU, and traditional ML models on NEPSE stock data.

### Data Collection
```bash
# Collect once
python -m data_collection.collector --mode once

# Continuous collection (every 5 minutes)
python -m data_collection.collector --mode continuous --interval 300
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Using Make Commands
```bash
make install          # Install dependencies
make test             # Run tests
make run-train        # Train models
make run-collect      # Collect data once
make collect-continuous  # Continuous data collection
make clean            # Clean cache and build files
make lint             # Check code style
```

---

## 🐳 Docker Deployment

### Build and Run with Docker
```bash
# Build image
docker build -t nepse-predictor .

# Run container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/logs:/app/logs \
           -v $(pwd)/models:/app/models \
           nepse-predictor
```

### Using Docker Compose (with PostgreSQL)
```bash
# Start both app and database
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

---

## ⚙️ Configuration

### Edit .env File
```bash
nano .env  # or use your preferred editor
```

**Key settings:**
- `DATABASE_URL` - SQLite or PostgreSQL connection string
- `LOG_LEVEL` - INFO, DEBUG, WARNING, ERROR
- `EPOCHS` - Number of training epochs (default: 50)
- `BATCH_SIZE` - Training batch size (default: 32)

---

## 📁 Project Structure

```
nepse-price-prediction/
├── data_collection/       # Fetch NEPSE data
├── preprocessing/         # Clean & engineer features
├── models/               # LSTM, GRU, traditional ML
├── utils/                # Helper utilities
├── tests/                # Unit & integration tests
├── logs/                 # Application logs
├── data/                 # Training data
├── models/trained/       # Saved trained models
├── train.py              # Main training script
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── Dockerfile            # Docker build config
├── docker-compose.yml    # Multi-container setup
├── Makefile              # Task automation
├── run.sh                # Startup script
└── README.md             # Full documentation
```

---

## 🧪 Verify Installation

Run this to verify everything is working:
```bash
python -c "import pandas, numpy, sklearn, tensorflow; print('✅ All dependencies installed!')"
pytest tests/ -v --tb=short
```

Expected output:
```
✅ All dependencies installed!
tests/test_config.py ............... PASSED
tests/test_train.py ................ PASSED
tests/test_integration.py .......... PASSED
```

---

## 📝 Typical Workflow

### 1. First Time Setup
```bash
./run.sh
# or
make setup
```

### 2. Collect Data
```bash
python -m data_collection.collector --mode once
```

### 3. Train Models
```bash
python train.py
```

### 4. Check Results
```bash
# View logs
tail -f logs/app.log

# Check trained models
ls -lah models/trained/
```

### 5. Run Tests (Always)
```bash
pytest tests/ -v
```

---

## 🐛 Troubleshooting

### "Module not found" Error
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### Database Connection Error
```bash
# Check .env file
cat .env

# Ensure DATABASE_URL is correct
# SQLite: sqlite:///./nepse.db
# PostgreSQL: postgresql://user:pass@localhost:5432/nepse_db
```

### Tests Failing
```bash
# Clean cache and reinstall
make clean
pip install -r requirements.txt --force-reinstall
pytest tests/ -v
```

### Permission Denied on run.sh
```bash
chmod +x run.sh
./run.sh
```

---

## 📚 What's Inside

- **50+ Unit Tests**: Configuration, training pipeline, data handling
- **Integration Tests**: End-to-end pipeline validation
- **3 ML Models**: LSTM, GRU, Ensemble + Traditional ML
- **Feature Engineering**: Technical indicators, lagged features, rolling windows
- **Data Validation**: Cleaning, normalization, quality checks
- **Logging**: Comprehensive application logs
- **Docker Support**: Containerized deployment ready

---

## ✅ Health Check

After running setup, you should see:

```
✅ Python version: 3.10+
✅ Virtual environment activated
✅ Dependencies installed
✅ Directories created
✅ .env file created
✅ Tests passed
```

---

## 🎯 Next Steps

1. **Edit .env** with your configuration
2. **Collect data** using data collection module
3. **Train models** using train.py
4. **Review logs** in logs/app.log
5. **Check results** in models/trained/ directory

---

## 📞 Support

- Check README.md for detailed documentation
- Review test files in `tests/` for usage examples
- Check logs/ directory for debugging information
- Visit: https://github.com/hamropolyclinic/nepse-price-prediction

---

**Status**: ✅ Ready to Run
**Last Updated**: 2026-09-02
