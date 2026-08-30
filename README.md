# NEPSE Stock Price Prediction System

## Overview

This is an end-to-end machine learning system for predicting NEPSE (Nepal Stock Exchange) stock prices. The system includes:

- **Data Collection**: Real-time and historical data collection from NEPSE API
- **Data Preprocessing**: Cleaning, normalization, and feature engineering
- **Machine Learning Models**: Multiple prediction models (LSTM, ARIMA, Prophet, etc.)
- **Backtesting & Evaluation**: Performance metrics and validation
- **API & Dashboard**: Web interface for predictions and analysis

## Project Structure

```
nepse-price-prediction/
├── data_collection/          # Data fetching and collection
│   ├── nepse_api_client.py   # NEPSE API client
│   ├── collector.py          # Main data collector
│   └── __init__.py
├── database/                 # Database models and connections
│   ├── models.py             # SQLAlchemy models
│   ├── connection.py         # Database session management
│   └── __init__.py
├── preprocessing/            # Data cleaning and feature engineering
│   ├── cleaner.py
│   ├── feature_engineering.py
│   └── __init__.py
├── models/                   # Machine learning models
│   ├── lstm_model.py
│   ├── arima_model.py
│   ├── prophet_model.py
│   └── __init__.py
├── evaluation/               # Model evaluation and backtesting
│   ├── metrics.py
│   ├── backtester.py
│   └── __init__.py
├── api/                      # REST API and web interface
│   ├── app.py
│   ├── routes.py
│   └── __init__.py
├── tests/                    # Unit and integration tests
├── logs/                     # Application logs
├── notebooks/               # Jupyter notebooks for analysis
├── requirements.txt         # Python dependencies
├── .env.example             # Environment configuration template
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- SQLite or PostgreSQL
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/hamropolyclinic/nepse-price-prediction.git
   cd nepse-price-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python -c "from database import init_db; init_db()"
   ```

## Quick Start

### Collect Data Once
```bash
python -m data_collection.collector --mode once
```

### Continuous Data Collection
```bash
python -m data_collection.collector --mode continuous --interval 300
```

### Options
- `--mode`: `once` (single collection) or `continuous` (periodic)
- `--interval`: Time between collections in seconds (default: 300/5 minutes)

## Data Collection

### NEPSEDataCollector

Main data collector class that fetches NEPSE stock data:

```python
from data_collection import NEPSEDataCollector

collector = NEPSEDataCollector()
collector.collect_all()  # Collect all data
collector.continuous_collection(interval_seconds=300)  # Continuous mode
```

### Database Models

**Company** - Stock company information
- symbol: Stock symbol (e.g., 'NABIL')
- name: Company name
- sector: Industry sector
- market_cap: Market capitalization

**DailyPrice** - OHLCV data
- symbol: Stock symbol
- date: Trading date
- open_price, high_price, low_price, close_price
- traded_shares, traded_amount
- price_change, percent_change

**MarketIndex** - NEPSE index data
- index_name: NEPSE, SENSITIVE, etc.
- date: Trading date
- open_value, high_value, low_value, close_value
- total_traded_shares, total_traded_amount

**DataFetchLog** - Collection operation logs
- source: Data source (NEPSE_API, WEB_SCRAPER)
- status: SUCCESS, FAILED, PARTIAL
- records_fetched: Number of records collected
- duration_seconds: Collection time

## API Endpoints

### Available Endpoints (To Be Implemented)
- `GET /api/stocks/today` - Today's price data
- `GET /api/stocks/{symbol}` - Company information
- `GET /api/stocks/{symbol}/history` - Historical data
- `GET /api/index/nepse` - NEPSE index data
- `POST /api/predict/{symbol}` - Price predictions
- `GET /api/analytics/gainers` - Top gaining stocks
- `GET /api/analytics/losers` - Top losing stocks

## Database Setup

### SQLite (Default)
Default SQLite database at `./nepse_data.db`

### PostgreSQL
Set `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/nepse_db
```

## Logging

Logs are stored in `logs/data_collector.log` with configurable levels:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
- Follow PEP 8
- Use type hints
- Document functions with docstrings

## Roadmap

- [x] Step 1: Data Collection Pipeline
  - [x] NEPSE API client
  - [x] Database models
  - [x] Data collector
- [ ] Step 2: Data Preprocessing
  - [ ] Data cleaning
  - [ ] Feature engineering
  - [ ] Normalization
- [ ] Step 3: ML Models
  - [ ] LSTM model
  - [ ] ARIMA model
  - [ ] Prophet model
- [ ] Step 4: Evaluation & Backtesting
  - [ ] Performance metrics
  - [ ] Backtesting framework
- [ ] Step 5: API & Dashboard
  - [ ] FastAPI/Flask API
  - [ ] Web dashboard
  - [ ] Real-time predictions

## Dependencies

Core dependencies:
- **requests**: HTTP requests
- **beautifulsoup4**: Web scraping
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **sqlalchemy**: ORM and database
- **tensorflow/pytorch**: ML frameworks (to be added)

See `requirements.txt` for complete list.

## Configuration

Key environment variables in `.env`:

```bash
# NEPSE API
NEPSE_API_URL=https://nepalstock.com/api/nots/nepse-data/today-price
NEPSE_AUTH_TOKEN=your_token_here

# Database
DATABASE_URL=sqlite:///./nepse_data.db

# Data Collection
DATA_FETCH_INTERVAL=300  # 5 minutes
RETRY_ATTEMPTS=3
RETRY_DELAY=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/nepse_collector.log
```

## Troubleshooting

### Database Connection Error
```
ensure .env has correct DATABASE_URL
init_db() must be called after importing models
```

### API Rate Limiting
```
Increment RETRY_DELAY in .env
Implement exponential backoff in nepse_api_client.py
```

### Missing Logs Directory
```bash
mkdir -p logs
```

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m 'Add your feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Open Pull Request

## License

MIT License - See LICENSE file

## Contact

For questions or suggestions:
- Email: hamropolyclinic@gmail.com
- GitHub Issues: [Report Issues](https://github.com/hamropolyclinic/nepse-price-prediction/issues)

## References

- [NEPSE Official Website](https://nepalstock.com/)
- [NEPSE API Documentation](https://nepalstock.com/api)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Status**: 🚀 Active Development
**Last Updated**: 2026-08-30
