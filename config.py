"""
Configuration settings for NEPSE stock price prediction
"""
import os
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "trained"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nepse.db")

# Data configuration
DATA_CONFIG = {
    'lookback_period': 60,  # Number of days for LSTM/GRU sequence
    'train_test_split': 0.2,  # 80% train, 20% test
    'normalization_method': 'minmax',  # 'minmax' or 'standard'
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'indicators': [
        'sma_10', 'sma_20', 'sma_50',
        'ema_12', 'ema_26',
        'rsi', 'macd', 'stochastic',
        'atr', 'bollinger',
        'obv', 'returns', 'volatility'
    ],
    'lagged_features': [1, 2, 3, 5],  # Lags in days
    'rolling_windows': [5, 10, 20],  # Rolling window sizes
}

# Neural Network configuration
NN_CONFIG = {
    'lstm': {
        'lookback': 60,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.1,
        'learning_rate': 0.001,
    },
    'gru': {
        'lookback': 60,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.1,
        'learning_rate': 0.001,
    },
    'ensemble': {
        'lstm_weight': 0.5,
        'gru_weight': 0.5,
    }
}

# Traditional ML configuration
ML_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
    },
    'adaboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
    },
    'svr': {
        'kernel': 'rbf',
        'C': 100.0,
        'gamma': 'scale',
    },
    'ridge': {
        'alpha': 1.0,
    },
    'lasso': {
        'alpha': 0.1,
    }
}

# Model paths
MODEL_PATHS = {
    'lstm': MODELS_DIR / "lstm_model.h5",
    'gru': MODELS_DIR / "gru_model.h5",
    'scaler_lstm': MODELS_DIR / "scaler_lstm.pkl",
    'scaler_gru': MODELS_DIR / "scaler_gru.pkl",
    'traditional_ml': MODELS_DIR / "traditional_ml_models.pkl",
    'scaler_ml': MODELS_DIR / "scaler_ml.pkl",
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'detailed',
            'level': 'INFO',
            'filename': LOGS_DIR / 'app.log',
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    }
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
}

# Stocks to track
STOCKS_TO_TRACK = [
    'NABIL', 'SCB', 'SBI', 'NEPSE', 'CCBL', 'HHPL', 'ADBL', 'EBL'
]

# Prediction confidence threshold
CONFIDENCE_THRESHOLD = 0.7  # R² score threshold

# Number of days to predict ahead
PREDICTION_DAYS_AHEAD = 1  # Predict next day's price
