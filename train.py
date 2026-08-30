"""
Training Pipeline for NEPSE stock price prediction models
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from datetime import datetime
import sys

from utils.data_loader import DataLoader
from preprocessing.cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineering
from models.neural_networks import LSTMModel, GRUModel, EnsembleModel
from models.traditional_ml import TraditionalMLModels
from utils.model_manager import ModelManager
from config import (
    DATA_CONFIG, NN_CONFIG, ML_CONFIG, 
    FEATURE_CONFIG, LOGS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete training pipeline"""
    
    def __init__(self, symbol: str = 'NABIL'):
        """
        Initialize training pipeline
        
        Args:
            symbol: Stock symbol to train on
        """
        self.symbol = symbol
        self.data_loader = DataLoader()
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineering()
        self.model_manager = ModelManager()
        
        self.raw_data = None
        self.cleaned_data = None
        self.featured_data = None
        self.X = None
        self.y = None
        
        self.lstm_model = None
        self.gru_model = None
        self.ml_models = None
        self.ensemble_model = None
        
        self.results = {}
        
        logger.info(f"Training Pipeline initialized for {symbol}")

    def fetch_data(self, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> bool:
        """
        Fetch stock data
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Fetching data for {self.symbol}...")
            
            self.raw_data = self.data_loader.fetch_from_yfinance(
                self.symbol, start_date, end_date
            )
            
            if self.raw_data is None or len(self.raw_data) == 0:
                logger.error("Failed to fetch data")
                return False
            
            logger.info(f"Successfully fetched {len(self.raw_data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return False

    def clean_data(self) -> bool:
        """
        Clean the fetched data
        
        Returns:
            True if successful
        """
        try:
            if self.raw_data is None:
                logger.error("No raw data to clean")
                return False
            
            logger.info("Cleaning data...")
            
            self.cleaned_data = self.data_loader.clean_data(self.symbol)
            
            if self.cleaned_data is None or len(self.cleaned_data) == 0:
                logger.error("Data cleaning failed")
                return False
            
            logger.info(f"Data cleaned successfully. Records: {len(self.cleaned_data)}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return False

    def engineer_features(self) -> bool:
        """
        Create features from cleaned data
        
        Returns:
            True if successful
        """
        try:
            if self.cleaned_data is None:
                logger.error("No cleaned data for feature engineering")
                return False
            
            logger.info("Engineering features...")
            
            # Extract price data
            price_data = self.cleaned_data['Close'].values
            
            # Create features
            self.featured_data = self.feature_engineer.create_all_features(price_data)
            
            if self.featured_data is None or len(self.featured_data) == 0:
                logger.error("Feature engineering failed")
                return False
            
            logger.info(f"Features created successfully. Shape: {self.featured_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return False

    def prepare_data_for_nn(self) -> bool:
        """
        Prepare data for neural network models
        
        Returns:
            True if successful
        """
        try:
            if self.cleaned_data is None:
                logger.error("No cleaned data to prepare")
                return False
            
            logger.info("Preparing data for neural networks...")
            
            price_data = self.cleaned_data['Close'].values
            
            # For NN models, we'll use raw price data with lookback
            lookback = DATA_CONFIG['lookback_period']
            
            if len(price_data) < lookback:
                logger.error(f"Not enough data. Need at least {lookback} records")
                return False
            
            logger.info(f"Data prepared. Total samples: {len(price_data)}")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return False

    def prepare_data_for_ml(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data for traditional ML models
        
        Returns:
            Tuple of (success, X, y)
        """
        try:
            if self.featured_data is None:
                logger.error("No featured data to prepare")
                return False, None, None
            
            logger.info("Preparing data for traditional ML models...")
            
            # Use featured data as X
            X = self.featured_data
            
            # Use next day's price as y
            price_data = self.cleaned_data['Close'].values
            y = np.roll(price_data, -1)[:-1]  # Shift prices up and remove last
            
            # Keep only matching indices
            if len(X) > len(y):
                X = X[:len(y)]
            elif len(y) > len(X):
                y = y[:len(X)]
            
            logger.info(f"ML data prepared. X shape: {X.shape}, y shape: {y.shape}")
            return True, X, y
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {str(e)}")
            return False, None, None

    def train_lstm(self) -> bool:
        """
        Train LSTM model
        
        Returns:
            True if successful
        """
        try:
            if self.cleaned_data is None:
                logger.error("No cleaned data for LSTM training")
                return False
            
            logger.info("Training LSTM model...")
            
            # Initialize LSTM model
            self.lstm_model = LSTMModel(
                lookback=NN_CONFIG['lstm']['lookback'],
                output_size=1
            )
            
            # Prepare data
            price_data = self.cleaned_data['Close'].values
            X_train, X_test, y_train, y_test = self.lstm_model.prepare_data(
                price_data, test_size=DATA_CONFIG['train_test_split']
            )
            
            # Build model
            self.lstm_model.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Train
            self.lstm_model.train(
                X_train, y_train,
                epochs=NN_CONFIG['lstm']['epochs'],
                batch_size=NN_CONFIG['lstm']['batch_size'],
                validation_split=NN_CONFIG['lstm']['validation_split']
            )
            
            # Evaluate
            metrics = self.lstm_model.evaluate(X_test, y_test)
            self.results['lstm'] = metrics
            
            logger.info(f"LSTM trained successfully. RMSE: {metrics['rmse']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
            return False

    def train_gru(self) -> bool:
        """
        Train GRU model
        
        Returns:
            True if successful
        """
        try:
            if self.cleaned_data is None:
                logger.error("No cleaned data for GRU training")
                return False
            
            logger.info("Training GRU model...")
            
            # Initialize GRU model
            self.gru_model = GRUModel(
                lookback=NN_CONFIG['gru']['lookback'],
                output_size=1
            )
            
            # Prepare data
            price_data = self.cleaned_data['Close'].values
            X_train, X_test, y_train, y_test = self.gru_model.prepare_data(
                price_data, test_size=DATA_CONFIG['train_test_split']
            )
            
            # Build model
            self.gru_model.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Train
            self.gru_model.train(
                X_train, y_train,
                epochs=NN_CONFIG['gru']['epochs'],
                batch_size=NN_CONFIG['gru']['batch_size'],
                validation_split=NN_CONFIG['gru']['validation_split']
            )
            
            # Evaluate
            metrics = self.gru_model.evaluate(X_test, y_test)
            self.results['gru'] = metrics
            
            logger.info(f"GRU trained successfully. RMSE: {metrics['rmse']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training GRU: {str(e)}")
            return False

    def train_traditional_ml(self) -> bool:
        """
        Train traditional ML models
        
        Returns:
            True if successful
        """
        try:
            logger.info("Training traditional ML models...")
            
            # Prepare data
            success, X, y = self.prepare_data_for_ml()
            if not success:
                logger.error("Failed to prepare data for ML")
                return False
            
            # Initialize ML models
            self.ml_models = TraditionalMLModels()
            
            # Prepare and split data
            X_train, X_test, y_train, y_test = self.ml_models.prepare_data(
                X, y, test_size=DATA_CONFIG['train_test_split']
            )
            
            # Train all models
            self.ml_models.train_all_models(X_train, y_train)
            
            # Evaluate all models
            results_df = self.ml_models.evaluate_all_models(X_test, y_test)
            
            # Store results
            for idx, row in results_df.iterrows():
                self.results[row['model']] = {
                    'mse': row['mse'],
                    'rmse': row['rmse'],
                    'mae': row['mae'],
                    'r2_score': row['r2_score']
                }
            
            logger.info(f"ML models trained. Best model: {self.ml_models.best_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
            return False

    def train_ensemble(self) -> bool:
        """
        Create and evaluate ensemble model
        
        Returns:
            True if successful
        """
        try:
            if self.lstm_model is None or self.gru_model is None:
                logger.error("LSTM and GRU models required for ensemble")
                return False
            
            logger.info("Creating ensemble model...")
            
            # Create ensemble
            weights = (
                NN_CONFIG['ensemble']['lstm_weight'],
                NN_CONFIG['ensemble']['gru_weight']
            )
            self.ensemble_model = EnsembleModel(self.lstm_model, self.gru_model, weights)
            
            # Evaluate ensemble (using same test data as LSTM)
            price_data = self.cleaned_data['Close'].values
            _, X_test, _, y_test = self.lstm_model.prepare_data(
                price_data, test_size=DATA_CONFIG['train_test_split']
            )
            
            metrics = self.ensemble_model.evaluate(X_test, y_test)
            self.results['ensemble'] = metrics
            
            logger.info(f"Ensemble model created. RMSE: {metrics['rmse']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            return False

    def save_models(self) -> bool:
        """
        Save all trained models
        
        Returns:
            True if successful
        """
        try:
            logger.info("Saving trained models...")
            
            if self.lstm_model:
                self.model_manager.save_lstm_model(
                    self.lstm_model, "lstm", self.results.get('lstm')
                )
            
            if self.gru_model:
                self.model_manager.save_gru_model(
                    self.gru_model, "gru", self.results.get('gru')
                )
            
            if self.ml_models:
                self.model_manager.save_traditional_ml_models(
                    self.ml_models, "traditional_ml"
                )
            
            logger.info("All models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False

    def print_results(self):
        """Print training results summary"""
        logger.info("=" * 80)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("=" * 80)
        
        for model_name, metrics in self.results.items():
            logger.info(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                if metric_name != 'model':
                    logger.info(f"  {metric_name}: {value:.4f}")
        
        logger.info("=" * 80)

    def run_complete_pipeline(self, start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> bool:
        """
        Run the complete training pipeline
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            True if all steps successful
        """
        try:
            logger.info("Starting complete training pipeline...")
            start_time = datetime.now()
            
            # Step 1: Fetch data
            if not self.fetch_data(start_date, end_date):
                logger.error("Pipeline failed at data fetching")
                return False
            
            # Step 2: Clean data
            if not self.clean_data():
                logger.error("Pipeline failed at data cleaning")
                return False
            
            # Step 3: Engineer features
            if not self.engineer_features():
                logger.error("Pipeline failed at feature engineering")
                return False
            
            # Step 4: Prepare data for NN
            if not self.prepare_data_for_nn():
                logger.error("Pipeline failed at NN data preparation")
                return False
            
            # Step 5: Train LSTM
            if not self.train_lstm():
                logger.warning("LSTM training failed, continuing...")
            
            # Step 6: Train GRU
            if not self.train_gru():
                logger.warning("GRU training failed, continuing...")
            
            # Step 7: Train Ensemble
            if self.lstm_model and self.gru_model:
                if not self.train_ensemble():
                    logger.warning("Ensemble training failed, continuing...")
            
            # Step 8: Train Traditional ML
            if not self.train_traditional_ml():
                logger.warning("ML training failed, continuing...")
            
            # Step 9: Save models
            if not self.save_models():
                logger.warning("Failed to save some models")
            
            # Print results
            self.print_results()
            
            elapsed_time = datetime.now() - start_time
            logger.info(f"Pipeline completed successfully in {elapsed_time}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    pipeline = TrainingPipeline(symbol='NABIL')
    pipeline.run_complete_pipeline(
        start_date='2023-01-01',
        end_date='2024-08-30'
    )
