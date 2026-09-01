"""
Machine Learning Models for NEPSE stock price prediction
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM model for stock price prediction"""
    
    def __init__(self, lookback: int = 60, output_size: int = 1):
        """
        Initialize LSTM model
        
        Args:
            lookback: Number of previous timesteps to use as input
            output_size: Number of output steps
        """
        self.lookback = lookback
        self.output_size = output_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: 1D input data array (shape: [n_samples]) or 2D (n_samples, 1)
            
        Returns:
            Tuple of (X sequences, y targets)
            - X shape: (n_sequences, lookback, 1)
            - y shape: (n_sequences, 1)
        """
        # Ensure we have a 1D array
        arr = np.array(data).reshape(-1)
        X, y = [], []
        for i in range(len(arr) - self.lookback):
            seq_x = arr[i:i + self.lookback].reshape(self.lookback, 1)
            seq_y = arr[i + self.lookback]
            X.append(seq_x)
            y.append(seq_y)
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        return X, y

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM neural network
        
        Args:
            input_shape: Shape of input data (lookback, features)
            
        Returns:
            Compiled Keras model
        """
        try:
            model = Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=input_shape),
                layers.Dropout(0.2),
                
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.2),
                
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.2),
                
                layers.Dense(16, activation='relu'),
                layers.Dense(self.output_size)
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            self.model = model
            logger.info("LSTM model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise

    def prepare_data(self, data: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and scale data
        
        Args:
            data: Input data (1D or 2D array)
            test_size: Test set size ratio
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Scale data to [0,1]
            arr = np.array(data).reshape(-1, 1)
            scaled = self.scaler.fit_transform(arr)
            # create sequences from scaled flattened values
            X, y = self.create_sequences(scaled.flatten())
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            logger.info(f"Data prepared: Train size: {len(X_train)}, Test size: {len(X_test)}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.1) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features (expected shape: samples, timesteps, features)
            y_train: Training targets (samples, 1)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training history dictionary
        """
        try:
            # Defensive shape handling
            if X_train.ndim == 2:
                # reshape to (samples, timesteps, 1)
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)

            if self.model is None:
                self.build_model((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            ]
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return self.history.history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X_test: Test features (samples, timesteps, features) or (samples, timesteps)
            
        Returns:
            Predictions (inverse-transformed to original scale)
        """
        try:
            if X_test.ndim == 2:
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            predictions = self.model.predict(X_test)
            # predictions shape expected (n_samples, 1)
            predictions = self.scaler.inverse_transform(predictions)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            y_pred = self.predict(X_test)
            y_test_scaled = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mse = mean_squared_error(y_test_scaled, y_pred)
            mae = mean_absolute_error(y_test_scaled, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_scaled, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            logger.info(f"Model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")


class GRUModel:
    """GRU model for stock price prediction"""
    
    def __init__(self, lookback: int = 60, output_size: int = 1):
        """
        Initialize GRU model
        """
        self.lookback = lookback
        self.output_size = output_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU training"""
        arr = np.array(data).reshape(-1)
        X, y = [], []
        for i in range(len(arr) - self.lookback):
            X.append(arr[i:i + self.lookback].reshape(self.lookback, 1))
            y.append(arr[i + self.lookback])
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        return X, y

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build GRU neural network"""
        try:
            model = Sequential([
                layers.GRU(128, return_sequences=True, input_shape=input_shape),
                layers.Dropout(0.2),
                
                layers.GRU(64, return_sequences=True),
                layers.Dropout(0.2),
                
                layers.GRU(32, return_sequences=False),
                layers.Dropout(0.2),
                
                layers.Dense(16, activation='relu'),
                layers.Dense(self.output_size)
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            self.model = model
            logger.info("GRU model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building GRU model: {str(e)}")
            raise

    def prepare_data(self, data: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and scale data"""
        try:
            arr = np.array(data).reshape(-1, 1)
            scaled = self.scaler.fit_transform(arr)
            X, y = self.create_sequences(scaled.flatten())
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            logger.info(f"Data prepared: Train size: {len(X_train)}, Test size: {len(X_test)}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.1) -> Dict:
        """Train the GRU model"""
        try:
            if X_train.ndim == 2:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)

            if self.model is None:
                self.build_model((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            ]
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("GRU model training completed successfully")
            return self.history.history
            
        except Exception as e:
            logger.error(f"Error training GRU model: {str(e)}")
            raise

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            if X_test.ndim == 2:
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            predictions = self.model.predict(X_test)
            predictions = self.scaler.inverse_transform(predictions)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        try:
            y_pred = self.predict(X_test)
            y_test_scaled = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mse = mean_squared_error(y_test_scaled, y_pred)
            mae = mean_absolute_error(y_test_scaled, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_scaled, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            logger.info(f"GRU evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating GRU model: {str(e)}")
            raise

    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            self.model.save(filepath)
            logger.info(f"GRU model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving GRU model: {str(e)}")

    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"GRU model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading GRU model: {str(e)}")


class EnsembleModel:
    """Ensemble model combining LSTM and GRU predictions"""
    
    def __init__(self, lstm_model: LSTMModel, gru_model: GRUModel, weights: Tuple[float, float] = (0.5, 0.5)):
        """
        Initialize ensemble model
        
        Args:
            lstm_model: Trained LSTM model
            gru_model: Trained GRU model
            weights: Weights for ensemble (lstm_weight, gru_weight)
        """
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.weights = weights

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X_test: Test features
            
        Returns:
            Ensemble predictions
        """
        try:
            lstm_pred = self.lstm_model.predict(X_test)
            gru_pred = self.gru_model.predict(X_test)
            
            # Ensure same shape
            if lstm_pred.shape != gru_pred.shape:
                logger.warning("LSTM and GRU prediction shapes differ; attempting to broadcast")
            ensemble_pred = (self.weights[0] * lstm_pred) + (self.weights[1] * gru_pred)
            
            logger.info("Ensemble predictions generated")
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate ensemble model performance"""
        try:
            y_pred = self.predict(X_test)
            y_test_scaled = self.lstm_model.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mse = mean_squared_error(y_test_scaled, y_pred)
            mae = mean_absolute_error(y_test_scaled, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_scaled, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            logger.info(f"Ensemble evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble model: {str(e)}")
            raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("ML Models module initialized")
