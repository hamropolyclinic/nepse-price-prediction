"""
Model Manager for saving, loading, and managing trained models
"""
import logging
import joblib
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import tensorflow as tf
from models.neural_networks import LSTMModel, GRUModel, EnsembleModel
from models.traditional_ml import TraditionalMLModels
from config import MODEL_PATHS, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model persistence and retrieval"""
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
        self.load_metadata()

    def load_metadata(self):
        """Load model metadata"""
        try:
            metadata_path = self.models_dir / "models_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            logger.info("Model metadata loaded")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self.metadata = {}

    def save_metadata(self):
        """Save model metadata"""
        try:
            metadata_path = self.models_dir / "models_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            logger.info("Model metadata saved")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")

    def save_lstm_model(self, lstm_model: LSTMModel, model_name: str = "lstm", 
                        metrics: Optional[Dict] = None) -> bool:
        """
        Save LSTM model and scaler
        
        Args:
            lstm_model: Trained LSTM model
            model_name: Name for the model
            metrics: Model evaluation metrics
            
        Returns:
            True if successful
        """
        try:
            model_path = self.models_dir / f"{model_name}_model.h5"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            # Save model
            lstm_model.model.save(str(model_path))
            
            # Save scaler
            joblib.dump(lstm_model.scaler, str(scaler_path))
            
            # Update metadata
            self.metadata[model_name] = {
                'type': 'LSTM',
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'lookback': lstm_model.lookback,
                'output_size': lstm_model.output_size,
                'saved_at': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            self.save_metadata()
            logger.info(f"LSTM model '{model_name}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {str(e)}")
            return False

    def save_gru_model(self, gru_model: GRUModel, model_name: str = "gru",
                       metrics: Optional[Dict] = None) -> bool:
        """
        Save GRU model and scaler
        
        Args:
            gru_model: Trained GRU model
            model_name: Name for the model
            metrics: Model evaluation metrics
            
        Returns:
            True if successful
        """
        try:
            model_path = self.models_dir / f"{model_name}_model.h5"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            # Save model
            gru_model.model.save(str(model_path))
            
            # Save scaler
            joblib.dump(gru_model.scaler, str(scaler_path))
            
            # Update metadata
            self.metadata[model_name] = {
                'type': 'GRU',
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'lookback': gru_model.lookback,
                'output_size': gru_model.output_size,
                'saved_at': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            self.save_metadata()
            logger.info(f"GRU model '{model_name}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving GRU model: {str(e)}")
            return False

    def save_traditional_ml_models(self, ml_models: TraditionalMLModels,
                                  model_name: str = "traditional_ml",
                                  metrics: Optional[Dict] = None) -> bool:
        """
        Save traditional ML models
        
        Args:
            ml_models: TraditionalMLModels instance
            model_name: Name for the model collection
            metrics: Model evaluation metrics
            
        Returns:
            True if successful
        """
        try:
            models_path = self.models_dir / f"{model_name}_models.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            # Save models
            joblib.dump(ml_models.models, str(models_path))
            
            # Save scaler
            joblib.dump(ml_models.scalers.get('default'), str(scaler_path))
            
            # Update metadata
            self.metadata[model_name] = {
                'type': 'TraditionalML',
                'models_path': str(models_path),
                'scaler_path': str(scaler_path),
                'model_names': list(ml_models.models.keys()),
                'best_model': ml_models.best_model_name,
                'saved_at': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            self.save_metadata()
            logger.info(f"Traditional ML models '{model_name}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving traditional ML models: {str(e)}")
            return False

    def load_lstm_model(self, model_name: str = "lstm") -> Optional[LSTMModel]:
        """
        Load LSTM model and scaler
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded LSTM model or None
        """
        try:
            if model_name not in self.metadata:
                logger.error(f"Model '{model_name}' not found in metadata")
                return None
            
            meta = self.metadata[model_name]
            
            lstm_model = LSTMModel(
                lookback=meta.get('lookback', 60),
                output_size=meta.get('output_size', 1)
            )
            
            # Load model
            lstm_model.model = tf.keras.models.load_model(meta['model_path'])
            
            # Load scaler
            lstm_model.scaler = joblib.load(meta['scaler_path'])
            
            logger.info(f"LSTM model '{model_name}' loaded successfully")
            return lstm_model
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            return None

    def load_gru_model(self, model_name: str = "gru") -> Optional[GRUModel]:
        """
        Load GRU model and scaler
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded GRU model or None
        """
        try:
            if model_name not in self.metadata:
                logger.error(f"Model '{model_name}' not found in metadata")
                return None
            
            meta = self.metadata[model_name]
            
            gru_model = GRUModel(
                lookback=meta.get('lookback', 60),
                output_size=meta.get('output_size', 1)
            )
            
            # Load model
            gru_model.model = tf.keras.models.load_model(meta['model_path'])
            
            # Load scaler
            gru_model.scaler = joblib.load(meta['scaler_path'])
            
            logger.info(f"GRU model '{model_name}' loaded successfully")
            return gru_model
            
        except Exception as e:
            logger.error(f"Error loading GRU model: {str(e)}")
            return None

    def load_traditional_ml_models(self, model_name: str = "traditional_ml") -> Optional[TraditionalMLModels]:
        """
        Load traditional ML models
        
        Args:
            model_name: Name of the model collection
            
        Returns:
            Loaded TraditionalMLModels instance or None
        """
        try:
            if model_name not in self.metadata:
                logger.error(f"Model collection '{model_name}' not found in metadata")
                return None
            
            meta = self.metadata[model_name]
            
            ml_models = TraditionalMLModels()
            
            # Load models
            ml_models.models = joblib.load(meta['models_path'])
            
            # Load scaler
            ml_models.scalers['default'] = joblib.load(meta['scaler_path'])
            
            # Set best model
            ml_models.best_model_name = meta.get('best_model')
            if ml_models.best_model_name:
                ml_models.best_model = ml_models.models[ml_models.best_model_name]
            
            logger.info(f"Traditional ML models '{model_name}' loaded successfully")
            return ml_models
            
        except Exception as e:
            logger.error(f"Error loading traditional ML models: {str(e)}")
            return None

    def create_ensemble_model(self, lstm_model_name: str = "lstm", 
                             gru_model_name: str = "gru",
                             weights: tuple = (0.5, 0.5)) -> Optional[EnsembleModel]:
        """
        Create ensemble model from saved LSTM and GRU models
        
        Args:
            lstm_model_name: Name of LSTM model
            gru_model_name: Name of GRU model
            weights: Weights for ensemble
            
        Returns:
            EnsembleModel instance or None
        """
        try:
            lstm = self.load_lstm_model(lstm_model_name)
            gru = self.load_gru_model(gru_model_name)
            
            if lstm is None or gru is None:
                logger.error("Failed to load LSTM or GRU model for ensemble")
                return None
            
            ensemble = EnsembleModel(lstm, gru, weights)
            logger.info("Ensemble model created successfully")
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {str(e)}")
            return None

    def list_models(self) -> Dict[str, Dict]:
        """
        List all saved models
        
        Returns:
            Dictionary with model metadata
        """
        return self.metadata

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata or None
        """
        return self.metadata.get(model_name)

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if successful
        """
        try:
            if model_name not in self.metadata:
                logger.error(f"Model '{model_name}' not found")
                return False
            
            meta = self.metadata[model_name]
            
            # Delete model files
            if 'model_path' in meta:
                Path(meta['model_path']).unlink(missing_ok=True)
            if 'models_path' in meta:
                Path(meta['models_path']).unlink(missing_ok=True)
            if 'scaler_path' in meta:
                Path(meta['scaler_path']).unlink(missing_ok=True)
            
            # Remove from metadata
            del self.metadata[model_name]
            self.save_metadata()
            
            logger.info(f"Model '{model_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return False

    def export_metadata(self, filepath: str) -> bool:
        """
        Export model metadata to JSON file
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            logger.info(f"Metadata exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting metadata: {str(e)}")
            return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = ModelManager()
    print("Available models:", manager.list_models())
