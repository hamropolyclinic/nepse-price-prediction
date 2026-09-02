"""Tests for configuration module"""
import pytest
from pathlib import Path
from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR,
    DATA_CONFIG, FEATURE_CONFIG, NN_CONFIG, ML_CONFIG,
    STOCKS_TO_TRACK, CONFIDENCE_THRESHOLD, PREDICTION_DAYS_AHEAD
)


class TestConfigDirectories:
    """Test configuration directories"""
    
    def test_base_dir_exists(self):
        """Test that BASE_DIR is properly set"""
        assert BASE_DIR is not None
        assert isinstance(BASE_DIR, Path)
    
    def test_data_dir_created(self):
        """Test that DATA_DIR is created"""
        assert DATA_DIR.exists()
        assert DATA_DIR.is_dir()
    
    def test_models_dir_created(self):
        """Test that MODELS_DIR is created"""
        assert MODELS_DIR.exists()
        assert MODELS_DIR.is_dir()
    
    def test_logs_dir_created(self):
        """Test that LOGS_DIR is created"""
        assert LOGS_DIR.exists()
        assert LOGS_DIR.is_dir()


class TestDataConfig:
    """Test data configuration"""
    
    def test_lookback_period_valid(self):
        """Test lookback period is valid"""
        assert DATA_CONFIG['lookback_period'] > 0
        assert isinstance(DATA_CONFIG['lookback_period'], int)
    
    def test_train_test_split_valid(self):
        """Test train-test split is between 0 and 1"""
        split = DATA_CONFIG['train_test_split']
        assert 0 < split < 1
    
    def test_normalization_method_valid(self):
        """Test normalization method is recognized"""
        method = DATA_CONFIG['normalization_method']
        assert method in ['minmax', 'standard']


class TestFeatureConfig:
    """Test feature configuration"""
    
    def test_indicators_not_empty(self):
        """Test indicators list is not empty"""
        assert len(FEATURE_CONFIG['indicators']) > 0
    
    def test_lagged_features_positive(self):
        """Test lagged features are positive integers"""
        for lag in FEATURE_CONFIG['lagged_features']:
            assert lag > 0
            assert isinstance(lag, int)
    
    def test_rolling_windows_positive(self):
        """Test rolling windows are positive integers"""
        for window in FEATURE_CONFIG['rolling_windows']:
            assert window > 0
            assert isinstance(window, int)


class TestNNConfig:
    """Test neural network configuration"""
    
    def test_lstm_config_exists(self):
        """Test LSTM configuration exists"""
        assert 'lstm' in NN_CONFIG
        assert 'epochs' in NN_CONFIG['lstm']
        assert 'batch_size' in NN_CONFIG['lstm']
    
    def test_gru_config_exists(self):
        """Test GRU configuration exists"""
        assert 'gru' in NN_CONFIG
        assert 'epochs' in NN_CONFIG['gru']
        assert 'batch_size' in NN_CONFIG['gru']
    
    def test_ensemble_weights_valid(self):
        """Test ensemble weights are valid"""
        assert 'ensemble' in NN_CONFIG
        weights_sum = (NN_CONFIG['ensemble']['lstm_weight'] + 
                      NN_CONFIG['ensemble']['gru_weight'])
        assert weights_sum == pytest.approx(1.0)


class TestMLConfig:
    """Test machine learning configuration"""
    
    def test_random_forest_config_exists(self):
        """Test random forest configuration exists"""
        assert 'random_forest' in ML_CONFIG
        assert 'n_estimators' in ML_CONFIG['random_forest']
    
    def test_gradient_boosting_config_exists(self):
        """Test gradient boosting configuration exists"""
        assert 'gradient_boosting' in ML_CONFIG
        assert 'learning_rate' in ML_CONFIG['gradient_boosting']
    
    def test_svr_config_exists(self):
        """Test SVR configuration exists"""
        assert 'svr' in ML_CONFIG
        assert 'kernel' in ML_CONFIG['svr']


class TestGeneralConfig:
    """Test general configuration"""
    
    def test_stocks_to_track_not_empty(self):
        """Test stocks to track list is not empty"""
        assert len(STOCKS_TO_TRACK) > 0
        assert 'NABIL' in STOCKS_TO_TRACK
    
    def test_confidence_threshold_valid(self):
        """Test confidence threshold is between 0 and 1"""
        assert 0 < CONFIDENCE_THRESHOLD < 1
    
    def test_prediction_days_ahead_positive(self):
        """Test prediction days ahead is positive"""
        assert PREDICTION_DAYS_AHEAD > 0
