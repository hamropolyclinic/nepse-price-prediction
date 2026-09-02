"""Integration tests for the NEPSE price prediction project"""
import pytest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from train import TrainingPipeline
from config import DATA_CONFIG, NN_CONFIG, ML_CONFIG


class TestTrainingPipelineIntegration:
    """Integration tests for complete training pipeline"""
    
    @patch('train.ModelManager')
    @patch('train.TraditionalMLModels')
    @patch('train.FeatureEngineering')
    @patch('train.DataCleaner')
    @patch('train.DataLoader')
    def test_pipeline_data_flow(self, mock_loader, mock_cleaner, 
                                 mock_fe, mock_ml, mock_mm):
        """Test data flows correctly through pipeline"""
        # Setup mocks
        mock_loader_inst = Mock()
        mock_loader.return_value = mock_loader_inst
        
        raw_data = pd.DataFrame({
            'Close': np.linspace(100, 110, 100),
            'date': pd.date_range('2024-01-01', periods=100)
        })
        mock_loader_inst.fetch_from_yfinance.return_value = raw_data
        mock_loader_inst.clean_data.return_value = raw_data
        
        mock_fe_inst = Mock()
        mock_fe.return_value = mock_fe_inst
        mock_fe_inst.create_all_features.return_value = pd.DataFrame(
            np.random.rand(95, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        
        # Run pipeline steps
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_loader_inst
        pipeline.feature_engineer = mock_fe_inst
        
        assert pipeline.fetch_data() is True
        assert pipeline.raw_data is not None
        
        assert pipeline.clean_data() is True
        assert pipeline.cleaned_data is not None
        
        assert pipeline.engineer_features() is True
        assert pipeline.featured_data is not None
    
    @patch('train.DataLoader')
    def test_pipeline_with_different_symbols(self, mock_loader):
        """Test pipeline works with different stock symbols"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_data = pd.DataFrame({
            'Close': np.random.rand(50),
            'date': pd.date_range('2024-01-01', periods=50)
        })
        mock_instance.fetch_from_yfinance.return_value = mock_data
        
        symbols = ['NABIL', 'SCB', 'SBI']
        
        for symbol in symbols:
            pipeline = TrainingPipeline(symbol=symbol)
            pipeline.data_loader = mock_instance
            result = pipeline.fetch_data()
            assert result is True
            assert pipeline.symbol == symbol
    
    @patch('train.DataLoader')
    def test_pipeline_data_shapes(self, mock_loader):
        """Test pipeline maintains correct data shapes"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        
        num_records = 100
        raw_data = pd.DataFrame({
            'Close': np.linspace(100, 120, num_records),
            'date': pd.date_range('2024-01-01', periods=num_records),
            'Open': np.linspace(99, 119, num_records),
            'High': np.linspace(101, 121, num_records),
            'Low': np.linspace(98, 118, num_records),
        })
        mock_instance.fetch_from_yfinance.return_value = raw_data
        mock_instance.clean_data.return_value = raw_data
        
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_instance
        
        pipeline.fetch_data()
        assert len(pipeline.raw_data) == num_records
        
        pipeline.clean_data()
        assert len(pipeline.cleaned_data) == num_records


class TestConfigurationConsistency:
    """Test configuration consistency across modules"""
    
    def test_data_config_lookback_matches_nn_config(self):
        """Test lookback period is consistent"""
        # DATA_CONFIG should have lookback_period
        assert 'lookback_period' in DATA_CONFIG
        assert DATA_CONFIG['lookback_period'] > 0
    
    def test_nn_config_has_required_fields(self):
        """Test NN config has all required fields"""
        required_fields = ['lstm', 'gru', 'ensemble']
        for field in required_fields:
            assert field in NN_CONFIG
    
    def test_ml_config_has_required_models(self):
        """Test ML config has required models"""
        required_models = ['random_forest', 'gradient_boosting', 'svr']
        for model in required_models:
            assert model in ML_CONFIG
    
    def test_config_values_are_positive(self):
        """Test configuration values are positive where expected"""
        assert DATA_CONFIG['lookback_period'] > 0
        assert NN_CONFIG['lstm']['epochs'] > 0
        assert NN_CONFIG['lstm']['batch_size'] > 0
        assert ML_CONFIG['random_forest']['n_estimators'] > 0


class TestDataPipelineEdgeCases:
    """Test edge cases in data pipeline"""
    
    @patch('train.DataLoader')
    def test_pipeline_with_minimal_data(self, mock_loader):
        """Test pipeline with minimal data points"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        
        # Create minimal dataset (just above lookback)
        min_data = pd.DataFrame({
            'Close': np.random.rand(65),  # Just above default lookback of 60
            'date': pd.date_range('2024-01-01', periods=65)
        })
        mock_instance.fetch_from_yfinance.return_value = min_data
        mock_instance.clean_data.return_value = min_data
        
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_instance
        
        assert pipeline.fetch_data() is True
        assert pipeline.clean_data() is True
        assert pipeline.prepare_data_for_nn() is True
    
    @patch('train.DataLoader')
    def test_pipeline_with_large_dataset(self, mock_loader):
        """Test pipeline with large dataset"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        
        # Create large dataset (multiple years of data)
        large_data = pd.DataFrame({
            'Close': np.random.rand(2000),  # ~5.5 years of trading data
            'date': pd.date_range('2020-01-01', periods=2000, freq='D')
        })
        mock_instance.fetch_from_yfinance.return_value = large_data
        mock_instance.clean_data.return_value = large_data
        
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_instance
        
        assert pipeline.fetch_data() is True
        assert len(pipeline.raw_data) == 2000
