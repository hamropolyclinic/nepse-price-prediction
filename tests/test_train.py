"""Tests for training module"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from train import TrainingPipeline


class TestTrainingPipelineInit:
    """Test TrainingPipeline initialization"""
    
    def test_pipeline_init_default_symbol(self):
        """Test pipeline initializes with default symbol"""
        pipeline = TrainingPipeline()
        assert pipeline.symbol == 'NABIL'
    
    def test_pipeline_init_custom_symbol(self):
        """Test pipeline initializes with custom symbol"""
        pipeline = TrainingPipeline(symbol='SCB')
        assert pipeline.symbol == 'SCB'
    
    def test_pipeline_init_attributes(self):
        """Test pipeline initializes all required attributes"""
        pipeline = TrainingPipeline()
        
        assert pipeline.raw_data is None
        assert pipeline.cleaned_data is None
        assert pipeline.featured_data is None
        assert pipeline.lstm_model is None
        assert pipeline.gru_model is None
        assert pipeline.ml_models is None
        assert pipeline.ensemble_model is None
        assert isinstance(pipeline.results, dict)


class TestTrainingPipelineFetchData:
    """Test data fetching functionality"""
    
    @patch('train.DataLoader')
    def test_fetch_data_success(self, mock_loader):
        """Test successful data fetching"""
        # Setup mock
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103],
            'date': pd.date_range('2024-01-01', periods=4)
        })
        mock_instance.fetch_from_yfinance.return_value = mock_data
        
        # Test
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_instance
        result = pipeline.fetch_data()
        
        assert result is True
        assert pipeline.raw_data is not None
    
    @patch('train.DataLoader')
    def test_fetch_data_failure_empty_data(self, mock_loader):
        """Test fetch data handles empty data"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_instance.fetch_from_yfinance.return_value = None
        
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_instance
        result = pipeline.fetch_data()
        
        assert result is False
    
    @patch('train.DataLoader')
    def test_fetch_data_with_date_range(self, mock_loader):
        """Test fetch data with date range"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_data = pd.DataFrame({'Close': [100, 101]})
        mock_instance.fetch_from_yfinance.return_value = mock_data
        
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_instance
        result = pipeline.fetch_data('2024-01-01', '2024-12-31')
        
        assert result is True
        mock_instance.fetch_from_yfinance.assert_called_once_with(
            'NABIL', '2024-01-01', '2024-12-31'
        )


class TestTrainingPipelineCleanData:
    """Test data cleaning functionality"""
    
    @patch('train.DataLoader')
    def test_clean_data_success(self, mock_loader):
        """Test successful data cleaning"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'date': pd.date_range('2024-01-01', periods=3)
        })
        mock_instance.clean_data.return_value = mock_data
        
        pipeline = TrainingPipeline()
        pipeline.raw_data = mock_data
        pipeline.data_loader = mock_instance
        result = pipeline.clean_data()
        
        assert result is True
        assert pipeline.cleaned_data is not None
    
    @patch('train.DataLoader')
    def test_clean_data_no_raw_data(self, mock_loader):
        """Test clean data fails when no raw data"""
        pipeline = TrainingPipeline()
        result = pipeline.clean_data()
        
        assert result is False
    
    @patch('train.DataLoader')
    def test_clean_data_empty_result(self, mock_loader):
        """Test clean data handles empty result"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_instance.clean_data.return_value = None
        
        pipeline = TrainingPipeline()
        pipeline.raw_data = pd.DataFrame({'Close': [100]})
        pipeline.data_loader = mock_instance
        result = pipeline.clean_data()
        
        assert result is False


class TestTrainingPipelinePrepareData:
    """Test data preparation functionality"""
    
    def test_prepare_data_for_nn_success(self):
        """Test successful data preparation for NN"""
        pipeline = TrainingPipeline()
        pipeline.cleaned_data = pd.DataFrame({
            'Close': np.random.rand(100)
        })
        result = pipeline.prepare_data_for_nn()
        
        assert result is True
    
    def test_prepare_data_for_nn_no_data(self):
        """Test prepare data for NN fails with no data"""
        pipeline = TrainingPipeline()
        result = pipeline.prepare_data_for_nn()
        
        assert result is False
    
    def test_prepare_data_for_nn_insufficient_data(self):
        """Test prepare data for NN fails with insufficient data"""
        pipeline = TrainingPipeline()
        pipeline.cleaned_data = pd.DataFrame({
            'Close': [100, 101]  # Less than lookback period
        })
        result = pipeline.prepare_data_for_nn()
        
        assert result is False
    
    @patch('train.FeatureEngineering')
    def test_prepare_data_for_ml_success(self, mock_fe):
        """Test successful data preparation for ML"""
        mock_instance = Mock()
        mock_fe.return_value = mock_instance
        
        featured_data = pd.DataFrame(
            np.random.rand(50, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        featured_data.index = pd.date_range('2024-01-01', periods=50)
        
        pipeline = TrainingPipeline()
        pipeline.featured_data = featured_data
        pipeline.cleaned_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=51),
            'Close': np.random.rand(51)
        })
        
        success, X, y = pipeline.prepare_data_for_ml()
        
        assert success is True
        assert X is not None
        assert y is not None


class TestTrainingPipelineResults:
    """Test results handling"""
    
    def test_results_dictionary_initialized(self):
        """Test results dictionary is initialized"""
        pipeline = TrainingPipeline()
        assert isinstance(pipeline.results, dict)
        assert len(pipeline.results) == 0
    
    def test_results_can_store_model_metrics(self):
        """Test results can store model metrics"""
        pipeline = TrainingPipeline()
        pipeline.results['lstm'] = {'rmse': 2.5, 'mae': 1.8}
        
        assert 'lstm' in pipeline.results
        assert pipeline.results['lstm']['rmse'] == 2.5


class TestTrainingPipelineErrorHandling:
    """Test error handling"""
    
    @patch('train.DataLoader')
    def test_fetch_data_exception_handling(self, mock_loader):
        """Test fetch data handles exceptions"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_instance.fetch_from_yfinance.side_effect = Exception("API Error")
        
        pipeline = TrainingPipeline()
        pipeline.data_loader = mock_instance
        result = pipeline.fetch_data()
        
        assert result is False
    
    @patch('train.DataLoader')
    def test_clean_data_exception_handling(self, mock_loader):
        """Test clean data handles exceptions"""
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        mock_instance.clean_data.side_effect = Exception("Cleaning Error")
        
        pipeline = TrainingPipeline()
        pipeline.raw_data = pd.DataFrame({'Close': [100]})
        pipeline.data_loader = mock_instance
        result = pipeline.clean_data()
        
        assert result is False
