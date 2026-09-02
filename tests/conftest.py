"""
Initialize tests package with pytest fixtures
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def mock_data():
    """Provide mock data for tests"""
    import pandas as pd
    import numpy as np
    
    return pd.DataFrame({
        'Close': np.linspace(100, 120, 100),
        'Open': np.linspace(99, 119, 100),
        'High': np.linspace(101, 121, 100),
        'Low': np.linspace(98, 118, 100),
        'date': pd.date_range('2024-01-01', periods=100)
    })


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model files"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir
