# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'gender': ['Male', 'Female', 'Male'],
        'contract_type': ['Month-to-Month', 'One year', 'Two year'],
        'tenure': [12, 24, 36],
        'monthly_charges': [50.0, 75.0, 100.0],
        'total_charges': [600.0, 1800.0, 3600.0],
        'churn': [0, 1, 0]
    })

def test_load_data(sample_data, tmp_path):
    """Test data loading functionality"""
    # Create temporary CSV file
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    preprocessor = DataPreprocessor()
    loaded_data = preprocessor.load_data(csv_path)
    
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.shape == sample_data.shape
    assert all(loaded_data.columns == sample_data.columns)

def test_preprocess_features(sample_data):
    """Test feature preprocessing"""
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_features(sample_data)
    
    # Check if categorical variables are converted to dummy variables
    assert 'gender_Male' in processed_data.columns
    assert 'gender_Female' in processed_data.columns
    assert 'contract_type_Month-to-Month' in processed_data.columns
    
    # Check if numerical features are scaled
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
    for col in numerical_cols:
        assert processed_data[col].mean() == pytest.approx(0, abs=1e-10)
        assert processed_data[col].std() == pytest.approx(1, abs=1e-10)

def test_prepare_data(sample_data):
    """Test data splitting functionality"""
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(sample_data)
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)
    
    # Check if target column is removed from features
    assert 'churn' not in X_train.columns
    assert 'churn' not in X_test.columns

def test_handle_missing_values(sample_data):
    """Test missing value handling"""
    # Add some missing values
    sample_data.loc[0, 'monthly_charges'] = np.nan
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.load_data(sample_data)
    
    # Check if missing values are handled
    assert processed_data['monthly_charges'].isna().sum() == 0
