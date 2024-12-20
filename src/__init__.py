# src/__init__.py
"""
Customer Churn Prediction Package

This package provides tools and utilities for predicting customer churn using
machine learning techniques. It includes modules for data preprocessing,
feature engineering, model training, and evaluation.

Modules:
    - data_preprocessing: Handle data cleaning and preparation
    - feature_engineering: Create and transform features
    - model: Implement the churn prediction model
    - evaluation: Evaluate model performance and generate insights
"""

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model import ChurnPredictor
from .evaluation import ModelEvaluator

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer', 
    'ChurnPredictor',
    'ModelEvaluator'
]

__version__ = '1.0.0'
