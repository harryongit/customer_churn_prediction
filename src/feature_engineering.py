# src/feature_engineering.py
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Create new features
        X_copy['revenue_per_tenure'] = X_copy['monthly_charges'] * X_copy['tenure']
        X_copy['is_new_customer'] = X_copy['tenure'].apply(lambda x: 1 if x <= 12 else 0)
        
        return X_copy

