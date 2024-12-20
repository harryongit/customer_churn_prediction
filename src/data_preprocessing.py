# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load and perform initial cleaning of data"""
        df = pd.read_csv(filepath)
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for model training"""
        # Convert categorical variables
        df = pd.get_dummies(df, columns=['gender', 'contract_type'])
        
        # Scale numerical features
        numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def prepare_data(self, df, target_col='churn'):
        """Split data into training and testing sets"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

