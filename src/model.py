# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance scores"""
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))
    
    def save_model(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load_model(filepath):
        """Load model from disk"""
        return joblib.load(filepath)

