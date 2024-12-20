# main.py
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import ChurnPredictor
from src.evaluation import ModelEvaluator

def main():
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    model = ChurnPredictor()
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    df = preprocessor.load_data('data/customer_data.csv')
    df_processed = preprocessor.preprocess_features(df)
    
    # Feature engineering
    df_featured = feature_engineer.transform(df_processed)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_featured)
    
    # Train model
    model.train(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    report, conf_matrix = evaluator.evaluate_model(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Plot ROC curve
    evaluator.plot_roc_curve(y_test, y_pred_proba)
    
    # Get feature importance
    importance = model.get_feature_importance(X_train.columns)
    print("\nFeature Importance:")
    for feature, score in sorted(importance.items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.4f}")

if __name__ == "__main__":
    main()

