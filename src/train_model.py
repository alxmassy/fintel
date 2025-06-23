import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib # For saving/loading model and data
import os
import sys
import shap

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function to create features and target from our previous script
from feature_engineering import create_features_and_target

def train_xgboost_model(model_output_path="models/xgboost_model.pkl"):
    """
    Trains an XGBoost Classifier on the prepared stock and news data.
    """
    print("\n--- Starting Model Training ---")

    # 1. Get prepared data
    # create_features_and_target will handle loading data, feature engineering, and splitting
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = create_features_and_target()

    if X_train_scaled is None:
        print("Failed to prepare data. Exiting training.")
        return None

    # 2. Remap Target Values from [-1, 0, 1] to [0, 1, 2] for XGBoost
    # Create a mapping dictionary: -1 -> 0, 0 -> 1, 1 -> 2
    label_map = {-1: 0, 0: 1, 1: 2}
    
    # Apply mapping to train and test labels
    if y_train is None or y_test is None:
        print("y_train or y_test is None. Exiting training.")
        return None

    try:
        y_train_mapped = y_train.map(label_map)
        y_test_mapped = y_test.map(label_map)

        if y_train_mapped.isnull().any() or y_test_mapped.isnull().any():
            raise ValueError("Mapping failed for some values in y_train or y_test. Check label_map.")
    except Exception as e:
        print(f"Error mapping labels: {e}")
        return None
    
    # Store the original mapping for prediction interpretation later
    original_labels = {v: k for k, v in label_map.items()}  # Reverse the mapping
    
    # Initialize and Train XGBoost Classifier
    # objective='multi:softmax' for multi-class classification
    # num_class=3 for Up (2), Down (0), Neutral (1) after mapping
    # eval_metric='mlogloss' is suitable for multi-class
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False, # Suppress deprecation warning
        n_estimators=100,       # Number of boosting rounds
        learning_rate=0.1,      # Step size shrinkage to prevent overfitting
        random_state=42         # For reproducibility
    )

    print("Training XGBoost model...")
    model.fit(X_train_scaled, y_train_mapped)
    print("Model training complete.")

    # 3. Evaluate the Model
    y_pred_mapped = model.predict(X_test_scaled)
    
    # Map predictions back to original labels [-1, 0, 1]
    y_pred_original = pd.Series(y_pred_mapped).map(original_labels)
    
    accuracy = accuracy_score(y_test, y_pred_original)
    report = classification_report(y_test, y_pred_original, target_names=['Down', 'Neutral', 'Up'])

    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report on Test Set:")
    print(report)

    # 4. Save the Trained Model and Label Mapping
    joblib.dump(model, model_output_path)
    print(f"Trained model saved to {model_output_path}")
    
    # Save feature names as well, helpful for consistent feature ordering later
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("Feature names saved to models/feature_names.pkl")
    
    # Save the original labels mapping for future predictions
    joblib.dump(original_labels, "models/label_mapping.pkl")
    print("Label mapping saved to models/label_mapping.pkl")

    # Add SHAP explanation generation after model prediction
    explainer = shap.Explainer(model, X_train_scaled)

    # Generate SHAP values for the test set
    shap_values = explainer(X_test_scaled)

    # Create a function to generate textual insights based on SHAP values
    def generate_insight(shap_values, feature_names):
        insights = []
        for i, shap_value in enumerate(shap_values):
            top_features = sorted(zip(feature_names, shap_value.values), key=lambda x: abs(x[1]), reverse=True)[:3]
            insight = f"Prediction influenced by: {', '.join([f'{feature} ({value:.2f})' for feature, value in top_features])}"
            insights.append(insight)
        return insights

    # Generate insights for the test set
    insights = generate_insight(shap_values, feature_names)

    print("--- Model Training Complete ---")
    return model

if __name__ == "__main__":
    trained_model = train_xgboost_model()
    if trained_model:
        print("\nFintel model is ready!")