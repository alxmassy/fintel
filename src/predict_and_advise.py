#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import shap
from datetime import datetime, timedelta
import os
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Import data acquisition and feature engineering functions for a "live" prediction scenario
from data_acquisition import get_apple_stock_data, get_apple_news_data
from feature_engineering import create_features_and_target # We'll reuse parts of this logic

# Define the target mapping (from feature_engineering.py)
TARGET_MAPPING = {1: "Up", -1: "Down", 0: "Neutral"}

def generate_advice(prediction_direction, shap_values, feature_names, current_news_summary):
    """
    Generates AI-driven advice based on model prediction and SHAP explanations.
    """
    advice_text = f"Fintel predicts the stock will move: **{TARGET_MAPPING[prediction_direction]}**.\n\n"

    # Get the absolute SHAP values for the predicted class
    # For multi-class, shap_values is a list of arrays (one per class).
    # We need the values for the predicted class's output.
    # Find the index corresponding to the predicted class (0 for Neutral, 1 for Up, 2 for Down based on XGBoost internal mapping for -1,0,1)
    # Note: XGBoost internally re-maps target labels 0, 1, -1 to 0, 1, 2 or similar.
    # Let's assume prediction_direction is already 0, 1, -1
    # For SHAP, typically you look at general feature importance across all classes, or specific to one output.
    # For simplicity, we'll look at the feature importance for the *overall* prediction.
    
    # If your SHAP returns values for each class, you need to select the one corresponding to the predicted class.
    # For prototype, we'll use shap_values[0] assuming a single output or a general explanation.
    
    # Ensure shap_values is an array-like object before proceeding
    if not isinstance(shap_values, (np.ndarray, pd.Series, list)):
        print("Warning: SHAP values not in expected format. Cannot generate detailed advice.")
        return advice_text + "No detailed explanation available."

    # Flatten SHAP values if they are nested (e.g., [array_for_class_0, array_for_class_1, ...])
    # For simplicity, let's assume `shap_values` is a single array of feature importances for a single prediction.
    if isinstance(shap_values, list) and all(isinstance(val, np.ndarray) for val in shap_values):
        # If multi-output SHAP, for simplicity, average across classes or pick one.
        # For tree explainers, shap_values[0] often refers to the first instance's explanation.
        # If XGBoost multi-class output (3 classes), SHAP will return a list of 3 arrays (one for each class).
        # We want the explanation for the *predicted* class.
        # Map prediction_direction (-1, 0, 1) to XGBoost's internal class index (often 0, 1, 2)
        # This is tricky without knowing exact internal mapping.
        # A common approach is to just use shap_values[0] (for the first instance in X_test)
        # and interpret the magnitude.
        
        # For prototype, let's simplify and assume `shap_values` is for the overall prediction influence.
        # Or, if shap_values is a list of arrays, let's take the first array for the first instance
        # and assume it represents general feature impact.
        if len(shap_values) == 3 and isinstance(shap_values[0], np.ndarray): # If multi-class SHAP
            # Find the index of the predicted class in the model's output (often 0 for down, 1 for neutral, 2 for up, or vice versa)
            # This mapping needs to be consistent with how XGBoost maps the labels.
            # For safety, let's just use the average magnitude for now.
            abs_shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
        else: # Assume single-instance explanation
            abs_shap_values = np.abs(np.array(shap_values))
    else: # Already a single array/series for one instance
        abs_shap_values = np.abs(np.array(shap_values))


    # Ensure `abs_shap_values` is 1-D array or extract values for predicted class
    if abs_shap_values.ndim > 1:
        # For multi-class model, the shape is (n_classes, n_features) or (n_features, n_classes)
        # Let's use the values for the predicted class (the one with the highest prediction)
        # For a 3-class model with 9 features, we expect either shape (3, 9) or (9, 3)
        
        # Check the shape and determine how to extract values for the predicted class
        if len(abs_shap_values) == len(feature_names):
            # Shape is likely (n_features, n_classes)
            # Sum across classes or take the values for the predicted class
            abs_shap_values = abs_shap_values.sum(axis=1)  # Sum across classes
        elif len(abs_shap_values) == 3 and len(abs_shap_values[0]) == len(feature_names):
            # Shape is likely (n_classes, n_features)
            # Take the values for the predicted class (index = prediction_direction + 1)
            class_idx = prediction_direction + 1  # Map -1->0, 0->1, 1->2
            abs_shap_values = abs_shap_values[class_idx]
        else:
            # If dimensions don't match expectations, take mean across all values
            abs_shap_values = np.mean(abs_shap_values, axis=0)
            # Ensure it matches feature_names length
            if len(abs_shap_values) != len(feature_names):
                # As last resort, flatten and truncate/pad to match feature_names
                abs_shap_values = abs_shap_values.flatten()[:len(feature_names)]
    
    # Create a Series for easy sorting
    feature_importances = pd.Series(abs_shap_values, index=feature_names)
    top_features = feature_importances.nlargest(3)

    advice_text += "This prediction is primarily influenced by:\n"
    for feature, importance in top_features.items():
        value = "N/A" # Default value if not found in current_news_summary for news features
        
        if 'News_Count' in feature and current_news_summary and 'News_Count' in current_news_summary:
            value = f"{current_news_summary['News_Count']} relevant articles"
            if current_news_summary['Avg_News_Sentiment'] > 0.05:
                advice_text += f"- **Positive sentiment** from recent news (e.g., {value}).\n"
            elif current_news_summary['Avg_News_Sentiment'] < -0.05:
                advice_text += f"- **Negative sentiment** from recent news (e.g., {value}).\n"
            else:
                advice_text += f"- **Neutral sentiment** from recent news (e.g., {value}).\n"
        elif 'Avg_News_Sentiment' in feature and current_news_summary and 'Avg_News_Sentiment' in current_news_summary:
            sentiment = current_news_summary['Avg_News_Sentiment']
            if sentiment > 0.05:
                advice_text += f"- Generally **positive news sentiment** (score: {sentiment:.2f}).\n"
            elif sentiment < -0.05:
                advice_text += f"- Generally **negative news sentiment** (score: {sentiment:.2f}).\n"
            else:
                advice_text += f"- Predominantly **neutral news sentiment** (score: {sentiment:.2f}).\n"
        elif 'Close_Lag1' in feature:
            advice_text += f"- The **previous day's closing price**.\n"
        elif 'SMA_10' in feature:
            advice_text += f"- The **10-day Simple Moving Average** (indicating short-term trend).\n"
        elif 'SMA_20' in feature:
            advice_text += f"- The **20-day Simple Moving Average** (indicating medium-term trend).\n"
        elif 'RSI' in feature:
            advice_text += f"- The **Relative Strength Index (RSI)** (indicating overbought/oversold conditions).\n"
        elif 'Volume_Lag1' in feature:
             advice_text += f"- The **previous day's trading volume**.\n"
        else:
            advice_text += f"- **{feature}** (a significant underlying factor).\n"
    
    advice_text += "\n*Note: This is a prototype system. Do not use for real financial decisions.*"
    return advice_text


def get_latest_data_and_predict(ticker_symbol="AAPL"):
    """
    Fetches the very latest data, preprocesses it, and makes a prediction.
    """
    print(f"\n--- Fintel Prediction for {ticker_symbol} ---")

    # 1. Load Model, Scaler, and Feature Names
    try:
        model = joblib.load("models/xgboost_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
    except FileNotFoundError as e:
        print(f"Error loading model components: {e}. Make sure training ran successfully.")
        return None, None, None, None

    # 2. Get Latest Stock Data (yesterday's close and volume)
    # Fetch only the last few days to ensure we have enough for lags/indicators
    print("Fetching latest stock data...")
    try:
        latest_stock_df = get_apple_stock_data(period="10d", output_path="data/aapl_latest_stock.csv")
        if latest_stock_df is None or latest_stock_df.empty:
            print("Could not fetch latest stock data.")
            return None, None, None, None
        
        # Get the very last available row (which is usually yesterday's data)
        current_day_data = latest_stock_df.iloc[-1]
        previous_day_data = latest_stock_df.iloc[-2] # For lag features

        # We need the most recent 'Close' for prediction context
        current_close_price = current_day_data['Close']
        
    except Exception as e:
        print(f"Error getting latest stock data: {e}")
        return None, None, None, None

    # 3. Get Latest News Data (for the last 24-48 hours, depending on when run)
    # We look back 2 days to ensure we capture news from after yesterday's close until now
    print("Fetching latest news data...")
    latest_news_df = get_apple_news_data(query=ticker_symbol, days_ago=2, output_path="data/aapl_latest_news.csv")

    current_news_summary = {}
    if latest_news_df is not None and not latest_news_df.empty:
        # Replicate news processing from feature_engineering.py for the latest news
        analyzer = SentimentIntensityAnalyzer()
        latest_news_df['text'] = latest_news_df['title'].fillna('') + ' ' + latest_news_df['description'].fillna('')
        latest_news_df['sentiment_compound'] = latest_news_df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        
        # Aggregate for the most recent day (today's news for tomorrow's prediction)
        current_news_summary = {
            'Avg_News_Sentiment': latest_news_df['sentiment_compound'].mean() if not latest_news_df.empty else 0,
            'News_Count': len(latest_news_df),
            'Positive_News_Count': (latest_news_df['sentiment_compound'] > 0.05).sum(),
            'Negative_News_Count': (latest_news_df['sentiment_compound'] < -0.05).sum()
        }
        print(f"Latest news summary: {current_news_summary}")
    else:
        print("No new news found or error fetching news. Assuming neutral sentiment.")
        current_news_summary = {
            'Avg_News_Sentiment': 0.0, 'News_Count': 0, 
            'Positive_News_Count': 0, 'Negative_News_Count': 0
        }

    # 4. Create Features for the NEW Prediction (matching training features)
    # We need to manually construct the single row of features for prediction
    # Ensure feature order matches `feature_names` saved during training.
    
    # Re-create a small part of the stock_df needed for feature calculation for the current day
    # For simplicity, we create a dummy DataFrame that includes `latest_stock_df` and
    # appends a row for the 'current' day with the required values to calculate lags/TAs.
    
    # This is a bit tricky for 'live' data. For prototype, we'll build a synthetic row.
    # Ensure `latest_stock_df` has enough rows for rolling means (e.g., 20 for SMA_20)
    if len(latest_stock_df) < 20: # Adjust based on max window of your TAs
        print("Not enough historical data to compute all features for current day.")
        # Fallback to a longer period if necessary or use simpler features
        latest_stock_df = get_apple_stock_data(period="30d", output_path="data/aapl_latest_stock.csv")
        if latest_stock_df is None or len(latest_stock_df) < 20:
            print("Still not enough data. Cannot proceed.")
            return None, None, None, None

    # Re-apply the feature engineering logic on the `latest_stock_df`
    # to get the same features as during training, for the *latest* rows.
    # This is a bit redundant but ensures consistency for a prototype.
    
    # We'll re-calculate TAs on the `latest_stock_df` to get the last valid row's features.
    temp_df = latest_stock_df.copy()
    temp_df['Close_Lag1'] = temp_df['Close'].shift(1)
    temp_df['Volume_Lag1'] = temp_df['Volume'].shift(1)
    temp_df['SMA_10'] = temp_df['Close'].rolling(window=10).mean()
    temp_df['SMA_20'] = temp_df['Close'].rolling(window=20).mean()
    delta = temp_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    temp_df['RSI'] = 100 - (100 / (1 + rs))

    # Get the row with the most recent *complete* features
    # This is the last row for which all features (including lags/rolling) are non-NaN
    features_for_prediction = temp_df.dropna().iloc[[-1]] # Use double brackets to keep it as DataFrame

    # Add aggregated news features to this single row
    # Ensure column names match what the model expects
    features_for_prediction['Avg_News_Sentiment'] = current_news_summary['Avg_News_Sentiment']
    features_for_prediction['News_Count'] = current_news_summary['News_Count']
    features_for_prediction['Positive_News_Count'] = current_news_summary['Positive_News_Count']
    features_for_prediction['Negative_News_Count'] = current_news_summary['Negative_News_Count']

    # Select only the columns that were used for training, in the correct order
    X_new = features_for_prediction[feature_names]

    # 5. Scale the new features
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=feature_names, index=X_new.index)
    
    print("\nFeatures prepared for prediction:")
    print(X_new_scaled_df)

    # 6. Make Prediction
    prediction_array = model.predict(X_new_scaled_df)
    prediction_direction = prediction_array[0] # Get the single prediction value
    
    print(f"\nRaw Prediction: {prediction_direction} -> {TARGET_MAPPING[prediction_direction]}")

    # 7. Generate SHAP values for explanation
    explainer = shap.TreeExplainer(model)
    # shap_values will be a list of arrays, one array per class output
    # For multi-class, it's typically a list of arrays (num_classes, num_features)
    # We want the explanation for the specific instance.
    
    # When calling shap_values for a multi-class model:
    # If `model.predict_proba` is used internally by SHAP, it might give a list of arrays.
    # If `model.predict` is used, it might be a single array.
    # For XGBoostClassifier, TreeExplainer often returns a list of arrays [shap_values_for_class0, shap_values_for_class1, ...]
    # For our three classes (-1, 0, 1), these will map to 0, 1, 2 internally.
    # Let's get the SHAP values for the predicted class's output.
    # A simplified approach for prototype: just use the raw SHAP values for the first instance.
    
    # Ensure `X_new_scaled_df` is not empty.
    if X_new_scaled_df.empty:
        print("Error: No data to explain.")
        shap_explanation_values = None
    else:
        # For multi-class (3 output classes in XGBoost), shap_values will be a list of 3 arrays (one for each class's output).
        # We want the contribution to the specific predicted class's probability/logit.
        # However, for general "advice" based on feature influence, the overall magnitude is often useful.
        # Let's take the SHAP values for the predicted class. This requires knowing XGBoost's internal mapping.
        # A simpler, more robust method for a prototype is to get a single SHAP explanation
        # by explaining the *predicted score* (e.g., the output of `model.predict_proba` for the predicted class).
        
        # For multi-class models, SHAP returns values for each class
        # We extract the values specifically for our task
        try:
            # First try to get the Explanation object and extract values for the first instance
            shap_explanation = explainer(X_new_scaled_df)
            
            # Check if we have a multi-class output (list of arrays, one per class)
            if hasattr(shap_explanation, 'values') and isinstance(shap_explanation.values, np.ndarray):
                # Direct values from Explanation object
                if shap_explanation.values.shape[0] == 1:
                    # Single instance, shape is (1, n_features) or (1, n_features, n_classes)
                    shap_explanation_values = shap_explanation.values[0]
                else:
                    # Multiple instances or classes in first dimension
                    shap_explanation_values = shap_explanation.values
            else:
                # Fall back to direct output (might be a list of arrays for multi-class)
                shap_values = explainer.shap_values(X_new_scaled_df)
                
                if isinstance(shap_values, list) and len(shap_values) == 3:  # 3 classes
                    # Get the values for the predicted class
                    # Map prediction_direction to the model's internal class index (-1->0, 0->1, 1->2)
                    class_idx = predicted_class_idx = int(prediction_direction) + 1
                    shap_explanation_values = np.array(shap_values[class_idx])
                else:
                    # Not a multi-class output or unexpected format
                    shap_explanation_values = np.array(shap_values)
        except Exception as e:
            print(f"Error getting SHAP values: {e}")
            # Create dummy SHAP values (all equal) as fallback
            shap_explanation_values = np.ones(len(feature_names))
    
    # 8. Generate Advice
    advice = generate_advice(prediction_direction, shap_explanation_values, feature_names, current_news_summary)

    print("\n--- Fintel's Advice ---")
    print(advice)

    return prediction_direction, advice, current_close_price, TARGET_MAPPING[prediction_direction]
"""
if __name__ == "__main__":
    # You can call this function directly to get a prediction and advice
    pred_direction, advice_text, current_price, pred_label = get_latest_data_and_predict(ticker_symbol="AAPL")
    
    if pred_direction is not None:
        print(f"\nPredicted movement for AAPL: {pred_label}")
        print(f"Current AAPL Close Price (from previous trading day): ${current_price:.2f}")"""