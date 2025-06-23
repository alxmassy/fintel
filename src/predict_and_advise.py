# This scrip was originally created to predict stock movements and provide advice based on the latest stock and news data.
# But due to some issues with the code and the SHAP model, we have simplified and integrated the logic into a single script of streamlit_app.py.
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

# Import generic data acquisition functions for multiple stocks
from data_acquisition import get_stock_data, get_news_data
from feature_engineering import create_features_and_target # We'll reuse parts of this logic

# Define the target mapping (from feature_engineering.py)
# Added '2: "Up"' to handle alternative model output mappings
TARGET_MAPPING = {1: "Up", -1: "Down", 0: "Neutral", 2: "Up"}

def generate_advice(prediction_direction, top_features=None, feature_names=None, current_news_summary=None):
    """
    Generates advice based on model prediction.
    
    Args:
        prediction_direction: The predicted direction (-1, 0, 1, or 2)
        top_features: Optional Series of important features with importance values
        feature_names: List of feature names used in the model
        current_news_summary: Dictionary of current news metrics
        
    Returns:
        String with prediction advice
    """
    # Convert NumPy integer types to standard Python int if needed
    if isinstance(prediction_direction, np.number):
        prediction_direction = int(prediction_direction)
        
    # Handle unexpected prediction values
    if prediction_direction not in TARGET_MAPPING:
        prediction_direction = 0  # Default to Neutral
        print(f"Unexpected prediction value: {prediction_direction}. Using fallback.")
        
    advice_text = f"Fintel predicts the stock will move: **{TARGET_MAPPING[prediction_direction]}**.\n\n"
    
    # If no top features provided, we'll just return the basic prediction
    if top_features is None or feature_names is None:
        advice_text += "\n*Note: This is a prototype system. Do not use for real financial decisions.*"
        return advice_text
        
    # If we have top features, include them in the advice
    advice_text += "This prediction is primarily influenced by:\n"
    
    for feature, importance in top_features.items():
        if "News_Count" in str(feature) and current_news_summary and 'News_Count' in current_news_summary:
            value = f"{current_news_summary['News_Count']} relevant articles"
            if current_news_summary['Avg_News_Sentiment'] > 0.05:
                advice_text += f"- **Positive sentiment** from recent news (e.g., {value}).\n"
            elif current_news_summary['Avg_News_Sentiment'] < -0.05:
                advice_text += f"- **Negative sentiment** from recent news (e.g., {value}).\n"
            else:
                advice_text += f"- **Neutral sentiment** from recent news (e.g., {value}).\n"
        elif 'Avg_News_Sentiment' in str(feature) and current_news_summary and 'Avg_News_Sentiment' in current_news_summary:
            sentiment = current_news_summary['Avg_News_Sentiment']
            if sentiment > 0.05:
                advice_text += f"- Generally **positive news sentiment** (score: {sentiment:.2f}).\n"
            elif sentiment < -0.05:
                advice_text += f"- Generally **negative news sentiment** (score: {sentiment:.2f}).\n"
            else:
                advice_text += f"- Predominantly **neutral news sentiment** (score: {sentiment:.2f}).\n"
        elif 'Close_Lag1' in str(feature):
            advice_text += f"- The **previous day's closing price**.\n"
        elif 'SMA_10' in str(feature):
            advice_text += f"- The **10-day Simple Moving Average** (indicating short-term trend).\n"
        elif 'SMA_20' in str(feature):
            advice_text += f"- The **20-day Simple Moving Average** (indicating medium-term trend).\n"
        elif 'RSI' in str(feature):
            advice_text += f"- The **Relative Strength Index (RSI)** (indicating overbought/oversold conditions).\n"
        elif 'Volume_Lag1' in str(feature):
             advice_text += f"- The **previous day's trading volume**.\n"
        else:
            advice_text += f"- **{feature}** (a significant underlying factor).\n"
    
    advice_text += "\n*Note: This is a prototype system. Do not use for real financial decisions.*"
    return advice_text


def get_latest_data_and_predict(ticker_symbol="AAPL"):
    """
    Fetches the very latest data, preprocesses it, and makes a prediction.
    
    Args:
        ticker_symbol: Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
        
    Returns:
        Tuple of (prediction_direction, advice_text, current_price, prediction_label)
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
        output_path = f"data/{ticker_symbol.lower()}_latest_stock.csv"
        latest_stock_df = get_stock_data(ticker=ticker_symbol, period="10d", output_path=output_path)
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
    output_path = f"data/{ticker_symbol.lower()}_latest_news.csv"
    latest_news_df = get_news_data(ticker=ticker_symbol, days_ago=2, output_path=output_path, strict_filtering=True)

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
    if len(latest_stock_df) < 20: # Adjust based on max window of your TAs
        print("Not enough historical data to compute all features for current day.")
        # Fallback to a longer period if necessary or use simpler features
        output_path = f"data/{ticker_symbol.lower()}_latest_stock.csv"
        latest_stock_df = get_stock_data(ticker=ticker_symbol, period="30d", output_path=output_path)
        if latest_stock_df is None or len(latest_stock_df) < 20:
            print("Still not enough data. Cannot proceed.")
            return None, None, None, None
    
    # We'll re-calculate TAs on the `latest_stock_df` to get the last valid row's features.
    temp_df = latest_stock_df.copy()
    # Ensure Close is numeric before calculations
    temp_df['Close'] = pd.to_numeric(temp_df['Close'], errors='coerce')
    temp_df['Close_Lag1'] = temp_df['Close'].shift(1)
    temp_df['Volume_Lag1'] = temp_df['Volume'].shift(1)
    temp_df['SMA_10'] = temp_df['Close'].rolling(window=10).mean()
    temp_df['SMA_20'] = temp_df['Close'].rolling(window=20).mean()
    delta = temp_df['Close'].diff()
    # Convert delta to numeric type before comparison
    delta = pd.to_numeric(delta, errors='coerce')
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

    # 7. Generate feature importance for explanation
    top_features = None
    try:
        # Use feature importances from the model instead of complex SHAP values
        if hasattr(model, 'feature_importances_'):
            # For tree-based models like XGBoost
            importances = model.feature_importances_
            top_features = pd.Series(importances, index=feature_names).nlargest(5)
        else:
            # Try to use a simplified SHAP approach if feature_importances_ is not available
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_new_scaled_df)
                
                # Handle different return types from shap
                if isinstance(shap_values, list):
                    # For multi-class, take the values for the predicted class or average
                    if len(shap_values) >= 3:  # Assume 3-class classification
                        class_idx = min(int(prediction_direction) + 1, len(shap_values) - 1)
                        abs_vals = np.abs(shap_values[class_idx])
                    else:
                        # Average across classes if mapping is unclear
                        abs_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                else:
                    # For single output models
                    abs_vals = np.abs(shap_values)
                
                # If we have multiple instances, take the first one
                if abs_vals.ndim > 1 and abs_vals.shape[0] > 1:
                    abs_vals = abs_vals[0]
                    
                # Create a Series for sorting
                top_features = pd.Series(abs_vals, index=feature_names).nlargest(5)
            except Exception as e:
                print(f"Error with SHAP: {e}")
                # Fall back to feature names only without importances
                top_features = None
    except Exception as e:
        print(f"Error calculating feature importances: {e}")
        top_features = None
    
    # 8. Generate Advice
    advice = generate_advice(prediction_direction, top_features, feature_names, current_news_summary)

    print("\n--- Fintel's Advice ---")
    print(advice)

    return prediction_direction, advice, current_close_price, TARGET_MAPPING[prediction_direction]
if __name__ == "__main__":
    import sys
    
    # Get ticker symbol from command line arguments or use default
    ticker = "AAPL"  # Default to Apple
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    
    print(f"Running prediction for {ticker}...")
    
    # Get prediction and advice
    pred_direction, advice_text, current_price, pred_label = get_latest_data_and_predict(ticker_symbol=ticker)
    
    if pred_direction is not None:
        print(f"\nPredicted movement for {ticker}: {pred_label}")
        print(f"Current {ticker} Close Price (from previous trading day): ${current_price:.2f}")
        print("\nDetailed advice:")
        print(advice_text)
    else:
        print(f"Could not generate prediction for {ticker}.")