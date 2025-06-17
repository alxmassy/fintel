# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from dotenv import load_dotenv # To load NEWS_API_KEY

# Import the functions from your backend scripts
from src.data_acquisition import get_apple_stock_data, get_apple_news_data
from src.feature_engineering import create_features_and_target

# Load environment variables (especially NEWS_API_KEY)
load_dotenv()

# Download VADER lexicon if not already available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- Replicate TARGET_MAPPING and generate_advice from predict_and_advise.py ---
# You could put these in a separate 'utils.py' file in src/ for cleaner code
TARGET_MAPPING = {1: "Up", -1: "Down", 0: "Neutral"}

def generate_advice(prediction_direction, shap_values, feature_names, current_news_summary):
    advice_text = f"Fintel predicts the stock will move: **{TARGET_MAPPING[prediction_direction]}**.\n\n"

    if not isinstance(shap_values, (np.ndarray, pd.Series, list)):
        st.warning("Warning: SHAP values not in expected format. Cannot generate detailed advice.")
        return advice_text + "No detailed explanation available."

    if isinstance(shap_values, list) and all(isinstance(val, np.ndarray) for val in shap_values):
        if len(shap_values) == 3 and isinstance(shap_values[0], np.ndarray):
            # Multi-class model output - get values for predicted class
            class_idx = prediction_direction + 1  # Map -1->0, 0->1, 1->2
            abs_shap_values = np.abs(np.array(shap_values[class_idx]))
        else:
            abs_shap_values = np.abs(np.array(shap_values))
    else:
        abs_shap_values = np.abs(np.array(shap_values))

    # Ensure `abs_shap_values` is 1-D array or extract values for predicted class
    if abs_shap_values.ndim > 1:
        # For multi-class model, the shape could be (n_classes, n_features) or (n_features, n_classes)
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
            # If shape doesn't match expectations, reshape to match features
            abs_shap_values = np.mean(abs_shap_values, axis=0)
            if len(abs_shap_values) != len(feature_names):
                # Flatten and take first n elements to match feature_names length
                abs_shap_values = abs_shap_values.flatten()[:len(feature_names)]
    
    feature_importances = pd.Series(abs_shap_values, index=feature_names)
    top_features = feature_importances.nlargest(3)

    advice_text += "This prediction is primarily influenced by:\n"
    for feature, importance in top_features.items():
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

# --- Main prediction function (similar to predict_and_advise.py) ---
@st.cache_resource # Cache the model loading for efficiency
def load_model_components():
    """Loads the trained model, scaler, and feature names."""
    try:
        model = joblib.load("models/xgboost_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Error loading model components: {e}. Please ensure you've run 'python src/train_model.py'.")
        return None, None, None

def get_prediction_and_advice(ticker_symbol="AAPL"):
    """
    Fetches the very latest data, preprocesses it, and makes a prediction.
    """
    model, scaler, feature_names = load_model_components()
    if model is None:
        return None, None, None, None

    # 1. Get Latest Stock Data
    latest_stock_df = get_apple_stock_data(period="30d", output_path="data/aapl_latest_stock.csv") # Fetch enough for TAs
    if latest_stock_df is None or latest_stock_df.empty:
        st.error("Could not fetch latest stock data for prediction.")
        return None, None, None, None
    current_day_data = latest_stock_df.iloc[-1]
    current_close_price = current_day_data['Close']

    # 2. Get Latest News Data
    latest_news_df = get_apple_news_data(query=ticker_symbol, days_ago=2, output_path="data/aapl_latest_news.csv")

    current_news_summary = {}
    if latest_news_df is not None and not latest_news_df.empty:
        analyzer = SentimentIntensityAnalyzer() # Initialize VADER
        latest_news_df['text'] = latest_news_df['title'].fillna('') + ' ' + latest_news_df['description'].fillna('')
        latest_news_df['sentiment_compound'] = latest_news_df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        
        current_news_summary = {
            'Avg_News_Sentiment': latest_news_df['sentiment_compound'].mean() if not latest_news_df.empty else 0,
            'News_Count': len(latest_news_df),
            'Positive_News_Count': (latest_news_df['sentiment_compound'] > 0.05).sum(),
            'Negative_News_Count': (latest_news_df['sentiment_compound'] < -0.05).sum()
        }
    else:
        current_news_summary = {'Avg_News_Sentiment': 0.0, 'News_Count': 0, 'Positive_News_Count': 0, 'Negative_News_Count': 0}

    # 3. Create Features for the NEW Prediction (replicate feature_engineering logic)
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

    features_for_prediction = temp_df.dropna().iloc[[-1]]
    
    # Add aggregated news features
    features_for_prediction['Avg_News_Sentiment'] = current_news_summary['Avg_News_Sentiment']
    features_for_prediction['News_Count'] = current_news_summary['News_Count']
    features_for_prediction['Positive_News_Count'] = current_news_summary['Positive_News_Count']
    features_for_prediction['Negative_News_Count'] = current_news_summary['Negative_News_Count']
    
    # Select and scale features
    try:
        X_new = features_for_prediction[feature_names]
        X_new_scaled = scaler.transform(X_new)
        X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=feature_names, index=X_new.index)
    except KeyError as e:
        st.error(f"Feature mismatch error: {e}. This usually means the feature engineering for live data doesn't perfectly match the training features. Please check column names and order.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error scaling new features: {e}. Ensure data is sufficient and correct.")
        return None, None, None, None

    # 4. Make Prediction
    prediction_array = model.predict(X_new_scaled_df)
    prediction_direction = prediction_array[0]
    
    # 5. Generate SHAP values for explanation
    explainer = shap.TreeExplainer(model)
    try:
        # For multi-class models, SHAP returns values for each class
        # Try different methods to get meaningful SHAP values
        shap_explanation = explainer(X_new_scaled_df)
        
        # Check if we have a multi-class output
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
                # XGBoost returns a list of arrays (one per class)
                shap_explanation_values = shap_values  # Handle in generate_advice
            else:
                # Not a multi-class output or unexpected format
                shap_explanation_values = np.array(shap_values)
                
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}. Advice might be limited.")
        # Create dummy SHAP values as fallback
        shap_explanation_values = np.ones(len(feature_names))

    # 6. Generate Advice
    advice = generate_advice(prediction_direction, shap_explanation_values, feature_names, current_news_summary)

    return prediction_direction, advice, current_close_price, TARGET_MAPPING[prediction_direction]


# --- Streamlit GUI Layout ---
st.set_page_config(
    page_title="Fintel: News-Driven Stock Prediction",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("📈 Fintel: News-Driven Stock Forecasting")
st.markdown(
    """
    Welcome to Fintel, your prototype AI for single-stock price movement and news-driven insights.
    Enter a stock ticker to get a prediction for the next trading day and AI-generated advice!
    """
)

st.warning("🚨 **Disclaimer:** This is a *prototype system* for educational purposes only. Do NOT use it for real financial decisions or trading.")

# Input for stock ticker
ticker_input = st.text_input(
    "Enter Stock Ticker (e.g., AAPL):",
    value="AAPL",
    max_chars=5
).upper() # Convert to uppercase to match common ticker formats

# Prediction button
if st.button("Get Fintel Prediction"):
    if not ticker_input:
        st.error("Please enter a stock ticker.")
    else:
        with st.spinner(f"Fetching data and generating prediction for {ticker_input}..."):
            pred_direction, advice_text, current_price, pred_label = get_prediction_and_advice(ticker_input)

        if pred_direction is not None:
            st.subheader(f"Prediction for {ticker_input} (Next Trading Day):")
            
            # Display current price if available
            if current_price is not None:
                st.info(f"Current Stock Close Price (from previous trading day): **${current_price:.2f}**")

            # Display prediction direction with styling
            if pred_direction == 1:
                st.success(f"**Predicted Movement: {pred_label}** ⬆️")
            elif pred_direction == -1:
                st.error(f"**Predicted Movement: {pred_label}** ⬇️")
            else:
                st.info(f"**Predicted Movement: {pred_label}** ➡️")

            st.subheader("Fintel's AI-Driven Advice:")
            st.markdown(advice_text)
        else:
            st.error("Could not complete prediction. Please check logs for details or try again later.")
            st.write("Possible reasons: NewsAPI.org rate limit, no recent news, or data fetching issues.")

st.sidebar.header("About Fintel")
st.sidebar.info(
    """
    Fintel is a machine learning prototype that combines historical stock price data with recent news sentiment to forecast short-term stock movements.
    It uses an XGBoost classifier for prediction and SHAP (SHapley Additive exPlanations) for generating the advice.

    **Developed by:** [Your Name/GitHub Link if you want]
    """
)
st.sidebar.text("Version: Prototype 1.0")