import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import plotly.graph_objects as go
import plotly.express as px
import os
import requests
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from dotenv import load_dotenv 
from src.data_acquisition import get_stock_data, get_news_data
from src.feature_engineering import create_features_and_target

load_dotenv()

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

TARGET_MAPPING = {1: "Up", -1: "Down", 0: "Neutral", 2: "Up"}

def generate_advice(prediction_direction):
    # Convert NumPy integer types to standard Python int
    if isinstance(prediction_direction, np.number):
        prediction_direction = int(prediction_direction)

    # Handle unexpected prediction values
    if prediction_direction not in TARGET_MAPPING:
        prediction_direction = 0  # Default to Neutral
        st.warning(f"Unexpected prediction value: {prediction_direction}. Using fallback.")

    advice_text = f"Fintel predicts the stock will move: {TARGET_MAPPING[prediction_direction]}.\n\n"
    return advice_text

# --- Main prediction function ---

@st.cache_resource 
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
    latest_stock_df = get_stock_data(ticker=ticker_symbol, period="30d", output_path=f"data/{ticker_symbol.lower()}_latest_stock.csv")
    if latest_stock_df is None or latest_stock_df.empty:
        st.error("Could not fetch latest stock data for prediction.")
        return None, None, None, None
    current_day_data = latest_stock_df.iloc[-1]
    current_close_price = current_day_data['Close']

    # 2. Get Latest News Data
    days_to_fetch = 7 if ticker_symbol == "GOOGL" else 2  # Use a longer window for Google
    latest_news_df = get_news_data(ticker=ticker_symbol, days_ago=days_to_fetch, output_path=f"data/{ticker_symbol.lower()}_latest_news.csv", strict_filtering=True)

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
    # Explicitly convert delta to numeric before comparisons
    delta = pd.to_numeric(delta, errors='coerce')
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
        if scaler is not None:
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
    raw_prediction = prediction_array[0]
    
    # Check if we need to map the prediction from [0, 1, 2] back to [-1, 0, 1]
    # Load the label mapping if available
    try:
        label_mapping = joblib.load('models/label_mapping.pkl')
        # label_mapping is {0: -1, 1: 0, 2: 1} (XGBoost class index to original label)
        prediction_direction = label_mapping.get(int(raw_prediction), 0)  # Default to 0 (Neutral) if mapping fails
    except (FileNotFoundError, KeyError):
        # If label mapping file doesn't exist or mapping fails, use simple heuristic
        if isinstance(raw_prediction, int):
            raw_prediction = int(raw_prediction)
        
        # Map predictions based on likely patterns
        if raw_prediction == 2:
            prediction_direction = 1  # Map 2 to 1 (Up)
        elif raw_prediction == 0:
            prediction_direction = -1  # Map 0 to -1 (Down)
        else:
            prediction_direction = raw_prediction  # Keep as is
    
    # 6. Generate Advice
    advice = generate_advice(prediction_direction)

    return prediction_direction, advice, current_close_price, TARGET_MAPPING[prediction_direction]

# --- Check for available tickers ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def check_ticker_data_availability(tickers_list):
    """
    Check which tickers have both price and news data available.
    Returns a filtered list of tickers with available data.
    """
    available_tickers = []
    
    for ticker in tickers_list:
        # Check if we already have data files for this ticker
        price_file = f"data/{ticker.lower()}_historical_prices.csv"
        news_file = f"data/{ticker.lower()}_recent_news.csv"
        
        has_price_data = os.path.exists(price_file)
        has_news_data = os.path.exists(news_file)
        
        # If we don't have files, do a quick check without saving
        if not has_price_data:
            try:
                # Try to get minimal data quickly
                ticker_obj = yf.Ticker(ticker)
                price_history = ticker_obj.history(period="1d")  # Just get one day to check
                has_price_data = not price_history.empty
            except Exception as e:
                print(f"Error checking price data for {ticker}: {e}")
                has_price_data = False
        
        if not has_news_data:
            # Check if we have a NEWS_API_KEY
            NEWS_API_KEY = os.getenv("NEWS_API_KEY")
            if not NEWS_API_KEY:
                # If no API key, just assume we have news data to avoid blocking the app
                has_news_data = True
            else:
                try:
                    # Check for news data without saving
                    company_names = {
                        "AAPL": "Apple",
                        "GOOGL": "Google",
                        "MSFT": "Microsoft",
                        "AMZN": "Amazon",
                        "META": "Meta",
                        "TSLA": "Tesla",
                        "NVDA": "NVIDIA",
                    }
                    
                    query = f'"{company_names.get(ticker, ticker)}" OR {ticker}'
                    
                    # Calculate dates for minimal query
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=1)  # Just check 1 day
                    
                    # NewsAPI.org URL
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": query,
                        "language": "en",
                        "pageSize": 1,  # Just get 1 result to check
                        "sortBy": "publishedAt",
                        "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
                        "to": to_date.strftime("%Y-%m-%dT%H:%M:%S"),
                        "apiKey": NEWS_API_KEY
                    }
                    
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get("articles", [])
                        has_news_data = len(articles) > 0
                    else:
                        print(f"API error checking news for {ticker}: {response.status_code}")
                        # If we get an error (like rate limit), assume we have data
                        has_news_data = True
                except Exception as e:
                    print(f"Error checking news data for {ticker}: {e}")
                    has_news_data = True  # Default to True on error to not block app
        
        # Include ticker if both data sources are available
        if has_price_data and has_news_data:
            available_tickers.append(ticker)
    
    return available_tickers

# --- Streamlit GUI Layout ---
st.set_page_config(
    page_title="Fintel",
    page_icon="📈",
    layout="wide",  # Use wide layout for better graphs
    initial_sidebar_state="collapsed",
)

# Custom CSS for better visual appeal
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.5em;
        color: #424242;
        margin-bottom: 1em;
    }
    .stock-ticker {
        font-size: 1.5em;
        font-weight: bold;
    }
    .prediction-card {
        border-radius: 5px;
        padding: 1.5em;
        margin: 1em 0;
    }
    .footer {
        margin-top: 3em;
        text-align: center;
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 Fintel: News-Driven Stock Forecasting</div>', unsafe_allow_html=True)


# Main content area for Stock Prediction
col1, col2 = st.columns([1, 2])
    
with col1:
        # Stock selection with common options
        st.markdown("### Select Stock")
        all_stock_options = {
            "AAPL": "Apple Inc.",
            "GOOGL": "Alphabet Inc. (Google)",
            "MSFT": "Microsoft Corp.",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc.",
            "TSLA": "Tesla Inc.",
            "NVDA": "NVIDIA Corp."
        }
        
        # Check which tickers have available data
        with st.spinner("Checking data availability..."):
            available_tickers = check_ticker_data_availability(list(all_stock_options.keys()))
            
            if not available_tickers:
                st.warning("No tickers with complete data found. Showing all options.")
                available_tickers = list(all_stock_options.keys())
            else:
                st.success(f"Found {len(available_tickers)} stocks with complete data.")
            
            # Create filtered stock options
            stock_options = {ticker: all_stock_options[ticker] for ticker in available_tickers}
        
        ticker_input = st.selectbox(
            "Choose a stock with available data:",
            options=available_tickers,
            format_func=lambda x: f"{x} - {all_stock_options[x]}"
        )
        
        # Alternative manual input
        custom_ticker = st.text_input("Or enter a custom ticker:", value="", max_chars=5)
        if custom_ticker:
            ticker_input = custom_ticker.upper()
            
        # Prediction button
        predict_button = st.button("Get Fintel Prediction", type="primary")
        
with col2:
        st.markdown("### Historical Price Chart")
        if 'ticker_input' in locals():
            with st.spinner(f"Loading historical data for {ticker_input}..."):
                # Get historical data for chart
                historical_data = get_stock_data(ticker=ticker_input, period="90d")
                if historical_data is not None:
                    # Create an interactive price chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=historical_data.index,
                        open=historical_data['Open'],
                        high=historical_data['High'],
                        low=historical_data['Low'],
                        close=historical_data['Close'],
                        name='Price'
                    ))
                    
                    # Add moving averages
                    historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data['SMA_20'],
                        line={'color': 'orange', 'width': 2},
                        name='20-Day MA'
                    ))
                    
                    fig.update_layout(
                        title=f'{ticker_input} Stock Price - Last 90 Days',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=500,
                        xaxis_rangeslider_visible=False,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Could not load historical data for {ticker_input}")

# Run prediction when button is clicked
if 'predict_button' in locals() and predict_button:
    if not ticker_input:
        st.error("Please enter a stock ticker.")
    else:
        with st.spinner(f"Fetching data and generating prediction for {ticker_input}..."):
            result = get_prediction_and_advice(ticker_input)
            if len(result) == 4:
                pred_direction, advice_text, current_price, pred_label = result
                top_features = None

            # Access the current_news_summary and latest_news_df from function
            current_news_summary = {}
            try:
                # Get latest news data for display
                days_to_fetch = 7 if ticker_input == "GOOGL" else 2  # Use a longer window for Google
                latest_news_df = get_news_data(ticker=ticker_input, days_ago=days_to_fetch, 
                                               output_path=f"data/{ticker_input.lower()}_latest_news.csv", 
                                               strict_filtering=True)
                if latest_news_df is not None and not latest_news_df.empty:
                    analyzer = SentimentIntensityAnalyzer()
                    latest_news_df['text'] = latest_news_df['title'].fillna('') + ' ' + latest_news_df['description'].fillna('')
                    latest_news_df['sentiment_compound'] = latest_news_df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

                    current_news_summary = {
                        'Avg_News_Sentiment': latest_news_df['sentiment_compound'].mean() if not latest_news_df.empty else 0,
                        'News_Count': len(latest_news_df),
                        'Positive_News_Count': (latest_news_df['sentiment_compound'] > 0.05).sum(),
                        'Negative_News_Count': (latest_news_df['sentiment_compound'] < -0.05).sum()
                    }
            except Exception as e:
                st.error(f"Error fetching news data: {e}")

        if pred_direction is not None:
            # Create a dashboard-like layout for the prediction results
            st.markdown("---")
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"### Prediction Results for {ticker_input}")

                # Price and movement prediction card with improved colors and styling
                movement_emoji = "⬆️" if pred_direction == 1 else "⬇️" if pred_direction == -1 else "➡️"
                
                # Improved color scheme with better contrast
                bg_color = "#f0fff0" if pred_direction == 1 else "#fff0f0" if pred_direction == -1 else "#f0f8ff"
                border_color = "#28a745" if pred_direction == 1 else "#dc3545" if pred_direction == -1 else "#17a2b8"
                text_color = "#0d6b1c" if pred_direction == 1 else "#8b0000" if pred_direction == -1 else "#0d557a"
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {bg_color}; border-left: 8px solid {border_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #333;">Current Price: <span style="font-weight: bold; color: #333;">${current_price:.2f}</span></h4>
                    <h2 style="color: {text_color}; margin: 15px 0;">Predicted Movement: <span style="font-weight: bold;">{pred_label} {movement_emoji}</span></h2>
                    <p style="color: #333; font-size: 1.1em;">{advice_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Stock performance visualization
                st.markdown("### Recent Performance")
                try:
                    # Get stock data for visualization if not already loaded
                    if 'historical_data' not in locals() or historical_data is None:
                        historical_data = get_stock_data(ticker_input, period="30d")
                    
                    if historical_data is not None:
                        # Create performance metrics
                        last_close = historical_data['Close'].iloc[-1]
                        prev_close = historical_data['Close'].iloc[-2] if len(historical_data) > 1 else None
                        
                        if prev_close:
                            daily_change = (last_close - prev_close) / prev_close * 100
                            weekly_change = (last_close - historical_data['Close'].iloc[-6] if len(historical_data) > 5 else last_close) / historical_data['Close'].iloc[-6] if len(historical_data) > 5 else 0 * 100
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric("Last Close", f"${last_close:.2f}", f"{daily_change:.2f}%")
                            
                            with metric_col2:
                                st.metric("1-Week Change", f"{weekly_change:.2f}%")
                            
                            with metric_col3:
                                avg_volume = historical_data['Volume'].mean()
                                st.metric("Avg Volume", f"{avg_volume/1000000:.1f}M")
                        
                        # Volume chart
                        volume_fig = px.bar(
                            historical_data,
                            x=historical_data.index,
                            y='Volume',
                            title=f"{ticker_input} Trading Volume"
                        )
                        volume_fig.update_layout(height=250)
                        st.plotly_chart(volume_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating performance visualization: {e}")
                
            with col2:
                # News sentiment visualization
                st.markdown("### News Sentiment Analysis")
                
                if current_news_summary and current_news_summary.get('News_Count', 0) > 0:
                    # Create a gauge chart for sentiment with improved colors
                    sentiment = current_news_summary['Avg_News_Sentiment']
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = sentiment,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "News Sentiment", 'font': {'color': 'white', 'size': 24}},
                        number = {'font': {'color': 'white', 'size': 20}},
                        gauge = {
                            'axis': {'range': [-1, 1], 'tickfont': {'color': '#333'}},
                            'bar': {'color': "#F4EE42"},
                            'steps': [
                                {'range': [-1, -0.05], 'color': "#FE7785"},
                                {'range': [-0.05, 0.05], 'color': "#50AEFC"},
                                {'range': [0.05, 1], 'color': "#69E16D"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': sentiment
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # News statistics
                    #st.markdown(f"**Article Count**: {current_news_summary['News_Count']}")
                    
                    # Pie chart of positive vs negative news
                    pos = current_news_summary['Positive_News_Count']
                    neg = current_news_summary['Negative_News_Count']
                    neutral = current_news_summary['News_Count'] - pos - neg
                    
                    sentiment_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                        'Count': [pos, neutral, neg]
                    })
                    
                    pie_fig = px.pie(
                        sentiment_df, 
                        values='Count', 
                        names='Sentiment',
                        color='Sentiment',
                        color_discrete_map={'Positive':'#66BB6A', 'Neutral':'#42A5F5', 'Negative':'#EF5350'},
                        hole=0.4
                    )
                    pie_fig.update_layout(height=250)
                    st.plotly_chart(pie_fig, use_container_width=True)
                else:
                    st.info("No news data available for sentiment analysis.")
                
            # Display news articles in an expandable section
            with st.expander("📰 Top News Articles"):
                if 'latest_news_df' in locals() and latest_news_df is not None and not latest_news_df.empty:
                    for i, (_, row) in enumerate(latest_news_df.head(5).iterrows()):
                        sentiment = row['sentiment_compound']
                        sentiment_label = "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
                        # Improved color scheme with better contrast
                        border_color = "#28a745" if sentiment > 0.05 else "#dc3545" if sentiment < -0.05 else "#17a2b8"
                        bg_color = "#f0fff0" if sentiment > 0.05 else "#fff0f0" if sentiment < -0.05 else "#f0f8ff"
                        text_color = "#0d6b1c" if sentiment > 0.05 else "#8b0000" if sentiment < -0.05 else "#0d557a"
                        
                        st.markdown(f"""
                        <div style="padding: 15px; margin-bottom: 15px; border-left: 5px solid {border_color}; background-color: {bg_color}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="color: {text_color}; margin-top: 0;">{row['title']}</h4>
                            <p style="color: #333;">{row['description']}</p>
                            <p style="text-align: right; font-style: italic; color: {text_color}; margin-bottom: 0;">
                                Sentiment: <strong>{sentiment_label}</strong> ({sentiment:.2f})
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No relevant news articles found.")
        else:
            st.error("Could not complete prediction. Please check logs for details or try again later.")
            st.write("Possible reasons: NewsAPI.org rate limit, no recent news, or data fetching issues.")

# --- Updated Sidebar ---
with st.sidebar:
    st.header("📊 Market Overview")
    
    # Display current market status
    market_status = "Open" if datetime.now().hour >= 9 and datetime.now().hour < 16 else "Closed"
    st.markdown(f"**Market Status**: {market_status}")
    
    # Quick stock lookup
    st.subheader("Quick Stock Lookup")
    
    all_quick_options = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA"]
    
    # Reuse our ticker availability check
    with st.spinner("Checking data availability..."):
        available_quick_tickers = check_ticker_data_availability(all_quick_options)
        
        if not available_quick_tickers:
            st.warning("No tickers with complete data found. Showing all options.")
            available_quick_tickers = all_quick_options
    
    quick_lookup = st.selectbox(
        "Select stock with data:",
        options=available_quick_tickers,
        key="quick_lookup"
    )
    
    if st.button("Show Chart", key="quick_chart"):
        with st.spinner("Loading chart..."):
            quick_data = get_stock_data(ticker=quick_lookup, period="30d")
            if quick_data is not None:
                fig = px.line(
                    quick_data, 
                    x=quick_data.index, 
                    y='Close',
                    title=f"{quick_lookup} - Last 30 Days"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not load data for {quick_lookup}")
    
    st.markdown("---")
    st.subheader("About Fintel")
    st.info(
        """
        Fintel is a machine learning prototype that combines historical stock price data with recent news sentiment to forecast short-term stock movements.
        It uses an XGBoost classifier for prediction.

        **Developed by:** [Team Recon]
        """
    )
    st.text("Version: 2.0")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def check_detailed_ticker_availability(tickers_list):
    """
    Check which tickers have both price and news data available.
    Returns a dictionary of tickers with a boolean indicating data availability.
    """
    available_tickers = {}
    
    for ticker in tickers_list:
        has_price_data = False
        has_news_data = False
        
        # Check for price data (quick check without saving)
        try:
            price_df = get_stock_data(ticker=ticker, period="7d")
            has_price_data = price_df is not None and not price_df.empty
        except Exception:
            has_price_data = False
            
        # Check for news data (quick check without saving)
        try:
            NEWS_API_KEY = os.getenv("NEWS_API_KEY")
            if NEWS_API_KEY:
                company_name = ticker
                if ticker in {"AAPL": "Apple", 
                              "GOOGL": "Google", 
                              "MSFT": "Microsoft",
                              "AMZN": "Amazon",
                              "META": "Meta",
                              "TSLA": "Tesla",
                              "NVDA": "NVIDIA"}:
                    company_name = {"AAPL": "Apple", 
                                   "GOOGL": "Google", 
                                   "MSFT": "Microsoft",
                                   "AMZN": "Amazon",
                                   "META": "Meta",
                                   "TSLA": "Tesla",
                                   "NVDA": "NVIDIA"}[ticker]
                
                # Build a query similar to get_news_data
                query = f'"{company_name}" OR {ticker}'
                
                # Calculate dates
                to_date = datetime.now()
                from_date = to_date - timedelta(days=2)  # Only check recent news
                
                # NewsAPI.org URL
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
                    "to": to_date.strftime("%Y-%m-%dT%H:%M:%S"),
                    "apiKey": NEWS_API_KEY
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    has_news_data = len(articles) > 0
        except Exception:
            has_news_data = False
        
        # Ticker is available if both price and news data exist
        available_tickers[ticker] = has_price_data and has_news_data
    
    return available_tickers