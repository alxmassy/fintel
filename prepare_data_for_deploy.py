"""
Prepare Data for Deployment

This script ensures that data files exist for the tickers we want to support
in our prototype deployment. It loads existing data or downloads new data if needed.
"""

import os
import pandas as pd
from src.data_acquisition import get_stock_data, get_news_data
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure nltk data is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# List of tickers to prepare data for
TICKERS_TO_PREPARE = [
    "AAPL",  # Apple
    "GOOGL", # Google
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta (Facebook)
    "TSLA",  # Tesla
    "NVDA",  # NVIDIA
]

def validate_data(ticker):
    """Check if data exists and is valid for the ticker"""
    print(f"\nChecking data for {ticker}...")
    price_file = f"data/{ticker.lower()}_historical_prices.csv"
    news_file = f"data/{ticker.lower()}_recent_news.csv"
    
    has_price_data = os.path.exists(price_file)
    has_news_data = os.path.exists(news_file)
    
    if has_price_data:
        try:
            df = pd.read_csv(price_file)
            if df.empty:
                print(f"✗ Price file exists but is empty: {price_file}")
                has_price_data = False
            else:
                print(f"✓ Price data found: {len(df)} records")
        except Exception as e:
            print(f"✗ Error reading price file {price_file}: {e}")
            has_price_data = False
    
    if has_news_data:
        try:
            df = pd.read_csv(news_file)
            if df.empty:
                print(f"✗ News file exists but is empty: {news_file}")
                has_news_data = False
            else:
                print(f"✓ News data found: {len(df)} articles")
                
                # Check for sentiment data
                if 'sentiment_compound' not in df.columns:
                    print("  Adding sentiment analysis to existing news data...")
                    analyzer = SentimentIntensityAnalyzer()
                    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
                    df['sentiment_compound'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
                    df.to_csv(news_file, index=False)
                    print("  ✓ Sentiment analysis added")
        except Exception as e:
            print(f"✗ Error reading news file {news_file}: {e}")
            has_news_data = False
    
    return has_price_data, has_news_data

def prepare_ticker_data(ticker):
    """Prepare data for the specified ticker"""
    print(f"\nPreparing data for {ticker}...")
    
    # Check if data already exists
    has_price_data, has_news_data = validate_data(ticker)
    
    # Download price data if needed
    if not has_price_data:
        print(f"Downloading price data for {ticker}...")
        try:
            price_df = get_stock_data(
                ticker=ticker, 
                period="90d", 
                output_path=f"data/{ticker.lower()}_historical_prices.csv"
            )
            
            if price_df is not None and not price_df.empty:
                print(f"✓ Successfully downloaded {len(price_df)} price records")
            else:
                print(f"✗ Failed to download price data for {ticker}")
        except Exception as e:
            print(f"✗ Error downloading price data: {e}")
    
    # Download news data if needed
    if not has_news_data:
        print(f"Downloading news data for {ticker}...")
        days = 7 if ticker == "GOOGL" else 2  # Use longer window for Google
        
        try:
            news_df = get_news_data(
                ticker=ticker, 
                days_ago=days,
                output_path=f"data/{ticker.lower()}_recent_news.csv",
                strict_filtering=True
            )
            
            if news_df is not None and not news_df.empty:
                print(f"✓ Successfully downloaded {len(news_df)} news articles")
            else:
                print(f"✗ Failed to download news data for {ticker}")
        except Exception as e:
            print(f"✗ Error downloading news data: {e}")
    
    # Check if model files exist
    if not os.path.exists("models/xgboost_model.pkl"):
        print("⚠️ Warning: Model file 'models/xgboost_model.pkl' not found.")
        print("   You may need to run training before deployment.")

def main():
    """Main function to prepare all data"""
    print("=== Fintel: Preparing Data for Deployment ===")
    
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory.")
    
    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created models directory.")
    
    # Prepare data for each ticker
    for ticker in TICKERS_TO_PREPARE:
        prepare_ticker_data(ticker)
    
    print("\n=== Data Preparation Complete ===")

if __name__ == "__main__":
    main()
