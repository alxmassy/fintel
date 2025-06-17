import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def get_apple_stock_data(period="3y", interval="1d", output_path="data/aapl_historical_prices.csv"):
    """
    Downloads historical stock data for Apple (AAPL) using yfinance.

    Args:
        period (str): Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval (str): Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        output_path (str): Path to save the CSV file.
    """
    print(f"Fetching historical stock data for AAPL for the last {period}...")
    try:
        ticker = yf.Ticker("AAPL")
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print("No stock data fetched. Check ticker or period.")
            return None
        
        df.index = df.index.tz_localize(None) # Remove timezone for easier merging
        df.to_csv(output_path)
        print(f"AAPL historical stock data saved to {output_path}")
        print(df.head())
        print(df.info())
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def get_apple_news_data(query="Apple OR AAPL", lang="en", output_path="data/aapl_recent_news.csv", days_ago=30):
    """
    Fetches recent news articles for Apple using NewsAPI.org.
    Note: Free tier usually limits to the last month.

    Args:
        query (str): Search query for news.
        lang (str): Language of the news articles (e.g., 'en' for English).
        output_path (str): Path to save the CSV file.
        days_ago (int): Number of days to look back for news (limited by NewsAPI free tier).
    """
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    if not NEWS_API_KEY:
        print("NEWS_API_KEY not found in .env. Please set it up.")
        return None

    # Calculate dates for the 'from' and 'to' parameters
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_ago)

    # NewsAPI.org URL for all articles
    url = "https://newsapi.org/v2/everything"
    
    # Parameters for the request
    params = {
        "q": query,
        "language": lang,
        "sortBy": "publishedAt",
        "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "to": to_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "apiKey": NEWS_API_KEY
    }

    print(f"\nFetching news data for '{query}' from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            print("No news articles found for the query and date range.")
            return None

        news_df = pd.DataFrame(articles)
        
        # Select relevant columns and clean up
        news_df = news_df[['publishedAt', 'title', 'description', 'source', 'url']]
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
        
        news_df.to_csv(output_path, index=False)
        print(f"AAPL recent news data saved to {output_path}")
        print(news_df.head())
        print(news_df.info())
        return news_df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news data: {e}")
        if response and response.status_code == 429:
            print("You might have hit the NewsAPI.org rate limit or free tier historical limit.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # --- Fetch Stock Data ---
    stock_df = get_apple_stock_data()

    # --- Fetch News Data ---
    # Note: Free NewsAPI.org accounts are limited to ~1 month of historical data.
    # This will mainly fetch very recent news for the prototype.
    news_df = get_apple_news_data()

    print("\n--- Data Acquisition Complete ---")
    if stock_df is not None:
        print(f"Stock data shape: {stock_df.shape}")
    if news_df is not None:
        print(f"News data shape: {news_df.shape}")