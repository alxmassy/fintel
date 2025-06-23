import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def get_stock_data(ticker="AAPL", period="3y", interval="1d", output_path=None):
    """
    Downloads historical stock data using yfinance.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")
        period (str): Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval (str): Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        output_path (str): Path to save the CSV file. If None, will generate a path based on ticker.
    """
    if output_path is None:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        output_path = f"data/{ticker.lower()}_historical_prices.csv"
        
    print(f"Fetching historical stock data for {ticker} for the last {period}...")
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No stock data fetched for {ticker}. Check ticker or period.")
            return None
        
        # Ensure the index is a DatetimeIndex before removing timezone information
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None)
        else:
            print("Index is not a DatetimeIndex. Skipping timezone localization.")
        
        df.to_csv(output_path)
        print(f"{ticker} historical stock data saved to {output_path}")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def get_news_data(ticker="AAPL", days_ago=30, lang="en", output_path=None, strict_filtering=True):
    """
    Fetches recent news articles for a given stock ticker using NewsAPI.org.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")
        days_ago (int): Number of days to look back for news (limited by NewsAPI free tier).
        lang (str): Language of the news articles (e.g., 'en' for English).
        output_path (str): Path to save the CSV file. If None, will generate a path based on ticker.
        strict_filtering (bool): If True, applies additional filtering to ensure articles are directly related
                                to the company.
    """
    # Get company name based on ticker for better news search
    company_names = {
        "AAPL": "Apple",
        "GOOGL": "Google OR Alphabet OR Android OR YouTube",  # Enhanced Google query
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "META": "Meta OR Facebook",
        "TSLA": "Tesla",
        "NVDA": "NVIDIA",
    }
    
    company_name = company_names.get(ticker, ticker)
    
    # Build more precise query with company name in quotes for exact matches
    # and include ticker symbol separately
    query = f'"{company_name}" OR {ticker}'
    
    # Enhanced query for Google to include more product news
    if ticker == "GOOGL":
        # Build a more comprehensive query that includes products and controversies
        query = f'"{company_name}" OR {ticker} OR "Google AI" OR "Google products" OR "Google Cloud" OR "Android" OR "Chrome" OR "YouTube"'
        
        # For Google, also try to include some news about regulations or controversies
        # which typically have more negative sentiment for balance
        query += ' OR "Google antitrust" OR "Google regulation" OR "Google lawsuit"'
    
    if output_path is None:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        output_path = f"data/{ticker.lower()}_recent_news.csv"
    
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
        
        if strict_filtering and not news_df.empty:
            print(f"Applying strict filtering for {ticker} news articles...")
            
            # Add a combined text field for searching
            news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
            
            # Convert all text to lowercase for case-insensitive matching
            news_df['text'] = news_df['text'].str.lower()
            ticker_lower = ticker.lower()
            
            # Get company name variations for filtering
            company_name_variations = []
            if ticker in company_names:
                # Extract company name and create variations
                company_name = company_names[ticker].split(' OR ')[0]
                company_name_variations = [company_name.lower()]
                
                # Add common variations like Inc., Corp., etc.
                if ticker == "AAPL":
                    company_name_variations.extend(['apple inc', 'iphone', 'ipad', 'mac', 'ios'])
                elif ticker == "GOOGL":
                    company_name_variations.extend([
                        'google llc', 'alphabet inc', 'android', 'pixel', 'chrome', 
                        'gmail', 'youtube', 'google cloud', 'waymo', 'deepmind',
                        'google search', 'google maps', 'google assistant',
                        'sundar pichai', 'bard', 'gemini', 'alphabet company'
                    ])
                elif ticker == "MSFT":
                    company_name_variations.extend(['microsoft corp', 'windows', 'azure', 'xbox'])
                elif ticker == "AMZN":
                    company_name_variations.extend(['amazon.com', 'aws', 'prime', 'alexa'])
                elif ticker == "META":
                    company_name_variations.extend(['facebook', 'instagram', 'whatsapp', 'oculus', 'meta platforms'])
                elif ticker == "TSLA":
                    company_name_variations.extend(['tesla motors', 'elon musk', 'model s', 'model 3', 'model x', 'model y'])
                elif ticker == "NVDA":
                    company_name_variations.extend(['nvidia corp', 'geforce', 'cuda', 'gpu'])
            
            # Filter articles that clearly mention the ticker or company name
            relevant_mask = news_df['text'].str.contains(ticker_lower)
            for variation in company_name_variations:
                relevant_mask |= news_df['text'].str.contains(variation)
                
            # Apply the filter
            relevant_articles = news_df[relevant_mask]
            
            # If we have relevant articles, use those; otherwise fall back to original results
            if not relevant_articles.empty:
                filtered_count = len(news_df) - len(relevant_articles)
                news_df = relevant_articles
                print(f"Filtered out {filtered_count} irrelevant articles. Kept {len(news_df)} articles about {ticker}.")
            else:
                print(f"Strict filtering removed all articles. Using original results.")
        
        # Add special case for financial stock holding news to avoid overwhelming positive bias
        if ticker == "GOOGL":
            # For Google specifically, if we have a lot of stock holding news,
            # limit them to a smaller percentage of the total articles
            stock_holding_pattern = r'(shares|stake|position|holdings|acquires|cuts|trims|boosts|raises|lowers)'
            stock_news_mask = news_df['text'].str.lower().str.contains(stock_holding_pattern, regex=True)
            product_news_mask = ~stock_news_mask
            
            # Calculate how many stock news articles we have
            stock_news_count = stock_news_mask.sum()
            product_news_count = product_news_mask.sum()
            total_news_count = len(news_df)
            
            # If more than 70% are stock holding news and we have at least 10 articles,
            # balance the distribution to include more product/tech news
            if stock_news_count > 10 and stock_news_count / total_news_count > 0.7:
                # Keep all product news and a subset of stock news
                target_stock_news = min(10, int(total_news_count * 0.5))
                stock_news_to_keep = news_df[stock_news_mask].sample(n=target_stock_news) if stock_news_count > target_stock_news else news_df[stock_news_mask]
                news_df = pd.concat([news_df[product_news_mask], stock_news_to_keep])
                print(f"Balanced Google news: kept {len(news_df[product_news_mask])} product news and {len(stock_news_to_keep)} stock news articles")
                
        news_df.to_csv(output_path, index=False)
        print(f"{ticker} recent news data saved to {output_path}")
        print(news_df.head())
        return news_df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news data: {e}")
        if hasattr(response, 'status_code') and response.status_code == 429:
            print("You might have hit the NewsAPI.org rate limit or free tier historical limit.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Keep backward compatibility
def get_apple_stock_data(period="3y", interval="1d", output_path="data/aapl_historical_prices.csv"):
    return get_stock_data(ticker="AAPL", period=period, interval=interval, output_path=output_path)

def get_apple_news_data(query="Apple OR AAPL", lang="en", output_path="data/aapl_recent_news.csv", days_ago=30):
    return get_news_data(ticker="AAPL", days_ago=days_ago, lang=lang, output_path=output_path)

if __name__ == "__main__":
    # --- Fetch Stock Data ---
    stock_df = get_stock_data("AAPL")
    google_df = get_stock_data("GOOGL")

    # --- Fetch News Data ---
    # Note: Free NewsAPI.org accounts are limited to ~1 month of historical data.
    news_df = get_news_data("AAPL")
    google_news_df = get_news_data("GOOGL")

    print("\n--- Data Acquisition Complete ---")
    if stock_df is not None:
        print(f"Stock data shape: {stock_df.shape}")
    if google_df is not None:
        print(f"Google stock data shape: {google_df.shape}")
    if news_df is not None:
        print(f"News data shape: {news_df.shape}")
    if google_news_df is not None:
        print(f"Google news data shape: {google_news_df.shape}")