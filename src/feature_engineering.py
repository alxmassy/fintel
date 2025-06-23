import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk import downloader
from sklearn.preprocessing import StandardScaler
import joblib # For saving/loading scaler
import os

# Download VADER lexicon (do this once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def create_features_and_target(stock_df_path="data/aapl_historical_prices.csv", 
                               news_df_path="data/aapl_recent_news.csv",
                               scaler_output_path="models/scaler.pkl"):
    """
    Loads stock and news data, creates features, defines the target variable,
    and splits into train/test sets.
    """
    print("\n--- Starting Feature Engineering ---")

    # 1. Load Data
    try:
        stock_df = pd.read_csv(stock_df_path, index_col='Date', parse_dates=True)
        news_df = pd.read_csv(news_df_path, parse_dates=['publishedAt'])
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure data acquisition ran successfully.")
        return None, None, None, None, None

    # Ensure index is sorted for time-series operations
    stock_df = stock_df.sort_index()

    # 2. Define Target Variable (Next Day's Price Direction)
    # Shift Close price to get next day's close for target calculation
    stock_df['Next_Day_Close'] = stock_df['Close'].shift(-1)
    
    # Calculate daily percentage change
    stock_df['Daily_Change_Pct'] = (stock_df['Next_Day_Close'] - stock_df['Close']) / stock_df['Close']

    # Define thresholds for 'Up', 'Down', 'Neutral'
    # Adjust these thresholds based on desired sensitivity (e.g., 0.005 = 0.5%)
    UP_THRESHOLD = 0.005
    DOWN_THRESHOLD = -0.005

    stock_df['Target_Direction'] = 0 # Default to Neutral
    stock_df.loc[stock_df['Daily_Change_Pct'] > UP_THRESHOLD, 'Target_Direction'] = 1 # Up
    stock_df.loc[stock_df['Daily_Change_Pct'] < DOWN_THRESHOLD, 'Target_Direction'] = -1 # Down

    # Drop the last row as its 'Next_Day_Close' and 'Target_Direction' will be NaN
    stock_df.dropna(subset=['Target_Direction'], inplace=True)
    stock_df['Target_Direction'] = stock_df['Target_Direction'].astype(int) # Ensure integer type

    print("Target variable 'Target_Direction' created.")
    print(stock_df['Target_Direction'].value_counts())

    # 3. Create Numerical Features (Lagged Prices & Technical Indicators)
    stock_df['Close_Lag1'] = stock_df['Close'].shift(1)
    stock_df['Volume_Lag1'] = stock_df['Volume'].shift(1)
    
    # Simple Moving Averages (SMA)
    stock_df['SMA_10'] = stock_df['Close'].rolling(window=10).mean()
    stock_df['SMA_20'] = stock_df['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI) - Simplified calculation for prototype
    # Real RSI is more complex, consider 'ta' library for production: pip install ta
    delta = stock_df['Close'].diff().astype(float)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_df['RSI'] = 100 - (100 / (1 + rs))
    print("Numerical features (lags, SMAs, RSI) created.")

    # 4. Textual Features (Sentiment Analysis from News)
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to news headlines and descriptions
    # Combine title and description for richer context
    news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
    news_df['sentiment_compound'] = news_df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    # Extract date from news publication timestamp
    news_df['news_date'] = news_df['publishedAt'].dt.normalize()

    # Aggregate news sentiment by day
    # For each stock day, consider news published on that day or the day before market open
    
    # We need to link news from day N (after close) and day N+1 (before close)
    # to the prediction for Day N+1's close.
    # Let's simplify for prototype: news for 'today' (up to market close) affects 'today's close',
    # and news from 'today after close' and 'tomorrow before open' affects 'tomorrow's close'.
    # For our daily prediction, we'll associate news published **before market close of day N+1**
    # with the target for **Day N+1**.

    # For the prototype, let's group all news by the day it was published
    # and consider it relevant for the NEXT trading day's prediction.
    # This simplifies the complex market open/close timing.
    daily_news_sentiment = news_df.groupby('news_date').agg(
        Avg_News_Sentiment=('sentiment_compound', 'mean'),
        News_Count=('sentiment_compound', 'count'),
        Positive_News_Count=('sentiment_compound', lambda x: (x > 0.05).sum()), # Count positive
        Negative_News_Count=('sentiment_compound', lambda x: (x < -0.05).sum()) # Count negative
    ).reset_index()

    # Shift news features to align with the *next* trading day's prediction
    daily_news_sentiment['news_date'] = daily_news_sentiment['news_date'] + pd.Timedelta(days=1)
    
    print("News sentiment features created and aggregated.")

    # 5. Merge Stock and News Data
    # Merge on the date. Use left merge to keep all stock days.
    # Ensure 'Date' index in stock_df and 'news_date' in daily_news_sentiment are aligned.
    # Convert stock_df index to datetime.date for merging with news_date
    stock_df_reset = stock_df.reset_index()
    stock_df_reset['Date_Only'] = stock_df_reset['Date'].dt.normalize()
    
    # Rename news_date to Date_Only for consistent merging
    daily_news_sentiment = daily_news_sentiment.rename(columns={'news_date': 'Date_Only'})

    # Ensure date columns are timezone-naive for merging
    daily_news_sentiment['Date_Only'] = daily_news_sentiment['Date_Only'].dt.tz_localize(None)

    df_merged = pd.merge(stock_df_reset, daily_news_sentiment, 
                         on='Date_Only', how='left')
    
    # Drop the temporary date column
    df_merged.drop(columns=['Date_Only'], inplace=True)
    df_merged.set_index('Date', inplace=True)


    # Fill NaN values for news features (e.g., if no news on a day)
    news_features_cols = ['Avg_News_Sentiment', 'News_Count', 'Positive_News_Count', 'Negative_News_Count']
    for col in news_features_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0) # Assume no news means neutral/zero count

    print("Stock and news data merged.")
    print(df_merged.head())
    print(df_merged.info())
    
    # 6. Final Data Cleaning (remove rows with NaNs after feature creation)
    df_final = df_merged.dropna()
    
    print(f"Initial data shape: {df_merged.shape}")
    print(f"Final data shape after dropping NaNs: {df_final.shape}")

    # 7. Define Features (X) and Target (y)
    features = [col for col in df_final.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Next_Day_Close', 'Daily_Change_Pct', 'Target_Direction']]
    X = df_final[features]
    y = df_final['Target_Direction']

    print(f"Features (X) columns: {X.columns.tolist()}")
    print(f"Target (y) shape: {y.shape}")

    # 8. Train-Test Split (Chronological)
    # 80% for training, 20% for testing
    split_point = int(len(df_final) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print(f"Train set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # 9. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to retain column names for SHAP and readability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Save the scaler for later use in prediction
    joblib.dump(scaler, scaler_output_path)
    print(f"Scaler saved to {scaler_output_path}")

    print("--- Feature Engineering Complete ---")
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist() # Return column names for later use

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = create_features_and_target()
    if X_train is not None:
        print("\nSuccessfully prepared data for training.")