# 📈 Fintel: News-Driven Stock Forecasting Prototype

## Overview

Fintel is a proof-of-concept machine learning system designed to predict short-term stock price movements (Up, Down, or Neutral) by leveraging both historical price data and recent news sentiment.

## Features

*   **Multi-Modal Data Integration**: Combines numerical stock with latest financial news.
*   **Sentiment Analysis**: Utilizes VADER (Valence Aware Dictionary and sEntiment Reasoner) to extract sentiment scores from news headlines and descriptions.
*   **Machine Learning Prediction**: Employs an XGBoost Classifier to predict the next day's stock movement direction (Up, Down, Neutral).
*   **Web-Based GUI**: A simple, intuitive graphical user interface built with Streamlit for easy interaction and demonstration.

## Click for Video Demo 
[![Watch the Demo](https://img.youtube.com/vi/6gytpbiClyE/hqdefault.jpg)](https://www.youtube.com/watch?v=6gytpbiClyE)
## How it Works (Under the Hood)

1.  **Data Acquisition**:
    *   **Stock Prices**: Uses `yfinance` to fetch historical data for the stock.
    *   **News Articles**: Connects to `NewsAPI.org` to retrieve recent news headlines and descriptions related to the stock.
2.  **Feature Engineering**:
    *   **Numerical Features**: Calculates lagged price data (e.g., previous day's close) and common technical indicators (e.g., Simple Moving Averages, RSI).
    *   **Textual Features**: Processes news articles to extract sentiment scores using VADER. These are then aggregated daily (e.g., average daily sentiment, count of positive/negative news).
    *   **Data Fusion**: Merges the daily stock features with the aggregated daily news features.
    *   **Target Definition**: Transforms the next day's predictions as: "Up", "Down", or "Neutral".
3.  **Model Training**:
    *   An `XGBoostClassifier` is trained on the combined historical features to predict the `Target_Direction`.
    *   The model and the `StandardScaler` (used for feature scaling) are saved for later use.
4.  **Prediction & Advice Generation**:
    *   When a prediction request is made, the system fetches the *latest* stock prices and news.
    *   These real-time inputs are transformed into features using the same preprocessing steps and the saved `StandardScaler`.
    *   The trained XGBoost model makes a prediction.
    *   The SHAP advice model is currently being implemented.
5.  **Web GUI**: Streamlit hosts an interactive interface that allows users to input a stock ticker (currently fixed to AAPL in the backend logic for this prototype) and view the prediction and advice.


