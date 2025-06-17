# 📈 Fintel: News-Driven Stock Forecasting Prototype

## Overview

Fintel is a proof-of-concept machine learning system designed to predict short-term stock price movements (Up, Down, or Neutral) for a single stock (Apple - AAPL) by leveraging both historical price data and recent news sentiment. Beyond just a prediction, Fintel aims to provide AI-driven "advice" that explains the rationale behind its forecast, primarily based on the most influential features identified by the model, including news sentiment.

This project serves as a comprehensive demonstration of key Machine Learning (ML), Natural Language Processing (NLP), and MLOps (Machine Learning Operations) concepts, from data acquisition and feature engineering to model training, explainability, and a user-friendly web interface.

## Features

*   **Multi-Modal Data Integration**: Combines numerical stock price data (e.g., historical close prices, volume, technical indicators) with unstructured text data from financial news.
*   **Sentiment Analysis**: Utilizes VADER (Valence Aware Dictionary and sEntiment Reasoner) to extract sentiment scores from news headlines and descriptions.
*   **Machine Learning Prediction**: Employs an XGBoost Classifier to predict the next day's stock movement direction (Up, Down, Neutral).
*   **Explainable AI (XAI)**: Integrates SHAP (SHapley Additive exPlanations) to identify the key features influencing a prediction, enabling the generation of human-readable advice.
*   **Web-Based GUI**: A simple, intuitive graphical user interface built with Streamlit for easy interaction and demonstration.
*   **Modular Codebase**: Organized Python scripts for data acquisition, feature engineering, model training, and prediction, promoting maintainability and scalability.

## How it Works (Under the Hood)

1.  **Data Acquisition**:
    *   **Stock Prices**: Uses `yfinance` to fetch historical `OHLCV` (Open, High, Low, Close, Volume) data for AAPL.
    *   **News Articles**: Connects to `NewsAPI.org` to retrieve recent news headlines and descriptions related to Apple.
2.  **Feature Engineering**:
    *   **Numerical Features**: Calculates lagged price data (e.g., previous day's close) and common technical indicators (e.g., Simple Moving Averages, RSI).
    *   **Textual Features**: Processes news articles to extract sentiment scores using NLTK's VADER. These are then aggregated daily (e.g., average daily sentiment, count of positive/negative news).
    *   **Data Fusion**: Merges the daily stock features with the aggregated daily news features.
    *   **Target Definition**: Transforms the next day's percentage change in close price into a categorical target: "Up", "Down", or "Neutral".
3.  **Model Training**:
    *   An `XGBoostClassifier` is trained on the combined historical features to predict the `Target_Direction`.
    *   The model and the `StandardScaler` (used for feature scaling) are saved for later use.
4.  **Prediction & Advice Generation**:
    *   When a prediction request is made, the system fetches the *latest* stock prices and news.
    *   These real-time inputs are transformed into features using the same preprocessing steps and the saved `StandardScaler`.
    *   The trained XGBoost model makes a prediction.
    *   SHAP values are computed for this specific prediction to understand which features contributed most.
    *   A custom logic then translates these influential features and the predicted direction into a concise, AI-driven advice message.
5.  **Web GUI**: Streamlit hosts an interactive interface that allows users to input a stock ticker (currently fixed to AAPL in the backend logic for this prototype) and view the prediction and advice.

## Getting Started

Follow these steps to set up and run the Fintel prototype on your local machine.

### Prerequisites

*   Python 3.8+
*   Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Fintel-Prototype.git
cd Fintel-Prototype