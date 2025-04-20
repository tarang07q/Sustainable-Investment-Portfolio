# Machine Learning Models for Sustainable Investment Portfolio

This directory contains the machine learning models used in the Sustainable Investment Portfolio application to provide AI-powered recommendations and analysis.

## Models Overview

### Portfolio Recommendation Model

The portfolio recommendation model uses a gradient boosting regressor to analyze assets based on:
- ESG scores (environmental, social, governance)
- Financial metrics (ROI, volatility, market cap)
- Market trends
- User preferences

The model provides personalized investment recommendations that balance financial returns with sustainability goals.

### Risk Assessment Model

The risk assessment model uses a random forest classifier to evaluate portfolio risk based on:
- Portfolio volatility
- Beta and market correlation
- Sharpe ratio
- ESG risk factors
- Sector volatility
- Liquidity metrics

The model categorizes risk into four levels (Low, Moderate, High, Very High) and provides detailed risk factor analysis.

### Sentiment Analysis Model

The sentiment analysis model uses natural language processing techniques to analyze market sentiment from news articles and social media data. It:
- Processes text data using TF-IDF vectorization
- Classifies sentiment as positive, neutral, or negative
- Calculates an overall sentiment score
- Provides trading implications based on sentiment analysis

## Technical Implementation

The models are implemented using scikit-learn and TensorFlow, with the following components:
- Data preprocessing and feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Model persistence using joblib
- Integration with the Streamlit application

## Usage

The models are integrated into the application through the `ml_integration.py` module, which provides functions to:
- Get portfolio recommendations
- Assess portfolio risk
- Analyze market sentiment
- Display ML-powered visualizations and insights

## Model Training

The models are trained on synthetic data generated to simulate real-world financial and ESG metrics. In a production environment, these would be replaced with real data from financial APIs and ESG rating providers.

## Future Enhancements

Planned enhancements for the ML models include:
- Deep learning models for more accurate predictions
- Time series forecasting for future performance prediction
- Reinforcement learning for portfolio optimization
- Integration with real-time market data
- Explainable AI features to provide more transparent recommendations
