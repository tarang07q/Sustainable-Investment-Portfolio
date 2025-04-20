"""
Market Sentiment Analysis Model

This module implements a machine learning model for analyzing market sentiment
based on news articles and social media data.
"""

import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime, timedelta

class SentimentAnalysisModel:
    """Machine learning model for market sentiment analysis."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Create model directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)
        
        # Try to load pre-trained model if it exists
        try:
            self.load_model()
            print("Pre-trained sentiment analysis model loaded successfully.")
        except:
            print("No pre-trained sentiment model found. Will train a new model when needed.")
    
    def preprocess_text(self, text):
        """Preprocess text data for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, data, text_col='text', target_col='sentiment'):
        """Train the model on the given data."""
        # Preprocess text
        processed_texts = [self.preprocess_text(text) for text in data[text_col]]
        
        # Vectorize text
        X = self.vectorizer.fit_transform(processed_texts)
        y = data[target_col].values
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_preds)
        val_accuracy = accuracy_score(y_val, val_preds)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_preds))
        
        # Save model
        self.save_model()
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }
    
    def predict(self, texts):
        """Predict sentiment based on the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        X = self.vectorizer.transform(processed_texts)
        
        # Generate predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, 'models/saved/sentiment_analysis_model.pkl')
        joblib.dump(self.vectorizer, 'models/saved/sentiment_analysis_vectorizer.pkl')
    
    def load_model(self):
        """Load a trained model from disk."""
        self.model = joblib.load('models/saved/sentiment_analysis_model.pkl')
        self.vectorizer = joblib.load('models/saved/sentiment_analysis_vectorizer.pkl')

# Function to generate synthetic training data
def generate_training_data(n_samples=1000):
    """Generate synthetic data for model training."""
    np.random.seed(42)
    
    # Positive sentiment phrases
    positive_phrases = [
        "strong growth potential", "exceeding expectations", "positive outlook",
        "bullish market", "impressive performance", "sustainable growth",
        "innovative solutions", "market leader", "strong buy recommendation",
        "record profits", "dividend increase", "successful expansion",
        "strategic acquisition", "cost reduction", "revenue growth",
        "positive earnings surprise", "strong balance sheet", "competitive advantage",
        "industry outperformer", "green investment opportunity"
    ]
    
    # Negative sentiment phrases
    negative_phrases = [
        "declining profits", "missed expectations", "negative outlook",
        "bearish market", "poor performance", "unsustainable practices",
        "outdated technology", "market laggard", "sell recommendation",
        "profit warning", "dividend cut", "failed expansion",
        "costly acquisition", "increasing costs", "revenue decline",
        "negative earnings surprise", "weak balance sheet", "losing market share",
        "industry underperformer", "environmental concerns"
    ]
    
    # Neutral sentiment phrases
    neutral_phrases = [
        "in line with expectations", "stable outlook", "mixed results",
        "market volatility", "average performance", "industry standard",
        "moderate growth", "hold recommendation", "steady dividend",
        "minor acquisition", "cost neutral", "flat revenue",
        "as expected earnings", "adequate capital", "maintaining market share",
        "industry average", "monitoring developments", "unchanged forecast",
        "continuing operations", "regulatory compliance"
    ]
    
    # Generate texts and sentiments
    texts = []
    sentiments = []
    
    for _ in range(n_samples):
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        
        if sentiment == 'positive':
            phrases = np.random.choice(positive_phrases, size=np.random.randint(3, 6), replace=True)
            filler = np.random.choice(neutral_phrases, size=np.random.randint(1, 3), replace=True)
        elif sentiment == 'negative':
            phrases = np.random.choice(negative_phrases, size=np.random.randint(3, 6), replace=True)
            filler = np.random.choice(neutral_phrases, size=np.random.randint(1, 3), replace=True)
        else:
            phrases = np.random.choice(neutral_phrases, size=np.random.randint(4, 7), replace=True)
            pos_filler = np.random.choice(positive_phrases, size=1)
            neg_filler = np.random.choice(negative_phrases, size=1)
            filler = np.concatenate([pos_filler, neg_filler])
        
        all_phrases = np.concatenate([phrases, filler])
        np.random.shuffle(all_phrases)
        
        text = "The company " + ". It is ".join(all_phrases) + "."
        texts.append(text)
        sentiments.append(sentiment)
    
    # Create DataFrame
    data = pd.DataFrame({
        'text': texts,
        'sentiment': sentiments
    })
    
    return data

# Function to generate market news for a ticker
def generate_market_news(ticker, n_articles=5):
    """
    Generate synthetic market news for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        n_articles: Number of news articles to generate
    
    Returns:
        DataFrame of news articles
    """
    np.random.seed(hash(ticker) % 10000)
    
    # Company name based on ticker
    company_name = f"{ticker.capitalize()} Inc."
    
    # Positive news templates
    positive_templates = [
        f"{company_name} reports strong quarterly earnings, exceeding analyst expectations",
        f"{company_name} announces new sustainable initiative to reduce carbon footprint",
        f"{company_name} expands into new markets with innovative products",
        f"Analysts upgrade {company_name} stock to 'Buy' citing growth potential",
        f"{company_name} partners with tech giant for next-generation solutions",
        f"{company_name} increases dividend by 10%, signaling financial strength",
        f"ESG rating agency upgrades {company_name}'s sustainability score"
    ]
    
    # Negative news templates
    negative_templates = [
        f"{company_name} misses earnings expectations, shares drop",
        f"{company_name} faces regulatory scrutiny over environmental practices",
        f"{company_name} announces layoffs amid restructuring efforts",
        f"Analysts downgrade {company_name} stock citing competitive pressures",
        f"{company_name} recalls products due to quality concerns",
        f"{company_name} cuts dividend amid cash flow concerns",
        f"ESG rating agency downgrades {company_name} citing governance issues"
    ]
    
    # Neutral news templates
    neutral_templates = [
        f"{company_name} reports earnings in line with expectations",
        f"{company_name} maintains current sustainability initiatives",
        f"{company_name} announces leadership transition plan",
        f"Analysts maintain 'Hold' rating for {company_name} stock",
        f"{company_name} completes previously announced acquisition",
        f"{company_name} maintains dividend at current levels",
        f"{company_name}'s ESG rating remains unchanged in latest review"
    ]
    
    # Generate news articles
    articles = []
    
    for i in range(n_articles):
        # Determine sentiment (weighted towards neutral)
        sentiment_weights = [0.3, 0.3, 0.4]  # positive, negative, neutral
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=sentiment_weights)
        
        # Select template based on sentiment
        if sentiment == 'positive':
            headline = np.random.choice(positive_templates)
        elif sentiment == 'negative':
            headline = np.random.choice(negative_templates)
        else:
            headline = np.random.choice(neutral_templates)
        
        # Generate publication date (within last 30 days)
        days_ago = np.random.randint(0, 30)
        pub_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Generate source
        sources = ['Financial Times', 'Bloomberg', 'Reuters', 'CNBC', 'Wall Street Journal', 'MarketWatch']
        source = np.random.choice(sources)
        
        articles.append({
            'ticker': ticker,
            'headline': headline,
            'sentiment': sentiment,
            'publication_date': pub_date,
            'source': source
        })
    
    return pd.DataFrame(articles)

# Function to analyze market sentiment for a ticker
def analyze_market_sentiment(ticker):
    """
    Analyze market sentiment for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dict with sentiment analysis results
    """
    # Create and train model if needed
    model = SentimentAnalysisModel()
    
    # If model not loaded, train it on synthetic data
    if model.model is None:
        training_data = generate_training_data()
        model.train(training_data)
    
    # Generate market news
    news_df = generate_market_news(ticker)
    
    try:
        # Predict sentiment
        sentiments, probabilities = model.predict(news_df['headline'].tolist())
        
        # Add predictions to DataFrame
        news_df['predicted_sentiment'] = sentiments
        
        # Calculate sentiment score (-100 to 100)
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        sentiment_values = [sentiment_map[s] for s in sentiments]
        sentiment_score = sum(sentiment_values) / len(sentiment_values) * 100
        
        # Count sentiments
        sentiment_counts = {
            'positive': sum(1 for s in sentiments if s == 'positive'),
            'neutral': sum(1 for s in sentiments if s == 'neutral'),
            'negative': sum(1 for s in sentiments if s == 'negative')
        }
        
        # Determine overall sentiment
        if sentiment_score > 30:
            overall_sentiment = 'Bullish'
        elif sentiment_score > 10:
            overall_sentiment = 'Somewhat Bullish'
        elif sentiment_score > -10:
            overall_sentiment = 'Neutral'
        elif sentiment_score > -30:
            overall_sentiment = 'Somewhat Bearish'
        else:
            overall_sentiment = 'Bearish'
        
        return {
            'ticker': ticker,
            'sentiment_score': sentiment_score,
            'overall_sentiment': overall_sentiment,
            'sentiment_counts': sentiment_counts,
            'news': news_df.to_dict('records')
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        # Fallback to random sentiment
        sentiment_score = np.random.uniform(-50, 50)
        
        if sentiment_score > 30:
            overall_sentiment = 'Bullish'
        elif sentiment_score > 10:
            overall_sentiment = 'Somewhat Bullish'
        elif sentiment_score > -10:
            overall_sentiment = 'Neutral'
        elif sentiment_score > -30:
            overall_sentiment = 'Somewhat Bearish'
        else:
            overall_sentiment = 'Bearish'
            
        return {
            'ticker': ticker,
            'sentiment_score': sentiment_score,
            'overall_sentiment': overall_sentiment,
            'error': str(e)
        }
