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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
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
        os.makedirs('models', exist_ok=True)
    
    def preprocess_text(self, text):
        """Preprocess text data for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, data, text_col='headline', target_col='sentiment'):
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
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_val, val_preds)
        
        # Generate word clouds
        self.generate_word_clouds(data, text_col, target_col)
        
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
        
        joblib.dump(self.model, 'models/sentiment_analysis_model.pkl')
        joblib.dump(self.vectorizer, 'models/sentiment_analysis_vectorizer.pkl')
    
    def load_model(self):
        """Load a trained model from disk."""
        self.model = joblib.load('models/sentiment_analysis_model.pkl')
        self.vectorizer = joblib.load('models/sentiment_analysis_vectorizer.pkl')
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=self.model.classes_)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.model.classes_, 
                    yticklabels=self.model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Sentiment Analysis Confusion Matrix')
        plt.tight_layout()
        plt.savefig('models/sentiment_confusion_matrix.png')
        plt.close()
    
    def generate_word_clouds(self, data, text_col, target_col):
        """Generate word clouds for each sentiment category."""
        # Group texts by sentiment
        sentiments = data[target_col].unique()
        
        for sentiment in sentiments:
            # Get texts for this sentiment
            texts = data[data[target_col] == sentiment][text_col].tolist()
            
            # Combine texts
            combined_text = ' '.join([self.preprocess_text(text) for text in texts])
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis',
                contour_width=1,
                contour_color='steelblue'
            ).generate(combined_text)
            
            # Plot word cloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
            plt.tight_layout()
            plt.savefig(f'models/wordcloud_{sentiment}.png')
            plt.close()

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
        'headline': texts,
        'sentiment': sentiments
    })
    
    return data

# Function to analyze market sentiment for a ticker
def analyze_market_sentiment(ticker, news_df=None):
    """
    Analyze market sentiment for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        news_df: DataFrame of news articles (optional)
    
    Returns:
        Dict with sentiment analysis results
    """
    try:
        # Create and train model
        model = SentimentAnalysisModel()
        
        # If news_df is provided, filter for the ticker
        if news_df is not None and not news_df.empty:
            ticker_news = news_df[news_df['ticker'] == ticker]
            
            # If no news for this ticker, generate some
            if ticker_news.empty:
                ticker_news = generate_market_news(ticker)
        else:
            # Generate market news if not provided
            ticker_news = generate_market_news(ticker)
        
        # Generate training data
        training_data = generate_training_data()
        
        # Add real news to training data
        training_data = pd.concat([training_data, ticker_news[['headline', 'sentiment']]])
        
        # Train the model
        model.train(training_data)
        
        # Predict sentiment for ticker news
        headlines = ticker_news['headline'].tolist()
        sentiments, probabilities = model.predict(headlines)
        
        # Add predictions to DataFrame
        ticker_news['predicted_sentiment'] = sentiments
        
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
        
        # Visualize sentiment
        visualize_sentiment(ticker, sentiment_score, sentiment_counts, overall_sentiment)
        
        return {
            'ticker': ticker,
            'sentiment_score': sentiment_score,
            'overall_sentiment': overall_sentiment,
            'sentiment_counts': sentiment_counts,
            'news': ticker_news.to_dict('records')
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

# Function to generate market news for a ticker
def generate_market_news(ticker, n_articles=8):
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
    company_name = f"{ticker}"
    
    # Positive news templates
    positive_templates = [
        f"{company_name} reports strong quarterly earnings, exceeding analyst expectations",
        f"{company_name} announces new sustainable initiative to reduce carbon footprint",
        f"{company_name} expands into new markets with innovative products",
        f"Analysts upgrade {company_name} stock to 'Buy' citing growth potential",
        f"{company_name} partners with tech giant for next-generation solutions",
        f"{company_name} increases dividend by 10%, signaling financial strength",
        f"ESG rating agency upgrades {company_name}'s sustainability score",
        f"{company_name} secures major contract with government agency",
        f"{company_name} launches innovative green technology solution",
        f"{company_name} reports record revenue growth in sustainable product line"
    ]
    
    # Negative news templates
    negative_templates = [
        f"{company_name} misses earnings expectations, shares drop",
        f"{company_name} faces regulatory scrutiny over environmental practices",
        f"{company_name} announces layoffs amid restructuring efforts",
        f"Analysts downgrade {company_name} stock citing competitive pressures",
        f"{company_name} recalls products due to quality concerns",
        f"{company_name} cuts dividend amid cash flow concerns",
        f"ESG rating agency downgrades {company_name} citing governance issues",
        f"{company_name} faces lawsuit over environmental damage claims",
        f"{company_name} struggles with supply chain disruptions",
        f"{company_name} reports higher than expected carbon emissions"
    ]
    
    # Neutral news templates
    neutral_templates = [
        f"{company_name} reports earnings in line with expectations",
        f"{company_name} maintains current sustainability initiatives",
        f"{company_name} announces leadership transition plan",
        f"Analysts maintain 'Hold' rating for {company_name} stock",
        f"{company_name} completes previously announced acquisition",
        f"{company_name} maintains dividend at current levels",
        f"{company_name}'s ESG rating remains unchanged in latest review",
        f"{company_name} hosts investor day, outlines 5-year strategy",
        f"{company_name} releases annual sustainability report",
        f"{company_name} refinances debt with new green bonds"
    ]
    
    # News sources
    sources = ['Financial Times', 'Bloomberg', 'Reuters', 'CNBC', 'Wall Street Journal', 'MarketWatch']
    
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
        source = np.random.choice(sources)
        
        articles.append({
            'ticker': ticker,
            'headline': headline,
            'sentiment': sentiment,
            'publication_date': pub_date,
            'source': source
        })
    
    return pd.DataFrame(articles)

# Function to visualize sentiment analysis results
def visualize_sentiment(ticker, sentiment_score, sentiment_counts, overall_sentiment):
    """
    Visualize sentiment analysis results.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_score: Sentiment score (-100 to 100)
        sentiment_counts: Dict with counts of positive, neutral, and negative sentiments
        overall_sentiment: Overall sentiment category
    """
    # Create pie chart of sentiment distribution
    plt.figure(figsize=(10, 6))
    
    # Define colors
    colors = {'positive': '#4CAF50', 'neutral': '#9E9E9E', 'negative': '#F44336'}
    
    # Create pie chart
    plt.pie(
        sentiment_counts.values(),
        labels=sentiment_counts.keys(),
        autopct='%1.1f%%',
        colors=[colors[s] for s in sentiment_counts.keys()],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Add title
    plt.title(f'Sentiment Distribution for {ticker}')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('models/sentiment_distribution.png')
    plt.close()
    
    # Create sentiment gauge
    plt.figure(figsize=(12, 6))
    
    # Create gauge
    gauge_colors = ['#F44336', '#FFC107', '#9E9E9E', '#8BC34A', '#4CAF50']  # Red to Green
    
    # Create a horizontal bar for the gauge
    plt.barh(['Sentiment'], [200], color='#E0E0E0', height=0.3, left=-100)
    
    # Add colored segments
    plt.barh(['Sentiment'], [40], color=gauge_colors[0], height=0.3, left=-100)
    plt.barh(['Sentiment'], [40], color=gauge_colors[1], height=0.3, left=-60)
    plt.barh(['Sentiment'], [40], color=gauge_colors[2], height=0.3, left=-20)
    plt.barh(['Sentiment'], [40], color=gauge_colors[3], height=0.3, left=20)
    plt.barh(['Sentiment'], [40], color=gauge_colors[4], height=0.3, left=60)
    
    # Add marker for the sentiment score
    plt.scatter([sentiment_score], ['Sentiment'], color='black', s=300, zorder=5, marker='v')
    
    # Add labels
    plt.text(-80, 0.85, 'Bearish', ha='center', fontsize=12)
    plt.text(-40, 0.85, 'Somewhat\nBearish', ha='center', fontsize=12)
    plt.text(0, 0.85, 'Neutral', ha='center', fontsize=12)
    plt.text(40, 0.85, 'Somewhat\nBullish', ha='center', fontsize=12)
    plt.text(80, 0.85, 'Bullish', ha='center', fontsize=12)
    
    # Add sentiment score text
    plt.text(sentiment_score, 1.15, f'{sentiment_score:.1f}', ha='center', fontsize=14, fontweight='bold')
    
    # Set x-axis limits
    plt.xlim(-110, 110)
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title(f'Market Sentiment for {ticker}: {overall_sentiment}', fontsize=16, pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('models/sentiment_gauge.png')
    plt.close()
