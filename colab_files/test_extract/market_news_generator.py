"""
Generate a large market news dataset for ML model demonstration
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_market_news(tickers, num_articles_per_ticker=5, seed=42):
    """
    Generate synthetic market news for a list of tickers
    
    Args:
        tickers: List of ticker symbols
        num_articles_per_ticker: Number of news articles to generate per ticker
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with generated news data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # News templates
    positive_templates = [
        "{company} reports strong quarterly earnings, exceeding analyst expectations",
        "{company} announces new sustainable initiative to reduce carbon footprint",
        "{company} expands into new markets with innovative products",
        "Analysts upgrade {company} stock to 'Buy' citing growth potential",
        "{company} partners with tech giant for next-generation solutions",
        "{company} increases dividend by 10%, signaling financial strength",
        "ESG rating agency upgrades {company}'s sustainability score",
        "{company} secures major contract with government agency",
        "{company} launches innovative green technology solution",
        "{company} reports record revenue growth in sustainable product line",
        "{company} achieves carbon neutrality ahead of schedule",
        "{company} recognized as industry leader in sustainability report",
        "{company} announces strategic acquisition to enhance green portfolio",
        "Institutional investors increase stake in {company} citing ESG leadership"
    ]
    
    negative_templates = [
        "{company} misses earnings expectations, shares drop",
        "{company} faces regulatory scrutiny over environmental practices",
        "{company} announces layoffs amid restructuring efforts",
        "Analysts downgrade {company} stock citing competitive pressures",
        "{company} recalls products due to quality concerns",
        "{company} cuts dividend amid cash flow concerns",
        "ESG rating agency downgrades {company} citing governance issues",
        "{company} faces lawsuit over environmental damage claims",
        "{company} struggles with supply chain disruptions",
        "{company} reports higher than expected carbon emissions",
        "{company} CEO under investigation for ethical violations",
        "{company} loses market share to more sustainable competitors",
        "{company} delays renewable energy transition plans",
        "Short seller report targets {company}'s sustainability claims"
    ]
    
    neutral_templates = [
        "{company} reports earnings in line with expectations",
        "{company} maintains current sustainability initiatives",
        "{company} announces leadership transition plan",
        "Analysts maintain 'Hold' rating for {company} stock",
        "{company} completes previously announced acquisition",
        "{company} maintains dividend at current levels",
        "{company}'s ESG rating remains unchanged in latest review",
        "{company} hosts investor day, outlines 5-year strategy",
        "{company} releases annual sustainability report",
        "{company} refinances debt with new green bonds",
        "{company} joins industry sustainability coalition",
        "{company} implements new governance policies",
        "{company} maintains market position despite industry challenges",
        "{company} announces regular quarterly dividend"
    ]
    
    # News sources
    sources = ['Financial Times', 'Bloomberg', 'Reuters', 'CNBC', 'Wall Street Journal', 
               'MarketWatch', 'Barron\'s', 'Forbes', 'The Economist', 'Morningstar']
    
    # Generate news articles
    articles = []
    
    # Current date for reference
    current_date = datetime.now()
    
    for ticker in tickers:
        # Company name is just the ticker for simplicity
        company = ticker
        
        for i in range(num_articles_per_ticker):
            # Determine sentiment (weighted towards neutral)
            sentiment_weights = [0.3, 0.3, 0.4]  # positive, negative, neutral
            sentiment = random.choices(['positive', 'negative', 'neutral'], weights=sentiment_weights)[0]
            
            # Select template based on sentiment
            if sentiment == 'positive':
                headline = random.choice(positive_templates).format(company=company)
            elif sentiment == 'negative':
                headline = random.choice(negative_templates).format(company=company)
            else:
                headline = random.choice(neutral_templates).format(company=company)
            
            # Generate publication date (within last 30 days)
            days_ago = random.randint(0, 30)
            pub_date = (current_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # Generate source
            source = random.choice(sources)
            
            articles.append({
                'ticker': ticker,
                'headline': headline,
                'sentiment': sentiment,
                'publication_date': pub_date,
                'source': source
            })
    
    return pd.DataFrame(articles)

# Generate and save the dataset
if __name__ == "__main__":
    # Load tickers from the portfolio dataset
    try:
        portfolio_df = pd.read_csv('large_portfolio_dataset.csv')
        tickers = portfolio_df['ticker'].unique().tolist()
    except:
        # Fallback to a small set of tickers if file not found
        tickers = ['GRNT', 'RNWP', 'SUFN', 'ECO', 'GRC', 'SUPH', 'CLRT', 'WTRT', 'CRCE', 'OCNT']
    
    # Generate news for a subset of tickers (to keep the dataset manageable)
    sample_size = min(50, len(tickers))
    sampled_tickers = random.sample(tickers, sample_size)
    
    news_df = generate_market_news(sampled_tickers, num_articles_per_ticker=8)
    news_df.to_csv('large_market_news_dataset.csv', index=False)
    
    print(f"Generated news dataset with {len(news_df)} articles")
    print(f"Number of tickers covered: {news_df['ticker'].nunique()}")
    
    # Print sentiment distribution
    sentiment_counts = news_df['sentiment'].value_counts()
    print("\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} ({count/len(news_df)*100:.1f}%)")
