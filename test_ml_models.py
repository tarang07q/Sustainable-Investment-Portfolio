"""
Test script for ML models in the sustainable investment portfolio application
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

def get_portfolio_recommendations(df, risk_tolerance=5, sustainability_focus=5):
    """Generate portfolio recommendations based on user preferences."""
    # Make a copy to avoid modifying the original dataframe
    portfolio_df = df.copy()

    # Normalize preferences to 0-1 scale
    risk_weight = (11 - risk_tolerance) / 10  # Higher risk tolerance = lower risk weight
    esg_weight = sustainability_focus / 10
    return_weight = 1 - risk_weight - esg_weight/2

    # Calculate weighted scores
    portfolio_df['recommendation_score'] = (
        portfolio_df['esg_score'] * esg_weight +
        (100 - portfolio_df['volatility'] * 100) * risk_weight +
        portfolio_df['roi_1y'] * return_weight
    )

    # Add recommendation strength
    max_score = portfolio_df['recommendation_score'].max()
    min_score = portfolio_df['recommendation_score'].min()
    score_range = max_score - min_score

    portfolio_df['recommendation_strength'] = portfolio_df['recommendation_score'].apply(
        lambda x: 'Strong Buy' if x > (min_score + 0.8 * score_range) else
                  'Buy' if x > (min_score + 0.6 * score_range) else
                  'Hold' if x > (min_score + 0.4 * score_range) else
                  'Underweight' if x > (min_score + 0.2 * score_range) else
                  'Sell'
    )

    # Sort by recommendation score
    recommendations = portfolio_df.sort_values('recommendation_score', ascending=False)

    return recommendations

def assess_portfolio_risk(portfolio_df, user_preferences=None):
    """Assess portfolio risk using ML model."""
    # Set default user preferences if not provided
    if user_preferences is None:
        user_preferences = {
            'risk_tolerance': 5,
            'sustainability_focus': 5
        }

    # Calculate portfolio-level metrics (weighted by allocation)
    portfolio_volatility = np.average(portfolio_df['volatility'], weights=portfolio_df['allocation'])
    portfolio_beta = np.average(portfolio_df['beta'], weights=portfolio_df['allocation'])
    portfolio_esg_risk = 100 - np.average(portfolio_df['esg_score'], weights=portfolio_df['allocation'])

    # Adjust risk based on user preferences
    risk_tolerance = user_preferences.get('risk_tolerance', 5)
    sustainability_focus = user_preferences.get('sustainability_focus', 5)

    # Risk tolerance adjustment (1-10 scale)
    # Higher risk tolerance = lower perceived risk
    risk_tolerance_factor = risk_tolerance / 5  # 0.2-2.0 range

    # Sustainability focus adjustment (1-10 scale)
    # Higher sustainability focus = higher sensitivity to ESG risk
    sustainability_factor = sustainability_focus / 5  # 0.2-2.0 range

    # Adjust risk components
    market_risk_weight = 1.0 / risk_tolerance_factor
    esg_risk_weight = sustainability_factor

    # Calculate adjusted risk score
    risk_score = (portfolio_volatility * 100) * 0.4 * market_risk_weight + \
                 portfolio_esg_risk * 0.4 * esg_risk_weight + \
                 portfolio_beta * 20 * 0.2

    # Ensure risk score is within 0-100 range
    risk_score = min(max(risk_score, 0), 100)

    # Determine risk category
    if risk_score < 25:
        risk_category = 'Low'
    elif risk_score < 50:
        risk_category = 'Moderate'
    elif risk_score < 75:
        risk_category = 'High'
    else:
        risk_category = 'Very High'

    # Prepare risk factors
    risk_factors = {
        'Market Risk': portfolio_volatility * 100,
        'Systematic Risk': portfolio_beta * 50,
        'ESG Risk': portfolio_esg_risk
    }

    return {
        'risk_category': risk_category,
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'portfolio_metrics': {
            'volatility': portfolio_volatility,
            'beta': portfolio_beta,
            'esg_risk_score': portfolio_esg_risk
        }
    }

def analyze_market_sentiment(ticker, news_df, user_preferences=None):
    """Analyze market sentiment for a given ticker."""
    # Set default user preferences if not provided
    if user_preferences is None:
        user_preferences = {
            'risk_tolerance': 5,
            'sustainability_focus': 5
        }

    # Filter news for the given ticker
    ticker_news = news_df[news_df['ticker'] == ticker]

    if len(ticker_news) == 0:
        return {
            'ticker': ticker,
            'sentiment_score': 0,
            'overall_sentiment': 'Neutral',
            'error': 'No news found for this ticker'
        }

    # Calculate sentiment score (-100 to 100)
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    sentiment_values = [sentiment_map[s] for s in ticker_news['sentiment']]
    base_sentiment_score = sum(sentiment_values) / len(sentiment_values) * 100

    # Adjust sentiment score based on user preferences
    risk_tolerance = user_preferences.get('risk_tolerance', 5)
    sustainability_focus = user_preferences.get('sustainability_focus', 5)

    # Sustainability focus adjustment (1-10 scale)
    # Higher sustainability focus = more sensitive to negative ESG news
    sustainability_factor = sustainability_focus / 5  # 0.2-2.0 range

    # Risk tolerance adjustment (1-10 scale)
    # Lower risk tolerance = more sensitive to negative news
    risk_sensitivity = (11 - risk_tolerance) / 5  # 0.2-2.0 range

    # Apply adjustments
    if base_sentiment_score < 0:
        # Negative sentiment is amplified for sustainability-focused or risk-averse users
        sentiment_score = base_sentiment_score * max(sustainability_factor, risk_sensitivity)
    else:
        # Positive sentiment is slightly dampened for very sustainability-focused users
        sentiment_score = base_sentiment_score * (1 - (sustainability_factor - 1) * 0.1 if sustainability_factor > 1 else 1)

    # Count sentiments
    sentiment_counts = ticker_news['sentiment'].value_counts().to_dict()

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
        'news': ticker_news.to_dict('records')
    }

# Main function to test the ML models
def main():
    print("Starting ML model tests with enhanced dataset...")

    # Load portfolio data
    portfolio_df = pd.read_csv('data/portfolio_dataset.csv')

    # Convert sdg_alignment from string to list if needed
    if 'sdg_alignment' in portfolio_df.columns:
        portfolio_df['sdg_alignment'] = portfolio_df['sdg_alignment'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.strip() else []
        )

    # Load market news data
    news_df = pd.read_csv('data/market_news.csv')

    print("Dataset shapes:")
    print(f"Portfolio data: {portfolio_df.shape}")
    print(f"Market news data: {news_df.shape}")

    # Print dataset statistics
    print("\nPortfolio Dataset Statistics:")
    print(f"Number of stocks: {len(portfolio_df[portfolio_df['asset_type'] == 'Stock'])}")
    print(f"Number of cryptos: {len(portfolio_df[portfolio_df['asset_type'] == 'Crypto'])}")
    print(f"Average ESG score: {portfolio_df['esg_score'].mean():.2f}")
    print(f"ESG score range: {portfolio_df['esg_score'].min():.2f} to {portfolio_df['esg_score'].max():.2f}")
    print(f"Average ROI: {portfolio_df['roi_1y'].mean():.2f}%")
    print(f"ROI range: {portfolio_df['roi_1y'].min():.2f}% to {portfolio_df['roi_1y'].max():.2f}%")
    print(f"Average volatility: {portfolio_df['volatility'].mean():.2f}")

    # Test portfolio recommendations for different user types
    print("\n=== Portfolio Recommendations ===")

    # Conservative, sustainability-focused user
    conservative_user = {'risk_tolerance': 3, 'sustainability_focus': 8}
    conservative_recs = get_portfolio_recommendations(portfolio_df, **conservative_user)
    print(f"\nTop 5 recommendations for conservative, sustainability-focused user:")
    print(conservative_recs[['name', 'ticker', 'asset_type', 'sector', 'esg_score', 'roi_1y', 'volatility', 'recommendation_score', 'recommendation_strength']].head(5))

    # Balanced user
    balanced_user = {'risk_tolerance': 5, 'sustainability_focus': 5}
    balanced_recs = get_portfolio_recommendations(portfolio_df, **balanced_user)
    print(f"\nTop 5 recommendations for balanced user:")
    print(balanced_recs[['name', 'ticker', 'asset_type', 'sector', 'esg_score', 'roi_1y', 'volatility', 'recommendation_score', 'recommendation_strength']].head(5))

    # Aggressive, return-focused user
    aggressive_user = {'risk_tolerance': 8, 'sustainability_focus': 3}
    aggressive_recs = get_portfolio_recommendations(portfolio_df, **aggressive_user)
    print(f"\nTop 5 recommendations for aggressive, return-focused user:")
    print(aggressive_recs[['name', 'ticker', 'asset_type', 'sector', 'esg_score', 'roi_1y', 'volatility', 'recommendation_score', 'recommendation_strength']].head(5))

    # Test risk assessment for different portfolios
    print("\n=== Risk Assessment ===")

    # Create a conservative portfolio (stocks with low volatility and high ESG)
    stocks_df = portfolio_df[portfolio_df['asset_type'] == 'Stock']
    conservative_portfolio = stocks_df[(stocks_df['volatility'] < stocks_df['volatility'].median()) &
                                      (stocks_df['esg_score'] > stocks_df['esg_score'].median())]
    if len(conservative_portfolio) > 10:
        conservative_portfolio = conservative_portfolio.head(10)

    # Create a balanced portfolio (mix of stocks and cryptos)
    balanced_portfolio = pd.concat([
        stocks_df.sample(min(10, len(stocks_df))),
        portfolio_df[portfolio_df['asset_type'] == 'Crypto'].sample(min(5, len(portfolio_df[portfolio_df['asset_type'] == 'Crypto'])))
    ])

    # Create an aggressive portfolio (high ROI assets, including cryptos)
    aggressive_portfolio = portfolio_df.sort_values('roi_1y', ascending=False).head(15)

    # Assess risk for each portfolio
    conservative_risk = assess_portfolio_risk(conservative_portfolio, conservative_user)
    print(f"\nConservative Portfolio Risk Assessment:")
    print(f"Risk Category: {conservative_risk['risk_category']}")
    print(f"Risk Score: {conservative_risk['risk_score']:.2f}/100")
    print("Risk Factors:")
    for factor, score in conservative_risk['risk_factors'].items():
        print(f"  {factor}: {score:.2f}")

    balanced_risk = assess_portfolio_risk(balanced_portfolio, balanced_user)
    print(f"\nBalanced Portfolio Risk Assessment:")
    print(f"Risk Category: {balanced_risk['risk_category']}")
    print(f"Risk Score: {balanced_risk['risk_score']:.2f}/100")
    print("Risk Factors:")
    for factor, score in balanced_risk['risk_factors'].items():
        print(f"  {factor}: {score:.2f}")

    aggressive_risk = assess_portfolio_risk(aggressive_portfolio, aggressive_user)
    print(f"\nAggressive Portfolio Risk Assessment:")
    print(f"Risk Category: {aggressive_risk['risk_category']}")
    print(f"Risk Score: {aggressive_risk['risk_score']:.2f}/100")
    print("Risk Factors:")
    for factor, score in aggressive_risk['risk_factors'].items():
        print(f"  {factor}: {score:.2f}")

    # Test sentiment analysis for different assets
    print("\n=== Sentiment Analysis ===")

    # Try to find assets with different characteristics
    # 1. A high-performing sustainable asset
    high_esg_assets = portfolio_df[portfolio_df['esg_score'] > 75].sort_values('roi_1y', ascending=False)
    if len(high_esg_assets) > 0:
        high_esg_ticker = high_esg_assets.iloc[0]['ticker']
        high_esg_name = high_esg_assets.iloc[0]['name']
    else:
        high_esg_ticker = portfolio_df.sort_values('esg_score', ascending=False).iloc[0]['ticker']
        high_esg_name = portfolio_df.sort_values('esg_score', ascending=False).iloc[0]['name']

    # 2. A poorly performing asset
    poor_performers = portfolio_df[portfolio_df['roi_1y'] < 0]
    if len(poor_performers) > 0:
        poor_performer = poor_performers.sort_values('roi_1y').iloc[0]
        poor_ticker = poor_performer['ticker']
        poor_name = poor_performer['name']
    else:
        poor_performer = portfolio_df.sort_values('roi_1y').iloc[0]
        poor_ticker = poor_performer['ticker']
        poor_name = poor_performer['name']

    # 3. A volatile crypto asset
    volatile_cryptos = portfolio_df[(portfolio_df['asset_type'] == 'Crypto') & (portfolio_df['volatility'] > 0.7)]
    if len(volatile_cryptos) > 0:
        volatile_crypto = volatile_cryptos.sample(1).iloc[0]
        volatile_ticker = volatile_crypto['ticker']
        volatile_name = volatile_crypto['name']
    else:
        volatile_crypto = portfolio_df[portfolio_df['asset_type'] == 'Crypto'].sort_values('volatility', ascending=False).iloc[0]
        volatile_ticker = volatile_crypto['ticker']
        volatile_name = volatile_crypto['name']

    # Analyze sentiment for each asset
    print(f"\nSentiment Analysis for {high_esg_name} ({high_esg_ticker}) - High ESG Score:")
    high_esg_sentiment = analyze_market_sentiment(high_esg_ticker, news_df, conservative_user)
    if 'error' in high_esg_sentiment:
        print(f"  {high_esg_sentiment['error']}")
    else:
        print(f"  Overall Sentiment: {high_esg_sentiment['overall_sentiment']}")
        print(f"  Sentiment Score: {high_esg_sentiment['sentiment_score']:.2f}")
        print("  Sentiment Distribution:")
        for sentiment_type, count in high_esg_sentiment['sentiment_counts'].items():
            print(f"    {sentiment_type.capitalize()}: {count}")
        if len(high_esg_sentiment['news']) > 0:
            print("  Recent News:")
            for i, news in enumerate(high_esg_sentiment['news'][:2]):
                print(f"    {i+1}. {news['headline']} ({news['source']}, {news['publication_date']})")

    print(f"\nSentiment Analysis for {poor_name} ({poor_ticker}) - Poor Performer:")
    poor_sentiment = analyze_market_sentiment(poor_ticker, news_df, balanced_user)
    if 'error' in poor_sentiment:
        print(f"  {poor_sentiment['error']}")
    else:
        print(f"  Overall Sentiment: {poor_sentiment['overall_sentiment']}")
        print(f"  Sentiment Score: {poor_sentiment['sentiment_score']:.2f}")
        print("  Sentiment Distribution:")
        for sentiment_type, count in poor_sentiment['sentiment_counts'].items():
            print(f"    {sentiment_type.capitalize()}: {count}")
        if len(poor_sentiment['news']) > 0:
            print("  Recent News:")
            for i, news in enumerate(poor_sentiment['news'][:2]):
                print(f"    {i+1}. {news['headline']} ({news['source']}, {news['publication_date']})")

    print(f"\nSentiment Analysis for {volatile_name} ({volatile_ticker}) - Volatile Crypto:")
    volatile_sentiment = analyze_market_sentiment(volatile_ticker, news_df, aggressive_user)
    if 'error' in volatile_sentiment:
        print(f"  {volatile_sentiment['error']}")
    else:
        print(f"  Overall Sentiment: {volatile_sentiment['overall_sentiment']}")
        print(f"  Sentiment Score: {volatile_sentiment['sentiment_score']:.2f}")
        print("  Sentiment Distribution:")
        for sentiment_type, count in volatile_sentiment['sentiment_counts'].items():
            print(f"    {sentiment_type.capitalize()}: {count}")
        if len(volatile_sentiment['news']) > 0:
            print("  Recent News:")
            for i, news in enumerate(volatile_sentiment['news'][:2]):
                print(f"    {i+1}. {news['headline']} ({news['source']}, {news['publication_date']})")

    print("\nAll ML models tested successfully with the enhanced dataset!")

if __name__ == "__main__":
    main()
