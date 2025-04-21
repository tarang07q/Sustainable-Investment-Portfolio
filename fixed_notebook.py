"""
This script contains the fixed code for the portfolio data exploration notebook.
We'll use this to extract the fixed functions and update the notebook.
"""

def prepare_data_for_recommendation(portfolio_df, user_preferences):
    """
    Prepare data for the portfolio recommendation model
    
    Args:
        portfolio_df: DataFrame of portfolio assets
        user_preferences: Dict of user preferences
        
    Returns:
        DataFrame ready for ML model
    """
    # Make a copy to avoid modifying the original dataframe
    df = portfolio_df.copy()
    
    # Ensure all required columns exist
    required_columns = [
        'ticker', 'name', 'asset_type', 'sector',
        'current_price', 'price_change_24h', 'market_cap_b', 'roi_1y', 'volatility',
        'environmental_score', 'social_score', 'governance_score', 'esg_score',
        'beta', 'sharpe_ratio', 'market_correlation', 'carbon_footprint'
    ]
    
    # Check for missing columns and add them with default values if needed
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataset. Adding with default values.")
            if col in ['ticker', 'name', 'asset_type', 'sector']:
                df[col] = f"Unknown {col}"
            else:
                df[col] = 50.0  # Default numeric value
    
    # Extract features for ML model
    ml_data = df[required_columns].copy()
    
    # Apply user preferences
    risk_tolerance = user_preferences.get('risk_tolerance', 5)
    sustainability_focus = user_preferences.get('sustainability_focus', 5)
    
    # Calculate weights based on user preferences
    risk_weight = (11 - risk_tolerance) / 10  # 1.0 to 0.1
    esg_weight = sustainability_focus / 10  # 0.1 to 1.0
    return_weight = 1 - risk_weight - esg_weight/2  # Balance the weights
    
    # Calculate custom score
    ml_data['custom_score'] = (
        ml_data['esg_score'] * esg_weight +
        (100 - ml_data['volatility'] * 100) * risk_weight +
        ml_data['roi_1y'] * return_weight
    )
    
    return ml_data

def assess_portfolio_risk(portfolio_df, user_preferences):
    """
    Assess the risk of a portfolio
    
    Args:
        portfolio_df: DataFrame of portfolio assets
        user_preferences: Dict of user preferences
        
    Returns:
        Dict with risk assessment results
    """
    # Make a copy to avoid modifying the original dataframe
    df = portfolio_df.copy()
    
    # Ensure all required columns exist
    required_columns = ['volatility', 'beta', 'esg_score', 'allocation']
    
    # Check for missing columns and add them with default values if needed
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataset. Adding with default values.")
            if col == 'allocation':
                # Equal allocation if missing
                df[col] = 1.0 / len(df)
            else:
                df[col] = 50.0  # Default numeric value
    
    # Calculate portfolio-level metrics (weighted by allocation)
    portfolio_volatility = np.average(df['volatility'], weights=df['allocation'])
    portfolio_beta = np.average(df['beta'], weights=df['allocation'])
    portfolio_esg_risk = 100 - np.average(df['esg_score'], weights=df['allocation'])
    
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

def analyze_market_sentiment(ticker, news_df, user_preferences):
    """
    Analyze market sentiment for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        news_df: DataFrame of news items
        user_preferences: Dict of user preferences
        
    Returns:
        Dict with sentiment analysis results
    """
    # Make a copy to avoid modifying the original dataframe
    df = news_df.copy()
    
    # Ensure all required columns exist
    required_columns = ['ticker', 'headline', 'sentiment', 'publication_date', 'source']
    
    # Check for missing columns and add them with default values if needed
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in news dataset. Adding with default values.")
            if col == 'ticker':
                df[col] = ticker
            elif col == 'sentiment':
                df[col] = 'neutral'
            elif col == 'publication_date':
                df[col] = pd.Timestamp.now().strftime('%Y-%m-%d')
            elif col == 'source':
                df[col] = 'Unknown Source'
            else:
                df[col] = f"No {col} available"
    
    # Filter news for the given ticker
    ticker_news = df[df['ticker'] == ticker]
    
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
