"""
ML Model Integration Utility

This module provides functions to integrate ML models with the Sustainable Investment Portfolio app.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

# Import ML models
from models.portfolio_recommendation import get_portfolio_recommendations
from models.risk_assessment import assess_portfolio_risk
from models.sentiment_analysis import analyze_market_sentiment

def get_ml_portfolio_recommendations(all_assets, user_preferences=None):
    """
    Get ML-powered portfolio recommendations.

    Args:
        all_assets: DataFrame of all available assets
        user_preferences: Dict of user preferences

    Returns:
        DataFrame of recommended assets
    """
    try:
        # Convert DataFrame to the format expected by the ML model
        assets_df = pd.DataFrame(all_assets) if not isinstance(all_assets, pd.DataFrame) else all_assets

        # Standardize column names if needed
        if 'ticker' not in assets_df.columns and 'symbol' in assets_df.columns:
            assets_df['ticker'] = assets_df['symbol']

        # Create empty portfolio DataFrame if needed
        portfolio_df = pd.DataFrame(columns=['ticker'])

        # Set default user preferences if not provided
        if user_preferences is None:
            user_preferences = {
                'risk_tolerance': 5,
                'sustainability_focus': 5,
                'investment_horizon': 'Medium-term (1-3 years)'
            }

        # Get recommendations
        recommendations = get_portfolio_recommendations(portfolio_df, assets_df, user_preferences)

        return recommendations.head(5)
    except Exception as e:
        st.error(f"Error generating ML recommendations: {str(e)}")
        # Fallback to simple filtering
        return all_assets.sort_values('esg_score', ascending=False).head(5)

def get_ml_risk_assessment(portfolio_assets, user_preferences=None):
    """
    Get ML-powered risk assessment for a portfolio.

    Args:
        portfolio_assets: DataFrame of portfolio assets

    Returns:
        Dict with risk assessment results
    """
    try:
        # Convert DataFrame to the format expected by the ML model
        portfolio_df = pd.DataFrame(portfolio_assets) if not isinstance(portfolio_assets, pd.DataFrame) else portfolio_assets

        # Get risk assessment
        if user_preferences is None:
            user_preferences = {'risk_tolerance': 5, 'sustainability_focus': 5}

        risk_assessment = assess_portfolio_risk(portfolio_df, user_preferences)

        return risk_assessment
    except Exception as e:
        st.error(f"Error generating risk assessment: {str(e)}")
        # Fallback to simple risk calculation
        return {
            'risk_category': 'Moderate',
            'risk_score': 50,
            'error': str(e)
        }

def get_ml_market_sentiment(ticker):
    """
    Get ML-powered market sentiment analysis for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with sentiment analysis results
    """
    try:
        # Get sentiment analysis
        sentiment_analysis = analyze_market_sentiment(ticker)

        return sentiment_analysis
    except Exception as e:
        st.error(f"Error generating sentiment analysis: {str(e)}")
        # Fallback to neutral sentiment
        return {
            'ticker': ticker,
            'sentiment_score': 0,
            'overall_sentiment': 'Neutral',
            'error': str(e)
        }

def display_ml_recommendations(recommendations, theme_colors):
    """
    Display ML-powered recommendations in Streamlit.

    Args:
        recommendations: DataFrame of recommended assets
        theme_colors: Dict of theme colors
    """
    st.markdown("### AI-Powered Investment Recommendations")
    st.markdown("Our machine learning model has analyzed market data, ESG criteria, and your preferences to generate these recommendations:")

    # Display recommendations
    for i, (_, asset) in enumerate(recommendations.iterrows()):
        col1, col2 = st.columns([1, 3])

        with col1:
            # Asset icon
            if asset['asset_type'] == 'Stock':
                icon = "📈"
            else:
                icon = "🪙"

            st.markdown(f"<h1 style='font-size: 3rem; margin: 0; text-align: center;'>{icon}</h1>", unsafe_allow_html=True)

            # Recommendation strength
            rec_color = "#4CAF50" if "Strong" in asset['recommendation_strength'] else "#FFC107" if "Moderate" in asset['recommendation_strength'] else "#F44336"
            st.markdown(f"<p style='text-align: center; color: {rec_color}; font-weight: bold;'>{asset['recommendation_strength']}</p>", unsafe_allow_html=True)

        with col2:
            # Asset details
            st.markdown(f"#### {asset['name']} ({asset['ticker']})")

            # Key metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

            with metrics_col1:
                st.metric("ESG Score", f"{asset['esg_score']:.1f}")

            with metrics_col2:
                st.metric("1Y ROI", f"{asset['roi_1y']:.1f}%")

            with metrics_col3:
                st.metric("Volatility", f"{asset['volatility']:.2f}")

            with metrics_col4:
                st.metric("ML Score", f"{asset['final_score']:.1f}")

            # Recommendation rationale
            st.markdown("**AI Recommendation Rationale:**")

            if "Strong" in asset['recommendation_strength']:
                rationale = f"This {asset['asset_type'].lower()} shows strong potential with excellent ESG credentials and favorable risk-return characteristics. Our ML model gives it a high score based on its sustainability profile and financial metrics."
            elif "Moderate" in asset['recommendation_strength']:
                rationale = f"This {asset['asset_type'].lower()} presents a balanced opportunity with decent ESG performance and acceptable risk-return characteristics. Our ML model suggests considering it as part of a diversified portfolio."
            else:
                rationale = f"This {asset['asset_type'].lower()} may present higher risks or lower ESG performance. Our ML model suggests careful consideration before investing."

            st.markdown(rationale)

        st.markdown("---")

def display_ml_risk_assessment(risk_assessment, theme_colors):
    """
    Display ML-powered risk assessment in Streamlit.

    Args:
        risk_assessment: Dict with risk assessment results
        theme_colors: Dict of theme colors
    """
    st.markdown("### AI-Powered Risk Assessment")
    st.markdown("Our machine learning model has analyzed your portfolio to assess its risk profile:")

    # Risk score gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_assessment['risk_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Portfolio Risk Score"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': theme_colors['text_color']},
            'bar': {'color': theme_colors['accent_color']},
            'bgcolor': theme_colors['secondary_bg'],
            'borderwidth': 2,
            'bordercolor': theme_colors['text_color'],
            'steps': [
                {'range': [0, 25], 'color': '#4CAF50'},
                {'range': [25, 50], 'color': '#8BC34A'},
                {'range': [50, 75], 'color': '#FFC107'},
                {'range': [75, 100], 'color': '#F44336'}
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=theme_colors['bg_color'],
        font={'color': theme_colors['text_color']}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Risk category
    risk_color = {
        'Low': '#4CAF50',
        'Moderate': '#8BC34A',
        'High': '#FFC107',
        'Very High': '#F44336'
    }.get(risk_assessment['risk_category'], theme_colors['accent_color'])

    st.markdown(f"<h4 style='text-align: center;'>Risk Category: <span style='color: {risk_color};'>{risk_assessment['risk_category']}</span></h4>", unsafe_allow_html=True)

    # Risk breakdown
    if 'risk_factors' in risk_assessment:
        st.markdown("#### Risk Factor Breakdown")

        # Create risk factors DataFrame
        risk_factors_df = pd.DataFrame({
            'Factor': list(risk_assessment['risk_factors'].keys()),
            'Score': list(risk_assessment['risk_factors'].values())
        })

        # Sort by score descending
        risk_factors_df = risk_factors_df.sort_values('Score', ascending=False)

        # Create horizontal bar chart
        fig = px.bar(
            risk_factors_df,
            x='Score',
            y='Factor',
            orientation='h',
            color='Score',
            color_continuous_scale=['green', 'yellow', 'red'],
            range_color=[0, 100],
            title="Risk Factor Analysis"
        )

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=theme_colors['bg_color'],
            plot_bgcolor=theme_colors['bg_color'],
            font={'color': theme_colors['text_color']},
            xaxis={'title': 'Risk Score (0-100)'},
            yaxis={'title': ''},
            coloraxis_colorbar={'title': 'Risk Level'}
        )

        st.plotly_chart(fig, use_container_width=True)

    # Risk explanation
    st.markdown("#### AI Risk Analysis")

    if risk_assessment['risk_category'] == 'Low':
        explanation = "Your portfolio demonstrates a conservative risk profile with strong stability metrics. The AI model indicates low volatility and good diversification across assets. This portfolio is well-positioned to withstand market fluctuations while maintaining sustainable investment principles."
    elif risk_assessment['risk_category'] == 'Moderate':
        explanation = "Your portfolio shows a balanced risk profile with reasonable stability metrics. The AI model detects moderate volatility and adequate diversification. This portfolio strikes a good balance between growth potential and risk management while maintaining sustainable investment principles."
    elif risk_assessment['risk_category'] == 'High':
        explanation = "Your portfolio exhibits an aggressive risk profile with elevated volatility metrics. The AI model identifies higher exposure to market fluctuations and potentially concentrated positions. Consider diversifying further to reduce risk while maintaining sustainable investment principles."
    else:  # Very High
        explanation = "Your portfolio demonstrates a highly aggressive risk profile with significant volatility metrics. The AI model detects substantial exposure to market fluctuations and concentrated positions. Consider rebalancing to reduce risk exposure while maintaining sustainable investment principles."

    st.markdown(explanation)

    # Risk mitigation recommendations
    st.markdown("#### AI Risk Mitigation Recommendations")

    if risk_assessment['risk_category'] in ['High', 'Very High']:
        st.markdown("""
        1. **Increase Diversification**: Add assets from different sectors and asset classes
        2. **Reduce Concentration**: Limit exposure to any single asset to no more than 10% of portfolio
        3. **Add Defensive Assets**: Consider adding assets with lower correlation to market movements
        4. **Implement Stop-Loss Strategy**: Set predetermined exit points to limit potential losses
        """)
    elif risk_assessment['risk_category'] == 'Moderate':
        st.markdown("""
        1. **Review Sector Allocation**: Ensure balanced exposure across different sectors
        2. **Monitor Volatility**: Keep an eye on assets with higher volatility metrics
        3. **Consider Hedging**: Evaluate partial hedging strategies for larger positions
        4. **Regular Rebalancing**: Maintain target allocations through periodic rebalancing
        """)
    else:  # Low
        st.markdown("""
        1. **Maintain Diversification**: Continue with current diversification strategy
        2. **Consider Growth Opportunities**: Evaluate adding selective growth assets
        3. **Regular Monitoring**: Continue monitoring portfolio performance and risk metrics
        4. **Optimize for Tax Efficiency**: Review portfolio for tax optimization opportunities
        """)

def display_ml_sentiment_analysis(sentiment_analysis, theme_colors):
    """
    Display ML-powered sentiment analysis in Streamlit.

    Args:
        sentiment_analysis: Dict with sentiment analysis results
        theme_colors: Dict of theme colors
    """
    st.markdown("### AI-Powered Market Sentiment Analysis")
    st.markdown(f"Our machine learning model has analyzed recent news and market sentiment for {sentiment_analysis['ticker']}:")

    # Sentiment score gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_analysis['sentiment_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Sentiment Score"},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': theme_colors['text_color']},
            'bar': {'color': theme_colors['accent_color']},
            'bgcolor': theme_colors['secondary_bg'],
            'borderwidth': 2,
            'bordercolor': theme_colors['text_color'],
            'steps': [
                {'range': [-100, -30], 'color': '#F44336'},
                {'range': [-30, -10], 'color': '#FFC107'},
                {'range': [-10, 10], 'color': '#9E9E9E'},
                {'range': [10, 30], 'color': '#8BC34A'},
                {'range': [30, 100], 'color': '#4CAF50'}
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=theme_colors['bg_color'],
        font={'color': theme_colors['text_color']}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Overall sentiment
    sentiment_color = {
        'Bullish': '#4CAF50',
        'Somewhat Bullish': '#8BC34A',
        'Neutral': '#9E9E9E',
        'Somewhat Bearish': '#FFC107',
        'Bearish': '#F44336'
    }.get(sentiment_analysis['overall_sentiment'], theme_colors['accent_color'])

    st.markdown(f"<h4 style='text-align: center;'>Overall Sentiment: <span style='color: {sentiment_color};'>{sentiment_analysis['overall_sentiment']}</span></h4>", unsafe_allow_html=True)

    # Sentiment breakdown
    if 'sentiment_counts' in sentiment_analysis:
        st.markdown("#### Sentiment Breakdown")

        # Create sentiment counts DataFrame
        sentiment_counts = sentiment_analysis['sentiment_counts']
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_counts.keys()),
            'Count': list(sentiment_counts.values())
        })

        # Create pie chart
        fig = px.pie(
            sentiment_df,
            values='Count',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'positive': '#4CAF50',
                'neutral': '#9E9E9E',
                'negative': '#F44336'
            },
            title="News Sentiment Distribution"
        )

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=theme_colors['bg_color'],
            font={'color': theme_colors['text_color']}
        )

        st.plotly_chart(fig, use_container_width=True)

    # Recent news
    if 'news' in sentiment_analysis:
        st.markdown("#### Recent News Analysis")

        for news in sentiment_analysis['news']:
            # Sentiment icon and color
            sentiment_icon = {
                'positive': '🟢',
                'neutral': '🟡',
                'negative': '🔴'
            }.get(news['predicted_sentiment'], '🟡')

            sentiment_color = {
                'positive': '#4CAF50',
                'neutral': '#9E9E9E',
                'negative': '#F44336'
            }.get(news['predicted_sentiment'], '#9E9E9E')

            # Display news item
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {sentiment_color}; margin-bottom: 10px; background-color: {theme_colors['secondary_bg']};">
                <p style="margin: 0; font-weight: bold;">{news['headline']}</p>
                <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: {theme_colors['text_color']}; opacity: 0.8;">
                    {news['source']} | {news['publication_date']} |
                    <span style="color: {sentiment_color};">AI Sentiment: {sentiment_icon} {news['predicted_sentiment'].capitalize()}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Sentiment explanation
    st.markdown("#### AI Sentiment Analysis")

    if sentiment_analysis['overall_sentiment'] in ['Bullish', 'Somewhat Bullish']:
        explanation = f"The AI model has detected predominantly positive sentiment around {sentiment_analysis['ticker']} based on recent news and market signals. This positive sentiment may indicate favorable market perception and potential upward price movement in the near term."
    elif sentiment_analysis['overall_sentiment'] == 'Neutral':
        explanation = f"The AI model has detected balanced sentiment around {sentiment_analysis['ticker']} based on recent news and market signals. This neutral sentiment suggests stable market perception without strong directional bias in the near term."
    else:  # Bearish or Somewhat Bearish
        explanation = f"The AI model has detected predominantly negative sentiment around {sentiment_analysis['ticker']} based on recent news and market signals. This negative sentiment may indicate concerns in the market and potential downward price pressure in the near term."

    st.markdown(explanation)

    # Trading implications
    st.markdown("#### Trading Implications")

    if sentiment_analysis['overall_sentiment'] in ['Bullish', 'Somewhat Bullish']:
        st.markdown("""
        - **Consider increasing position** if aligned with your investment strategy
        - **Monitor for confirmation** through technical indicators and volume
        - **Set appropriate stop-loss levels** to protect against unexpected reversals
        - **Consider ESG implications** alongside positive sentiment signals
        """)
    elif sentiment_analysis['overall_sentiment'] == 'Neutral':
        st.markdown("""
        - **Maintain current position** if already invested
        - **Watch for emerging trends** in either direction
        - **Consider selling covered calls** to generate income during sideways movement
        - **Focus on fundamentals** rather than short-term sentiment
        """)
    else:  # Bearish or Somewhat Bearish
        st.markdown("""
        - **Consider reducing exposure** if already invested
        - **Implement hedging strategies** to protect against downside
        - **Watch for potential overselling** that might create entry opportunities
        - **Evaluate if negative sentiment is related to ESG concerns** or purely financial
        """)
