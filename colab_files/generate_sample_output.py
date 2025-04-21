"""
Generate sample output from ML models for demonstration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set plot style for dark theme
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Create output directory
os.makedirs('sample_output', exist_ok=True)

# Generate dataset
print("Generating large dataset...")
from large_portfolio_dataset import generate_large_dataset
portfolio_df = generate_large_dataset()
portfolio_df.to_csv('large_portfolio_dataset.csv', index=False)
print(f"Generated dataset with {len(portfolio_df)} assets")

# Generate market news
print("\nGenerating market news...")
from market_news_generator import generate_market_news
tickers = portfolio_df['ticker'].unique().tolist()
sample_size = min(100, len(tickers))
np.random.seed(42)
sampled_tickers = np.random.choice(tickers, sample_size, replace=False)
news_df = generate_market_news(sampled_tickers.tolist(), num_articles_per_ticker=10)
news_df.to_csv('large_market_news_dataset.csv', index=False)
print(f"Generated news dataset with {len(news_df)} articles")

# Create a sample portfolio
print("\nCreating sample portfolio...")
np.random.seed(42)
portfolio_assets = portfolio_df.sample(15).copy()
portfolio_assets.to_csv('sample_portfolio.csv', index=False)
print(f"Created sample portfolio with {len(portfolio_assets)} assets")

# Set user preferences
user_preferences = {
    'risk_tolerance': 7,  # 1-10 (higher = more risk tolerant)
    'sustainability_focus': 8  # 1-10 (higher = more sustainability focused)
}

print(f"\nUser Preferences: Risk Tolerance = {user_preferences['risk_tolerance']}/10, Sustainability Focus = {user_preferences['sustainability_focus']}/10")

# 1. Portfolio Recommendation Model
print("\n1. Running Portfolio Recommendation Model...")
from portfolio_recommendation_model import PortfolioRecommendationModel, generate_training_data, get_portfolio_recommendations, visualize_recommendations

# Get portfolio recommendations
recommendations = get_portfolio_recommendations(portfolio_assets, portfolio_df, user_preferences)

# Save top recommendations to CSV
recommendations.head(20).to_csv('sample_output/top_recommendations.csv', index=False)
print(f"Saved top 20 recommendations to sample_output/top_recommendations.csv")

# Visualize recommendations
visualize_recommendations(recommendations)
print("Saved recommendation visualizations to models/ directory")

# 2. Risk Assessment Model
print("\n2. Running Risk Assessment Model...")
from risk_assessment_model import RiskAssessmentModel, generate_training_data, assess_portfolio_risk, visualize_risk_factors

# Assess portfolio risk
risk_assessment = assess_portfolio_risk(portfolio_assets)

# Save risk assessment to CSV
risk_factors_df = pd.DataFrame({
    'Factor': list(risk_assessment['risk_factors'].keys()),
    'Score': list(risk_assessment['risk_factors'].values())
})
risk_factors_df.to_csv('sample_output/risk_factors.csv', index=False)

risk_probs_df = pd.DataFrame({
    'Category': list(risk_assessment['risk_probabilities'].keys()),
    'Probability': list(risk_assessment['risk_probabilities'].values())
})
risk_probs_df.to_csv('sample_output/risk_probabilities.csv', index=False)

# Save portfolio metrics
pd.DataFrame([risk_assessment['portfolio_metrics']]).to_csv('sample_output/portfolio_metrics.csv', index=False)

print(f"Risk Category: {risk_assessment['risk_category']}")
print(f"Risk Score: {risk_assessment['risk_score']:.2f}/100")
print("Saved risk assessment data to sample_output/ directory")

# 3. Sentiment Analysis Model
print("\n3. Running Sentiment Analysis Model...")
from sentiment_analysis_model import SentimentAnalysisModel, generate_training_data, analyze_market_sentiment, generate_market_news, visualize_sentiment

# Select a ticker from the portfolio
selected_ticker = portfolio_assets['ticker'].iloc[0]
print(f"Selected ticker for sentiment analysis: {selected_ticker}")

# Analyze market sentiment
sentiment_analysis = analyze_market_sentiment(selected_ticker, news_df)

# Save sentiment analysis to CSV
sentiment_counts_df = pd.DataFrame({
    'Sentiment': list(sentiment_analysis['sentiment_counts'].keys()),
    'Count': list(sentiment_analysis['sentiment_counts'].values())
})
sentiment_counts_df.to_csv('sample_output/sentiment_counts.csv', index=False)

# Save news with sentiment
news_df = pd.DataFrame(sentiment_analysis['news'])
news_df.to_csv('sample_output/ticker_news_with_sentiment.csv', index=False)

print(f"Overall Sentiment: {sentiment_analysis['overall_sentiment']}")
print(f"Sentiment Score: {sentiment_analysis['sentiment_score']:.2f} (-100 to 100)")
print("Saved sentiment analysis data to sample_output/ directory")

# Create a summary report
print("\nCreating summary report...")
with open('sample_output/ml_models_summary.txt', 'w') as f:
    f.write("ML MODELS FOR SUSTAINABLE INVESTMENT PORTFOLIO\n")
    f.write("=============================================\n\n")
    f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATASET STATISTICS\n")
    f.write("-----------------\n")
    f.write(f"Total assets: {len(portfolio_df)}\n")
    f.write(f"Stocks: {len(portfolio_df[portfolio_df['asset_type'] == 'Stock'])}\n")
    f.write(f"Cryptocurrencies: {len(portfolio_df[portfolio_df['asset_type'] == 'Crypto'])}\n")
    f.write(f"Average ESG score: {portfolio_df['esg_score'].mean():.2f}\n")
    f.write(f"Average volatility: {portfolio_df['volatility'].mean():.2f}\n")
    f.write(f"Average ROI (1Y): {portfolio_df['roi_1y'].mean():.2f}%\n\n")
    
    f.write("PORTFOLIO RECOMMENDATIONS\n")
    f.write("------------------------\n")
    f.write(f"User preferences: Risk Tolerance = {user_preferences['risk_tolerance']}/10, Sustainability Focus = {user_preferences['sustainability_focus']}/10\n")
    f.write("Top 5 recommendations:\n")
    for i, (_, asset) in enumerate(recommendations.head(5).iterrows()):
        f.write(f"{i+1}. {asset['name']} ({asset['ticker']}) - {asset['asset_type']}\n")
        f.write(f"   ESG Score: {asset['esg_score']:.1f}, ROI: {asset['roi_1y']:.1f}%, Volatility: {asset['volatility']:.2f}\n")
        f.write(f"   Final Score: {asset['final_score']:.1f}, Recommendation: {asset['recommendation_strength']}\n\n")
    
    f.write("RISK ASSESSMENT\n")
    f.write("--------------\n")
    f.write(f"Risk Category: {risk_assessment['risk_category']}\n")
    f.write(f"Risk Score: {risk_assessment['risk_score']:.2f}/100\n\n")
    f.write("Risk Probabilities:\n")
    for category, prob in risk_assessment['risk_probabilities'].items():
        f.write(f"- {category}: {prob:.2%}\n")
    f.write("\nTop Risk Factors:\n")
    for factor, score in sorted(risk_assessment['risk_factors'].items(), key=lambda x: x[1], reverse=True)[:5]:
        f.write(f"- {factor}: {score:.2f}/100\n")
    
    f.write("\nSENTIMENT ANALYSIS\n")
    f.write("-----------------\n")
    f.write(f"Ticker: {selected_ticker}\n")
    f.write(f"Overall Sentiment: {sentiment_analysis['overall_sentiment']}\n")
    f.write(f"Sentiment Score: {sentiment_analysis['sentiment_score']:.2f} (-100 to 100)\n\n")
    f.write("Sentiment Distribution:\n")
    for sentiment, count in sentiment_analysis['sentiment_counts'].items():
        f.write(f"- {sentiment.capitalize()}: {count} ({count/sum(sentiment_analysis['sentiment_counts'].values())*100:.1f}%)\n")
    f.write("\nRecent News Headlines:\n")
    for i, news in enumerate(sentiment_analysis['news'][:5]):
        f.write(f"{i+1}. {news['headline']}\n")
        f.write(f"   Source: {news['source']}, Date: {news['publication_date']}\n")
        f.write(f"   Sentiment: {news['predicted_sentiment']}\n\n")
    
    f.write("\nCONCLUSION\n")
    f.write("----------\n")
    f.write("The ML models have successfully analyzed the portfolio data and provided:\n")
    f.write("1. Personalized investment recommendations based on ESG criteria and user preferences\n")
    f.write("2. Comprehensive risk assessment with detailed risk factor breakdown\n")
    f.write("3. Market sentiment analysis for selected assets\n\n")
    f.write("These insights can help investors make more informed decisions that align with both their financial goals and sustainability values.")

print("Created summary report at sample_output/ml_models_summary.txt")
print("\nAll sample outputs generated successfully!")
