import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import yfinance as yf

# Set page config
st.set_page_config(
    page_title="Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Generate dummy data
def generate_dummy_data() -> pd.DataFrame:
    np.random.seed(42)
    
    companies = {
        'Name': [
            'GreenTech Solutions', 'EcoEnergy Corp', 'Sustainable Pharma',
            'CleanRetail Inc', 'Future Mobility', 'Smart Agriculture',
            'Renewable Power', 'Circular Economy', 'WaterTech',
            'Sustainable Finance', 'Green Construction', 'EcoTransport'
        ],
        'Sector': [
            'Technology', 'Energy', 'Healthcare',
            'Retail', 'Transportation', 'Agriculture',
            'Energy', 'Manufacturing', 'Utilities',
            'Finance', 'Construction', 'Transportation'
        ]
    }
    
    df = pd.DataFrame(companies)
    df['ROI'] = np.random.uniform(4, 15, len(df))
    df['ESG_Score'] = np.random.uniform(0, 100, len(df))
    df['SDG8_Score'] = np.random.uniform(0, 10, len(df))
    
    # Calculate final score and recommendation
    df['Final_Score'] = (df['ROI'] * 0.5 + df['ESG_Score'] * 0.4 + df['SDG8_Score'] * 0.1)
    df['Recommendation'] = df['Final_Score'].apply(
        lambda x: '🟢 Eco-Leader' if x > 80 else '🟡 Neutral' if x > 60 else '🔴 Risk'
    )
    
    # Sort by final score
    df = df.sort_values('Final_Score', ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    
    return df

# Load data
df = generate_dummy_data()

# Header
st.title("🌱 Sustainable Investment Portfolio")
st.markdown("*Powered by ESG insights & SDG 8 alignment*")

# Sidebar for portfolio configuration
with st.sidebar:
    st.header("Portfolio Configuration")
    
    # Sector selection
    available_sectors = sorted(df['Sector'].unique())
    selected_sectors = st.multiselect(
        "Select Sectors",
        available_sectors,
        default=available_sectors[:2]
    )
    
    # Investment amount
    investment_amount = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    # Profit vs Sustainability slider
    profit_weight = st.slider(
        "Weight: Profitability vs Sustainability",
        min_value=0,
        max_value=100,
        value=50,
        help="Slide left for more sustainability focus, right for more profit focus"
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Company Performance")
    
    # Filter data based on selected sectors
    filtered_df = df[df['Sector'].isin(selected_sectors)]
    
    # Create ROI vs ESG Score scatter plot
    fig = px.scatter(
        filtered_df,
        x='ESG_Score',
        y='ROI',
        color='Recommendation',
        size='Final_Score',
        hover_data=['Name', 'Sector', 'SDG8_Score'],
        title='ROI vs ESG Score by Company'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display company table
    st.dataframe(
        filtered_df[['Name', 'Sector', 'ESG_Score', 'ROI', 'SDG8_Score', 'Rank', 'Recommendation']],
        use_container_width=True
    )

with col2:
    st.header("Top Recommendations")
    
    # Display top 3 recommended companies
    top_companies = filtered_df.head(3)
    for _, company in top_companies.iterrows():
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>{company['Name']}</h3>
            <p>📊 ESG Score: {company['ESG_Score']:.1f}</p>
            <p>💰 ROI: {company['ROI']:.1f}%</p>
            <p>🎯 {company['Recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Portfolio Impact Summary
    st.header("Portfolio Impact")
    total_esg = filtered_df['ESG_Score'].mean()
    total_roi = filtered_df['ROI'].mean()
    sdg_impact = len(filtered_df[filtered_df['SDG8_Score'] > 7])
    
    st.metric("Average ESG Score", f"{total_esg:.1f}")
    st.metric("Expected ROI", f"{total_roi:.1f}%")
    st.metric("SDG 8 Impact", f"Contributing to {sdg_impact} SDGs")

# Step 3: Load and preprocess data
def load_portfolio_data():
    """Load and preprocess portfolio data"""
    # Download stock data using yfinance
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'WMT']
    data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(period="1y")
        except:
            print(f"Could not download data for {ticker}")
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame()
    
    for ticker, df in data.items():
        if not df.empty:
            # Calculate metrics
            returns = df['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            roi = (df['Close'][-1] / df['Close'][0] - 1) * 100
            
            # Add to portfolio
            portfolio_df = portfolio_df.append({
                'ticker': ticker,
                'name': stock.info.get('longName', ticker),
                'sector': stock.info.get('sector', 'Unknown'),
                'current_price': df['Close'][-1],
                'volatility': volatility,
                'roi_1y': roi,
                'market_cap_b': stock.info.get('marketCap', 0) / 1e9,
                'beta': stock.info.get('beta', 1.0),
                'sharpe_ratio': roi / (volatility * 100) if volatility != 0 else 0
            }, ignore_index=True)
    
    return portfolio_df

# Load the data
portfolio_df = load_portfolio_data()
print("Portfolio Data Shape:", portfolio_df.shape)
portfolio_df.head() 