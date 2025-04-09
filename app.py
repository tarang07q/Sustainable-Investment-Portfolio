import streamlit as st
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.supabase import is_authenticated, get_current_user
from utils.quotes import get_random_quote
from utils.financial_data import get_all_assets, get_market_trends
from utils.theme import apply_theme_css

st.set_page_config(
    page_title="Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Theme is now handled by the theme utility

# Apply theme-specific styles
theme_colors = apply_theme_css()
plotly_template = theme_colors['plotly_template']

# Check if user is authenticated
if not is_authenticated():
    # Authentication banner
    st.markdown("""
    <div class="auth-banner" style="display: flex; justify-content: space-between; align-items: center; background-color: #1e2530; padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.2); border-left: 4px solid #4CAF50;">
        <div>
            <h3 style="margin: 0; font-size: 1.2rem; display: flex; align-items: center;">
                <span style="margin-right: 8px; font-size: 1.3rem;">🔐</span> Sign in to access all features
            </h3>
            <p style="margin: 0.3rem 0 0 0; color: #e0e0e0;">Create an account or sign in to save portfolios and get personalized recommendations.</p>
        </div>
        <div>
            <a href="/Authentication" target="_self" style="text-decoration: none;">
                <button style="background-color: #4CAF50; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: 500; transition: all 0.3s ease; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    Sign In / Sign Up
                </button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Welcome message for authenticated users
    user = get_current_user()
    st.markdown(f"""<div style="background-color: #1e2530; padding: 1.2rem; border-radius: 10px; margin-bottom: 2rem; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
        <h3 style="margin: 0; color: #ffffff;">👋 Welcome back, {user['full_name']}!</h3>
        <p style="margin: 0.5rem 0 0 0; color: #e0e0e0;">Your sustainable investment journey continues. Check out the latest market trends and recommendations.</p>
    </div>""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='margin-bottom: 0.2rem; font-size: 2.5rem;'>🌱 Sustainable Investment Portfolio</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; margin-bottom: 2rem; font-style: italic;'>AI-powered investment platform for sustainable and profitable investing</p>", unsafe_allow_html=True)

# Display a random quote
quote = get_random_quote()
st.markdown(f"""
<div class="quote-container">
    <p style="font-style: italic; font-size: 1.1rem;">"{quote['quote']}"</p>
    <p style="text-align: right; font-weight: 500;">— {quote['author']}</p>
</div>
""", unsafe_allow_html=True)

# Main navigation
st.markdown("### Welcome to your sustainable investing journey!")
st.write("""
This platform helps you make investment decisions that are both profitable and aligned with your values.
Our AI-powered analytics combine financial metrics with ESG (Environmental, Social, and Governance) criteria
to provide personalized recommendations.
""")

# Feature highlights
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add some spacing

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div>
            <h3 style="margin-top: 0;"><span style="font-size: 1.5rem;">📊</span> Data-Driven Insights</h3>
            <p style="margin-bottom: 1.5rem;">
                Access comprehensive ESG ratings and financial metrics for stocks and cryptocurrencies.
                Our AI analyzes multiple data sources to provide you with the most accurate information.
            </p>
        </div>
        <div>
            <button onclick="window.location.href='/Market_Explorer'" style="width: 100%; background-color: #4CAF50; color: white; border: none; padding: 0.6rem; border-radius: 4px; cursor: pointer; font-weight: 500;">Explore Analytics</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div>
            <h3 style="margin-top: 0;"><span style="font-size: 1.5rem;">💼</span> Portfolio Management</h3>
            <p style="margin-bottom: 1.5rem;">
                Create and manage custom portfolios that align with your financial goals and sustainability values.
                Track performance, assess risks, and understand your impact.
            </p>
        </div>
        <div>
            <button onclick="window.location.href='/Portfolio_Manager'" style="width: 100%; background-color: #4CAF50; color: white; border: none; padding: 0.6rem; border-radius: 4px; cursor: pointer; font-weight: 500;">Manage Portfolios</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div>
            <h3 style="margin-top: 0;"><span style="font-size: 1.5rem;">🤖</span> AI Recommendations</h3>
            <p style="margin-bottom: 1.5rem;">
                Receive personalized investment recommendations based on your risk profile, financial goals,
                and sustainability preferences. Our AI continuously learns from market trends and ESG developments.
            </p>
        </div>
        <div>
            <button onclick="window.location.href='/AI_Recommendations'" style="width: 100%; background-color: #4CAF50; color: white; border: none; padding: 0.6rem; border-radius: 4px; cursor: pointer; font-weight: 500;">Get Recommendations</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Market trends section
st.markdown("<hr style='margin: 2rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-bottom: 1.5rem;'>Latest Market Trends</h2>", unsafe_allow_html=True)

# Get market trends
market_trends = get_market_trends()

# Create columns for trends
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

# Display top 3 market trends
for i, trend in enumerate(market_trends[:3]):
    with cols[i]:
        # Determine impact color
        impact_color = "#4CAF50"  # Default green
        if "negative" in trend['impact'].lower():
            impact_color = "#F44336"  # Red for negative
        elif "neutral" in trend['impact'].lower():
            impact_color = "#FF9800"  # Orange for neutral

        st.markdown(f"""
        <div class="trend-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <h4 style="margin: 0; font-size: 1.1rem;">{trend['title']}</h4>
                <span style="font-size: 0.8rem; color: #e0e0e0; background-color: #2c3440; padding: 0.2rem 0.5rem; border-radius: 4px;">Confidence: {trend['confidence']}%</span>
            </div>
            <p style="margin-bottom: 1rem; font-size: 0.95rem;">{trend['description']}</p>
            <div style="display: flex; align-items: center;">
                <span style="font-weight: 600; margin-right: 0.5rem;">Impact:</span>
                <span style="color: {impact_color}; font-weight: 500;">{trend['impact']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Top performing assets
st.markdown("<h2 style='margin-top: 2rem;'>Top Performing Sustainable Assets</h2>", unsafe_allow_html=True)

# Get all assets
all_assets = get_all_assets()

# Sort by ROI
top_assets = sorted(all_assets, key=lambda x: x['roi_1y'], reverse=True)[:5]

# Display top assets
col1, col2, col3, col4, col5 = st.columns(5)
cols = [col1, col2, col3, col4, col5]

for i, asset in enumerate(top_assets):
    with cols[i]:
        price_change = asset['price_change_24h']
        price_color = "#4CAF50" if price_change > 0 else "#F44336"

        # Calculate a performance indicator (0-100) for the progress bar
        performance = min(100, max(0, asset['roi_1y'] / 2))

        st.markdown(f"""
        <div class="asset-card" style="height: auto;">
            <h4 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1.1rem;">{asset['name']} <span style="font-size: 0.9rem; color: #666;">({asset['symbol']})</span></h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 500;">Price:</span>
                <span>${asset['current_price']:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 500;">24h:</span>
                <span style="color:{price_color};">{price_change:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 500;">1Y ROI:</span>
                <span>{asset['roi_1y']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 500;">ESG Score:</span>
                <span>{asset['esg_score']:.1f}/100</span>
            </div>
            <div style="margin-top: 0.8rem;">
                <div style="height: 6px; background-color: #2c3440; border-radius: 3px; overflow: hidden;">
                    <div style="height: 100%; width: {performance}%; background-color: #4CAF50;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 0.2rem;">
                    <span>Performance</span>
                    <span>{performance:.0f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Getting started section
st.markdown("---")
st.markdown("## Getting Started")
st.write("""
1. **Explore Assets**: Browse our database of stocks and cryptocurrencies with ESG ratings
2. **Create a Portfolio**: Build a portfolio that matches your investment goals and values
3. **Analyze Performance**: Track your portfolio's financial returns and sustainability impact
4. **Get AI Recommendations**: Receive personalized suggestions to optimize your investments
""")

# Call to action
if not is_authenticated():
    st.markdown("### Ready to start your sustainable investing journey?")
    if st.button("Create Your Free Account", key="cta_signup"):
        st.switch_page("pages/0_Authentication.py")

# Footer
st.markdown("---")
st.markdown("### About the Platform")
st.write("""
The Sustainable Investment Portfolio App is designed to democratize access to responsible investing.
By combining financial analytics with ESG criteria, we help investors make decisions that generate returns
while contributing to a more sustainable future.
""")
