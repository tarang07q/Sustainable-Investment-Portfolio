import streamlit as st
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.supabase import is_authenticated, get_current_user
from utils.quotes import get_random_quote
from utils.financial_data import get_all_assets, get_market_trends

# Initialize theme in session state if not present
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

st.set_page_config(
    page_title="Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Theme toggle in sidebar
with st.sidebar:
    if st.button("🌓 Toggle Theme"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Apply theme-specific styles
theme_bg_color = "#0e1117" if st.session_state.theme == "dark" else "#ffffff"
theme_text_color = "#ffffff" if st.session_state.theme == "dark" else "#0e1117"
theme_secondary_bg = "#1e2530" if st.session_state.theme == "dark" else "#f0f2f6"

# Custom CSS with dynamic theming
st.markdown(f"""
    <style>
    .main {{
        padding: 2rem;
        background-color: {theme_bg_color};
        color: {theme_text_color};
    }}
    .stButton>button {{
        width: 100%;
    }}
    .recommendation-card {{
        background-color: {theme_secondary_bg};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: {theme_text_color};
    }}
    .css-1v3fvcr {{
        background-color: {theme_bg_color};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {theme_secondary_bg};
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: {theme_text_color};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #4CAF50;
        color: white;
    }}
    .quote-container {{
        background-color: {theme_secondary_bg};
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0 2rem 0;
        border-left: 5px solid #4CAF50;
        color: {theme_text_color};
    }}
    .auth-banner {{
        background-color: {theme_secondary_bg};
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: {theme_text_color};
    }}
    .trend-card {{
        background-color: {theme_secondary_bg};
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin: 0.5rem 0;
        border-left: 3px solid #2196F3;
        color: {theme_text_color};
    }}
    .asset-card {{
        background-color: {theme_secondary_bg};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: {theme_text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Check if user is authenticated
if not is_authenticated():
    # Authentication banner
    st.markdown("""
    <div class="auth-banner">
        <div>
            <h3 style="margin: 0;">🔐 Sign in to access all features</h3>
            <p style="margin: 0;">Create an account or sign in to save portfolios and get personalized recommendations.</p>
        </div>
        <div>
            <a href="/Authentication" target="_self"><button style="background-color: #4CAF50; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer;">Sign In / Sign Up</button></a>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Welcome message for authenticated users
    user = get_current_user()
    st.markdown(f"""<div style="background-color: #e8f5e9; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="margin: 0;">👋 Welcome back, {user['full_name']}!</h3>
        <p style="margin: 0;">Your sustainable investment journey continues. Check out the latest market trends and recommendations.</p>
    </div>""", unsafe_allow_html=True)

# Header
st.title("🌱 Sustainable Investment Portfolio")
st.markdown("*AI-powered investment platform for sustainable and profitable investing*")

# Display a random quote
quote = get_random_quote()
st.markdown(f"""
<div class="quote-container">
    <p style="font-style: italic; font-size: 1.1rem;">"{quote['quote']}"</p>
    <p style="text-align: right; font-weight: bold;">— {quote['author']}</p>
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
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Data-Driven Insights")
    st.write("""
    Access comprehensive ESG ratings and financial metrics for stocks and cryptocurrencies.
    Our AI analyzes multiple data sources to provide you with the most accurate information.
    """)
    if st.button("Explore Analytics", key="analytics_btn"):
        st.switch_page("pages/1_Market_Explorer.py")

with col2:
    st.markdown("### 💼 Portfolio Management")
    st.write("""
    Create and manage custom portfolios that align with your financial goals and sustainability values.
    Track performance, assess risks, and understand your impact.
    """)
    if st.button("Manage Portfolios", key="portfolio_btn"):
        st.switch_page("pages/2_Portfolio_Manager.py")

with col3:
    st.markdown("### 🤖 AI Recommendations")
    st.write("""
    Receive personalized investment recommendations based on your risk profile, financial goals,
    and sustainability preferences. Our AI continuously learns from market trends and ESG developments.
    """)
    if st.button("Get Recommendations", key="recommendations_btn"):
        st.switch_page("pages/3_AI_Recommendations.py")

# Market trends section
st.markdown("---")
st.markdown("## Latest Market Trends")

# Get market trends
market_trends = get_market_trends()

# Display top 3 market trends
for i, trend in enumerate(market_trends[:3]):
    st.markdown(f"""
    <div class="trend-card">
        <h4>{trend['title']} <span style="float:right;font-size:0.8rem;color:#666;">Confidence: {trend['confidence']}%</span></h4>
        <p>{trend['description']}</p>
        <p><strong>Impact:</strong> {trend['impact']}</p>
    </div>
    """, unsafe_allow_html=True)

# Top performing assets
st.markdown("## Top Performing Sustainable Assets")

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
        price_color = "green" if price_change > 0 else "red"

        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; height: 100%;">
            <h4>{asset['name']} ({asset['symbol']})</h4>
            <p><strong>Price:</strong> ${asset['current_price']:.2f}</p>
            <p><strong>24h:</strong> <span style="color:{price_color};">{price_change:.2f}%</span></p>
            <p><strong>1Y ROI:</strong> {asset['roi_1y']}%</p>
            <p><strong>ESG Score:</strong> {asset['esg_score']:.1f}/100</p>
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
