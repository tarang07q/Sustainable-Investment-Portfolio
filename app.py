import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.supabase import is_authenticated, get_current_user
from utils.quotes import get_random_quote
from utils.financial_data import get_all_assets, get_market_trends
from utils.theme import apply_theme_css

st.set_page_config(
    page_title="Sustainable Investment Portfolio - Home",
    page_icon="🌱",
    layout="wide"
)

# Apply theme-specific styles
theme_colors = apply_theme_css()
plotly_template = theme_colors['plotly_template']

# Add Font Awesome for icons
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)

# Add custom CSS for enhanced styling
st.markdown("""
<style>
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom fonts and typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 600;
    }

    p, div, span, li {
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Custom sidebar styling */
    .css-1d391kg, .css-1lcbmhc {  /* Sidebar background */
        background-color: #1a1f2b;
    }

    /* Dashboard styles */
    .dashboard-container {
        background-color: #1e2530;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }

    .dashboard-container::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(76, 175, 80, 0.1) 0%, rgba(0, 0, 0, 0) 70%);
        z-index: 0;
    }

    .dashboard-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }

    .dashboard-icon {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-right: 1.5rem;
        background-color: rgba(76, 175, 80, 0.1);
        width: 70px;
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .metric-card {
        background-color: rgba(30, 37, 48, 0.7);
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--metric-color) 0%, transparent 100%);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }

    .metric-icon {
        font-size: 1.8rem;
        margin-right: 1rem;
        background-color: rgba(var(--metric-bg), 0.15);
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
    }

    /* Enhanced feature cards */
    .feature-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border-left: 4px solid #4CAF50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, rgba(76, 175, 80, 0.05) 0%, rgba(0, 0, 0, 0) 70%);
        z-index: 0;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }

    .feature-button {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        font-weight: 500;
        text-align: center;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .feature-button:hover {
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        transform: translateY(-2px);
    }

    /* Enhanced trend cards */
    .trend-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4CAF50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .trend-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Enhanced asset cards */
    .asset-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
        border-left: 4px solid #4CAF50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .asset-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Button styling */
    .feature-button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.8rem;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        position: relative;
        overflow: hidden;
    }

    .feature-button:hover {
        background-color: #3e8e41;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .feature-button:after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: rgba(255, 255, 255, 0.5);
        opacity: 0;
        border-radius: 100%;
        transform: scale(1, 1) translate(-50%);
        transform-origin: 50% 50%;
    }

    .feature-button:focus:not(:active)::after {
        animation: ripple 1s ease-out;
    }

    @keyframes ripple {
        0% {
            transform: scale(0, 0);
            opacity: 0.5;
        }
        20% {
            transform: scale(25, 25);
            opacity: 0.3;
        }
        100% {
            opacity: 0;
            transform: scale(40, 40);
        }
    }

    /* Confidence badge */
    .confidence-badge {
        font-size: 0.8rem;
        color: #e0e0e0;
        background-color: #2c3440;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        display: inline-block;
    }

    /* Data sources section */
    .data-source-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1rem;
    }

    .data-source-item {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border-left: 3px solid #4CAF50;
    }

    .data-source-item:hover {
        background-color: #2c3440;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }

    /* Form styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1e2530;
        color: #ffffff;
        border: 1px solid #2c3440;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 1px #4CAF50;
    }

    /* Custom header */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #333;
    }

    /* Custom navigation */
    .nav-link {
        color: #e0e0e0;
        text-decoration: none;
        margin-right: 1.5rem;
        font-weight: 500;
        transition: color 0.3s ease;
        display: inline-flex;
        align-items: center;
    }

    .nav-link:hover {
        color: #4CAF50;
    }

    .nav-link i {
        margin-right: 0.5rem;
    }

    /* Custom input with icon */
    .input-with-icon {
        position: relative;
    }

    .input-with-icon i {
        position: absolute;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        color: #888;
    }

    .input-with-icon input {
        padding-left: 35px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header section with navigation and authentication
st.markdown("""
<div class="custom-header">
    <div style="display: flex; align-items: center;">
        <span style="font-size: 1.8rem; margin-right: 10px;">🌱</span>
        <h1 style="margin: 0; font-size: 1.8rem;">Sustainable Investment Portfolio</h1>
    </div>
    <div>
        <a href="/" class="nav-link"><i class="fas fa-home"></i> Home</a>
        <a href="/ESG_Education" class="nav-link"><i class="fas fa-graduation-cap"></i> ESG Education</a>
        <a href="/Market_Explorer" class="nav-link"><i class="fas fa-chart-line"></i> Markets</a>
        <a href="/Portfolio_Manager" class="nav-link"><i class="fas fa-briefcase"></i> Portfolio</a>
        <a href="/AI_Recommendations" class="nav-link"><i class="fas fa-robot"></i> AI Advisor</a>
""", unsafe_allow_html=True)

# Check if user is authenticated
if not is_authenticated():
    # Authentication button
    st.markdown("""
        <a href="/Authentication" target="_self" style="text-decoration: none;">
            <button style="background-color: #4CAF50; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: 500; transition: all 0.3s ease; display: flex; align-items: center;">
                <i class="fas fa-user" style="margin-right: 8px;"></i> Sign In / Sign Up
            </button>
        </a>
    </div>
</div>

<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3rem; background-color: #1e2530; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <div style="max-width: 60%;">
        <h2 style="margin: 0; font-size: 2.2rem; margin-bottom: 1rem; background: linear-gradient(90deg, #4CAF50, #2196F3); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Sustainable investing for a better future</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">Our AI-powered platform helps you build profitable portfolios while making a positive impact on the planet.</p>
        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
            <div style="display: flex; align-items: center;">
                <i class="fas fa-check-circle" style="color: #4CAF50; margin-right: 8px; font-size: 1.2rem;"></i>
                <span>ESG-focused investments</span>
            </div>
            <div style="display: flex; align-items: center;">
                <i class="fas fa-check-circle" style="color: #4CAF50; margin-right: 8px; font-size: 1.2rem;"></i>
                <span>AI-powered recommendations</span>
            </div>
            <div style="display: flex; align-items: center;">
                <i class="fas fa-check-circle" style="color: #4CAF50; margin-right: 8px; font-size: 1.2rem;"></i>
                <span>Real-time analytics</span>
            </div>
        </div>
        <div style="margin-top: 1.5rem;">
            <a href="/ESG_Education" style="text-decoration: none; color: #4CAF50; font-weight: 500; display: flex; align-items: center;">
                <i class="fas fa-graduation-cap" style="margin-right: 8px;"></i> Learn about ESG investing in our free Education Center
                <i class="fas fa-arrow-right" style="margin-left: 8px; font-size: 0.9rem;"></i>
            </a>
        </div>
    </div>
    <div>
        <a href="/Authentication" target="_self" style="text-decoration: none;">
            <button style="background-color: #4CAF50; color: white; border: none; padding: 1rem 2rem; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 1.1rem; transition: all 0.3s ease; display: flex; align-items: center;">
                <i class="fas fa-rocket" style="margin-right: 10px;"></i> Get Started
            </button>
        </a>
    </div>
</div>
    """, unsafe_allow_html=True)
else:
    # Welcome message for authenticated users
    user = get_current_user()

    # User welcome header
    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <span style="color: #e0e0e0; font-size: 1rem; margin-right: 1rem;">Welcome, {user['full_name']}</span>
            <a href="/User_Profile" style="text-decoration: none;">
                <button style="background-color: transparent; color: #e0e0e0; border: 1px solid #4CAF50; padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.9rem; transition: all 0.3s ease; display: flex; align-items: center;">
                    <i class="fas fa-user-circle" style="margin-right: 6px;"></i> My Profile
                </button>
            </a>
        </div>
    </div>
</div>
    """, unsafe_allow_html=True)

    # Dashboard container
    st.markdown(f"""
    <div class="dashboard-container">
        <div class="dashboard-header">
            <div class="dashboard-icon">
                <i class="fas fa-solar-panel"></i>
            </div>
            <div>
                <h2 style="margin: 0; font-size: 1.8rem; margin-bottom: 0.5rem;">Welcome to your dashboard, {user['full_name'].split()[0]}</h2>
                <p style="font-size: 1.1rem; opacity: 0.8; margin: 0;">Your sustainable investment journey continues</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card" style="--metric-color: #4CAF50;">
            <div style="display: flex; align-items: center;">
                <div class="metric-icon" style="--metric-bg: 76, 175, 80; color: #4CAF50;">
                    <i class="fas fa-leaf"></i>
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 1rem; color: #aaa;">ESG Impact</h4>
                    <p style="margin: 0; font-size: 1.4rem; font-weight: 600; color: #fff;">High</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="--metric-color: #2196F3;">
            <div style="display: flex; align-items: center;">
                <div class="metric-icon" style="--metric-bg: 33, 150, 243; color: #2196F3;">
                    <i class="fas fa-chart-pie"></i>
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 1rem; color: #aaa;">Portfolio Value</h4>
                    <p style="margin: 0; font-size: 1.4rem; font-weight: 600; color: #fff;">$24,680</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="--metric-color: #FF9800;">
            <div style="display: flex; align-items: center;">
                <div class="metric-icon" style="--metric-bg: 255, 152, 0; color: #FF9800;">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div>
                    <h4 style="margin: 0; font-size: 1rem; color: #aaa;">Performance</h4>
                    <p style="margin: 0; font-size: 1.4rem; font-weight: 600; color: #fff;">+12.4%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Quote section
quote = get_random_quote()
st.markdown(f"""
<div style="margin: 2rem 0; padding-left: 2rem; border-left: 3px solid #4CAF50;">
    <p style="font-style: italic; font-size: 1.2rem; margin-bottom: 0.5rem; color: #e0e0e0;">"{quote['quote']}"</p>
    <p style="font-weight: 500; color: #aaa;">— {quote['author']}</p>
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
            <h3 style="margin-top: 0; display: flex; align-items: center;">
                <i class="fas fa-chart-bar" style="color: #4CAF50; font-size: 1.5rem; margin-right: 10px;"></i>
                Data-Driven Insights
            </h3>
            <p style="margin-bottom: 1.5rem;">
                Access comprehensive ESG ratings and financial metrics for stocks and cryptocurrencies.
                Our AI analyzes multiple data sources to provide you with the most accurate information.
            </p>
        </div>
        <div>
            <a href="/Market_Explorer" target="_self" style="text-decoration: none; display: block;">
                <div class="feature-button">
                    Explore Analytics
                    <i class="fas fa-arrow-right" style="margin-left: 8px; font-size: 0.9rem;"></i>
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div>
            <h3 style="margin-top: 0; display: flex; align-items: center;">
                <i class="fas fa-briefcase" style="color: #2196F3; font-size: 1.5rem; margin-right: 10px;"></i>
                Portfolio Management
            </h3>
            <p style="margin-bottom: 1.5rem;">
                Create and manage custom portfolios that align with your financial goals and sustainability values.
                Track performance, assess risks, and understand your impact.
            </p>
        </div>
        <div>
            <a href="/Portfolio_Manager" target="_self" style="text-decoration: none; display: block;">
                <div class="feature-button">
                    Manage Portfolios
                    <i class="fas fa-arrow-right" style="margin-left: 8px; font-size: 0.9rem;"></i>
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div>
            <h3 style="margin-top: 0; display: flex; align-items: center;">
                <i class="fas fa-robot" style="color: #FF9800; font-size: 1.5rem; margin-right: 10px;"></i>
                AI Recommendations
            </h3>
            <p style="margin-bottom: 1.5rem;">
                Receive personalized investment recommendations based on your risk profile, financial goals,
                and sustainability preferences. Our AI continuously learns from market trends and ESG developments.
            </p>
        </div>
        <div>
            <a href="/AI_Recommendations" target="_self" style="text-decoration: none; display: block;">
                <div class="feature-button">
                    Get Recommendations
                    <i class="fas fa-arrow-right" style="margin-left: 8px; font-size: 0.9rem;"></i>
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Market trends section
st.markdown("<hr style='margin: 2rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-bottom: 1.5rem;'>Latest Market Trends <span style='font-size: 0.8rem; color: #888; font-weight: normal;'>ⓘ</span></h2>", unsafe_allow_html=True)

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

        # Create confidence indicator
        confidence = trend['confidence']
        confidence_color = "#F44336" if confidence < 70 else "#FF9800" if confidence < 85 else "#4CAF50"

        st.markdown(f"""
        <div class="trend-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <h4 style="margin: 0; font-size: 1.1rem;">{trend['title']}</h4>
                <div class="confidence-badge">Confidence: {confidence}%</div>
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
top_assets = sorted(all_assets, key=lambda x: x['roi_1y'], reverse=True)[:8]

# Display top assets in two rows of 4
row1_cols = st.columns(4)
row2_cols = st.columns(4)
cols = row1_cols + row2_cols

for i, asset in enumerate(top_assets):
    with cols[i]:
        price_change = asset['price_change_24h']
        price_color = "#4CAF50" if price_change > 0 else "#F44336"

        # Calculate a performance indicator (0-100) for the progress bar
        performance = min(100, max(0, asset['roi_1y'] / 2))

        # Determine asset type icon
        asset_icon = "💰" if asset['asset_type'] == "Stock" else "🔷"

        st.markdown(f"""
        <div class="asset-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h4 style="margin: 0; font-size: 1.1rem;">{asset_icon} {asset['name']}</h4>
                <span style="font-size: 0.8rem; color: #888;">({asset['symbol']})</span>
            </div>
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

# Data sources section
# Get current date for the update timestamp
current_date = datetime.now().strftime('%Y-%m-%d')

st.markdown("<h3 style='margin-top: 2rem;'>Data Sources & Information Providers</h3>", unsafe_allow_html=True)
st.markdown(f"""
<div class="data-source-container">
    <a href="https://www.alphavantage.co/documentation/" target="_blank" style="text-decoration: none; color: inherit;">
        <div class="data-source-item">
            <span style="font-size: 1.2rem;">📈</span>
            <span>Alpha Vantage - Market Data</span>
        </div>
    </a>
    <a href="https://www.kaggle.com/datasets/debashish311601/esg-scores-and-ratings" target="_blank" style="text-decoration: none; color: inherit;">
        <div class="data-source-item">
            <span style="font-size: 1.2rem;">🌿</span>
            <span>Kaggle - ESG Scores Dataset</span>
        </div>
    </a>
    <a href="https://finance.yahoo.com/" target="_blank" style="text-decoration: none; color: inherit;">
        <div class="data-source-item">
            <span style="font-size: 1.2rem;">🔍</span>
            <span>Yahoo Finance - Financial Data</span>
        </div>
    </a>
    <a href="https://www.nasdaq.com/esg" target="_blank" style="text-decoration: none; color: inherit;">
        <div class="data-source-item">
            <span style="font-size: 1.2rem;">📊</span>
            <span>Nasdaq - ESG Data Hub</span>
        </div>
    </a>
    <a href="https://data.worldbank.org/topic/environment" target="_blank" style="text-decoration: none; color: inherit;">
        <div class="data-source-item">
            <span style="font-size: 1.2rem;">🌐</span>
            <span>World Bank - Environmental Data</span>
        </div>
    </a>
</div>

<p style="margin-top: 1rem; font-size: 0.9rem; color: #888;">Data is updated daily. Last update: {current_date}</p>
""", unsafe_allow_html=True)