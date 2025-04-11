import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theme import apply_theme_css
from utils.sustainability_data import generate_sector_recommendations, generate_portfolio_impact_metrics

# Set page config
st.set_page_config(
    page_title="Portfolio Manager - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Apply theme-specific styles
theme_colors = apply_theme_css()
plotly_template = theme_colors['plotly_template']

# Add authentication check
from utils.auth_redirect import check_authentication
check_authentication()

# Generate dummy portfolio data
def generate_portfolio_data():
    # Create a sample portfolio
    portfolio = {
        'name': 'My Sustainable Portfolio',
        'created_date': '2023-01-15',
        'last_updated': '2023-04-09',
        'total_value': 10000,
        'assets': [
            {
                'name': 'GreenTech Solutions',
                'ticker': 'GRNT',
                'asset_type': 'Stock',
                'shares': 10,
                'purchase_price': 120.50,
                'current_price': 145.75,
                'esg_score': 85.2,
                'sector': 'Technology',
                'carbon_footprint': 28.5,
                'allocation': 0.12
            },
            {
                'name': 'Renewable Power',
                'ticker': 'RNWP',
                'asset_type': 'Stock',
                'shares': 25,
                'purchase_price': 45.25,
                'current_price': 52.30,
                'esg_score': 92.1,
                'sector': 'Energy',
                'carbon_footprint': 12.3,
                'allocation': 0.11
            },
            {
                'name': 'Sustainable Finance',
                'ticker': 'SUFN',
                'asset_type': 'Stock',
                'shares': 15,
                'purchase_price': 78.90,
                'current_price': 82.15,
                'esg_score': 79.8,
                'sector': 'Finance',
                'carbon_footprint': 18.7,
                'allocation': 0.10
            },
            {
                'name': 'EcoToken',
                'ticker': 'ECO',
                'asset_type': 'Crypto',
                'shares': 50,
                'purchase_price': 25.10,
                'current_price': 32.45,
                'esg_score': 76.5,
                'sector': 'Renewable Energy',
                'carbon_footprint': 35.2,
                'allocation': 0.13
            },
            {
                'name': 'GreenCoin',
                'ticker': 'GRC',
                'asset_type': 'Crypto',
                'shares': 100,
                'purchase_price': 12.30,
                'current_price': 15.80,
                'esg_score': 68.9,
                'sector': 'Green Technology',
                'carbon_footprint': 42.8,
                'allocation': 0.13
            },
            {
                'name': 'Sustainable Pharma',
                'ticker': 'SUPH',
                'asset_type': 'Stock',
                'shares': 20,
                'purchase_price': 65.40,
                'current_price': 72.25,
                'esg_score': 81.3,
                'sector': 'Healthcare',
                'carbon_footprint': 22.1,
                'allocation': 0.12
            },
            {
                'name': 'CleanRetail Inc',
                'ticker': 'CLRT',
                'asset_type': 'Stock',
                'shares': 30,
                'purchase_price': 42.60,
                'current_price': 39.75,
                'esg_score': 77.2,
                'sector': 'Retail',
                'carbon_footprint': 31.5,
                'allocation': 0.09
            },
            {
                'name': 'WaterTech',
                'ticker': 'WTRT',
                'asset_type': 'Stock',
                'shares': 15,
                'purchase_price': 68.20,
                'current_price': 74.50,
                'esg_score': 88.7,
                'sector': 'Utilities',
                'carbon_footprint': 15.3,
                'allocation': 0.09
            },
            {
                'name': 'Circular Economy',
                'ticker': 'CRCE',
                'asset_type': 'Stock',
                'shares': 12,
                'purchase_price': 55.40,
                'current_price': 62.80,
                'esg_score': 86.5,
                'sector': 'Manufacturing',
                'carbon_footprint': 19.8,
                'allocation': 0.06
            },
            {
                'name': 'OceanToken',
                'ticker': 'OCNT',
                'asset_type': 'Crypto',
                'shares': 40,
                'purchase_price': 8.75,
                'current_price': 10.20,
                'esg_score': 82.3,
                'sector': 'Marine Conservation',
                'carbon_footprint': 18.5,
                'allocation': 0.05
            }
        ]
    }

    # Calculate current values and gains/losses
    for asset in portfolio['assets']:
        asset['current_value'] = asset['shares'] * asset['current_price']
        asset['purchase_value'] = asset['shares'] * asset['purchase_price']
        asset['gain_loss'] = asset['current_value'] - asset['purchase_value']
        asset['gain_loss_pct'] = (asset['current_price'] / asset['purchase_price'] - 1) * 100

    return portfolio

# Generate historical portfolio performance
def generate_portfolio_history(days=180):
    np.random.seed(42)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate portfolio value with some trend and volatility
    base_value = 10000
    portfolio_values = []
    market_values = []
    esg_values = []

    # Portfolio performs slightly better than market
    portfolio_trend = 0.00025  # ~9% annual return
    portfolio_volatility = 0.008

    # Market benchmark
    market_trend = 0.0002  # ~7.5% annual return
    market_volatility = 0.01

    # ESG benchmark performs between portfolio and market
    esg_trend = 0.00022  # ~8% annual return
    esg_volatility = 0.009

    portfolio_value = base_value
    market_value = base_value
    esg_value = base_value

    for _ in dates:
        # Random daily returns with trend
        portfolio_return = np.random.normal(portfolio_trend, portfolio_volatility)
        market_return = np.random.normal(market_trend, market_volatility)
        esg_return = np.random.normal(esg_trend, esg_volatility)

        # Update values
        portfolio_value *= (1 + portfolio_return)
        market_value *= (1 + market_return)
        esg_value *= (1 + esg_return)

        portfolio_values.append(portfolio_value)
        market_values.append(market_value)
        esg_values.append(esg_value)

    # Create DataFrame
    history_df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': portfolio_values,
        'Market_Benchmark': market_values,
        'ESG_Benchmark': esg_values
    })

    return history_df

# Generate risk metrics using our utility function
def generate_risk_metrics():
    # Get portfolio assets
    portfolio_assets = portfolio['assets']

    # Generate impact metrics
    impact_metrics = generate_portfolio_impact_metrics(portfolio_assets)

    # Create risk metrics dictionary
    risk_metrics = {
        'portfolio_volatility': 12.5,  # Annualized volatility (%)
        'sharpe_ratio': 1.8,
        'max_drawdown': 15.2,  # Maximum drawdown (%)
        'var_95': 2.3,  # 95% Value at Risk (%)
        'esg_risk_score': 18.5,  # Lower is better
        'carbon_intensity': impact_metrics['carbon']['intensity'],  # tCO2e/$M revenue
        'water_usage': impact_metrics['water']['usage'],  # Cubic meters/$M revenue
        'waste_reduction': impact_metrics['waste']['reduction'],  # % reduction from baseline
        'renewable_energy': impact_metrics['energy']['renewable_percentage'],  # % of energy from renewable sources
        'diversity_score': impact_metrics['social']['diversity_score'],  # Diversity and inclusion score
        'sdg_alignment': impact_metrics['sdg_alignment']['count'],  # Number of SDGs aligned with
        'sdg_details': impact_metrics['sdg_alignment']['aligned_sdgs'],
        'controversy_exposure': 'Low',
        'climate_risk_exposure': 'Medium',
        'transition_risk': 'Low',
        'physical_risk': 'Medium-Low'
    }
    return risk_metrics

# Load data
portfolio = generate_portfolio_data()
portfolio_history = generate_portfolio_history()
risk_metrics = generate_risk_metrics()

# Convert portfolio assets to DataFrame for easier manipulation
assets_df = pd.DataFrame(portfolio['assets'])

# Header
st.title("💼 Portfolio Manager")
st.markdown("*Create, analyze, and optimize your sustainable investment portfolio*")

# Add a stylish introduction
st.markdown("""
<div style="padding: 20px; border-radius: 10px; background-color: rgba(76, 175, 80, 0.1); border-left: 5px solid #4CAF50; margin-bottom: 25px;">
    <h3 style="margin-top: 0;">Welcome to Your Sustainable Portfolio</h3>
    <p>Track your investments' financial performance and sustainability impact in one place. Use the tools below to analyze your portfolio and receive AI-powered recommendations for optimization.</p>
</div>
""", unsafe_allow_html=True)

# Portfolio summary
st.markdown("## Portfolio Summary")
st.markdown("<p style='margin-bottom:20px;'>Key metrics and performance indicators for your sustainable investment portfolio.</p>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_value = sum(asset['current_value'] for asset in portfolio['assets'])
    initial_value = sum(asset['purchase_value'] for asset in portfolio['assets'])
    total_gain_loss = total_value - initial_value
    total_gain_loss_pct = (total_value / initial_value - 1) * 100

    st.metric(
        "Total Value",
        f"${total_value:,.2f}",
        delta=f"{total_gain_loss_pct:.2f}%"
    )

with col2:
    avg_esg = np.average(
        [asset['esg_score'] for asset in portfolio['assets']],
        weights=[asset['allocation'] for asset in portfolio['assets']]
    )
    st.metric("ESG Score", f"{avg_esg:.1f}/100")

with col3:
    stock_allocation = sum(asset['allocation'] for asset in portfolio['assets'] if asset['asset_type'] == 'Stock')
    crypto_allocation = sum(asset['allocation'] for asset in portfolio['assets'] if asset['asset_type'] == 'Crypto')
    st.metric("Stock Allocation", f"{stock_allocation*100:.1f}%")

with col4:
    st.metric("Crypto Allocation", f"{crypto_allocation*100:.1f}%")

# Portfolio performance chart
st.markdown("### Portfolio Performance")
st.markdown("<p style='margin-bottom:15px;'>Track your portfolio's value over time and compare against sustainability benchmarks.</p>", unsafe_allow_html=True)

# Time period selector
time_period = st.selectbox(
    "Time Period",
    options=["1M", "3M", "6M", "YTD", "1Y", "All"],
    index=2
)

# Filter history based on selected time period
if time_period == "1M":
    filtered_history = portfolio_history.iloc[-30:]
elif time_period == "3M":
    filtered_history = portfolio_history.iloc[-90:]
elif time_period == "6M":
    filtered_history = portfolio_history
elif time_period == "YTD":
    start_of_year = datetime(datetime.now().year, 1, 1)
    filtered_history = portfolio_history[portfolio_history['Date'] >= start_of_year]
elif time_period == "1Y":
    filtered_history = portfolio_history  # We only have 6 months of data in our example
else:  # All
    filtered_history = portfolio_history

# Create performance chart
fig = go.Figure()

# Add traces for portfolio, market benchmark, and ESG benchmark
fig.add_trace(go.Scatter(
    x=filtered_history['Date'],
    y=filtered_history['Portfolio_Value'],
    mode='lines',
    name='Your Portfolio',
    line=dict(color='#4CAF50', width=3)
))

fig.add_trace(go.Scatter(
    x=filtered_history['Date'],
    y=filtered_history['Market_Benchmark'],
    mode='lines',
    name='Market Benchmark',
    line=dict(color='#2196F3', width=2, dash='dash')
))

fig.add_trace(go.Scatter(
    x=filtered_history['Date'],
    y=filtered_history['ESG_Benchmark'],
    mode='lines',
    name='ESG Benchmark',
    line=dict(color='#9C27B0', width=2, dash='dot')
))

# Update layout
fig.update_layout(
    title='Portfolio Performance vs Benchmarks',
    xaxis_title='Date',
    yaxis_title='Value ($)',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template=plotly_template
)

st.plotly_chart(fig, use_container_width=True)

# Portfolio allocation
st.markdown("### Portfolio Allocation")

col1, col2 = st.columns(2)

with col1:
    # Asset allocation pie chart
    fig = px.pie(
        assets_df,
        values='allocation',
        names='name',
        title='Asset Allocation',
        hole=0.4,
        template=plotly_template
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Asset type allocation pie chart
    asset_type_allocation = assets_df.groupby('asset_type')['allocation'].sum().reset_index()
    fig = px.pie(
        asset_type_allocation,
        values='allocation',
        names='asset_type',
        title='Asset Type Allocation',
        hole=0.4,
        color_discrete_map={'Stock': '#4CAF50', 'Crypto': '#2196F3'},
        template=plotly_template
    )
    st.plotly_chart(fig, use_container_width=True)

# Portfolio assets table
st.markdown("### Portfolio Assets")
st.dataframe(
    assets_df[[
        'name', 'ticker', 'asset_type', 'shares', 'purchase_price',
        'current_price', 'current_value', 'gain_loss', 'gain_loss_pct', 'esg_score'
    ]].sort_values('current_value', ascending=False),
    use_container_width=True
)

# Risk assessment
st.markdown("## Risk Assessment")

col1, col2 = st.columns(2)

with col1:
    # Financial risk metrics
    st.markdown("### Financial Risk Metrics")

    financial_metrics = {
        'Volatility (Annualized)': f"{risk_metrics['portfolio_volatility']}%",
        'Sharpe Ratio': f"{risk_metrics['sharpe_ratio']}",
        'Maximum Drawdown': f"{risk_metrics['max_drawdown']}%",
        'Value at Risk (95%)': f"{risk_metrics['var_95']}%"
    }

    for metric, value in financial_metrics.items():
        st.markdown(f"""
        <div class="metric-card">
            <h4>{metric}</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Sustainability risk metrics
    st.markdown("### Sustainability Risk Metrics")

    sustainability_metrics = {
        'ESG Risk Score': f"{risk_metrics['esg_risk_score']} (Low Risk)",
        'Carbon Intensity': f"{risk_metrics['carbon_intensity']} tCO2e/$M",
        'SDG Alignment': f"{risk_metrics['sdg_alignment']} SDGs",
        'Controversy Exposure': risk_metrics['controversy_exposure']
    }

    for metric, value in sustainability_metrics.items():
        st.markdown(f"""
        <div class="metric-card">
            <h4>{metric}</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

# Portfolio optimization
st.markdown("## Portfolio Optimization")
st.markdown("""
Our AI-powered portfolio optimization tool helps you balance financial returns with sustainability goals.
Adjust the sliders below to set your preferences and get personalized recommendations.
""")

col1, col2 = st.columns(2)

with col1:
    risk_tolerance = st.slider(
        "Risk Tolerance",
        min_value=1,
        max_value=10,
        value=5,
        help="1 = Very Conservative, 10 = Very Aggressive"
    )

with col2:
    sustainability_focus = st.slider(
        "Sustainability Focus",
        min_value=1,
        max_value=10,
        value=7,
        help="1 = Return Focused, 10 = Impact Focused"
    )

# Optimization recommendations
st.markdown("### Optimization Recommendations")

# Generate recommendations based on sliders
if st.button("Generate Recommendations"):
    st.markdown("#### Recommended Portfolio Adjustments")
    st.markdown("<p style='margin-bottom:15px;'>Based on your risk and sustainability preferences, we recommend the following adjustments:</p>", unsafe_allow_html=True)

    # Get unique sectors in the portfolio
    portfolio_sectors = set(asset['sector'] for asset in portfolio['assets'])

    # Generate sector-specific recommendations based on risk and sustainability preferences
    sector_recommendations = {}
    sector_insights = []

    for sector in portfolio_sectors:
        sector_rec = generate_sector_recommendations(sector, risk_tolerance, sustainability_focus)
        sector_recommendations[sector] = sector_rec

        # Add sector insights based on the recommendation
        if 'rationale' in sector_rec:
            insight = f"**{sector} Sector**: {sector_rec['rationale'].split(':', 1)[1].strip() if ':' in sector_rec['rationale'] else sector_rec['rationale']}"
            sector_insights.append(insight)

    # Generate portfolio recommendations based on risk tolerance and sustainability focus
    if risk_tolerance < 4 and sustainability_focus > 7:
        st.markdown("""
        Based on your low risk tolerance and high sustainability focus, we recommend:

        1. **Reduce** your position in GreenCoin (GRC) by 30% due to its high volatility and moderate ESG score
        2. **Increase** your position in Renewable Power (RNWP) by 20% for its strong ESG score (92.1/100) and stable returns
        3. **Add** a new position in WaterTech (WTRT) for exposure to the water sustainability sector and low carbon footprint
        4. **Consider** adding Ocean Conservation (OCNC) for marine biodiversity exposure and strong ESG alignment
        5. **Reduce** exposure to technology sector by 5% to decrease overall portfolio volatility
        """)

    elif risk_tolerance > 7 and sustainability_focus < 4:
        st.markdown("""
        Based on your high risk tolerance and lower sustainability focus, we recommend:

        1. **Increase** your position in EcoToken (ECO) by 25% for higher potential returns in the renewable energy sector
        2. **Add** a new position in FutureCoin (FUTC) for growth exposure in emerging sustainable technologies
        3. **Add** Carbon Capture Inc (CCAP) for exposure to the growing carbon management market
        4. **Reduce** your position in CleanRetail Inc (CLRT) by 40% due to underperformance and sector headwinds
        5. **Consider** Green Hydrogen (GRHD) for high-growth potential in the clean energy transition
        """)

    else:
        st.markdown("""
        Based on your balanced risk and sustainability preferences, we recommend:

        1. **Maintain** your current allocation to GreenTech Solutions (GRNT) for technology sector exposure
        2. **Increase** your position in Sustainable Finance (SUFN) by 10% to benefit from growing ESG investment trends
        3. **Add** a new position in Circular Economy (CRCE) for diversification into sustainable manufacturing
        4. **Add** a small position (3%) in Biodiversity Fund (BIOD) to gain exposure to nature-based solutions
        5. **Reduce** your position in CleanRetail Inc (CLRT) by 15% due to recent underperformance
        6. **Consider** rebalancing your crypto exposure toward OceanToken (OCNT) for its higher ESG score
        """)

    # Display sector-specific insights
    st.markdown("#### Sector Insights")
    for insight in sector_insights[:5]:  # Show top 5 insights
        st.markdown(f"- {insight}")

    # Show more insights in an expander if there are more than 5 sectors
    if len(sector_insights) > 5:
        with st.expander("View more sector insights"):
            for insight in sector_insights[5:]:
                st.markdown(f"- {insight}")

    # Show projected impact with more detailed metrics
    st.markdown("#### Projected Impact of Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Adjust projected return based on risk tolerance
        projected_return = 6.8 if risk_tolerance < 4 else 10.2 if risk_tolerance > 7 else 8.5
        current_return = 5.6 if risk_tolerance < 4 else 9.0 if risk_tolerance > 7 else 7.3
        return_delta = projected_return - current_return
        st.metric("Projected Annual Return", f"{projected_return}%", delta=f"{return_delta:.1f}%")

    with col2:
        # Adjust projected risk based on risk tolerance
        projected_risk = 8.3 if risk_tolerance < 4 else 15.7 if risk_tolerance > 7 else 11.5
        current_risk = 9.1 if risk_tolerance < 4 else 16.5 if risk_tolerance > 7 else 12.3
        risk_delta = projected_risk - current_risk
        st.metric("Projected Volatility", f"{projected_risk}%", delta=f"{risk_delta:.1f}%", delta_color="inverse")

    with col3:
        # Adjust projected ESG score based on sustainability focus
        projected_esg = 86.5 if sustainability_focus > 7 else 78.2 if sustainability_focus < 4 else 82.3
        current_esg = 82.4 if sustainability_focus > 7 else 76.5 if sustainability_focus < 4 else 80.2
        esg_delta = projected_esg - current_esg
        st.metric("Projected ESG Score", f"{projected_esg:.1f}", delta=f"{esg_delta:.1f}")

    # Additional projected impacts
    col1, col2, col3 = st.columns(3)

    with col1:
        # Carbon footprint reduction
        carbon_reduction = 18.5 if sustainability_focus > 7 else 8.2 if sustainability_focus < 4 else 12.7
        st.metric("Carbon Footprint Reduction", f"{carbon_reduction}%")

    with col2:
        # SDG alignment improvement
        current_sdgs = len(risk_metrics['sdg_details'])
        projected_sdgs = current_sdgs + (1 if sustainability_focus > 7 else 0)
        sdg_delta = projected_sdgs - current_sdgs
        st.metric("SDG Alignment", f"{projected_sdgs} SDGs", delta=sdg_delta if sdg_delta > 0 else None)

    with col3:
        # Sharpe ratio improvement
        current_sharpe = risk_metrics['sharpe_ratio']
        projected_sharpe = current_sharpe + (0.3 if risk_tolerance < 4 else 0.1 if risk_tolerance > 7 else 0.2)
        sharpe_delta = projected_sharpe - current_sharpe
        st.metric("Sharpe Ratio", f"{projected_sharpe:.2f}", delta=f"{sharpe_delta:.2f}")

# Portfolio metrics
st.markdown("## Portfolio Metrics")
st.markdown("<p style='margin-bottom:20px;'>Detailed analysis of your portfolio's risk, return, and sustainability characteristics.</p>", unsafe_allow_html=True)

# Impact reporting
st.markdown("## Impact Reporting")
st.markdown("""
Understand how your investments are contributing to a more sustainable future.
This section shows the environmental and social impact of your portfolio.
""")

# Impact metrics
st.markdown("## Impact Metrics")
st.markdown("<p style='margin-bottom:25px;'>Measure your portfolio's environmental and social impact with these key sustainability indicators.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Carbon Impact")

    carbon_avoided = 12.5  # Tons of CO2 equivalent
    carbon_intensity = risk_metrics['carbon_intensity']
    st.markdown(f"""
    <div class="metric-card">
        <h4>Carbon Emissions Avoided</h4>
        <p style="font-size: 1.8rem; font-weight: bold;">{carbon_avoided} tCO2e</p>
        <p>Equivalent to taking 2.7 cars off the road for a year</p>
        <hr style="margin: 10px 0;">
        <h4>Carbon Intensity</h4>
        <p style="font-size: 1.5rem; font-weight: bold;">{carbon_intensity} tCO2e/$M</p>
        <p>18% below industry average</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Resource Efficiency")

    water_usage = risk_metrics['water_usage']
    waste_reduction = risk_metrics['waste_reduction']
    renewable_energy = risk_metrics['renewable_energy']
    st.markdown(f"""
    <div class="metric-card">
        <h4>Water Usage</h4>
        <p style="font-size: 1.5rem; font-weight: bold;">{water_usage} m³/$M</p>
        <p>22% below industry average</p>
        <hr style="margin: 10px 0;">
        <h4>Waste Reduction</h4>
        <p style="font-size: 1.5rem; font-weight: bold;">{waste_reduction}%</p>
        <p>Reduction from baseline year</p>
        <hr style="margin: 10px 0;">
        <h4>Renewable Energy</h4>
        <p style="font-size: 1.5rem; font-weight: bold;">{renewable_energy}%</p>
        <p>Of total energy consumption</p>
    </div>
    """, unsafe_allow_html=True)

# Social impact and SDG alignment
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Social Impact")

    jobs_supported = 85
    diversity_score = risk_metrics['diversity_score']
    st.markdown(f"""
    <div class="metric-card">
        <h4>Jobs Supported</h4>
        <p style="font-size: 1.5rem; font-weight: bold;">{jobs_supported}</p>
        <p>Through investments in companies with fair labor practices</p>
        <hr style="margin: 10px 0;">
        <h4>Diversity & Inclusion</h4>
        <p style="font-size: 1.5rem; font-weight: bold;">{diversity_score}/100</p>
        <p>Portfolio companies' average D&I score</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### SDG Alignment")

    aligned_sdgs = list(risk_metrics['sdg_details'].keys())
    sdg_badges = " ".join([f"<span style='background-color:#f0f2f6;padding:5px 10px;border-radius:10px;margin-right:5px;'>SDG {sdg}</span>" for sdg in aligned_sdgs])

    # Create SDG contribution details
    sdg_details = ""
    for sdg, details in risk_metrics['sdg_details'].items():
        sdg_details += f"<p><strong>SDG {sdg}:</strong> {details['name']} - {details['contribution']} contribution</p>"

    st.markdown(f"""
    <div class="metric-card">
        <h4>SDG Contribution</h4>
        <p>{sdg_badges}</p>
        <p>Your portfolio contributes to {len(aligned_sdgs)} of the UN Sustainable Development Goals</p>
        <hr style="margin: 10px 0;">
        <h4>Contribution Details</h4>
        {sdg_details}
    </div>
    """, unsafe_allow_html=True)

# Impact visualization
st.markdown("## Impact Visualization")
st.markdown("<p style='margin-bottom:25px;'>Visualize how your investments are contributing to sustainability goals and sector-specific impacts.</p>", unsafe_allow_html=True)

# Create a sector-based impact visualization
sector_impact = {}
for asset in portfolio['assets']:
    sector = asset['sector']
    if sector not in sector_impact:
        sector_impact[sector] = {
            'allocation': 0,
            'esg_score': 0,
            'carbon_footprint': 0
        }
    sector_impact[sector]['allocation'] += asset['allocation']
    sector_impact[sector]['esg_score'] += asset['esg_score'] * asset['allocation']
    sector_impact[sector]['carbon_footprint'] += asset['carbon_footprint'] * asset['allocation']

# Normalize ESG scores and carbon footprints by allocation
for sector in sector_impact:
    if sector_impact[sector]['allocation'] > 0:
        sector_impact[sector]['esg_score'] /= sector_impact[sector]['allocation']
        sector_impact[sector]['carbon_footprint'] /= sector_impact[sector]['allocation']

# Convert to DataFrame for visualization
sector_impact_df = pd.DataFrame([
    {
        'Sector': sector,
        'Allocation': data['allocation'] * 100,
        'ESG_Score': data['esg_score'],
        'Carbon_Footprint': data['carbon_footprint']
    } for sector, data in sector_impact.items()
])

# Sort by allocation
sector_impact_df = sector_impact_df.sort_values('Allocation', ascending=False)

# Create two columns for visualizations
col1, col2 = st.columns(2)

with col1:
    # Sector allocation
    fig = px.pie(
        sector_impact_df,
        values='Allocation',
        names='Sector',
        title='Portfolio Allocation by Sector',
        hole=0.4,
        template=plotly_template
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # ESG score by sector
    fig = px.bar(
        sector_impact_df,
        x='Sector',
        y='ESG_Score',
        title='ESG Score by Sector',
        color='Carbon_Footprint',
        color_continuous_scale='RdYlGn_r',  # Reversed so red is high carbon (bad)
        labels={'ESG_Score': 'ESG Score', 'Carbon_Footprint': 'Carbon Footprint'},
        template=plotly_template
    )
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)

# Climate risk analysis
st.markdown("## Climate Risk Analysis")
st.markdown("<p style='margin-bottom:25px;'>Understand how your portfolio is positioned for climate-related risks and opportunities.</p>", unsafe_allow_html=True)
st.markdown("""
Understand how your portfolio is positioned for climate-related risks and opportunities.
This analysis evaluates physical risks, transition risks, and climate-related opportunities.
""")

# Climate risk metrics
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Physical Risk Exposure")

    # Create a gauge chart for physical risk
    physical_risk_value = 35  # On a scale of 0-100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=physical_risk_value,
        title={'text': "Physical Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "green"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))

    fig.update_layout(
        height=250,
        template=plotly_template
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Physical Risks:**
    - Water stress exposure: Medium (32/100)
    - Extreme weather vulnerability: Medium-Low (28/100)
    - Sea level rise exposure: Low (15/100)
    - Biodiversity loss impact: Medium (45/100)
    """)

with col2:
    st.markdown("### Transition Risk Exposure")

    # Create a gauge chart for transition risk
    transition_risk_value = 42  # On a scale of 0-100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=transition_risk_value,
        title={'text': "Transition Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "green"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))

    fig.update_layout(
        height=250,
        template=plotly_template
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Transition Risks:**
    - Policy & legal risk: Medium (38/100)
    - Technology risk: Medium (42/100)
    - Market risk: Medium (45/100)
    - Reputation risk: Medium-Low (30/100)
    """)

# Climate opportunities
st.markdown("### Climate-Related Opportunities")

# Create data for climate opportunities
climate_opportunities = pd.DataFrame({
    'Opportunity': [
        'Resource Efficiency', 'Energy Source', 'Products & Services',
        'Markets', 'Resilience'
    ],
    'Exposure': [65, 78, 58, 45, 52],
    'Potential_Impact': [3.2, 4.5, 2.8, 2.1, 2.4]
})

# Create bubble chart for climate opportunities
fig = px.scatter(
    climate_opportunities,
    x='Exposure',
    y='Potential_Impact',
    size='Potential_Impact',
    color='Opportunity',
    text='Opportunity',
    title='Climate-Related Opportunities',
    labels={
        'Exposure': 'Portfolio Exposure (%)',
        'Potential_Impact': 'Potential Financial Impact (% of portfolio value)'
    },
    template=plotly_template
)

fig.update_traces(textposition='top center')
fig.update_layout(height=400)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Top Climate Opportunities:**
1. **Energy Source (78% exposure)**: Portfolio companies transitioning to lower-emission energy sources
2. **Resource Efficiency (65% exposure)**: Companies improving resource efficiency in operations
3. **Products & Services (58% exposure)**: Development of low-emission products and services
""")

# Add a section for climate risk analysis
st.markdown("### Climate Risk Analysis")

# Create a radar chart for climate risk categories
categories = ['Physical Risk', 'Transition Risk', 'Policy & Legal',
              'Technology', 'Market', 'Reputation']
values = [35, 42, 38, 42, 45, 30]

# Create radar chart
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Portfolio Climate Risk',
    line_color='rgba(31, 119, 180, 0.8)',
    fillcolor='rgba(31, 119, 180, 0.3)'
))

# Add benchmark comparison
benchmark_values = [45, 48, 52, 40, 50, 35]
fig.add_trace(go.Scatterpolar(
    r=benchmark_values,
    theta=categories,
    fill='toself',
    name='Market Benchmark',
    line_color='rgba(214, 39, 40, 0.8)',
    fillcolor='rgba(214, 39, 40, 0.3)'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    showlegend=True,
    title="Climate Risk Profile vs Benchmark",
    template=plotly_template
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Climate Risk Summary:**

Your portfolio demonstrates lower climate risk exposure compared to the market benchmark, particularly in policy & legal and market risk categories. This positioning may provide resilience against climate-related regulatory changes and market shifts.

**Key Risk Factors:**
- **Physical Risk**: Your portfolio has moderate exposure to physical climate risks, primarily through holdings in the utilities and agriculture sectors.
- **Transition Risk**: Your portfolio's transition risk is below market average, reflecting investments in companies well-positioned for the low-carbon transition.
- **Market Risk**: Your portfolio shows reduced market risk due to diversification across climate solutions providers.
""")
values = [85, 92, 65, 78, 60, 82]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Your Portfolio',
    line_color='#4CAF50'
))

fig.add_trace(go.Scatterpolar(
    r=[70, 65, 60, 55, 65, 60],
    theta=categories,
    fill='toself',
    name='Market Average',
    line_color='#2196F3'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    showlegend=True,
    template=plotly_template
)

st.plotly_chart(fig, use_container_width=True)

# Download reports
st.markdown("### Download Reports")
col1, col2, col3 = st.columns(3)

# Generate PDF report content
def generate_portfolio_report():
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Create content
    content = []

    # Title
    title_style = styles["Title"]
    content.append(Paragraph("Sustainable Investment Portfolio Report", title_style))
    content.append(Spacer(1, 12))

    # Calculate portfolio values from the current portfolio data
    total_value = sum(asset['current_value'] for asset in portfolio['assets'])
    initial_value = sum(asset['purchase_value'] for asset in portfolio['assets'])
    annual_return = ((total_value / initial_value) ** (1/1.5) - 1) * 100  # Assuming 1.5 years since purchase
    avg_esg = sum(asset['esg_score'] * asset['allocation'] for asset in portfolio['assets'])

    # Calculate asset type allocation
    allocation = {}
    for asset in portfolio['assets']:
        asset_type = asset['asset_type']
        if asset_type not in allocation:
            allocation[asset_type] = 0
        allocation[asset_type] += asset['allocation'] * 100

    # Portfolio summary
    content.append(Paragraph("Portfolio Summary", styles["Heading1"]))
    content.append(Paragraph(f"Total Value: ${total_value:,.2f}", styles["Normal"]))
    content.append(Paragraph(f"Annual Return: {annual_return:.2f}%", styles["Normal"]))
    content.append(Paragraph(f"ESG Score: {avg_esg:.1f}/100", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Asset allocation table
    content.append(Paragraph("Asset Allocation", styles["Heading2"]))
    allocation_data = [["Asset Type", "Allocation %"]]
    for asset_type, alloc in allocation.items():
        allocation_data.append([asset_type, f"{alloc:.1f}%"])

    allocation_table = Table(allocation_data, colWidths=[300, 100])
    allocation_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(allocation_table)
    content.append(Spacer(1, 12))

    # Holdings table
    content.append(Paragraph("Holdings", styles["Heading2"]))
    holdings_data = [["Asset", "Ticker", "Shares", "Value", "ESG Score"]]
    for asset in portfolio['assets']:
        holdings_data.append([
            asset['name'],
            asset['ticker'],
            str(asset['shares']),
            f"${asset['current_value']:,.2f}",
            f"{asset['esg_score']:.1f}"
        ])

    holdings_table = Table(holdings_data, colWidths=[150, 60, 60, 100, 80])
    holdings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(holdings_table)

    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

# Generate impact report
def generate_impact_report():
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Create content
    content = []

    # Title
    title_style = styles["Title"]
    content.append(Paragraph("Sustainable Investment Impact Report", title_style))
    content.append(Spacer(1, 12))

    # Use risk metrics from the global variable
    # Impact summary
    content.append(Paragraph("Impact Summary", styles["Heading1"]))
    content.append(Paragraph(f"Carbon Footprint: {risk_metrics['carbon_intensity']} tCO2e/$M", styles["Normal"]))
    content.append(Paragraph(f"SDG Alignment: {risk_metrics['sdg_alignment']} SDGs", styles["Normal"]))
    content.append(Paragraph(f"Controversy Exposure: {risk_metrics['controversy_exposure']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # SDG Contributions
    content.append(Paragraph("SDG Contributions", styles["Heading2"]))
    sdg_data = [["SDG", "Contribution Level"]]
    for sdg, level in zip([1, 7, 12, 13], ["High", "Medium", "High", "Medium"]):
        sdg_data.append([f"SDG {sdg}", level])

    sdg_table = Table(sdg_data, colWidths=[200, 200])
    sdg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(sdg_table)
    content.append(Spacer(1, 12))

    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

with col1:
    try:
        pdf_buffer = generate_portfolio_report()
        st.download_button(
            label="Download Portfolio Report",
            data=pdf_buffer,
            file_name="portfolio_report.pdf",
            mime="application/pdf",
            key="portfolio_report"
        )
    except Exception as e:
        st.error(f"Error generating portfolio report: {str(e)}")
        st.download_button(
            label="Download Portfolio Report",
            data="This is a placeholder for the portfolio report PDF",
            file_name="portfolio_report.pdf",
            mime="application/pdf"
        )

with col2:
    try:
        impact_buffer = generate_impact_report()
        st.download_button(
            label="Download Impact Report",
            data=impact_buffer,
            file_name="impact_report.pdf",
            mime="application/pdf",
            key="impact_report"
        )
    except Exception as e:
        st.error(f"Error generating impact report: {str(e)}")
        st.download_button(
            label="Download Impact Report",
            data="This is a placeholder for the impact report PDF",
            file_name="impact_report.pdf",
            mime="application/pdf"
        )

with col3:
    st.download_button(
        label="Export Portfolio Data",
        data=json.dumps(portfolio, indent=2),
        file_name="portfolio_data.json",
        mime="application/json",
        key="portfolio_data"
    )
