import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from datetime import datetime, timedelta  # Not used
import random
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sustainability data
from utils.sustainability_data import SDG_DATA, SUSTAINABILITY_TRENDS, generate_sector_recommendations

# Set page config
st.set_page_config(
    page_title="AI Recommendations - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Get theme from session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Apply theme-specific styles
theme_bg_color = "#0e1117" if st.session_state.theme == "dark" else "#ffffff"
theme_text_color = "#ffffff" if st.session_state.theme == "dark" else "#0e1117"

# Add authentication check
from utils.auth_redirect import check_authentication
check_authentication()
theme_secondary_bg = "#1e2530" if st.session_state.theme == "dark" else "#f0f2f6"
theme_card_bg = "#262730" if st.session_state.theme == "dark" else "white"

# Set Plotly theme based on app theme
plotly_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

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
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        color: {theme_text_color};
    }}
    .insight-card {{
        background-color: {theme_card_bg};
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin: 0.5rem 0;
        border-left: 3px solid #2196F3;
        color: {theme_text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Generate dummy stock data
def generate_stock_data() -> pd.DataFrame:
    np.random.seed(42)

    companies = {
        'Name': [
            'GreenTech Solutions', 'EcoEnergy Corp', 'Sustainable Pharma',
            'CleanRetail Inc', 'Future Mobility', 'Smart Agriculture',
            'Renewable Power', 'Circular Economy', 'WaterTech',
            'Sustainable Finance', 'Green Construction', 'EcoTransport',
            'Ocean Conservation', 'Biodiversity Fund', 'Clean Air Technologies',
            'Sustainable Materials', 'Ethical AI Systems', 'Carbon Capture Inc',
            'Sustainable Forestry', 'Green Hydrogen', 'Waste Management Solutions'
        ],
        'Ticker': [
            'GRNT', 'ECOE', 'SUPH', 'CLRT', 'FUTM', 'SMAG',
            'RNWP', 'CRCE', 'WTRT', 'SUFN', 'GRCN', 'ECTR',
            'OCNC', 'BIOD', 'CAIR', 'SMAT', 'EAIS', 'CCAP',
            'SFST', 'GRHD', 'WAMS'
        ],
        'Sector': [
            'Technology', 'Energy', 'Healthcare',
            'Retail', 'Transportation', 'Agriculture',
            'Energy', 'Manufacturing', 'Utilities',
            'Finance', 'Construction', 'Transportation',
            'Marine Conservation', 'Biodiversity', 'Clean Air',
            'Materials', 'Technology', 'Carbon Management',
            'Forestry', 'Clean Energy', 'Waste Management'
        ],
        'Asset_Type': ['Stock'] * 21
    }

    df = pd.DataFrame(companies)
    df['Current_Price'] = np.random.uniform(50, 500, len(df))
    df['Price_Change_24h'] = np.random.uniform(-5, 8, len(df))
    df['Market_Cap_B'] = np.random.uniform(1, 100, len(df))
    df['ROI_1Y'] = np.random.uniform(4, 25, len(df))
    df['Volatility'] = np.random.uniform(0.1, 0.5, len(df))

    # ESG Scores
    df['Environmental_Score'] = np.random.uniform(50, 95, len(df))
    df['Social_Score'] = np.random.uniform(40, 90, len(df))
    df['Governance_Score'] = np.random.uniform(60, 95, len(df))
    df['ESG_Score'] = (df['Environmental_Score'] + df['Social_Score'] + df['Governance_Score']) / 3

    # SDG Alignment
    df['SDG_Alignment'] = [
        random.sample(range(1, 18), k=random.randint(2, 5)) for _ in range(len(df))
    ]

    # Carbon Footprint
    df['Carbon_Footprint'] = np.random.uniform(10, 100, len(df))
    df['Carbon_Reduction_Target'] = np.random.uniform(10, 50, len(df))

    # AI Risk Score (lower is better)
    df['AI_Risk_Score'] = 100 - (df['ESG_Score'] * 0.4 + (100 - df['Volatility'] * 100) * 0.4 + df['ROI_1Y'] * 0.2)

    # AI Recommendation
    df['AI_Recommendation'] = df['AI_Risk_Score'].apply(
        lambda x: '🟢 Strong Buy' if x < 30 else '🟡 Hold' if x < 50 else '🔴 Caution'
    )

    return df

# Generate dummy crypto data
def generate_crypto_data() -> pd.DataFrame:
    np.random.seed(43)

    cryptos = {
        'Name': [
            'GreenCoin', 'EcoToken', 'SustainChain',
            'CleanCrypto', 'FutureCoin', 'AgriToken',
            'RenewCoin', 'CircularToken', 'OceanToken',
            'BiodiversityCoin', 'CarbonOffset', 'ForestCoin',
            'EthicalAI Token', 'WaterCredit', 'SolarCoin'
        ],
        'Ticker': [
            'GRC', 'ECO', 'SUST', 'CLNC', 'FUTC', 'AGRT',
            'RNWC', 'CIRC', 'OCNT', 'BIOC', 'CRBO', 'FRST',
            'EAIT', 'WATR', 'SOLC'
        ],
        'Sector': [
            'Green Technology', 'Renewable Energy', 'Sustainable Supply Chain',
            'Carbon Credits', 'Future Tech', 'Sustainable Agriculture',
            'Renewable Energy', 'Circular Economy', 'Marine Conservation',
            'Biodiversity', 'Carbon Management', 'Forestry',
            'Ethical Technology', 'Water Management', 'Solar Energy'
        ],
        'Asset_Type': ['Crypto'] * 15
    }

    df = pd.DataFrame(cryptos)
    df['Current_Price'] = np.random.uniform(0.1, 2000, len(df))
    df['Price_Change_24h'] = np.random.uniform(-10, 15, len(df))
    df['Market_Cap_B'] = np.random.uniform(0.1, 50, len(df))
    df['ROI_1Y'] = np.random.uniform(10, 100, len(df))
    df['Volatility'] = np.random.uniform(0.3, 0.8, len(df))

    # ESG Scores - Cryptos typically have different ESG profiles
    df['Environmental_Score'] = np.random.uniform(30, 90, len(df))  # Some cryptos have high energy usage
    df['Social_Score'] = np.random.uniform(50, 85, len(df))
    df['Governance_Score'] = np.random.uniform(40, 80, len(df))  # Decentralization affects governance
    df['ESG_Score'] = (df['Environmental_Score'] + df['Social_Score'] + df['Governance_Score']) / 3

    # SDG Alignment
    df['SDG_Alignment'] = [
        random.sample(range(1, 18), k=random.randint(1, 4)) for _ in range(len(df))
    ]

    # Energy Consumption (specific to crypto)
    df['Energy_Consumption'] = np.random.uniform(10, 1000, len(df))
    df['Renewable_Energy_Pct'] = np.random.uniform(10, 90, len(df))

    # Carbon Footprint
    df['Carbon_Footprint'] = df['Energy_Consumption'] * (1 - df['Renewable_Energy_Pct']/100) / 10
    df['Carbon_Reduction_Target'] = np.random.uniform(5, 40, len(df))

    # AI Risk Score (lower is better)
    df['AI_Risk_Score'] = 100 - (df['ESG_Score'] * 0.3 + (100 - df['Volatility'] * 100) * 0.5 + df['ROI_1Y'] * 0.2 / 4)

    # AI Recommendation
    df['AI_Recommendation'] = df['AI_Risk_Score'].apply(
        lambda x: '🟢 Strong Buy' if x < 30 else '🟡 Hold' if x < 50 else '🔴 Caution'
    )

    return df

# Use sustainability trends from imported data
def generate_market_trends():
    return SUSTAINABILITY_TRENDS

# Load data
stocks_df = generate_stock_data()
crypto_df = generate_crypto_data()
all_assets_df = pd.concat([stocks_df, crypto_df]).reset_index(drop=True)
market_trends = generate_market_trends()

# Header
st.title("🤖 AI Recommendations")
st.markdown("*Personalized investment insights powered by AI analysis*")

# User preferences
st.markdown("## Your Investment Preferences")
st.markdown("Set your preferences to receive tailored recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    investment_horizon = st.selectbox(
        "Investment Horizon",
        options=["Short-term (< 1 year)", "Medium-term (1-3 years)", "Long-term (> 3 years)"],
        index=1
    )

with col2:
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Very Low", "Low", "Moderate", "High", "Very High"],
        value="Moderate"
    )

with col3:
    sustainability_focus = st.select_slider(
        "Sustainability Focus",
        options=["Financial Returns First", "Balanced Approach", "Impact First"],
        value="Balanced Approach"
    )

# Investment amount
investment_amount = st.number_input(
    "Investment Amount ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

# Asset preferences
col1, col2 = st.columns(2)

with col1:
    preferred_asset_types = st.multiselect(
        "Preferred Asset Types",
        options=["Stocks", "Cryptocurrencies"],
        default=["Stocks", "Cryptocurrencies"]
    )

with col2:
    preferred_sectors = st.multiselect(
        "Preferred Sectors",
        options=sorted(list(set(all_assets_df['Sector'].unique()))),
        default=["Renewable Energy", "Green Technology"]
    )

# Generate recommendations button
if st.button("Generate Personalized Recommendations"):
    st.markdown("## Your Personalized Recommendations")

    # Filter assets based on user preferences
    filtered_assets = all_assets_df.copy()

    # Filter by asset type
    if "Stocks" in preferred_asset_types and "Cryptocurrencies" not in preferred_asset_types:
        filtered_assets = filtered_assets[filtered_assets['Asset_Type'] == 'Stock']
    elif "Cryptocurrencies" in preferred_asset_types and "Stocks" not in preferred_asset_types:
        filtered_assets = filtered_assets[filtered_assets['Asset_Type'] == 'Crypto']

    # Filter by sector if preferences are set
    if preferred_sectors:
        filtered_assets = filtered_assets[filtered_assets['Sector'].isin(preferred_sectors)]

    # Adjust scoring based on user preferences
    if risk_tolerance == "Very Low":
        risk_weight = 0.6
        return_weight = 0.2
    elif risk_tolerance == "Low":
        risk_weight = 0.5
        return_weight = 0.3
    elif risk_tolerance == "Moderate":
        risk_weight = 0.4
        return_weight = 0.4
    elif risk_tolerance == "High":
        risk_weight = 0.3
        return_weight = 0.5
    else:  # Very High
        risk_weight = 0.2
        return_weight = 0.6

    if sustainability_focus == "Financial Returns First":
        esg_weight = 0.2
    elif sustainability_focus == "Balanced Approach":
        esg_weight = 0.4
    else:  # Impact First
        esg_weight = 0.6

    # Calculate custom score
    filtered_assets['Custom_Score'] = (
        filtered_assets['ESG_Score'] * esg_weight +
        (100 - filtered_assets['Volatility'] * 100) * risk_weight +
        filtered_assets['ROI_1Y'] * return_weight
    )

    # Sort by custom score
    filtered_assets = filtered_assets.sort_values('Custom_Score', ascending=False)

    # Display top recommendations
    top_recommendations = filtered_assets.head(5)

    for i, (_, asset) in enumerate(top_recommendations.iterrows()):
        recommendation_type = "Strong Buy" if asset['AI_Recommendation'] == '🟢 Strong Buy' else "Consider" if asset['AI_Recommendation'] == '🟡 Hold' else "Research Further"

        # Generate more detailed sector-specific recommendation using our utility function
        sector_rec = generate_sector_recommendations(asset['Sector'], risk_tolerance, sustainability_focus)

        # Create sector-specific insight based on the recommendation
        sector_specific_insight = f"{asset['Name']} {sector_rec['rationale'].split(',', 1)[1].strip() if ',' in sector_rec['rationale'] else sector_rec['rationale']}"

        # Add relevant ESG metrics for the sector
        key_metrics = []
        if sector_rec['key_esg_metrics']['environmental']:
            key_metrics.append(f"key environmental focus: {sector_rec['key_esg_metrics']['environmental'][0]}")
        if sector_rec['key_esg_metrics']['social']:
            key_metrics.append(f"social consideration: {sector_rec['key_esg_metrics']['social'][0]}")

        # Add climate risk information if available
        climate_risk = f"Climate risk exposure: {sector_rec['climate_risk_exposure']}"

        # Combine insights
        sector_specific_insight = f"{sector_specific_insight} With {', '.join(key_metrics)}. {climate_risk}"

        # Generate risk assessment based on volatility and ESG score
        if asset['Volatility'] < 0.3 and asset['ESG_Score'] > 80:
            risk_assessment = "Low risk profile with strong sustainability credentials"
        elif asset['Volatility'] < 0.3 and asset['ESG_Score'] > 60:
            risk_assessment = "Low market volatility with moderate sustainability performance"
        elif asset['Volatility'] < 0.5 and asset['ESG_Score'] > 80:
            risk_assessment = "Moderate market volatility balanced by strong sustainability practices"
        elif asset['Volatility'] < 0.5 and asset['ESG_Score'] > 60:
            risk_assessment = "Moderate risk profile across both market and sustainability dimensions"
        elif asset['Volatility'] >= 0.5 and asset['ESG_Score'] > 80:
            risk_assessment = "Higher market volatility offset by excellent sustainability performance"
        else:
            risk_assessment = "Higher risk profile requiring careful position sizing"

        # Generate investment horizon recommendation
        if investment_horizon == "Short-term (< 1 year)":
            horizon_fit = "short-term" if asset['Volatility'] < 0.4 and asset['Price_Change_24h'] > 0 else "medium to long-term"
        elif investment_horizon == "Medium-term (1-3 years)":
            horizon_fit = "medium-term" if asset['ESG_Score'] > 70 or asset['ROI_1Y'] > 10 else "long-term"
        else:  # Long-term
            horizon_fit = "long-term" if asset['ESG_Score'] > 75 else "medium-term"

        st.markdown(f"""
        <div class="recommendation-card">
            <h3>#{i+1}: {recommendation_type} - {asset['Name']} ({asset['Ticker']})</h3>
            <p><strong>Asset Type:</strong> {asset['Asset_Type']} | <strong>Sector:</strong> {asset['Sector']}</p>
            <p><strong>Current Price:</strong> ${asset['Current_Price']:.2f} | <strong>24h Change:</strong> {asset['Price_Change_24h']:.2f}%</p>
            <p><strong>ESG Score:</strong> {asset['ESG_Score']:.1f}/100 | <strong>1Y ROI:</strong> {asset['ROI_1Y']:.2f}% | <strong>Volatility:</strong> {asset['Volatility']:.2f}</p>
            <p><strong>Why we recommend this:</strong> {asset['Name']} aligns well with your {sustainability_focus.lower()} and {risk_tolerance.lower()} risk tolerance preferences.
            It has {('strong ESG credentials' if asset['ESG_Score'] > 75 else 'moderate ESG performance')} and {('excellent' if asset['ROI_1Y'] > 15 else 'solid' if asset['ROI_1Y'] > 8 else 'stable')} return potential.</p>
            <p><strong>Sector Insight:</strong> {sector_specific_insight}</p>
            <p><strong>Risk Assessment:</strong> {risk_assessment}</p>
            <p><strong>Investment Horizon:</strong> Best suited for {horizon_fit} investors</p>
            <p><strong>Suggested Allocation:</strong> {25 if i == 0 else 20 if i == 1 else 15 if i == 2 else 10}% of your portfolio (${investment_amount * (0.25 if i == 0 else 0.2 if i == 1 else 0.15 if i == 2 else 0.1):,.2f})</p>
        </div>
        """, unsafe_allow_html=True)

    # Portfolio allocation recommendation
    st.markdown("### Recommended Portfolio Allocation")

    # Calculate allocations based on user preferences
    if risk_tolerance in ["Very Low", "Low"]:
        stock_allocation = 0.8
        crypto_allocation = 0.2
    elif risk_tolerance == "Moderate":
        stock_allocation = 0.7
        crypto_allocation = 0.3
    else:  # High or Very High
        stock_allocation = 0.6
        crypto_allocation = 0.4

    # Adjust based on sustainability focus
    if sustainability_focus == "Impact First":
        high_esg_allocation = 0.7
        moderate_esg_allocation = 0.3
    elif sustainability_focus == "Balanced Approach":
        high_esg_allocation = 0.5
        moderate_esg_allocation = 0.5
    else:  # Financial Returns First
        high_esg_allocation = 0.3
        moderate_esg_allocation = 0.7

    # Create allocation data
    allocation_data = {
        'Category': [
            'High ESG Stocks', 'Moderate ESG Stocks',
            'High ESG Crypto', 'Moderate ESG Crypto'
        ],
        'Allocation': [
            stock_allocation * high_esg_allocation,
            stock_allocation * moderate_esg_allocation,
            crypto_allocation * high_esg_allocation,
            crypto_allocation * moderate_esg_allocation
        ]
    }
    allocation_df = pd.DataFrame(allocation_data)

    # Create pie chart
    fig = px.pie(
        allocation_df,
        values='Allocation',
        names='Category',
        title='Recommended Portfolio Allocation',
        color_discrete_map={
            'High ESG Stocks': '#4CAF50',
            'Moderate ESG Stocks': '#8BC34A',
            'High ESG Crypto': '#2196F3',
            'Moderate ESG Crypto': '#64B5F6'
        },
        template=plotly_template
    )
    st.plotly_chart(fig, use_container_width=True)

    # Expected performance
    st.markdown("### Expected Performance")

    # Generate expected performance based on user preferences
    if risk_tolerance in ["Very Low", "Low"]:
        expected_return = np.random.uniform(5, 8)
        expected_volatility = np.random.uniform(5, 10)
    elif risk_tolerance == "Moderate":
        expected_return = np.random.uniform(8, 12)
        expected_volatility = np.random.uniform(10, 15)
    else:  # High or Very High
        expected_return = np.random.uniform(12, 20)
        expected_volatility = np.random.uniform(15, 25)

    # Adjust based on sustainability focus
    if sustainability_focus == "Impact First":
        expected_return *= 0.9  # Slightly lower returns for impact focus
        expected_esg = np.random.uniform(80, 95)
    elif sustainability_focus == "Balanced Approach":
        expected_esg = np.random.uniform(70, 85)
    else:  # Financial Returns First
        expected_return *= 1.1  # Slightly higher returns for financial focus
        expected_esg = np.random.uniform(60, 75)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Expected Annual Return", f"{expected_return:.1f}%")

    with col2:
        st.metric("Expected Volatility", f"{expected_volatility:.1f}%")

    with col3:
        st.metric("Expected ESG Score", f"{expected_esg:.1f}/100")

    # Market insights
    st.markdown("## Market Insights")
    st.markdown("Our AI analyzes market trends and ESG developments to provide you with actionable insights")

    # Filter trends relevant to selected sectors if sectors are selected
    relevant_trends = []
    if preferred_sectors:
        for trend in market_trends:
            if 'All' in trend['related_sectors'] or any(sector in trend['related_sectors'] for sector in preferred_sectors):
                relevant_trends.append(trend)
    else:
        relevant_trends = market_trends

    for trend in relevant_trends:
        # Determine relevance level based on confidence and sector match
        relevance = "High" if trend['confidence'] > 85 else "Medium" if trend['confidence'] > 75 else "Moderate"
        if preferred_sectors and any(sector in trend['related_sectors'] for sector in preferred_sectors):
            relevance = "Very High" if relevance == "High" else "High" if relevance == "Medium" else "Medium"

        # Generate actionable recommendation based on the trend
        if "Positive" in trend['impact']:
            action = "Consider increasing allocation to" if relevance in ["Very High", "High"] else "Monitor"
        else:
            action = "Consider reducing exposure to" if relevance in ["Very High", "High"] else "Review holdings in"

        sectors_affected = ", ".join(trend['related_sectors']) if trend['related_sectors'] != ["All"] else "all sectors"

        st.markdown(f"""
        <div class="insight-card">
            <h4>{trend['title']} <span style="float:right;font-size:0.8rem;color:#666;">Confidence: {trend['confidence']}% | Relevance: {relevance}</span></h4>
            <p>{trend['description']}</p>
            <p><strong>Impact:</strong> {trend['impact']}</p>
            <p><strong>Recommendation:</strong> {action} {sectors_affected}</p>
        </div>
        """, unsafe_allow_html=True)

    # ESG focus areas
    st.markdown("## ESG Focus Areas")
    st.markdown("Based on your preferences, these are the key ESG areas to consider in your investment strategy")

    # Generate ESG focus areas based on user preferences
    if sustainability_focus == "Impact First":
        focus_areas = [
            {"area": "Climate Action", "importance": 95, "description": "Companies with strong climate commitments and carbon reduction targets"},
            {"area": "Renewable Energy", "importance": 90, "description": "Investments in solar, wind, and other renewable energy sources"},
            {"area": "Sustainable Supply Chains", "importance": 85, "description": "Companies implementing sustainable and ethical supply chain practices"},
            {"area": "Diversity & Inclusion", "importance": 80, "description": "Organizations with strong diversity policies and inclusive practices"}
        ]
    elif sustainability_focus == "Balanced Approach":
        focus_areas = [
            {"area": "Energy Efficiency", "importance": 85, "description": "Companies improving energy efficiency in operations and products"},
            {"area": "Sustainable Innovation", "importance": 80, "description": "Organizations developing sustainable technologies and solutions"},
            {"area": "Corporate Governance", "importance": 75, "description": "Companies with transparent governance and ethical business practices"},
            {"area": "Water Management", "importance": 70, "description": "Businesses implementing responsible water usage and conservation"}
        ]
    else:  # Financial Returns First
        focus_areas = [
            {"area": "ESG Risk Management", "importance": 75, "description": "Companies effectively managing ESG risks to protect shareholder value"},
            {"area": "Resource Efficiency", "importance": 70, "description": "Organizations optimizing resource usage for cost savings and sustainability"},
            {"area": "Regulatory Compliance", "importance": 65, "description": "Businesses well-positioned for evolving ESG regulations"},
            {"area": "Reputation Management", "importance": 60, "description": "Companies maintaining strong brand reputation through ESG practices"}
        ]

    # Create horizontal bar chart
    focus_df = pd.DataFrame(focus_areas)

    fig = px.bar(
        focus_df,
        x='importance',
        y='area',
        orientation='h',
        title='ESG Focus Areas by Importance',
        labels={'importance': 'Importance Score', 'area': 'ESG Area'},
        color='importance',
        color_continuous_scale='Viridis',
        template=plotly_template
    )

    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Display descriptions
    for area in focus_areas:
        st.markdown(f"**{area['area']}:** {area['description']}")

    # Call to action
    st.markdown("## Next Steps")
    st.markdown("""
    Based on these recommendations, you can:
    1. Explore the recommended assets in detail in the Market Explorer
    2. Add them to your portfolio in the Portfolio Manager
    3. Set up alerts for price changes or ESG developments
    4. Schedule a portfolio review to track performance against your goals
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.button("Explore Recommended Assets")

    with col2:
        st.button("Add to Portfolio")
else:
    # Show placeholder content when recommendations haven't been generated yet
    st.markdown("## How Our AI Recommendation Engine Works")

    st.markdown("""
    Our AI-powered recommendation engine analyzes multiple data sources to provide personalized investment advice:

    1. **Financial Analysis**: We evaluate traditional metrics like ROI, volatility, and market trends

    2. **ESG Integration**: We incorporate environmental, social, and governance factors into our analysis

    3. **Risk Assessment**: We calculate risk scores based on multiple factors, including market volatility and ESG risks

    4. **Personalization**: We tailor recommendations to your specific preferences and goals

    5. **Continuous Learning**: Our AI models continuously improve based on market developments and ESG trends

    Set your preferences above and click "Generate Personalized Recommendations" to receive tailored investment advice.
    """)

    # Sample recommendation visualization
    st.markdown("### Sample Recommendation Process")

    # Create a sample flow chart
    fig = go.Figure()

    # Add nodes
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4, 2],
        y=[0, 0, 0, 0, 0, 1],
        mode='markers+text',
        marker=dict(size=30, color=['#2196F3', '#4CAF50', '#9C27B0', '#FF9800', '#F44336', '#009688']),
        text=['Market<br>Data', 'ESG<br>Data', 'AI<br>Analysis', 'Personalized<br>Scoring', 'Final<br>Recommendations', 'Your<br>Preferences'],
        textposition='bottom center'
    ))

    # Add arrows
    fig.add_annotation(
        x=0.5, y=0,
        ax=0, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#666'
    )

    fig.add_annotation(
        x=1.5, y=0,
        ax=1, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#666'
    )

    fig.add_annotation(
        x=2.5, y=0,
        ax=2, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#666'
    )

    fig.add_annotation(
        x=3.5, y=0,
        ax=3, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#666'
    )

    # Add arrow from preferences to AI analysis
    fig.add_annotation(
        x=2, y=0.1,
        ax=2, ay=0.9,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#666'
    )

    # Update layout
    fig.update_layout(
        title='AI Recommendation Process',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        template=plotly_template
    )

    st.plotly_chart(fig, use_container_width=True)

# Educational content
st.markdown("## Understanding ESG Investing")
st.markdown("""
ESG investing considers Environmental, Social, and Governance factors alongside financial metrics:

- **Environmental factors** include climate impact, resource usage, pollution, and waste management
- **Social factors** cover labor practices, community relations, diversity, and human rights
- **Governance factors** involve corporate structure, executive compensation, and business ethics

Research shows that companies with strong ESG practices often demonstrate better long-term financial performance and resilience during market downturns.
""")

# SDG alignment explanation
st.markdown("## UN Sustainable Development Goals (SDGs)")

# Convert imported SDG_DATA to DataFrame format for display
sdg_data = {
    'SDG': [],
    'Description': [],
    'Related_Sectors': []
}

for sdg_num, data in SDG_DATA.items():
    sdg_data['SDG'].append(f"SDG {sdg_num}: {data['name']}")
    sdg_data['Description'].append(data['description'])
    sdg_data['Related_Sectors'].append(', '.join(data['related_sectors']))

sdg_df = pd.DataFrame(sdg_data)

# Display SDGs in an expander
with st.expander("View UN Sustainable Development Goals (SDGs)"):
    # Create three columns for better display
    col1, col2, col3 = st.columns(3)

    # Distribute SDGs across columns
    sdgs_per_column = len(sdg_df) // 3 + (1 if len(sdg_df) % 3 > 0 else 0)

    with col1:
        for i in range(0, sdgs_per_column):
            st.markdown(f"**{sdg_df.iloc[i]['SDG']}**")
            st.markdown(f"{sdg_df.iloc[i]['Description']}")
            st.markdown(f"*Related sectors: {sdg_df.iloc[i]['Related_Sectors']}*")
            st.markdown("---")

    with col2:
        for i in range(sdgs_per_column, min(2*sdgs_per_column, len(sdg_df))):
            st.markdown(f"**{sdg_df.iloc[i]['SDG']}**")
            st.markdown(f"{sdg_df.iloc[i]['Description']}")
            st.markdown(f"*Related sectors: {sdg_df.iloc[i]['Related_Sectors']}*")
            st.markdown("---")

    with col3:
        for i in range(2*sdgs_per_column, len(sdg_df)):
            st.markdown(f"**{sdg_df.iloc[i]['SDG']}**")
            st.markdown(f"{sdg_df.iloc[i]['Description']}")
            st.markdown(f"*Related sectors: {sdg_df.iloc[i]['Related_Sectors']}*")
            st.markdown("---")

# FAQ section
st.markdown("## Frequently Asked Questions")

with st.expander("How are ESG scores calculated?"):
    st.markdown("""
    ESG scores are calculated using data from multiple sources, including:

    - Company disclosures and sustainability reports
    - Third-party ESG rating agencies
    - News and media analysis
    - Industry benchmarks and standards

    Our AI analyzes these data points to create comprehensive ESG profiles for each asset, considering factors like:

    - Carbon emissions and climate strategy
    - Resource usage and waste management
    - Labor practices and human rights
    - Diversity and inclusion
    - Corporate governance and business ethics

    Each factor is weighted based on its materiality to the specific industry and company.
    """)

with st.expander("How does sustainable investing affect returns?"):
    st.markdown("""
    Research on sustainable investing and returns shows:

    - **Long-term performance**: Multiple studies indicate that ESG-focused investments can match or outperform conventional investments over the long term.

    - **Risk mitigation**: Companies with strong ESG practices often face fewer regulatory issues, lawsuits, and reputational damage, potentially reducing downside risk.

    - **Resilience**: During market downturns, sustainable investments have frequently demonstrated greater resilience.

    - **Future-proofing**: Companies addressing sustainability challenges may be better positioned for future regulatory changes and market shifts.

    While past performance doesn't guarantee future results, the evidence suggests that sustainable investing doesn't require sacrificing returns and may offer advantages in risk management.
    """)

with st.expander("How often are recommendations updated?"):
    st.markdown("""
    Our AI recommendation engine updates its analysis based on:

    - **Market data**: Daily updates of price movements, volatility, and trading volumes

    - **ESG developments**: Weekly updates of ESG scores and sustainability metrics

    - **News and events**: Real-time monitoring of significant news that might impact ESG ratings or financial performance

    - **Regulatory changes**: Updates whenever relevant ESG regulations or reporting requirements change

    We recommend reviewing your portfolio and our recommendations at least quarterly to ensure alignment with your investment goals and sustainability preferences.
    """)

with st.expander("How can I balance profitability with sustainability?"):
    st.markdown("""
    Balancing profitability with sustainability involves:

    1. **Define your priorities**: Use our preference settings to determine the relative importance of financial returns and sustainability impact

    2. **Diversification**: Include a mix of assets with different ESG profiles and financial characteristics

    3. **Focus on material ESG factors**: Prioritize ESG issues that are financially material to each industry

    4. **Consider time horizon**: Sustainable investments often outperform over longer time horizons

    5. **Regular rebalancing**: Periodically review and adjust your portfolio to maintain your desired balance

    Our AI recommendations are designed to help you find this balance based on your specific preferences and goals.
    """)

# Feedback section
st.markdown("## Feedback")
st.markdown("Help us improve our recommendations by providing feedback")

feedback = st.text_area("Your feedback on the recommendations", height=100)
rating = st.slider("Rate the quality of recommendations", 1, 5, 3)

if st.button("Submit Feedback"):
    st.success("Thank you for your feedback! We'll use it to improve our recommendation engine.")
