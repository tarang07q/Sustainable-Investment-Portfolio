import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theme import apply_theme_css

# Set page config
st.set_page_config(
    page_title="Market Explorer - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Apply theme-specific styles
theme_colors = apply_theme_css()
plotly_template = theme_colors['plotly_template']

# Generate dummy stock data
def generate_stock_data() -> pd.DataFrame:
    np.random.seed(42)

    companies = {
        'Name': [
            'GreenTech Solutions', 'EcoEnergy Corp', 'Sustainable Pharma',
            'CleanRetail Inc', 'Future Mobility', 'Smart Agriculture',
            'Renewable Power', 'Circular Economy', 'WaterTech',
            'Sustainable Finance', 'Green Construction', 'EcoTransport'
        ],
        'Ticker': [
            'GRNT', 'ECOE', 'SUPH', 'CLRT', 'FUTM', 'SMAG',
            'RNWP', 'CRCE', 'WTRT', 'SUFN', 'GRCN', 'ECTR'
        ],
        'Sector': [
            'Technology', 'Energy', 'Healthcare',
            'Retail', 'Transportation', 'Agriculture',
            'Energy', 'Manufacturing', 'Utilities',
            'Finance', 'Construction', 'Transportation'
        ],
        'Asset_Type': ['Stock'] * 12
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
            'RenewCoin', 'CircularToken'
        ],
        'Ticker': [
            'GRC', 'ECO', 'SUST', 'CLNC', 'FUTC', 'AGRT',
            'RNWC', 'CIRC'
        ],
        'Sector': [
            'Green Technology', 'Renewable Energy', 'Sustainable Supply Chain',
            'Carbon Credits', 'Future Tech', 'Sustainable Agriculture',
            'Renewable Energy', 'Circular Economy'
        ],
        'Asset_Type': ['Crypto'] * 8
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

# Generate historical price data
def generate_historical_data(ticker, days=180, volatility=0.2, trend=0.1):
    np.random.seed(hash(ticker) % 10000)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate price movement with some trend and volatility
    returns = np.random.normal(trend/100, volatility/100, size=len(dates))
    price = 100
    prices = [price]

    for r in returns:
        price = price * (1 + r)
        prices.append(price)

    prices = prices[:-1]  # Remove the extra price

    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })

    return df

# Load data
stocks_df = generate_stock_data()
crypto_df = generate_crypto_data()
all_assets_df = pd.concat([stocks_df, crypto_df]).reset_index(drop=True)

# Header
st.title("🔍 Market Explorer")
st.markdown("*Discover sustainable investment opportunities across stocks and cryptocurrencies*")

# Filters
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    asset_type = st.selectbox(
        "Asset Type",
        options=["All", "Stock", "Crypto"],
        index=0
    )

with col2:
    if asset_type == "All":
        available_sectors = sorted(all_assets_df['Sector'].unique())
    elif asset_type == "Stock":
        available_sectors = sorted(stocks_df['Sector'].unique())
    else:
        available_sectors = sorted(crypto_df['Sector'].unique())

    sector = st.selectbox(
        "Sector",
        options=["All"] + available_sectors,
        index=0
    )

with col3:
    esg_min = st.slider(
        "Minimum ESG Score",
        min_value=0,
        max_value=100,
        value=50
    )

with col4:
    sort_by = st.selectbox(
        "Sort By",
        options=["AI Recommendation", "ESG Score", "ROI (1Y)", "Price Change (24h)"],
        index=0
    )

# Filter data based on selections
filtered_df = all_assets_df.copy()

if asset_type != "All":
    filtered_df = filtered_df[filtered_df['Asset_Type'] == asset_type]

if sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == sector]

filtered_df = filtered_df[filtered_df['ESG_Score'] >= esg_min]

# Sort data
if sort_by == "AI Recommendation":
    filtered_df = filtered_df.sort_values('AI_Risk_Score')
elif sort_by == "ESG Score":
    filtered_df = filtered_df.sort_values('ESG_Score', ascending=False)
elif sort_by == "ROI (1Y)":
    filtered_df = filtered_df.sort_values('ROI_1Y', ascending=False)
else:  # Price Change (24h)
    filtered_df = filtered_df.sort_values('Price_Change_24h', ascending=False)

# Display market overview
st.markdown("## Market Overview")

# Market metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_esg = filtered_df['ESG_Score'].mean()
    st.metric("Average ESG Score", f"{avg_esg:.1f}")

with col2:
    avg_roi = filtered_df['ROI_1Y'].mean()
    st.metric("Average ROI (1Y)", f"{avg_roi:.1f}%")

with col3:
    avg_price_change = filtered_df['Price_Change_24h'].mean()
    st.metric("Avg Price Change (24h)", f"{avg_price_change:.2f}%",
              delta=f"{avg_price_change:.2f}%")

with col4:
    green_assets = len(filtered_df[filtered_df['AI_Recommendation'] == '🟢 Strong Buy'])
    total_assets = len(filtered_df)
    st.metric("Recommended Assets", f"{green_assets}/{total_assets}")

# ESG vs ROI scatter plot
st.markdown("### ESG Score vs ROI")

# Set Plotly theme based on app theme
plotly_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

fig = px.scatter(
    filtered_df,
    x='ESG_Score',
    y='ROI_1Y',
    color='AI_Recommendation',
    size='Market_Cap_B',
    hover_data=['Name', 'Ticker', 'Sector', 'Current_Price', 'Price_Change_24h'],
    title='ESG Score vs ROI by Asset',
    labels={'ESG_Score': 'ESG Score', 'ROI_1Y': 'ROI (1 Year)'},
    color_discrete_map={'🟢 Strong Buy': 'green', '🟡 Hold': 'gold', '🔴 Caution': 'red'},
    template=plotly_template
)
st.plotly_chart(fig, use_container_width=True)

# Asset list
st.markdown("## Asset List")
st.dataframe(
    filtered_df[[
        'Name', 'Ticker', 'Asset_Type', 'Sector', 'Current_Price',
        'Price_Change_24h', 'ESG_Score', 'ROI_1Y', 'AI_Recommendation'
    ]],
    use_container_width=True
)

# Asset details
st.markdown("## Asset Details")

# Check if there are any assets in the filtered dataframe
if not filtered_df.empty:
    selected_asset = st.selectbox(
        "Select an asset to view details",
        options=filtered_df['Name'].tolist(),
        index=0
    )

    # Get the matching rows
    matching_rows = filtered_df[filtered_df['Name'] == selected_asset]

    # Check if there are any matching rows
    if not matching_rows.empty:
        asset_data = matching_rows.iloc[0]
    else:
        st.warning(f"No data found for {selected_asset}. Please select another asset.")
        st.stop()
else:
    st.warning("No assets match the current filters. Please adjust your filter criteria.")
    st.stop()

# Only display asset details if we have valid data
if 'asset_data' in locals():
    col1, col2 = st.columns([2, 1])

    with col1:
        # Historical price chart
        historical_data = generate_historical_data(
            asset_data['Ticker'],
            volatility=asset_data['Volatility']*100,
            trend=asset_data['ROI_1Y']/365
        )

        fig = px.line(
            historical_data,
            x='Date',
            y='Price',
            title=f"{asset_data['Name']} ({asset_data['Ticker']}) - 6 Month Price History",
            template=plotly_template
        )
        st.plotly_chart(fig, use_container_width=True)

        # ESG breakdown
        st.markdown("### ESG Score Breakdown")
        esg_data = {
            'Category': ['Environmental', 'Social', 'Governance'],
            'Score': [
                asset_data['Environmental_Score'],
                asset_data['Social_Score'],
                asset_data['Governance_Score']
            ]
        }
        esg_df = pd.DataFrame(esg_data)

        fig = px.bar(
            esg_df,
            x='Category',
            y='Score',
            color='Category',
            title='ESG Component Scores',
            color_discrete_map={
                'Environmental': 'green',
                'Social': 'blue',
                'Governance': 'purple'
            },
            template=plotly_template
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"### {asset_data['Name']} ({asset_data['Ticker']})")
        st.markdown(f"**Sector:** {asset_data['Sector']}")
        st.markdown(f"**Asset Type:** {asset_data['Asset_Type']}")
        st.markdown(f"**Current Price:** ${asset_data['Current_Price']:.2f}")

        # Price change with color
        price_change = asset_data['Price_Change_24h']
        price_color = "green" if price_change > 0 else "red"
        st.markdown(f"**24h Change:** <span style='color:{price_color}'>{price_change:.2f}%</span>", unsafe_allow_html=True)

        st.markdown(f"**Market Cap:** ${asset_data['Market_Cap_B']:.2f}B")
        st.markdown(f"**1Y ROI:** {asset_data['ROI_1Y']:.2f}%")
        st.markdown(f"**Volatility:** {asset_data['Volatility']:.2f}")
        st.markdown(f"**ESG Score:** {asset_data['ESG_Score']:.1f}/100")

        # SDG alignment
        st.markdown("**SDG Alignment:**")
        sdg_list = asset_data['SDG_Alignment']
        sdg_badges = " ".join([f"<span style='background-color:#f0f2f6;padding:3px 8px;border-radius:10px;margin-right:5px;'>SDG {sdg}</span>" for sdg in sdg_list])
        st.markdown(sdg_badges, unsafe_allow_html=True)

        # Carbon footprint
        st.markdown(f"**Carbon Footprint:** {asset_data['Carbon_Footprint']:.1f} tons CO2e")
        st.markdown(f"**Carbon Reduction Target:** {asset_data['Carbon_Reduction_Target']:.1f}%")

        # AI recommendation
        recommendation = asset_data['AI_Recommendation']
        rec_color = "green" if recommendation == '🟢 Strong Buy' else "gold" if recommendation == '🟡 Hold' else "red"
        st.markdown(f"**AI Recommendation:** <span style='color:{rec_color};font-weight:bold'>{recommendation}</span>", unsafe_allow_html=True)

        # Add to portfolio button
        st.button("Add to Portfolio", key=f"add_{asset_data['Ticker']}")

        # Set price alert button
        st.button("Set Price Alert", key=f"alert_{asset_data['Ticker']}")
