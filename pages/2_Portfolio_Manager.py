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

# Set page config
st.set_page_config(
    page_title="Portfolio Manager - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Apply theme-specific styles
theme_colors = apply_theme_css()
plotly_template = theme_colors['plotly_template']

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
                'allocation': 0.15
            },
            {
                'name': 'Renewable Power',
                'ticker': 'RNWP',
                'asset_type': 'Stock',
                'shares': 25,
                'purchase_price': 45.25,
                'current_price': 52.30,
                'esg_score': 92.1,
                'allocation': 0.13
            },
            {
                'name': 'Sustainable Finance',
                'ticker': 'SUFN',
                'asset_type': 'Stock',
                'shares': 15,
                'purchase_price': 78.90,
                'current_price': 82.15,
                'esg_score': 79.8,
                'allocation': 0.12
            },
            {
                'name': 'EcoToken',
                'ticker': 'ECO',
                'asset_type': 'Crypto',
                'shares': 50,
                'purchase_price': 25.10,
                'current_price': 32.45,
                'esg_score': 76.5,
                'allocation': 0.16
            },
            {
                'name': 'GreenCoin',
                'ticker': 'GRC',
                'asset_type': 'Crypto',
                'shares': 100,
                'purchase_price': 12.30,
                'current_price': 15.80,
                'esg_score': 68.9,
                'allocation': 0.16
            },
            {
                'name': 'Sustainable Pharma',
                'ticker': 'SUPH',
                'asset_type': 'Stock',
                'shares': 20,
                'purchase_price': 65.40,
                'current_price': 72.25,
                'esg_score': 81.3,
                'allocation': 0.14
            },
            {
                'name': 'CleanRetail Inc',
                'ticker': 'CLRT',
                'asset_type': 'Stock',
                'shares': 30,
                'purchase_price': 42.60,
                'current_price': 39.75,
                'esg_score': 77.2,
                'allocation': 0.12
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

# Generate risk metrics
def generate_risk_metrics():
    risk_metrics = {
        'portfolio_volatility': 12.5,  # Annualized volatility (%)
        'sharpe_ratio': 1.8,
        'max_drawdown': 15.2,  # Maximum drawdown (%)
        'var_95': 2.3,  # 95% Value at Risk (%)
        'esg_risk_score': 18.5,  # Lower is better
        'carbon_intensity': 65.3,  # tCO2e/$M revenue
        'sdg_alignment': 6,  # Number of SDGs aligned with
        'controversy_exposure': 'Low'
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

# Portfolio summary
st.markdown("## Portfolio Summary")

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

    if risk_tolerance < 4 and sustainability_focus > 7:
        st.markdown("""
        Based on your low risk tolerance and high sustainability focus, we recommend:

        1. **Reduce** your position in GreenCoin (GRC) by 30% due to its high volatility
        2. **Increase** your position in Renewable Power (RNWP) by 20% for its strong ESG score and stable returns
        3. **Add** a new position in WaterTech (WTRT) for exposure to the water sustainability sector
        """)
    elif risk_tolerance > 7 and sustainability_focus < 4:
        st.markdown("""
        Based on your high risk tolerance and lower sustainability focus, we recommend:

        1. **Increase** your position in EcoToken (ECO) by 25% for higher potential returns
        2. **Add** a new position in FutureCoin (FUTC) for growth exposure
        3. **Reduce** your position in CleanRetail Inc (CLRT) due to underperformance
        """)
    else:
        st.markdown("""
        Based on your balanced risk and sustainability preferences, we recommend:

        1. **Maintain** your current allocation to GreenTech Solutions (GRNT)
        2. **Increase** your position in Sustainable Finance (SUFN) by 10%
        3. **Add** a new position in Circular Economy (CRCE) for diversification
        4. **Reduce** your position in CleanRetail Inc (CLRT) by 15% due to recent underperformance
        """)

    # Show projected impact
    st.markdown("#### Projected Impact of Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Projected Return", "+8.5%", delta="+1.2%")

    with col2:
        st.metric("Projected Risk", "-5.2%", delta="-0.8%", delta_color="inverse")

    with col3:
        st.metric("Projected ESG Score", "84.3", delta="+2.1")

# Impact reporting
st.markdown("## Impact Reporting")
st.markdown("""
Understand how your investments are contributing to a more sustainable future.
This section shows the environmental and social impact of your portfolio.
""")

# Impact metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Carbon Impact")

    carbon_avoided = 12.5  # Tons of CO2 equivalent
    st.markdown(f"""
    <div class="metric-card">
        <h4>Carbon Emissions Avoided</h4>
        <p style="font-size: 1.8rem; font-weight: bold;">{carbon_avoided} tCO2e</p>
        <p>Equivalent to taking 2.7 cars off the road for a year</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Social Impact")

    jobs_supported = 85
    st.markdown(f"""
    <div class="metric-card">
        <h4>Jobs Supported</h4>
        <p style="font-size: 1.8rem; font-weight: bold;">{jobs_supported}</p>
        <p>Through investments in companies with fair labor practices</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("### SDG Alignment")

    aligned_sdgs = [7, 8, 9, 12, 13]
    sdg_badges = " ".join([f"<span style='background-color:#f0f2f6;padding:5px 10px;border-radius:10px;margin-right:5px;'>SDG {sdg}</span>" for sdg in aligned_sdgs])

    st.markdown(f"""
    <div class="metric-card">
        <h4>SDG Contribution</h4>
        <p>{sdg_badges}</p>
        <p>Your portfolio contributes to 5 of the UN Sustainable Development Goals</p>
    </div>
    """, unsafe_allow_html=True)

# Impact visualization
st.markdown("### Impact Visualization")

# Create a radar chart for impact categories
categories = ['Climate Action', 'Clean Energy', 'Sustainable Cities',
              'Responsible Consumption', 'Gender Equality', 'Decent Work']
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
