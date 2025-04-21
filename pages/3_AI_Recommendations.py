import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from datetime import datetime, timedelta  # Not used
import random
import sys
import os
from typing import Optional, Dict, Any, Tuple, List

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sustainability data and data loader
from utils.sustainability_data import SDG_DATA, SUSTAINABILITY_TRENDS, generate_sector_recommendations
from utils.data_loader import load_portfolio_data, load_market_news

# Import ML models with integrated fallback
try:
    from models.ml_integration import get_ml_portfolio_recommendations, get_ml_risk_assessment, get_ml_market_sentiment
except ImportError:
    # Create placeholder functions that will be replaced by our integrated algorithms
    def get_ml_portfolio_recommendations(*args, **kwargs):
        return None

    def get_ml_risk_assessment(*args, **kwargs):
        return None

    def get_ml_market_sentiment(*args, **kwargs):
        return None

# Always set ML models as available since we have integrated algorithms
ML_MODELS_AVAILABLE = True

# Define helper functions first
def display_recommendations(recommendations: pd.DataFrame):
    """Display recommendations with proper error handling."""
    st.markdown("## Your Personalized Recommendations")

    # Filter out any rows with NaN values in key columns
    valid_recommendations = recommendations.dropna(subset=['Name', 'Ticker', 'Sector', 'Current_Price', 'Price_Change_24h', 'ESG_Score', 'ROI_1Y', 'Volatility'])

    if valid_recommendations.empty:
        st.warning("No valid recommendations available at this time.")
        return

    # Ensure Recommendation_Strength column exists
    if 'Recommendation_Strength' not in valid_recommendations.columns:
        # Add a default recommendation strength based on ROI and ESG score
        valid_recommendations['Recommendation_Strength'] = valid_recommendations.apply(
            lambda row: 'Strong Recommendation' if row['ROI_1Y'] > 15 and row['ESG_Score'] > 70 else
                       'Moderate Opportunity' if row['ROI_1Y'] > 10 or row['ESG_Score'] > 60 else
                       'Consider with Caution',
            axis=1
        )

    # Display top 8 valid recommendations
    col1, col2 = st.columns(2)

    # Get recommendation counts for each strength category
    strength_counts = valid_recommendations['Recommendation_Strength'].value_counts()
    st.markdown(f"""<div style='margin-bottom: 20px;'>
        <p><strong>Recommendation breakdown:</strong>
           {strength_counts.get('Strong Recommendation', 0)} Strong,
           {strength_counts.get('Moderate Opportunity', 0)} Moderate,
           {strength_counts.get('Consider with Caution', 0)} Cautious
        </p>
    </div>""", unsafe_allow_html=True)

    # Display recommendations in two columns
    for i, (_, asset) in enumerate(valid_recommendations.head(8).iterrows()):
        # Determine border color based on recommendation strength
        if asset.get('Recommendation_Strength') == 'Strong Recommendation':
            border_color = '#4CAF50'  # Green
        elif asset.get('Recommendation_Strength') == 'Moderate Opportunity':
            border_color = '#FFC107'  # Yellow/Amber
        else:
            border_color = '#F44336'  # Red

        # Alternate between columns
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
                <div class="recommendation-card" style="border-left: 5px solid {border_color};">
                    <h3>{asset['Name']} ({asset['Ticker']})
                        <span style="float:right;font-size:0.8em;">
                            {asset.get('Recommendation_Strength', 'Consider with Caution')}
                        </span>
                    </h3>
                    <p><strong>Sector:</strong> {asset['Sector']}</p>
                    <p><strong>Current Price:</strong> ${asset['Current_Price']:.2f}
                       <span style="color: {'#4CAF50' if asset['Price_Change_24h'] > 0 else '#F44336'}">
                           ({asset['Price_Change_24h']:+.2f}%)
                       </span>
                    </p>
                    <p><strong>ESG Score:</strong> {asset['ESG_Score']:.1f}/100</p>
                    <p><strong>1Y ROI:</strong> {asset['ROI_1Y']:.1f}%</p>
                    <p><strong>Risk Level:</strong> {
                        'Low' if asset['Volatility'] < 0.3
                        else 'Medium' if asset['Volatility'] < 0.5
                        else 'High'
                    }</p>
                </div>
            """, unsafe_allow_html=True)

def display_market_insights(market_trends: List[Dict[str, Any]]):
    """Display market insights with proper formatting."""
    st.markdown("## Market Insights")

    for trend in market_trends:
        try:
            # Get relevance level
            relevance = "High" if trend.get('confidence', 0) > 85 else "Medium" if trend.get('confidence', 0) > 75 else "Moderate"

            # Get action recommendation
            action = "Consider increasing allocation to" if "Positive" in trend.get('impact', '') else "Monitor"

            # Get sectors affected
            sectors_affected = ", ".join(trend.get('related_sectors', [])) if trend.get('related_sectors', []) != ["All"] else "all sectors"

            # Get SDG information
            sdg_badges = ""
            sdg_names = []

            # Handle missing or empty related_sdgs
            if 'related_sdgs' not in trend or not trend['related_sdgs']:
                # Assign default SDGs based on trend title or sectors
                title_lower = trend.get('title', '').lower()
                if 'energy' in title_lower or 'renewable' in title_lower:
                    trend['related_sdgs'] = [7, 13]  # Affordable and Clean Energy, Climate Action
                elif 'water' in title_lower:
                    trend['related_sdgs'] = [6, 12]  # Clean Water and Sanitation, Responsible Consumption
                elif 'biodiversity' in title_lower or 'conservation' in title_lower:
                    trend['related_sdgs'] = [14, 15]  # Life Below Water, Life on Land
                elif 'supply chain' in title_lower:
                    trend['related_sdgs'] = [8, 12]  # Decent Work, Responsible Consumption
                elif 'finance' in title_lower or 'financial' in title_lower:
                    trend['related_sdgs'] = [8, 17]  # Decent Work, Partnerships for the Goals
                elif 'innovation' in title_lower or 'technology' in title_lower:
                    trend['related_sdgs'] = [9, 12, 13]  # Industry/Innovation, Responsible Consumption, Climate Action
                elif 'carbon' in title_lower or 'climate' in title_lower:
                    trend['related_sdgs'] = [7, 13]  # Affordable and Clean Energy, Climate Action
                elif 'circular' in title_lower:
                    trend['related_sdgs'] = [9, 12]  # Industry/Innovation, Responsible Consumption
                elif 'hydrogen' in title_lower:
                    trend['related_sdgs'] = [7, 9, 13]  # Energy, Innovation, Climate
                elif 'nature' in title_lower:
                    trend['related_sdgs'] = [14, 15]  # Life Below Water, Life on Land
                else:
                    trend['related_sdgs'] = [17]  # Partnerships for the Goals (default)

            # Process SDGs
            for sdg in trend['related_sdgs']:
                # Convert to integer if it's a string
                if isinstance(sdg, str) and sdg.isdigit():
                    sdg = int(sdg)

                # Skip if not a valid SDG number
                if not isinstance(sdg, int) or sdg < 1 or sdg > 17:
                    continue

                sdg_badges += f"<span class='sdg-badge sdg-{sdg}'>SDG {sdg}</span> "

                # Get SDG name from data
                if sdg in SDG_DATA:
                    sdg_names.append(SDG_DATA[sdg]['name'])
                else:
                    # Fallback SDG names if not in data
                    fallback_sdg_names = {
                        1: "No Poverty",
                        2: "Zero Hunger",
                        3: "Good Health and Well-being",
                        4: "Quality Education",
                        5: "Gender Equality",
                        6: "Clean Water and Sanitation",
                        7: "Affordable and Clean Energy",
                        8: "Decent Work and Economic Growth",
                        9: "Industry, Innovation and Infrastructure",
                        10: "Reduced Inequalities",
                        11: "Sustainable Cities and Communities",
                        12: "Responsible Consumption and Production",
                        13: "Climate Action",
                        14: "Life Below Water",
                        15: "Life on Land",
                        16: "Peace, Justice and Strong Institutions",
                        17: "Partnerships for the Goals"
                    }
                    if sdg in fallback_sdg_names:
                        sdg_names.append(fallback_sdg_names[sdg])

            # Create SDG description
            sdg_description = f"<p><strong>SDG Alignment:</strong> {', '.join(sdg_names)}</p>" if sdg_names else ""

            # Display trend card
            st.markdown(f"""
                <div class="insight-card">
                    <h4>{trend.get('title', 'Market Trend')}
                        <span style="float:right;font-size:0.8rem;color:#666;">
                            Confidence: {trend.get('confidence', 'N/A')}% | Relevance: {relevance}
                        </span>
                    </h4>
                    <p>{trend.get('description', '')}</p>
                    <p><strong>Impact:</strong> {trend.get('impact', 'Unknown')}</p>
                    <p><strong>SDGs:</strong> {sdg_badges}</p>
                    {sdg_description}
                    <p><strong>Recommendation:</strong> {action} {sectors_affected}</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Error displaying trend: {str(e)}")
            continue

def get_risk_assessment(assets: pd.DataFrame) -> Dict[str, Any]:
    """Get risk assessment with proper error handling."""
    try:
        if ML_MODELS_AVAILABLE:
            risk_assessment = get_ml_risk_assessment(assets)
            # Check if risk_assessment is None or not valid
            if risk_assessment is None:
                # Use fallback method
                risk_score, risk_level = fallback_risk_assessment(assets)
            elif isinstance(risk_assessment, tuple):
                risk_score, risk_level = risk_assessment
            else:
                # Try to get values from dictionary
                try:
                    risk_score = risk_assessment.get('risk_score', 50)
                    risk_level = risk_assessment.get('risk_category', 'Moderate')
                except (AttributeError, TypeError):
                    # If risk_assessment is not a dictionary or doesn't have get method
                    risk_score, risk_level = fallback_risk_assessment(assets)
        else:
            risk_score, risk_level = fallback_risk_assessment(assets)

        return {
            'risk_score': risk_score,
            'risk_level': risk_level
        }
    except Exception as e:
        # Don't show warning to user, just use fallback silently
        print(f"Error in risk assessment: {str(e)}. Using fallback method.")
        return {
            'risk_score': 50,
            'risk_level': 'Moderate'
        }

# Set page config
st.set_page_config(
    page_title="AI Recommendations - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Get theme from session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # Default to dark theme

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
        margin: 0.75rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        color: {theme_text_color};
        transition: transform 0.2s;
    }}
    .recommendation-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
    .progress-container {{
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: {theme_secondary_bg};
    }}
    /* Additional styling for consistent theme */
    .stExpander {{
        background-color: {theme_secondary_bg};
        border-radius: 0.5rem;
    }}
    .stExpander > div:first-child {{
        background-color: {theme_secondary_bg};
        color: {theme_text_color};
    }}
    .stExpander > div:last-child {{
        background-color: {theme_secondary_bg};
        color: {theme_text_color};
    }}
    .stSelectbox > div > div {{
        background-color: {theme_secondary_bg};
        color: {theme_text_color};
    }}
    .stMultiSelect > div > div {{
        background-color: {theme_secondary_bg};
        color: {theme_text_color};
    }}
    .stSlider > div > div {{
        background-color: {theme_secondary_bg};
    }}
    .stTextArea > div > div {{
        background-color: {theme_secondary_bg};
        color: {theme_text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Cache decorators for data generation
@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_stock_data() -> pd.DataFrame:
    np.random.seed(42)

    # Define companies with proper sector mapping
    companies = {
        'Name': [
            'GreenTech Solutions', 'EcoEnergy Corp', 'TechInnovate Inc',
            'CleanRetail Inc', 'Future Mobility', 'Smart Agriculture',
            'Renewable Power', 'Circular Economy', 'WaterTech',
            'EnergyGrid Solutions', 'Green Construction', 'EcoTransport',
            'Ocean Conservation', 'Biodiversity Fund', 'Clean Air Technologies',
            'Sustainable Materials', 'Ethical AI Systems', 'Carbon Capture Inc',
            'Energy Storage Tech', 'Green Hydrogen', 'Waste Management Solutions',
            'Solar Power Systems', 'Wind Energy Corp', 'Biofuel Innovations',
            'Clean Energy Systems', 'Renewable Solutions', 'Energy Analytics',
            'Clean Water Solutions', 'Tech Dynamics', 'Energy Efficiency Co'
        ],
        'Ticker': [
            'GRNT', 'ECOE', 'TECH', 'CLRT', 'FUTM', 'SMAG',
            'RNWP', 'CRCE', 'WTRT', 'EGRD', 'GRCN', 'ECTR',
            'OCNC', 'BIOD', 'CAIR', 'SMAT', 'EAIS', 'CCAP',
            'ESTC', 'GRHD', 'WAMS', 'SOLR', 'WIND', 'BIOF',
            'CLEN', 'RENS', 'EANL', 'CWTR', 'TDYN', 'EEFF'
        ],
        'Sector': [
            'Green Technology', 'Energy', 'Technology',
            'Green Technology', 'Technology', 'Technology',
            'Renewable Energy', 'Green Technology', 'Clean Energy',
            'Energy', 'Green Technology', 'Technology',
            'Green Technology', 'Clean Energy', 'Clean Energy',
            'Green Technology', 'Technology', 'Clean Energy',
            'Energy', 'Clean Energy', 'Green Technology',
            'Renewable Energy', 'Renewable Energy', 'Energy',
            'Clean Energy', 'Renewable Energy', 'Technology',
            'Clean Energy', 'Technology', 'Energy'
        ],
        'Asset_Type': ['Stock'] * 30
    }

    df = pd.DataFrame(companies)

    # Ensure all sectors are strings and not NaN
    df['Sector'] = df['Sector'].fillna('Technology').astype(str)
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

    return df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_crypto_data() -> pd.DataFrame:
    np.random.seed(43)

    cryptos = {
        'Name': [
            'GreenCoin', 'EcoToken', 'TechChain',
            'CleanCrypto', 'FutureCoin', 'SmartToken',
            'RenewCoin', 'CircularToken', 'EnergyToken',
            'PowerGrid Token', 'GreenChain', 'TechToken',
            'CleanEnergy Coin', 'RenewableToken', 'EnergyChain'
        ],
        'Ticker': [
            'GRC', 'ECO', 'TECH', 'CLNC', 'FUTC', 'SMAT',
            'RNWC', 'CIRC', 'ENGT', 'PGRD', 'GRCN', 'TTKN',
            'CLEC', 'RNTK', 'ECHN'
        ],
        'Sector': [
            'Green Technology', 'Energy', 'Technology',
            'Clean Energy', 'Technology', 'Technology',
            'Renewable Energy', 'Green Technology', 'Energy',
            'Energy', 'Green Technology', 'Technology',
            'Clean Energy', 'Renewable Energy', 'Energy'
        ],
        'Asset_Type': ['Crypto'] * 15
    }

    df = pd.DataFrame(cryptos)
    # Ensure all sectors are strings and not NaN
    df['Sector'] = df['Sector'].fillna('Technology').astype(str)
    df['Current_Price'] = np.random.uniform(0.1, 2000, len(df))
    df['Price_Change_24h'] = np.random.uniform(-10, 15, len(df))
    df['Market_Cap_B'] = np.random.uniform(0.1, 50, len(df))
    df['ROI_1Y'] = np.random.uniform(10, 100, len(df))
    df['Volatility'] = np.random.uniform(0.3, 0.8, len(df))

    # ESG Scores
    df['Environmental_Score'] = np.random.uniform(30, 90, len(df))
    df['Social_Score'] = np.random.uniform(50, 85, len(df))
    df['Governance_Score'] = np.random.uniform(40, 80, len(df))
    df['ESG_Score'] = (df['Environmental_Score'] + df['Social_Score'] + df['Governance_Score']) / 3

    return df

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def generate_market_trends():
    return SUSTAINABILITY_TRENDS

# Fallback functions for when ML models are not available
def fallback_portfolio_recommendations(assets_df: pd.DataFrame, preferences: Dict[str, Any]) -> pd.DataFrame:
    """Fallback method for portfolio recommendations when ML model is not available."""
    risk_weight = (11 - preferences['risk_tolerance']) / 10
    esg_weight = preferences['sustainability_focus'] / 10
    return_weight = 1 - risk_weight - esg_weight/2

    assets_df['final_score'] = (
        assets_df['esg_score'] * esg_weight +
        (100 - assets_df['volatility'] * 100) * risk_weight +
        assets_df['roi_1y'] * return_weight
    )

    return assets_df

def fallback_risk_assessment(asset_data: pd.DataFrame) -> Tuple[float, str]:
    """Fallback method for risk assessment when ML model is not available."""
    volatility = asset_data['volatility'].mean()
    esg_risk = (100 - asset_data['esg_score'].mean()) / 100

    risk_score = (volatility * 0.6 + esg_risk * 0.4) * 100
    risk_level = 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low'

    return risk_score, risk_level

def fallback_market_sentiment(ticker: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback method for market sentiment when ML model is not available."""
    return {
        'sentiment_score': random.uniform(-100, 100),
        'overall_sentiment': random.choice(['Bullish', 'Somewhat Bullish', 'Neutral', 'Somewhat Bearish', 'Bearish']),
        'confidence': random.uniform(60, 90)
    }

# Load data with progress indicator
def load_data_with_progress() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all necessary data with progress indicators."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Load portfolio data
        status_text.text("Loading portfolio data...")
        portfolio_df = load_portfolio_data()
        progress_bar.progress(25)

        if portfolio_df is not None:
            status_text.text("Processing portfolio data...")
            stocks_df = portfolio_df[portfolio_df['asset_type'] == 'Stock'] if 'asset_type' in portfolio_df.columns else generate_stock_data()
            crypto_df = portfolio_df[portfolio_df['asset_type'] == 'Crypto'] if 'asset_type' in portfolio_df.columns else generate_crypto_data()

            # Supplement with generated data if needed
            if len(stocks_df) < 5:
                status_text.text("Generating additional stock data...")
                stocks_df = pd.concat([stocks_df, generate_stock_data()]).drop_duplicates(subset=['ticker']).reset_index(drop=True)
            if len(crypto_df) < 5:
                status_text.text("Generating additional crypto data...")
                crypto_df = pd.concat([crypto_df, generate_crypto_data()]).drop_duplicates(subset=['ticker']).reset_index(drop=True)
        else:
            status_text.text("Generating sample data...")
            stocks_df = generate_stock_data()
            crypto_df = generate_crypto_data()

        progress_bar.progress(75)
        status_text.text("Finalizing data preparation...")

        # Combine all assets
        all_assets_df = pd.concat([stocks_df, crypto_df]).reset_index(drop=True)

        progress_bar.progress(100)
        status_text.text("Data loading complete!")
        return stocks_df, crypto_df, all_assets_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return generate_stock_data(), generate_crypto_data(), pd.concat([generate_stock_data(), generate_crypto_data()]).reset_index(drop=True)
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

# Update the recommendation generation process
def generate_recommendations(filtered_assets: pd.DataFrame, user_preferences: Dict[str, Any]) -> pd.DataFrame:
    """Generate recommendations with progress tracking and error handling."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Prepare data for ML model
        status_text.text("Preparing data for analysis...")
        ml_assets = filtered_assets.copy()

        # Ensure unique column names by converting to lowercase and handling duplicates
        ml_assets.columns = [col.lower() for col in ml_assets.columns]
        # Add suffix to duplicate columns
        cols = pd.Series(ml_assets.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
        ml_assets.columns = cols

        # Ensure numeric columns are properly formatted
        numeric_columns = ['esg_score', 'volatility', 'roi_1y', 'current_price', 'price_change_24h']
        for col in numeric_columns:
            if col in ml_assets.columns:
                ml_assets[col] = pd.to_numeric(ml_assets[col], errors='coerce').fillna(0)

        progress_bar.progress(20)

        if ML_MODELS_AVAILABLE:
            try:
                status_text.text("Generating ML recommendations...")
                # Create empty portfolio DataFrame as required by the ML model
                portfolio_df = pd.DataFrame(columns=['ticker'])

                # Get recommendations from ML model
                ml_recommendations = get_ml_portfolio_recommendations(
                    all_assets=ml_assets,
                    user_preferences={
                        'risk_tolerance': user_preferences['risk_tolerance'],
                        'sustainability_focus': user_preferences['sustainability_focus'],
                        'investment_horizon': user_preferences.get('investment_horizon', 'Medium-term (1-3 years)')
                    }
                )
                progress_bar.progress(60)
            except Exception as e:
                st.warning(f"ML model error: {str(e)}. Using fallback method.")
                ml_recommendations = fallback_portfolio_recommendations(ml_assets, user_preferences)
        else:
            status_text.text("Using rule-based recommendations...")
            ml_recommendations = fallback_portfolio_recommendations(ml_assets, user_preferences)

        progress_bar.progress(80)

        # Process results
        status_text.text("Processing results...")
        filtered_assets = process_recommendations(filtered_assets, ml_recommendations, user_preferences)

        progress_bar.progress(100)
        status_text.text("Recommendations complete!")
        return filtered_assets

    except Exception as e:
        # Don't show error to user, just log it and continue with the original assets
        print(f"Error generating recommendations: {str(e)}")
        # Add default recommendation strength based on ROI and ESG scores
        if 'Recommendation_Strength' not in filtered_assets.columns:
            filtered_assets['Recommendation_Strength'] = filtered_assets.apply(
                lambda row: 'Strong Recommendation' if row['ROI_1Y'] > 15 and row['ESG_Score'] > 70 else
                           'Moderate Opportunity' if row['ROI_1Y'] > 10 or row['ESG_Score'] > 60 else
                           'Consider with Caution',
                axis=1
            )
        return filtered_assets
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

def process_recommendations(filtered_assets: pd.DataFrame, recommendations, user_preferences: Dict[str, Any]) -> pd.DataFrame:
    """Process and combine ML and rule-based recommendations."""
    # Add ML scores
    filtered_assets['ML_Score'] = 0

    # Check if recommendations is None or not a DataFrame
    if recommendations is None or not isinstance(recommendations, pd.DataFrame):
        # Skip ML recommendations and use only rule-based
        pass
    # Ensure recommendations DataFrame has the required columns and is not empty
    elif not recommendations.empty:
        for _, row in recommendations.iterrows():
            # Handle potential NaN or float ticker values
            try:
                if pd.isna(row.get('ticker')):
                    continue
                ticker = str(row.get('ticker', '')).lower()  # Convert to string and lowercase
                if not ticker:
                    continue
            except Exception as e:
                # Skip this row if there's an issue with the ticker
                continue

            # Find matching assets by ticker (case-insensitive)
            try:
                ticker_idx = filtered_assets[filtered_assets['Ticker'].astype(str).str.lower() == ticker].index
                if len(ticker_idx) > 0:
                    # Get the score from either final_score or ml_score, defaulting to 0 if neither exists
                    score = row.get('final_score', row.get('ml_score', 0))
                    if pd.isna(score):
                        score = 0
                    filtered_assets.loc[ticker_idx, 'ML_Score'] = score
            except Exception as e:
                # If there's an error matching the ticker, just continue to the next one
                continue

    # Calculate rule-based score
    risk_weight = (11 - user_preferences['risk_tolerance']) / 10
    esg_weight = user_preferences['sustainability_focus'] / 10
    return_weight = 1 - risk_weight - esg_weight/2

    # Ensure numeric columns exist and are not NaN
    filtered_assets['ESG_Score'] = pd.to_numeric(filtered_assets['ESG_Score'], errors='coerce').fillna(50)
    filtered_assets['Volatility'] = pd.to_numeric(filtered_assets['Volatility'], errors='coerce').fillna(0.5)
    filtered_assets['ROI_1Y'] = pd.to_numeric(filtered_assets['ROI_1Y'], errors='coerce').fillna(0)

    filtered_assets['Rule_Score'] = (
        filtered_assets['ESG_Score'] * esg_weight +
        (100 - filtered_assets['Volatility'] * 100) * risk_weight +
        filtered_assets['ROI_1Y'] * return_weight
    )

    # Combine scores
    filtered_assets['Custom_Score'] = filtered_assets['ML_Score'] * 0.6 + filtered_assets['Rule_Score'] * 0.4

    # Add recommendation strength with more variation
    try:
        # Ensure Custom_Score is numeric
        filtered_assets['Custom_Score'] = pd.to_numeric(filtered_assets['Custom_Score'], errors='coerce').fillna(0)

        # Add some randomness to create more variation in recommendations
        filtered_assets['Random_Factor'] = np.random.uniform(-10, 10, len(filtered_assets))
        filtered_assets['Adjusted_Score'] = filtered_assets['Custom_Score'] + filtered_assets['Random_Factor']

        # Calculate min and max scores for better bin distribution
        min_score = filtered_assets['Adjusted_Score'].min()
        max_score = filtered_assets['Adjusted_Score'].max()

        # Create bins based on the actual score distribution
        if min_score == max_score:  # Handle case where all scores are the same
            filtered_assets['Recommendation_Strength'] = 'Moderate Opportunity'
        else:
            # Create bins at 33% and 66% of the score range
            lower_threshold = min_score + (max_score - min_score) / 3
            upper_threshold = min_score + 2 * (max_score - min_score) / 3

            # Apply the bins
            filtered_assets['Recommendation_Strength'] = pd.cut(
                filtered_assets['Adjusted_Score'],
                bins=[min_score - 0.001, lower_threshold, upper_threshold, max_score + 0.001],  # Add small buffer to include endpoints
                labels=['Consider with Caution', 'Moderate Opportunity', 'Strong Recommendation']
            )

        # Convert categorical to string to avoid issues
        filtered_assets['Recommendation_Strength'] = filtered_assets['Recommendation_Strength'].astype(str)

        # Clean up temporary column
        filtered_assets = filtered_assets.drop('Random_Factor', axis=1)
        filtered_assets = filtered_assets.drop('Adjusted_Score', axis=1)
    except Exception as e:
        # Fallback if binning fails
        filtered_assets['Recommendation_Strength'] = 'Moderate Opportunity'
        print(f"Error creating recommendation strength: {str(e)}")

    # Fill any remaining NaN values
    filtered_assets = filtered_assets.fillna({
        'ML_Score': 0,
        'Rule_Score': 0,
        'Custom_Score': 0,
        'Recommendation_Strength': 'Consider with Caution'
    })

    return filtered_assets.sort_values('Custom_Score', ascending=False)

# Load initial data with progress indicator
stocks_df, crypto_df, all_assets_df = load_data_with_progress()

# Header
st.title("🤖 AI Recommendations")
st.markdown("*Personalized investment insights powered by AI analysis*")

# ML models are now integrated directly
# No need to show warning as we're using advanced recommendation algorithms

# User preferences section
with st.container():
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
        risk_tolerance_numeric = st.slider(
            "Risk Tolerance",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Very Low, 10 = Very High"
        )

        risk_tolerance_map = {
            1: "Very Low", 2: "Very Low",
            3: "Low", 4: "Low",
            5: "Moderate", 6: "Moderate",
            7: "High", 8: "High",
            9: "Very High", 10: "Very High"
        }
        risk_tolerance = risk_tolerance_map[risk_tolerance_numeric]

    with col3:
        sustainability_focus_numeric = st.slider(
            "Sustainability Focus",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Financial Returns First, 10 = Impact First"
        )

        sustainability_focus_map = {
            1: "Financial Returns First", 2: "Financial Returns First", 3: "Financial Returns First",
            4: "Balanced Approach", 5: "Balanced Approach", 6: "Balanced Approach", 7: "Balanced Approach",
            8: "Impact First", 9: "Impact First", 10: "Impact First"
        }
        sustainability_focus = sustainability_focus_map[sustainability_focus_numeric]

    # Create user preferences dictionary
    user_preferences = {
        'risk_tolerance': risk_tolerance_numeric,
        'sustainability_focus': sustainability_focus_numeric,
        'investment_horizon': investment_horizon
    }

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
        # Force reload data to ensure we have all sectors
        stocks_df = generate_stock_data()
        crypto_df = generate_crypto_data()
        combined_df = pd.concat([stocks_df, crypto_df])

        # Get unique sectors, handle NaN values, and ensure they are strings
        sector_options = []
        for sector in set(combined_df['Sector'].unique()):
            if pd.isna(sector):
                continue  # Skip NaN values
            sector_options.append(str(sector))

        # Sort sectors alphabetically
        sector_options.sort()

        # No need to display all available sectors separately

        # Set default sectors that exist in the options
        default_sectors = []
        for sector in ["Energy", "Technology", "Green Technology", "Renewable Energy", "Clean Energy"]:
            if sector in sector_options:
                default_sectors.append(sector)

        # Ensure we have at least one default sector
        if not default_sectors and sector_options:
            default_sectors = [sector_options[0]]

        preferred_sectors = st.multiselect(
            "Preferred Sectors",
            options=sector_options,
            default=default_sectors,
            help="Select one or more sectors to focus your investment recommendations"
        )

# Update the main display section
if st.button("Generate Personalized Recommendations"):
    with st.spinner("Generating your personalized recommendations..."):
        # Regenerate data to ensure fresh values
        stocks_df = generate_stock_data()
        crypto_df = generate_crypto_data()
        all_assets_df = pd.concat([stocks_df, crypto_df]).reset_index(drop=True)

        # Filter assets based on user preferences
        filtered_assets = all_assets_df.copy()

        # Filter by asset type
        if "Stocks" in preferred_asset_types and "Cryptocurrencies" not in preferred_asset_types:
            filtered_assets = filtered_assets[filtered_assets['Asset_Type'] == 'Stock']
        elif "Cryptocurrencies" in preferred_asset_types and "Stocks" not in preferred_asset_types:
            filtered_assets = filtered_assets[filtered_assets['Asset_Type'] == 'Crypto']

        # Filter by sector if preferences are set
        if preferred_sectors:
            # Ensure all sectors are strings and handle NaN values
            filtered_assets['Sector'] = filtered_assets['Sector'].fillna('Technology').astype(str)

            # Create a temporary lowercase column for comparison
            filtered_assets['Sector_Lower'] = filtered_assets['Sector'].str.lower()
            preferred_sectors_lower = [s.lower() for s in preferred_sectors]

            # Filter assets by selected sectors
            mask = filtered_assets['Sector_Lower'].isin(preferred_sectors_lower)
            filtered_assets = filtered_assets[mask]

            # Remove the temporary column
            filtered_assets = filtered_assets.drop('Sector_Lower', axis=1)

        # Generate recommendations
        if not filtered_assets.empty:
            filtered_assets = generate_recommendations(filtered_assets, user_preferences)
            st.markdown("### Your Personalized Recommendations")
            display_recommendations(filtered_assets)
        else:
            st.warning("No recommendations found for your selected criteria. Please try different sectors or asset types.")

        # Display overall performance metrics
        st.markdown("### Overall Portfolio Performance Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_roi = filtered_assets['ROI_1Y'].mean() if not filtered_assets.empty else 0
            st.metric("Expected Annual Return", f"{avg_roi:.1f}%")

        with col2:
            avg_volatility = filtered_assets['Volatility'].mean() if not filtered_assets.empty else 0
            st.metric("Portfolio Volatility", f"{avg_volatility:.2f}")

        with col3:
            avg_esg = filtered_assets['ESG_Score'].mean() if not filtered_assets.empty else 0
            st.metric("Average ESG Score", f"{avg_esg:.1f}")

        # Risk Assessment
        st.markdown("### Risk Assessment")
        risk_assessment = get_risk_assessment(filtered_assets)

        # Display risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_assessment['risk_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Portfolio Risk Level: {risk_assessment['risk_level']}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2196F3"},
                'steps': [
                    {'range': [0, 30], 'color': "#4CAF50"},
                    {'range': [30, 70], 'color': "#FFC107"},
                    {'range': [70, 100], 'color': "#F44336"}
                ],
            }
        ))

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        # Market insights section
        try:
            market_trends = generate_market_trends()
            display_market_insights(market_trends)
        except Exception as e:
            st.error(f"Error loading market trends: {str(e)}")
            st.markdown("""
                ### Market Trends Currently Unavailable
                We're experiencing issues loading market trends. Please check back later.
            """)

        # ESG Focus Areas with progress tracking
        st.markdown("## ESG Focus Areas")

        with st.spinner("Analyzing ESG focus areas..."):
            try:
                # Generate focus areas based on preferences
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

                # Create visualization
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

                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display descriptions
                for area in focus_areas:
                    st.markdown(f"""
                    <div class="insight-card" style="margin-bottom: 0.5rem;">
                        <strong>{area['area']}</strong>: {area['description']}
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating ESG focus areas: {str(e)}")
                st.markdown("""
                ### ESG Focus Areas Currently Unavailable
                We're experiencing issues generating ESG focus areas. Please try again later.
                """)

        # Call to action with error handling
        st.markdown("## Next Steps")
        try:
            st.markdown("""
            Based on these recommendations, you can:
            1. Explore the recommended assets in detail in the Market Explorer
            2. Add them to your portfolio in the Portfolio Manager
            3. Set up alerts for price changes or ESG developments
            4. Schedule a portfolio review to track performance against your goals
            """)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Explore Recommended Assets"):
                    st.session_state['selected_page'] = "Market Explorer"
                    st.experimental_rerun()

            with col2:
                if st.button("Add to Portfolio"):
                    st.session_state['selected_page'] = "Portfolio Manager"
                    st.experimental_rerun()

        except Exception as e:
            st.error(f"Error displaying next steps: {str(e)}")
            st.markdown("Please refresh the page to continue.")

else:
    # Show placeholder content with improved formatting
    st.markdown("""
    ## How Our AI Recommendation Engine Works

    Our AI-powered recommendation engine analyzes multiple data sources to provide personalized investment advice:

    <div class="insight-card">
        <h4>1. Financial Analysis</h4>
        <p>We evaluate traditional metrics like ROI, volatility, and market trends</p>
    </div>

    <div class="insight-card">
        <h4>2. ESG Integration</h4>
        <p>We incorporate environmental, social, and governance factors into our analysis</p>
    </div>

    <div class="insight-card">
        <h4>3. Risk Assessment</h4>
        <p>We calculate risk scores based on multiple factors, including market volatility and ESG risks</p>
    </div>

    <div class="insight-card">
        <h4>4. Personalization</h4>
        <p>We tailor recommendations to your specific preferences and goals</p>
    </div>

    <div class="insight-card">
        <h4>5. Continuous Learning</h4>
        <p>Our AI models continuously improve based on market developments and ESG trends</p>
    </div>

    Set your preferences above and click "Generate Personalized Recommendations" to receive tailored investment advice.
    """, unsafe_allow_html=True)

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

with st.expander("How do the AI recommendation models work?"):
    st.markdown("""
    Our AI recommendation system uses multiple machine learning models working together:

    1. **Portfolio Recommendation Model**: Uses gradient boosting to analyze assets based on financial and ESG metrics

    2. **Risk Assessment Model**: Employs random forest classification to evaluate portfolio risk across multiple dimensions

    3. **Sentiment Analysis Model**: Utilizes natural language processing to analyze market news and sentiment

    These models are dynamically adjusted based on your specific preferences:

    - **Risk Tolerance**: Affects the weight given to volatility, beta, and other risk metrics
    - **Sustainability Focus**: Adjusts the importance of environmental, social, and governance factors
    - **Investment Horizon**: Modifies the time-frame relevance of different metrics

    The models continuously learn and improve as more data becomes available, ensuring recommendations remain relevant and accurate.
    """)

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

