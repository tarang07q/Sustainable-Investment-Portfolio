"""
Generate a comprehensive portfolio dataset for ML model demonstration

This script generates a realistic dataset of stocks and cryptocurrencies with all the
necessary fields for ML models, including financial metrics, ESG scores, risk metrics,
and portfolio data. It also generates market news and SDG alignments.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

# SDG data for asset alignment
SDG_DATA = {
    1: {"name": "No Poverty", "description": "End poverty in all its forms everywhere", "related_sectors": ["Finance", "Agriculture", "All"]},
    2: {"name": "Zero Hunger", "description": "End hunger, achieve food security and improved nutrition", "related_sectors": ["Agriculture", "Food", "Sustainable Agriculture"]},
    3: {"name": "Good Health and Well-being", "description": "Ensure healthy lives and promote well-being for all", "related_sectors": ["Healthcare", "Pharmaceuticals"]},
    4: {"name": "Quality Education", "description": "Ensure inclusive and equitable quality education", "related_sectors": ["Education", "Technology"]},
    5: {"name": "Gender Equality", "description": "Achieve gender equality and empower all women and girls", "related_sectors": ["All"]},
    6: {"name": "Clean Water and Sanitation", "description": "Ensure availability and sustainable management of water", "related_sectors": ["Utilities", "Water Management"]},
    7: {"name": "Affordable and Clean Energy", "description": "Ensure access to affordable, reliable, sustainable energy", "related_sectors": ["Energy", "Renewable Energy", "Clean Energy"]},
    8: {"name": "Decent Work and Economic Growth", "description": "Promote sustained, inclusive economic growth", "related_sectors": ["Finance", "Technology", "All"]},
    9: {"name": "Industry, Innovation and Infrastructure", "description": "Build resilient infrastructure, promote inclusive industrialization", "related_sectors": ["Manufacturing", "Technology", "Construction"]},
    10: {"name": "Reduced Inequality", "description": "Reduce inequality within and among countries", "related_sectors": ["Finance", "Education", "All"]},
    11: {"name": "Sustainable Cities and Communities", "description": "Make cities inclusive, safe, resilient and sustainable", "related_sectors": ["Construction", "Transportation", "Utilities"]},
    12: {"name": "Responsible Consumption and Production", "description": "Ensure sustainable consumption and production patterns", "related_sectors": ["Retail", "Manufacturing", "All"]},
    13: {"name": "Climate Action", "description": "Take urgent action to combat climate change and its impacts", "related_sectors": ["Energy", "Transportation", "All"]},
    14: {"name": "Life Below Water", "description": "Conserve and sustainably use the oceans, seas and marine resources", "related_sectors": ["Marine Conservation", "Biodiversity"]},
    15: {"name": "Life on Land", "description": "Protect, restore and promote sustainable use of terrestrial ecosystems", "related_sectors": ["Forestry", "Agriculture", "Biodiversity"]},
    16: {"name": "Peace, Justice and Strong Institutions", "description": "Promote peaceful and inclusive societies for sustainable development", "related_sectors": ["Finance", "All"]},
    17: {"name": "Partnerships for the Goals", "description": "Strengthen the means of implementation and revitalize partnerships", "related_sectors": ["All"]}
}

# Market trends for news generation
SUSTAINABILITY_TRENDS = [
    {
        "title": "Renewable Energy Growth Accelerates",
        "description": "Investment in renewable energy sources is growing at an unprecedented rate, with solar and wind capacity additions outpacing fossil fuels.",
        "impact": "Positive for clean energy companies, negative for traditional fossil fuel producers",
        "confidence": 92,
        "related_sectors": ["Energy", "Renewable Energy", "Clean Energy"],
        "related_sdgs": [7, 13]
    },
    {
        "title": "ESG Reporting Standards Consolidation",
        "description": "Global ESG reporting frameworks are consolidating, with the International Sustainability Standards Board (ISSB) emerging as the leading standard.",
        "impact": "Positive for companies with strong ESG practices, challenging for laggards",
        "confidence": 85,
        "related_sectors": ["All"],
        "related_sdgs": [12, 17]
    },
    {
        "title": "Circular Economy Business Models Gain Traction",
        "description": "Companies adopting circular economy principles are seeing improved resource efficiency and new revenue streams from waste reduction.",
        "impact": "Positive for manufacturing and retail companies with circular initiatives",
        "confidence": 78,
        "related_sectors": ["Manufacturing", "Retail", "Circular Economy"],
        "related_sdgs": [9, 12]
    },
    {
        "title": "Biodiversity Loss Recognized as Financial Risk",
        "description": "Financial institutions are beginning to assess biodiversity loss as a material financial risk, similar to climate change.",
        "impact": "Negative for companies with high biodiversity impacts, positive for conservation solutions",
        "confidence": 80,
        "related_sectors": ["Finance", "Agriculture", "Forestry", "Marine Conservation", "Biodiversity"],
        "related_sdgs": [14, 15]
    },
    {
        "title": "Green Hydrogen Economy Emerges",
        "description": "Green hydrogen production costs are falling rapidly, opening up new applications in industry, transportation, and energy storage.",
        "impact": "Positive for clean energy, industrial decarbonization, and fuel cell technologies",
        "confidence": 88,
        "related_sectors": ["Energy", "Clean Energy", "Transportation", "Manufacturing"],
        "related_sdgs": [7, 9, 13]
    }
]

def generate_portfolio_dataset(num_stocks=20, num_crypto=10):
    """
    Generate a simplified dataset of stocks and cryptocurrencies

    Args:
        num_stocks: Number of stock assets to generate (default: 20)
        num_crypto: Number of crypto assets to generate (default: 10)

    Returns:
        DataFrame with generated data
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Pre-defined stock data
    stocks = [
        {"name": "Green Tech", "ticker": "GRTC", "sector": "Technology"},
        {"name": "Eco Energy", "ticker": "ECEN", "sector": "Energy"},
        {"name": "Sustainable Pharma", "ticker": "SUPH", "sector": "Healthcare"},
        {"name": "Clean Retail", "ticker": "CLRT", "sector": "Retail"},
        {"name": "Future Mobility", "ticker": "FUTM", "sector": "Transportation"},
        {"name": "Smart Agriculture", "ticker": "SMAG", "sector": "Agriculture"},
        {"name": "Renewable Power", "ticker": "RNWP", "sector": "Energy"},
        {"name": "Circular Economy", "ticker": "CRCE", "sector": "Manufacturing"},
        {"name": "Water Tech", "ticker": "WTRT", "sector": "Utilities"},
        {"name": "Sustainable Finance", "ticker": "SUFN", "sector": "Finance"},
        {"name": "Green Construction", "ticker": "GRCN", "sector": "Construction"},
        {"name": "Eco Transport", "ticker": "ECTR", "sector": "Transportation"},
        {"name": "Ocean Conservation", "ticker": "OCNC", "sector": "Marine Conservation"},
        {"name": "Biodiversity Fund", "ticker": "BIOD", "sector": "Biodiversity"},
        {"name": "Clean Air Tech", "ticker": "CAIR", "sector": "Clean Air"},
        {"name": "Sustainable Materials", "ticker": "SMAT", "sector": "Materials"},
        {"name": "Ethical AI Systems", "ticker": "EAIS", "sector": "Technology"},
        {"name": "Carbon Capture", "ticker": "CCAP", "sector": "Carbon Management"},
        {"name": "Sustainable Forestry", "ticker": "SFST", "sector": "Forestry"},
        {"name": "Green Hydrogen", "ticker": "GRHD", "sector": "Clean Energy"}
    ]

    # Pre-defined crypto data
    cryptos = [
        {"name": "GreenCoin", "ticker": "GRC", "sector": "Green Technology"},
        {"name": "EcoToken", "ticker": "ECO", "sector": "Renewable Energy"},
        {"name": "SustainChain", "ticker": "SUST", "sector": "Sustainable Supply Chain"},
        {"name": "CleanCrypto", "ticker": "CLNC", "sector": "Carbon Credits"},
        {"name": "FutureCoin", "ticker": "FUTC", "sector": "Future Tech"},
        {"name": "AgriToken", "ticker": "AGRT", "sector": "Sustainable Agriculture"},
        {"name": "RenewCoin", "ticker": "RNWC", "sector": "Renewable Energy"},
        {"name": "CircularToken", "ticker": "CIRC", "sector": "Circular Economy"},
        {"name": "OceanToken", "ticker": "OCNT", "sector": "Marine Conservation"},
        {"name": "BiodiversityCoin", "ticker": "BIOC", "sector": "Biodiversity"}
    ]

    # Limit to requested number
    stocks = stocks[:num_stocks]
    cryptos = cryptos[:num_crypto]

    # Generate additional data for stocks
    for stock in stocks:
        stock["asset_type"] = "Stock"
        stock["current_price"] = round(np.random.uniform(10, 500), 2)
        stock["price_change_24h"] = round(np.random.uniform(-5, 8), 1)
        stock["market_cap_b"] = round(np.random.uniform(1, 100), 1)
        stock["roi_1y"] = round(np.random.uniform(4, 25), 1)
        stock["volatility"] = round(np.random.uniform(0.1, 0.5), 2)

        # ESG Scores
        stock["environmental_score"] = round(np.random.uniform(50, 95), 1)
        stock["social_score"] = round(np.random.uniform(40, 90), 1)
        stock["governance_score"] = round(np.random.uniform(60, 95), 1)
        stock["esg_score"] = round((stock["environmental_score"] + stock["social_score"] + stock["governance_score"]) / 3, 1)

        # Additional metrics
        stock["beta"] = round(np.random.uniform(0.5, 1.5), 1)
        stock["sharpe_ratio"] = round(np.random.uniform(0.8, 2.5), 1)
        stock["market_correlation"] = round(np.random.uniform(0.4, 0.8), 2)
        stock["carbon_footprint"] = round(np.random.uniform(10, 50), 1)
        stock["carbon_reduction_target"] = round(np.random.uniform(10, 50), 1)

        # Portfolio data
        stock["shares"] = np.random.randint(5, 100)
        stock["purchase_price"] = round(stock["current_price"] * np.random.uniform(0.8, 1.2), 2)
        stock["allocation"] = stock["shares"] * stock["current_price"]

        # SDG Alignment - make it sector-specific
        sector = stock["sector"]
        sector_sdgs = []
        for sdg_num, sdg_data in SDG_DATA.items():
            if sector in sdg_data["related_sectors"] or "All" in sdg_data["related_sectors"]:
                sector_sdgs.append(sdg_num)

        # If we found sector-specific SDGs, use them (with some randomness)
        if sector_sdgs:
            # Take 2-4 SDGs that are relevant to the sector
            num_sdgs = min(len(sector_sdgs), random.randint(2, 4))
            stock["sdg_alignment"] = random.sample(sector_sdgs, k=num_sdgs)
        else:
            # Fallback to random SDGs if no sector match
            stock["sdg_alignment"] = random.sample(range(1, 18), k=random.randint(2, 3))

    # Generate additional data for cryptos
    for crypto in cryptos:
        crypto["asset_type"] = "Crypto"
        crypto["current_price"] = round(np.random.uniform(0.1, 2000), 2)
        crypto["price_change_24h"] = round(np.random.uniform(-10, 15), 1)
        crypto["market_cap_b"] = round(np.random.uniform(0.1, 50), 1)
        crypto["roi_1y"] = round(np.random.uniform(10, 100), 1)
        crypto["volatility"] = round(np.random.uniform(0.3, 0.8), 2)

        # ESG Scores - Cryptos typically have different ESG profiles
        crypto["environmental_score"] = round(np.random.uniform(30, 90), 1)
        crypto["social_score"] = round(np.random.uniform(50, 85), 1)
        crypto["governance_score"] = round(np.random.uniform(40, 80), 1)
        crypto["esg_score"] = round((crypto["environmental_score"] + crypto["social_score"] + crypto["governance_score"]) / 3, 1)

        # Additional metrics
        crypto["beta"] = round(np.random.uniform(1.0, 2.5), 1)
        crypto["sharpe_ratio"] = round(np.random.uniform(0.5, 2.0), 1)
        crypto["market_correlation"] = round(np.random.uniform(0.3, 0.7), 2)
        crypto["carbon_footprint"] = round(np.random.uniform(20, 100), 1)
        crypto["carbon_reduction_target"] = round(np.random.uniform(5, 40), 1)

        # Portfolio data
        crypto["shares"] = np.random.randint(10, 1000)
        crypto["purchase_price"] = round(crypto["current_price"] * np.random.uniform(0.7, 1.3), 2)
        crypto["allocation"] = crypto["shares"] * crypto["current_price"]

        # Define crypto-relevant SDGs
        crypto_relevant_sdgs = {
            "GreenCoin": [7, 13],  # Clean energy, Climate action
            "EcoToken": [12, 13, 15],  # Responsible consumption, Climate action, Life on land
            "SustainChain": [9, 12, 17],  # Industry/innovation, Responsible consumption, Partnerships
            "CleanCrypto": [7, 13],  # Clean energy, Climate action
            "FutureCoin": [9, 11],  # Industry/innovation, Sustainable cities
            "AgriToken": [2, 12, 15],  # Zero hunger, Responsible consumption, Life on land
            "RenewCoin": [7, 13],  # Clean energy, Climate action
            "CircularToken": [12],  # Responsible consumption
            "OceanToken": [14],  # Life below water
            "BiodiversityCoin": [14, 15]  # Life below water, Life on land
        }

        # Assign SDGs based on crypto name
        name = crypto["name"]
        if name in crypto_relevant_sdgs:
            # Add some randomness - don't always include all SDGs
            sdgs = crypto_relevant_sdgs[name]
            num_sdgs = min(len(sdgs), random.randint(1, len(sdgs)))
            crypto["sdg_alignment"] = random.sample(sdgs, k=num_sdgs)
        else:
            # Fallback
            crypto["sdg_alignment"] = random.sample([7, 9, 12, 13], k=random.randint(1, 2))

        # Energy Consumption (specific to crypto)
        crypto["energy_consumption"] = round(np.random.uniform(10, 1000), 1)
        crypto["renewable_energy_pct"] = round(np.random.uniform(10, 90), 1)

    # Combine stocks and cryptos
    all_assets = stocks + cryptos
    df = pd.DataFrame(all_assets)

    # Normalize allocations to sum to 1
    total_allocation = df["allocation"].sum()
    df["allocation"] = df["allocation"] / total_allocation

    # Add risk metrics
    df["esg_risk_score"] = 100 - df["esg_score"]
    df["sector_volatility"] = df.groupby("sector")["volatility"].transform("mean")
    df["liquidity_ratio"] = np.random.uniform(0.5, 5.0, len(df))

    return df

def generate_market_news(assets_df, num_news_items=50):
    """
    Generate synthetic market news for assets in the dataset

    Args:
        assets_df: DataFrame of assets
        num_news_items: Number of news items to generate

    Returns:
        DataFrame of news items
    """
    np.random.seed(42)
    random.seed(42)

    # News templates
    positive_templates = [
        "{company} reports strong quarterly earnings, exceeding analyst expectations",
        "{company} announces new sustainable initiative to reduce carbon footprint",
        "{company} expands into new markets with innovative products",
        "Analysts upgrade {company} stock to 'Buy' citing growth potential",
        "{company} partners with tech giant for next-generation solutions",
        "{company} increases dividend by 10%, signaling financial strength",
        "ESG rating agency upgrades {company}'s sustainability score"
    ]

    negative_templates = [
        "{company} misses earnings expectations, shares drop",
        "{company} faces regulatory scrutiny over environmental practices",
        "{company} announces layoffs amid restructuring efforts",
        "Analysts downgrade {company} stock citing competitive pressures",
        "{company} recalls products due to quality concerns",
        "{company} cuts dividend amid cash flow concerns",
        "ESG rating agency downgrades {company} citing governance issues"
    ]

    neutral_templates = [
        "{company} reports earnings in line with expectations",
        "{company} maintains current sustainability initiatives",
        "{company} announces leadership transition plan",
        "Analysts maintain 'Hold' rating for {company} stock",
        "{company} completes previously announced acquisition",
        "{company} maintains dividend at current levels",
        "{company}'s ESG rating remains unchanged in latest review"
    ]

    # News sources
    sources = ['Financial Times', 'Bloomberg', 'Reuters', 'CNBC', 'Wall Street Journal', 'MarketWatch']

    # Generate news items
    news_items = []

    for _ in range(num_news_items):
        # Select a random asset
        asset = assets_df.sample(1).iloc[0]
        company = asset['name']
        ticker = asset['ticker']

        # Determine sentiment (weighted towards neutral)
        sentiment_weights = [0.3, 0.3, 0.4]  # positive, negative, neutral
        sentiment = random.choices(['positive', 'negative', 'neutral'], weights=sentiment_weights)[0]

        # Select template based on sentiment
        if sentiment == 'positive':
            headline = random.choice(positive_templates).format(company=company)
        elif sentiment == 'negative':
            headline = random.choice(negative_templates).format(company=company)
        else:
            headline = random.choice(neutral_templates).format(company=company)

        # Generate publication date (within last 30 days)
        days_ago = random.randint(0, 30)
        pub_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        # Generate source
        source = random.choice(sources)

        news_items.append({
            'ticker': ticker,
            'company': company,
            'headline': headline,
            'sentiment': sentiment,
            'publication_date': pub_date,
            'source': source
        })

    return pd.DataFrame(news_items)

# Generate and save the dataset
if __name__ == "__main__":
    try:
        print("Generating portfolio dataset...")
        df = generate_portfolio_dataset()

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Save portfolio data
        df.to_csv("data/portfolio_dataset.csv", index=False)
        print(f"Successfully generated portfolio dataset with {len(df)} assets")

        # Generate and save market news
        print("Generating market news...")
        news_df = generate_market_news(df)
        news_df.to_csv("data/market_news.csv", index=False)
        print(f"Successfully generated {len(news_df)} market news items")

        # Save SDG data for reference
        with open("data/sdg_data.json", "w") as f:
            json.dump(SDG_DATA, f, indent=2)

        # Save sustainability trends for reference
        with open("data/sustainability_trends.json", "w") as f:
            json.dump(SUSTAINABILITY_TRENDS, f, indent=2)

        # Print some statistics
        print(f"\nPortfolio Statistics:")
        print(f"Number of stocks: {len(df[df['asset_type'] == 'Stock'])}")
        print(f"Number of cryptos: {len(df[df['asset_type'] == 'Crypto'])}")
        print(f"Average ESG score: {df['esg_score'].mean():.2f}")
        print(f"Average volatility: {df['volatility'].mean():.2f}")

        # Print sample data
        print("\nSample portfolio data (first 5 rows):")
        print(df[["name", "ticker", "asset_type", "current_price", "esg_score"]].head())

        print("\nSample news data (first 5 rows):")
        print(news_df[["ticker", "headline", "sentiment", "publication_date"]].head())

        print("\nAll data files saved to the 'data' directory.")

    except Exception as e:
        print(f"Error generating dataset: {str(e)}")
