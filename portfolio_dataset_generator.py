"""
Generate a comprehensive portfolio dataset for ML model demonstration

This script generates a realistic dataset of stocks and cryptocurrencies with all the
necessary fields for ML models, including financial metrics, ESG scores, risk metrics,
and portfolio data.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_portfolio_dataset(num_stocks=200, num_crypto=100, seed=42):
    """
    Generate a dataset of stocks and cryptocurrencies for ML model demonstration

    Args:
        num_stocks: Number of stock assets to generate
        num_crypto: Number of crypto assets to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with generated data
    """
    np.random.seed(seed)
    random.seed(seed)

    # Stock company name components for generating more stocks
    prefixes = ['Green', 'Eco', 'Sustainable', 'Clean', 'Future', 'Smart', 'Renewable',
                'Circular', 'Bio', 'Carbon', 'Solar', 'Wind', 'Hydro', 'Geo', 'Ocean',
                'Global', 'Advanced', 'Innovative', 'Modern', 'Next', 'Progressive', 'Strategic',
                'Integrated', 'Digital', 'Quantum', 'Nano', 'Micro', 'Macro', 'Meta', 'Hyper',
                'Ultra', 'Super', 'Mega', 'Giga', 'Tera', 'Peta', 'Exa', 'Zetta', 'Yotta']

    suffixes = ['Tech', 'Energy', 'Power', 'Solutions', 'Systems', 'Industries', 'Corp',
                'Inc', 'Group', 'Global', 'International', 'Ventures', 'Partners', 'Holdings',
                'Enterprises', 'Networks', 'Dynamics', 'Innovations', 'Labs', 'Research',
                'Development', 'Manufacturing', 'Products', 'Services', 'Consulting', 'Analytics',
                'Robotics', 'Automation', 'Intelligence', 'Computing', 'Data', 'Cloud', 'Mobile',
                'Communications', 'Telecom', 'Media', 'Entertainment', 'Health', 'Medical', 'Pharma']

    # Sectors
    sectors = ['Technology', 'Energy', 'Finance', 'Healthcare', 'Retail',
              'Utilities', 'Manufacturing', 'Transportation', 'Construction',
              'Agriculture', 'Materials', 'Telecommunications', 'Renewable Energy',
              'Sustainable Agriculture', 'Green Construction', 'Clean Energy',
              'Carbon Management', 'Forestry', 'Marine Conservation', 'Biodiversity',
              'Clean Air', 'Water Management', 'Waste Management', 'Circular Economy',
              'Education', 'Entertainment', 'Food Production', 'Mining', 'Oil & Gas',
              'Chemicals', 'Pharmaceuticals', 'Biotechnology', 'Aerospace', 'Defense',
              'Automotive', 'Real Estate', 'Hospitality', 'Tourism', 'Media', 'Publishing']

    # Pre-defined stock data (base set)
    base_stocks = [
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

    # Generate additional stocks
    stocks = base_stocks.copy()
    used_tickers = set([stock["ticker"] for stock in stocks])

    # Generate more stocks to reach the desired number
    while len(stocks) < num_stocks:
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        name = f"{prefix} {suffix}"

        # Generate ticker (2-4 letters)
        ticker_length = random.randint(3, 4)
        if ticker_length == 3:
            ticker = prefix[:2].upper() + suffix[0].upper()
        else:
            ticker = prefix[:2].upper() + suffix[:2].upper()

        # Skip if ticker already exists
        if ticker in used_tickers:
            continue

        used_tickers.add(ticker)
        sector = random.choice(sectors)

        stocks.append({"name": name, "ticker": ticker, "sector": sector})

    # Crypto name components for generating more cryptos
    crypto_prefixes = ['Green', 'Eco', 'Sustain', 'Clean', 'Future', 'Smart', 'Renew',
                      'Circular', 'Bio', 'Carbon', 'Solar', 'Wind', 'Hydro', 'Geo', 'Ocean',
                      'Global', 'Advanced', 'Innovative', 'Modern', 'Next', 'Progressive', 'Strategic',
                      'Digital', 'Quantum', 'Nano', 'Micro', 'Meta', 'Hyper', 'Ultra', 'Super',
                      'Mega', 'Giga', 'Terra', 'Luna', 'Cosmos', 'Galaxy', 'Star', 'Planet', 'Earth']

    crypto_suffixes = ['Coin', 'Token', 'Chain', 'Credit', 'Cash', 'Pay', 'Money', 'Finance',
                      'Swap', 'Exchange', 'Trade', 'Market', 'Wallet', 'Asset', 'Fund', 'Capital',
                      'Invest', 'Yield', 'Stake', 'Mine', 'Block', 'Ledger', 'Hash', 'Node',
                      'Net', 'Web', 'Protocol', 'Standard', 'Index', 'Metric', 'Data', 'AI',
                      'ML', 'Bot', 'Algo', 'Script', 'Code', 'Dev', 'Tech', 'System']

    # Crypto sectors
    crypto_sectors = ['Green Technology', 'Renewable Energy', 'Sustainable Supply Chain',
                     'Carbon Credits', 'Future Tech', 'Sustainable Agriculture',
                     'Circular Economy', 'Marine Conservation', 'Biodiversity',
                     'DeFi', 'NFT', 'Metaverse', 'Gaming', 'Social Media', 'Content Creation',
                     'Data Storage', 'Cloud Computing', 'AI & ML', 'IoT', 'Smart Contracts',
                     'Identity Verification', 'Supply Chain', 'Healthcare', 'Insurance',
                     'Real Estate', 'Energy Trading', 'Carbon Offsets', 'Climate Action',
                     'Water Management', 'Waste Management', 'Education', 'Charity']

    # Pre-defined crypto data (base set)
    base_cryptos = [
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

    # Generate additional cryptos
    cryptos = base_cryptos.copy()

    # Generate more cryptos to reach the desired number
    while len(cryptos) < num_crypto:
        prefix = random.choice(crypto_prefixes)
        suffix = random.choice(crypto_suffixes)
        name = f"{prefix}{suffix}"

        # Generate ticker (3-4 letters)
        if len(prefix) >= 3:
            ticker = prefix[:3].upper()
        else:
            ticker = prefix.upper() + suffix[0].upper()

        # Skip if ticker already exists
        if ticker in used_tickers:
            continue

        used_tickers.add(ticker)
        sector = random.choice(crypto_sectors)

        cryptos.append({"name": name, "ticker": ticker, "sector": sector})

    # Limit to requested number
    stocks = stocks[:num_stocks]
    cryptos = cryptos[:num_crypto]

    # Generate additional data for stocks
    for i, stock in enumerate(stocks):
        stock["asset_type"] = "Stock"

        # Add more variability and some negative scenarios
        # For the first 20% of stocks, create some underperforming assets
        if i < len(stocks) * 0.2:
            stock["current_price"] = round(np.random.uniform(1, 50), 2)  # Lower price range
            stock["price_change_24h"] = round(np.random.uniform(-15, -1), 1)  # Negative price change
            stock["market_cap_b"] = round(np.random.uniform(0.1, 5), 1)  # Smaller market cap
            stock["roi_1y"] = round(np.random.uniform(-20, 0), 1)  # Negative ROI
            stock["volatility"] = round(np.random.uniform(0.5, 0.9), 2)  # Higher volatility

            # Lower ESG scores for some sectors
            if stock["sector"] in ["Oil & Gas", "Mining", "Chemicals", "Defense"]:
                stock["environmental_score"] = round(np.random.uniform(10, 40), 1)
                stock["social_score"] = round(np.random.uniform(20, 50), 1)
                stock["governance_score"] = round(np.random.uniform(30, 60), 1)
            else:
                stock["environmental_score"] = round(np.random.uniform(30, 60), 1)
                stock["social_score"] = round(np.random.uniform(30, 70), 1)
                stock["governance_score"] = round(np.random.uniform(40, 75), 1)

            # Higher risk metrics
            stock["beta"] = round(np.random.uniform(1.5, 2.5), 1)
            stock["sharpe_ratio"] = round(np.random.uniform(0.1, 0.8), 1)
            stock["market_correlation"] = round(np.random.uniform(0.7, 0.9), 2)
            stock["carbon_footprint"] = round(np.random.uniform(50, 100), 1)
            stock["carbon_reduction_target"] = round(np.random.uniform(0, 10), 1)

        # For the next 60% of stocks, create average performing assets
        elif i < len(stocks) * 0.8:
            stock["current_price"] = round(np.random.uniform(10, 500), 2)
            stock["price_change_24h"] = round(np.random.uniform(-5, 8), 1)
            stock["market_cap_b"] = round(np.random.uniform(1, 100), 1)
            stock["roi_1y"] = round(np.random.uniform(-5, 15), 1)  # Some negative, mostly positive
            stock["volatility"] = round(np.random.uniform(0.2, 0.5), 2)

            # Average ESG scores
            stock["environmental_score"] = round(np.random.uniform(40, 80), 1)
            stock["social_score"] = round(np.random.uniform(40, 80), 1)
            stock["governance_score"] = round(np.random.uniform(50, 85), 1)

            # Average risk metrics
            stock["beta"] = round(np.random.uniform(0.7, 1.5), 1)
            stock["sharpe_ratio"] = round(np.random.uniform(0.8, 1.8), 1)
            stock["market_correlation"] = round(np.random.uniform(0.4, 0.7), 2)
            stock["carbon_footprint"] = round(np.random.uniform(20, 60), 1)
            stock["carbon_reduction_target"] = round(np.random.uniform(10, 30), 1)

        # For the last 20% of stocks, create high-performing sustainable assets
        else:
            stock["current_price"] = round(np.random.uniform(50, 1000), 2)
            stock["price_change_24h"] = round(np.random.uniform(1, 15), 1)  # All positive
            stock["market_cap_b"] = round(np.random.uniform(10, 500), 1)  # Larger market cap
            stock["roi_1y"] = round(np.random.uniform(15, 40), 1)  # High ROI
            stock["volatility"] = round(np.random.uniform(0.1, 0.3), 2)  # Lower volatility

            # High ESG scores
            stock["environmental_score"] = round(np.random.uniform(75, 98), 1)
            stock["social_score"] = round(np.random.uniform(70, 95), 1)
            stock["governance_score"] = round(np.random.uniform(80, 98), 1)

            # Better risk metrics
            stock["beta"] = round(np.random.uniform(0.5, 1.2), 1)
            stock["sharpe_ratio"] = round(np.random.uniform(1.8, 3.0), 1)
            stock["market_correlation"] = round(np.random.uniform(0.3, 0.6), 2)
            stock["carbon_footprint"] = round(np.random.uniform(5, 25), 1)
            stock["carbon_reduction_target"] = round(np.random.uniform(30, 70), 1)

        # Calculate ESG score as average of components
        stock["esg_score"] = round((stock["environmental_score"] + stock["social_score"] + stock["governance_score"]) / 3, 1)

        # Portfolio data
        stock["shares"] = np.random.randint(5, 1000)
        stock["purchase_price"] = round(stock["current_price"] * np.random.uniform(0.7, 1.3), 2)
        stock["allocation"] = stock["shares"] * stock["current_price"]

        # SDG Alignment (stored as a list)
        if stock["esg_score"] > 70:
            # High ESG stocks align with more SDGs
            stock["sdg_alignment"] = random.sample(range(1, 18), k=random.randint(3, 6))
        elif stock["esg_score"] > 50:
            # Medium ESG stocks align with some SDGs
            stock["sdg_alignment"] = random.sample(range(1, 18), k=random.randint(1, 3))
        else:
            # Low ESG stocks align with fewer SDGs
            stock["sdg_alignment"] = random.sample(range(1, 18), k=random.randint(0, 2))

    # Generate additional data for cryptos
    for i, crypto in enumerate(cryptos):
        crypto["asset_type"] = "Crypto"

        # Add more variability and some negative scenarios
        # For the first 30% of cryptos, create some underperforming/high-risk assets
        if i < len(cryptos) * 0.3:
            crypto["current_price"] = round(np.random.uniform(0.001, 5), 3)  # Very low price
            crypto["price_change_24h"] = round(np.random.uniform(-30, -5), 1)  # Significant negative change
            crypto["market_cap_b"] = round(np.random.uniform(0.01, 1), 2)  # Very small market cap
            crypto["roi_1y"] = round(np.random.uniform(-70, -10), 1)  # Negative ROI
            crypto["volatility"] = round(np.random.uniform(0.8, 1.5), 2)  # Extreme volatility

            # Poor ESG scores - especially environmental due to energy usage
            crypto["environmental_score"] = round(np.random.uniform(10, 30), 1)
            crypto["social_score"] = round(np.random.uniform(20, 50), 1)
            crypto["governance_score"] = round(np.random.uniform(10, 40), 1)

            # High risk metrics
            crypto["beta"] = round(np.random.uniform(2.0, 4.0), 1)
            crypto["sharpe_ratio"] = round(np.random.uniform(0.1, 0.5), 1)
            crypto["market_correlation"] = round(np.random.uniform(0.1, 0.4), 2)  # Low correlation

            # High energy consumption, low renewable
            crypto["energy_consumption"] = round(np.random.uniform(500, 2000), 1)
            crypto["renewable_energy_pct"] = round(np.random.uniform(1, 20), 1)
            crypto["carbon_footprint"] = round(np.random.uniform(80, 200), 1)
            crypto["carbon_reduction_target"] = round(np.random.uniform(0, 10), 1)

        # For the next 40% of cryptos, create average/mixed performance
        elif i < len(cryptos) * 0.7:
            crypto["current_price"] = round(np.random.uniform(1, 500), 2)
            crypto["price_change_24h"] = round(np.random.uniform(-15, 20), 1)  # Mixed performance
            crypto["market_cap_b"] = round(np.random.uniform(0.5, 20), 1)
            crypto["roi_1y"] = round(np.random.uniform(-30, 50), 1)  # Mixed ROI
            crypto["volatility"] = round(np.random.uniform(0.5, 0.9), 2)  # High volatility

            # Mixed ESG scores
            crypto["environmental_score"] = round(np.random.uniform(30, 60), 1)
            crypto["social_score"] = round(np.random.uniform(40, 70), 1)
            crypto["governance_score"] = round(np.random.uniform(30, 70), 1)

            # Mixed risk metrics
            crypto["beta"] = round(np.random.uniform(1.2, 2.5), 1)
            crypto["sharpe_ratio"] = round(np.random.uniform(0.4, 1.2), 1)
            crypto["market_correlation"] = round(np.random.uniform(0.3, 0.6), 2)

            # Moderate energy metrics
            crypto["energy_consumption"] = round(np.random.uniform(100, 800), 1)
            crypto["renewable_energy_pct"] = round(np.random.uniform(20, 60), 1)
            crypto["carbon_footprint"] = round(np.random.uniform(40, 100), 1)
            crypto["carbon_reduction_target"] = round(np.random.uniform(10, 30), 1)

        # For the last 30% of cryptos, create high-performing sustainable cryptos
        else:
            crypto["current_price"] = round(np.random.uniform(100, 5000), 2)  # Higher price
            crypto["price_change_24h"] = round(np.random.uniform(5, 40), 1)  # Strong positive change
            crypto["market_cap_b"] = round(np.random.uniform(10, 200), 1)  # Larger market cap
            crypto["roi_1y"] = round(np.random.uniform(50, 300), 1)  # High ROI
            crypto["volatility"] = round(np.random.uniform(0.3, 0.6), 2)  # Lower volatility (for crypto)

            # Better ESG scores - sustainable cryptos
            crypto["environmental_score"] = round(np.random.uniform(60, 90), 1)
            crypto["social_score"] = round(np.random.uniform(65, 90), 1)
            crypto["governance_score"] = round(np.random.uniform(70, 95), 1)

            # Better risk metrics
            crypto["beta"] = round(np.random.uniform(0.8, 1.8), 1)
            crypto["sharpe_ratio"] = round(np.random.uniform(1.2, 3.0), 1)
            crypto["market_correlation"] = round(np.random.uniform(0.5, 0.8), 2)

            # Low energy consumption, high renewable
            crypto["energy_consumption"] = round(np.random.uniform(10, 200), 1)
            crypto["renewable_energy_pct"] = round(np.random.uniform(60, 100), 1)
            crypto["carbon_footprint"] = round(np.random.uniform(5, 40), 1)
            crypto["carbon_reduction_target"] = round(np.random.uniform(30, 80), 1)

        # Calculate ESG score as average of components
        crypto["esg_score"] = round((crypto["environmental_score"] + crypto["social_score"] + crypto["governance_score"]) / 3, 1)

        # Portfolio data
        crypto["shares"] = np.random.randint(10, 10000)
        crypto["purchase_price"] = round(crypto["current_price"] * np.random.uniform(0.5, 2.0), 2)  # More price volatility in purchase
        crypto["allocation"] = crypto["shares"] * crypto["current_price"]

        # SDG Alignment based on ESG score
        if crypto["esg_score"] > 70:
            # High ESG cryptos align with more SDGs
            crypto["sdg_alignment"] = random.sample(range(1, 18), k=random.randint(2, 5))
        elif crypto["esg_score"] > 40:
            # Medium ESG cryptos align with some SDGs
            crypto["sdg_alignment"] = random.sample(range(1, 18), k=random.randint(1, 3))
        else:
            # Low ESG cryptos align with fewer or no SDGs
            sdg_count = random.randint(0, 1)
            if sdg_count == 0:
                crypto["sdg_alignment"] = []
            else:
                crypto["sdg_alignment"] = random.sample(range(1, 18), k=1)

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

def generate_market_news(assets_df, num_news_items=500, seed=42):
    """
    Generate synthetic market news for assets in the dataset

    Args:
        assets_df: DataFrame of assets
        num_news_items: Number of news items to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame of news items
    """
    np.random.seed(seed)
    random.seed(seed)

    # News templates - expanded with more varied news
    positive_templates = [
        "{company} reports strong quarterly earnings, exceeding analyst expectations",
        "{company} announces new sustainable initiative to reduce carbon footprint",
        "{company} expands into new markets with innovative products",
        "Analysts upgrade {company} stock to 'Buy' citing growth potential",
        "{company} partners with tech giant for next-generation solutions",
        "{company} increases dividend by 10%, signaling financial strength",
        "ESG rating agency upgrades {company}'s sustainability score",
        "{company} secures major government contract worth millions",
        "{company} launches breakthrough renewable energy technology",
        "{company} achieves carbon neutrality ahead of schedule",
        "{company} reports record-breaking user growth in Q2",
        "{company} stock surges after positive clinical trial results",
        "{company} announces strategic acquisition to expand market share",
        "{company} joins sustainability coalition with industry leaders",
        "{company} receives industry award for ethical business practices",
        "{company} exceeds renewable energy targets for third consecutive year",
        "{company} announces share buyback program, boosting investor confidence",
        "{company} reports significant reduction in waste production",
        "{company} partners with UN for sustainable development initiatives",
        "{company} stock hits all-time high on strong growth forecast"
    ]

    negative_templates = [
        "{company} misses earnings expectations, shares drop",
        "{company} faces regulatory scrutiny over environmental practices",
        "{company} announces layoffs amid restructuring efforts",
        "Analysts downgrade {company} stock citing competitive pressures",
        "{company} recalls products due to quality concerns",
        "{company} cuts dividend amid cash flow concerns",
        "ESG rating agency downgrades {company} citing governance issues",
        "{company} faces class-action lawsuit from investors",
        "{company} under investigation for potential accounting irregularities",
        "{company} CEO resigns amid controversy, stock plummets",
        "{company} reports significant data breach affecting millions of users",
        "{company} factory shutdown due to environmental violations",
        "{company} loses major contract to competitor, shares tumble",
        "{company} profit margins shrink due to rising material costs",
        "{company} faces backlash over controversial marketing campaign",
        "{company} delays product launch, disappointing investors",
        "{company} reports higher than expected carbon emissions",
        "{company} faces supply chain disruptions affecting production",
        "{company} announces unexpected quarterly loss, shares plunge",
        "{company} executive team faces scrutiny over excessive compensation"
    ]

    neutral_templates = [
        "{company} reports earnings in line with expectations",
        "{company} maintains current sustainability initiatives",
        "{company} announces leadership transition plan",
        "Analysts maintain 'Hold' rating for {company} stock",
        "{company} completes previously announced acquisition",
        "{company} maintains dividend at current levels",
        "{company}'s ESG rating remains unchanged in latest review",
        "{company} hosts annual shareholder meeting, no major announcements",
        "{company} reaffirms annual guidance despite market uncertainty",
        "{company} releases sustainability report with mixed results",
        "{company} announces routine executive reshuffle",
        "{company} maintains market share despite increased competition",
        "{company} completes scheduled facility maintenance",
        "{company} postpones decision on expansion plans",
        "{company} reports stable user growth, meeting expectations",
        "{company} refinances debt with similar terms",
        "{company} makes minor updates to product lineup",
        "{company} continues research partnership with no new breakthroughs",
        "{company} maintains pricing despite industry pressure",
        "{company} reports quarterly results with no significant surprises"
    ]

    # News sources - expanded list
    sources = ['Financial Times', 'Bloomberg', 'Reuters', 'CNBC', 'Wall Street Journal', 'MarketWatch',
              'The Economist', 'Forbes', 'Business Insider', 'Yahoo Finance', 'Barron\'s', 'Morningstar',
              'Seeking Alpha', 'The Street', 'Investopedia', 'ESG Today', 'Sustainable Investing',
              'Green Finance News', 'Climate Tech Review', 'Crypto Daily', 'DeFi Pulse']

    # Generate news items
    news_items = []

    for _ in range(num_news_items):
        # Select a random asset
        asset = assets_df.sample(1).iloc[0]
        company = asset['name']
        ticker = asset['ticker']

        # Determine sentiment based on asset performance and sector
        # Adjust sentiment weights based on asset characteristics
        if asset['asset_type'] == 'Stock':
            # For stocks with negative ROI, more likely to have negative news
            if asset['roi_1y'] < 0:
                sentiment_weights = [0.2, 0.5, 0.3]  # positive, negative, neutral
            # For stocks with high ESG scores, more likely to have positive news
            elif asset['esg_score'] > 75:
                sentiment_weights = [0.5, 0.2, 0.3]  # positive, negative, neutral
            # For high volatility stocks, more mixed news
            elif asset['volatility'] > 0.6:
                sentiment_weights = [0.3, 0.4, 0.3]  # positive, negative, neutral
            # Default for average stocks
            else:
                sentiment_weights = [0.3, 0.3, 0.4]  # positive, negative, neutral
        else:  # Crypto
            # For cryptos with negative ROI, more likely to have negative news
            if asset['roi_1y'] < 0:
                sentiment_weights = [0.1, 0.6, 0.3]  # positive, negative, neutral
            # For cryptos with high price change, more likely to have extreme news
            elif abs(asset['price_change_24h']) > 20:
                if asset['price_change_24h'] > 0:
                    sentiment_weights = [0.6, 0.2, 0.2]  # positive, negative, neutral
                else:
                    sentiment_weights = [0.2, 0.6, 0.2]  # positive, negative, neutral
            # For high ESG score cryptos (sustainable), more positive news
            elif asset['esg_score'] > 70:
                sentiment_weights = [0.5, 0.2, 0.3]  # positive, negative, neutral
            # Default for average cryptos
            else:
                sentiment_weights = [0.3, 0.4, 0.3]  # positive, negative, neutral

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
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    print("Generating portfolio dataset...")
    portfolio_df = generate_portfolio_dataset()
    portfolio_df.to_csv("data/portfolio_dataset.csv", index=False)
    print(f"Successfully generated portfolio dataset with {len(portfolio_df)} assets")

    print("Generating market news...")
    news_df = generate_market_news(portfolio_df)
    news_df.to_csv("data/market_news.csv", index=False)
    print(f"Successfully generated {len(news_df)} market news items")

    # Print some statistics
    print(f"\nPortfolio Statistics:")
    print(f"Number of stocks: {len(portfolio_df[portfolio_df['asset_type'] == 'Stock'])}")
    print(f"Number of cryptos: {len(portfolio_df[portfolio_df['asset_type'] == 'Crypto'])}")
    print(f"Average ESG score: {portfolio_df['esg_score'].mean():.2f}")
    print(f"Average volatility: {portfolio_df['volatility'].mean():.2f}")

    # Print sample data
    print("\nSample portfolio data (first 5 rows):")
    print(portfolio_df[["name", "ticker", "asset_type", "current_price", "esg_score"]].head())

    print("\nSample news data (first 5 rows):")
    print(news_df[["ticker", "headline", "sentiment", "publication_date"]].head())

    print("\nAll data files saved to the 'data' directory.")
