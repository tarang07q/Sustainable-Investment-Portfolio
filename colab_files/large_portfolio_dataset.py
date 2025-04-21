"""
Generate a large portfolio dataset for ML model demonstration
"""

import pandas as pd
import numpy as np
import random

def generate_large_dataset(num_stocks=350, num_crypto=150, seed=42):
    """
    Generate a large dataset of stocks and cryptocurrencies for ML model demonstration

    Args:
        num_stocks: Number of stock assets to generate
        num_crypto: Number of crypto assets to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with generated data
    """
    np.random.seed(seed)
    random.seed(seed)

    # Stock company name components
    prefixes = ['Green', 'Eco', 'Sustainable', 'Clean', 'Future', 'Smart', 'Renewable',
                'Circular', 'Bio', 'Carbon', 'Solar', 'Wind', 'Hydro', 'Geo', 'Ocean',
                'Ethical', 'Responsible', 'Progressive', 'Innovative', 'Advanced',
                'Natural', 'Organic', 'Pure', 'Efficient', 'Digital', 'Quantum', 'Nano',
                'Global', 'Urban', 'Rural', 'Agri', 'Aqua', 'Terra', 'Aero', 'Fusion',
                'Neuro', 'Cyber', 'Meta', 'Omni', 'Ultra']

    suffixes = ['Tech', 'Energy', 'Power', 'Solutions', 'Systems', 'Industries', 'Corp',
                'Inc', 'Group', 'Global', 'International', 'Ventures', 'Partners',
                'Networks', 'Innovations', 'Materials', 'Resources', 'Capital', 'Investments',
                'Dynamics', 'Nexus', 'Fusion', 'Logic', 'Metrics', 'Synergy', 'Alliance',
                'Collective', 'Foundation', 'Initiative', 'Labs', 'Works', 'Forge', 'Grid',
                'Hub', 'Sphere', 'Wave', 'Pulse', 'Stream', 'Cloud', 'Edge', 'Core']

    # Sectors
    stock_sectors = ['Technology', 'Energy', 'Finance', 'Healthcare', 'Retail',
                    'Utilities', 'Manufacturing', 'Transportation', 'Construction',
                    'Agriculture', 'Materials', 'Telecommunications', 'Consumer Goods',
                    'Pharmaceuticals', 'Aerospace', 'Automotive', 'Education', 'Real Estate',
                    'Food & Beverage', 'Media & Entertainment', 'Biotechnology', 'Renewable Energy',
                    'Sustainable Agriculture', 'Green Construction', 'Clean Transportation',
                    'Circular Economy', 'Water Management', 'Waste Management', 'Carbon Capture',
                    'Smart Cities', 'Sustainable Tourism', 'Green Chemistry', 'Eco-friendly Packaging',
                    'Sustainable Fashion', 'Plant-based Foods', 'Precision Agriculture']

    crypto_sectors = ['Renewable Energy', 'Green Technology', 'Sustainable Supply Chain',
                     'Carbon Credits', 'Future Tech', 'Sustainable Agriculture',
                     'Circular Economy', 'Marine Conservation', 'Biodiversity',
                     'Carbon Management', 'Forestry', 'Ethical Technology',
                     'Water Management', 'Solar Energy', 'Clean Energy',
                     'Decentralized Finance', 'Green Infrastructure', 'Sustainable Logistics',
                     'Climate Action', 'Conservation', 'Regenerative Agriculture',
                     'Ocean Cleanup', 'Plastic Reduction', 'Renewable Materials',
                     'Sustainable Mining', 'Green AI', 'Eco-friendly Computing',
                     'Sustainable Gaming', 'Carbon-neutral Blockchain', 'Green NFTs',
                     'Sustainable Metaverse', 'Eco-friendly Cloud', 'Green Data Centers',
                     'Sustainable IoT', 'Eco-friendly Wearables', 'Green Mobility']

    # Generate stock data
    stocks = []
    used_tickers = set()

    for i in range(num_stocks):
        # Generate unique company name and ticker
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        name = f"{prefix} {suffix}"

        # Generate ticker (2-4 letters)
        while True:
            ticker_length = random.randint(3, 4)
            if ticker_length == 3:
                ticker = prefix[:2].upper() + suffix[0].upper()
            else:
                ticker = prefix[:2].upper() + suffix[:2].upper()

            if ticker not in used_tickers:
                used_tickers.add(ticker)
                break

        # Generate other attributes
        sector = random.choice(stock_sectors)
        current_price = np.random.uniform(10, 500)
        purchase_price = current_price * np.random.uniform(0.8, 1.2)
        shares = np.random.randint(5, 100)

        # ESG scores - stocks tend to have higher ESG scores
        environmental_score = np.random.uniform(50, 95)
        social_score = np.random.uniform(40, 90)
        governance_score = np.random.uniform(60, 95)
        esg_score = (environmental_score + social_score + governance_score) / 3

        # Financial metrics
        volatility = np.random.uniform(0.1, 0.4)
        beta = np.random.uniform(0.5, 1.5)
        sharpe_ratio = np.random.uniform(0.8, 2.5)
        market_correlation = np.random.uniform(0.4, 0.8)
        roi_1y = np.random.uniform(5, 25)

        # Sustainability metrics
        carbon_footprint = np.random.uniform(10, 50)
        carbon_reduction_target = np.random.uniform(10, 50)

        # Calculate allocation (will be normalized later)
        allocation = shares * current_price

        stocks.append({
            'name': name,
            'ticker': ticker,
            'asset_type': 'Stock',
            'sector': sector,
            'shares': shares,
            'purchase_price': round(purchase_price, 2),
            'current_price': round(current_price, 2),
            'esg_score': round(esg_score, 1),
            'environmental_score': round(environmental_score, 1),
            'social_score': round(social_score, 1),
            'governance_score': round(governance_score, 1),
            'carbon_footprint': round(carbon_footprint, 1),
            'carbon_reduction_target': round(carbon_reduction_target, 1),
            'volatility': round(volatility, 2),
            'beta': round(beta, 1),
            'sharpe_ratio': round(sharpe_ratio, 1),
            'market_correlation': round(market_correlation, 2),
            'roi_1y': round(roi_1y, 1),
            'allocation': allocation,
            'price_change_24h': round(np.random.uniform(-5, 8), 1),
            'market_cap_b': round(np.random.uniform(1, 100), 1)
        })

    # Generate crypto data
    cryptos = []

    for i in range(num_crypto):
        # Generate unique crypto name and ticker
        prefix = random.choice(prefixes)
        suffix = random.choice(['Coin', 'Token', 'Chain', 'Credit', 'Cash', 'Pay', 'Ledger', 'Block'])
        name = f"{prefix}{suffix}"

        # Generate ticker (3-4 letters)
        while True:
            if len(prefix) >= 3:
                ticker = prefix[:3].upper()
            else:
                ticker = prefix.upper() + suffix[0].upper()

            if ticker not in used_tickers:
                used_tickers.add(ticker)
                break

        # Generate other attributes
        sector = random.choice(crypto_sectors)
        current_price = np.random.uniform(0.1, 2000)
        purchase_price = current_price * np.random.uniform(0.7, 1.3)
        shares = np.random.randint(10, 1000)

        # ESG scores - cryptos tend to have lower ESG scores due to energy usage
        environmental_score = np.random.uniform(30, 85)
        social_score = np.random.uniform(40, 80)
        governance_score = np.random.uniform(30, 75)
        esg_score = (environmental_score + social_score + governance_score) / 3

        # Financial metrics
        volatility = np.random.uniform(0.3, 0.8)
        beta = np.random.uniform(1.0, 2.5)
        sharpe_ratio = np.random.uniform(0.5, 2.0)
        market_correlation = np.random.uniform(0.3, 0.7)
        roi_1y = np.random.uniform(10, 100)

        # Sustainability metrics
        carbon_footprint = np.random.uniform(20, 100)
        carbon_reduction_target = np.random.uniform(5, 40)

        # Calculate allocation (will be normalized later)
        allocation = shares * current_price

        cryptos.append({
            'name': name,
            'ticker': ticker,
            'asset_type': 'Crypto',
            'sector': sector,
            'shares': shares,
            'purchase_price': round(purchase_price, 2),
            'current_price': round(current_price, 2),
            'esg_score': round(esg_score, 1),
            'environmental_score': round(environmental_score, 1),
            'social_score': round(social_score, 1),
            'governance_score': round(governance_score, 1),
            'carbon_footprint': round(carbon_footprint, 1),
            'carbon_reduction_target': round(carbon_reduction_target, 1),
            'volatility': round(volatility, 2),
            'beta': round(beta, 1),
            'sharpe_ratio': round(sharpe_ratio, 1),
            'market_correlation': round(market_correlation, 2),
            'roi_1y': round(roi_1y, 1),
            'allocation': allocation,
            'price_change_24h': round(np.random.uniform(-10, 15), 1),
            'market_cap_b': round(np.random.uniform(0.1, 50), 1)
        })

    # Combine stocks and cryptos
    all_assets = stocks + cryptos
    df = pd.DataFrame(all_assets)

    # Normalize allocations to sum to 1
    total_allocation = df['allocation'].sum()
    df['allocation'] = df['allocation'] / total_allocation

    # Add risk metrics
    df['esg_risk_score'] = 100 - df['esg_score']
    df['sector_volatility'] = df.groupby('sector')['volatility'].transform('mean')
    df['liquidity_ratio'] = np.random.uniform(0.5, 5.0, len(df))

    return df

# Generate and save the dataset
if __name__ == "__main__":
    large_df = generate_large_dataset()
    large_df.to_csv('large_portfolio_dataset.csv', index=False)
    print(f"Generated dataset with {len(large_df)} assets")

    # Print some statistics
    print(f"Number of stocks: {len(large_df[large_df['asset_type'] == 'Stock'])}")
    print(f"Number of cryptos: {len(large_df[large_df['asset_type'] == 'Crypto'])}")
    print(f"Average ESG score: {large_df['esg_score'].mean():.2f}")
    print(f"Average volatility: {large_df['volatility'].mean():.2f}")
