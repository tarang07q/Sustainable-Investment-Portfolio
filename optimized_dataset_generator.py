"""
Optimized dataset generator for sustainable investment portfolio analysis
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
import time

def generate_portfolio_dataset(num_stocks=100, num_crypto=50, seed=42):
    """
    Generate a dataset of stocks and cryptocurrencies with optimized performance
    
    Args:
        num_stocks: Number of stock assets to generate
        num_crypto: Number of crypto assets to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with generated data
    """
    print(f"Generating {num_stocks} stocks and {num_crypto} cryptocurrencies...")
    start_time = time.time()
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Stock name components
    prefixes = ['Green', 'Eco', 'Sustainable', 'Clean', 'Future', 'Smart', 'Renewable', 
                'Circular', 'Bio', 'Carbon', 'Solar', 'Wind', 'Hydro', 'Geo', 'Ocean',
                'Global', 'Advanced', 'Innovative', 'Modern', 'Next', 'Progressive']
    
    suffixes = ['Tech', 'Energy', 'Power', 'Solutions', 'Systems', 'Industries', 'Corp', 
                'Inc', 'Group', 'Global', 'International', 'Ventures', 'Partners', 'Holdings',
                'Enterprises', 'Networks', 'Dynamics', 'Innovations', 'Labs', 'Research']
    
    # Sectors
    sectors = ['Technology', 'Energy', 'Finance', 'Healthcare', 'Retail', 
              'Utilities', 'Manufacturing', 'Transportation', 'Construction', 
              'Agriculture', 'Materials', 'Telecommunications', 'Renewable Energy',
              'Sustainable Agriculture', 'Green Construction', 'Clean Energy',
              'Carbon Management', 'Forestry', 'Marine Conservation', 'Biodiversity']
    
    # Generate stock data
    stocks = []
    used_tickers = set()
    
    # Generate stocks
    for i in range(num_stocks):
        # Progress indicator
        if i % 50 == 0 and i > 0:
            print(f"Generated {i} stocks...")
        
        # Generate company name and ticker
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        name = f"{prefix} {suffix}"
        
        # Generate ticker (3-4 letters)
        ticker_length = random.randint(3, 4)
        if ticker_length == 3:
            ticker = prefix[:2].upper() + suffix[0].upper()
        else:
            ticker = prefix[:2].upper() + suffix[:2].upper()
        
        # Skip if ticker already exists
        if ticker in used_tickers:
            ticker = prefix[:1].upper() + suffix[:2].upper() + str(random.randint(1, 9))
        
        used_tickers.add(ticker)
        sector = random.choice(sectors)
        
        # Determine performance category
        if i < num_stocks * 0.2:  # 20% underperforming
            performance_category = "underperforming"
        elif i < num_stocks * 0.8:  # 60% average
            performance_category = "average"
        else:  # 20% high-performing
            performance_category = "high-performing"
        
        # Generate data based on performance category
        if performance_category == "underperforming":
            current_price = round(np.random.uniform(1, 50), 2)
            price_change_24h = round(np.random.uniform(-15, -1), 1)
            market_cap_b = round(np.random.uniform(0.1, 5), 1)
            roi_1y = round(np.random.uniform(-20, 0), 1)
            volatility = round(np.random.uniform(0.5, 0.9), 2)
            
            # Lower ESG scores for some sectors
            if sector in ["Oil & Gas", "Mining", "Chemicals"]:
                environmental_score = round(np.random.uniform(10, 40), 1)
                social_score = round(np.random.uniform(20, 50), 1)
                governance_score = round(np.random.uniform(30, 60), 1)
            else:
                environmental_score = round(np.random.uniform(30, 60), 1)
                social_score = round(np.random.uniform(30, 70), 1)
                governance_score = round(np.random.uniform(40, 75), 1)
            
            beta = round(np.random.uniform(1.5, 2.5), 1)
            sharpe_ratio = round(np.random.uniform(0.1, 0.8), 1)
            market_correlation = round(np.random.uniform(0.7, 0.9), 2)
            carbon_footprint = round(np.random.uniform(50, 100), 1)
            carbon_reduction_target = round(np.random.uniform(0, 10), 1)
            
        elif performance_category == "average":
            current_price = round(np.random.uniform(10, 500), 2)
            price_change_24h = round(np.random.uniform(-5, 8), 1)
            market_cap_b = round(np.random.uniform(1, 100), 1)
            roi_1y = round(np.random.uniform(-5, 15), 1)
            volatility = round(np.random.uniform(0.2, 0.5), 2)
            
            environmental_score = round(np.random.uniform(40, 80), 1)
            social_score = round(np.random.uniform(40, 80), 1)
            governance_score = round(np.random.uniform(50, 85), 1)
            
            beta = round(np.random.uniform(0.7, 1.5), 1)
            sharpe_ratio = round(np.random.uniform(0.8, 1.8), 1)
            market_correlation = round(np.random.uniform(0.4, 0.7), 2)
            carbon_footprint = round(np.random.uniform(20, 60), 1)
            carbon_reduction_target = round(np.random.uniform(10, 30), 1)
            
        else:  # high-performing
            current_price = round(np.random.uniform(50, 1000), 2)
            price_change_24h = round(np.random.uniform(1, 15), 1)
            market_cap_b = round(np.random.uniform(10, 500), 1)
            roi_1y = round(np.random.uniform(15, 40), 1)
            volatility = round(np.random.uniform(0.1, 0.3), 2)
            
            environmental_score = round(np.random.uniform(75, 98), 1)
            social_score = round(np.random.uniform(70, 95), 1)
            governance_score = round(np.random.uniform(80, 98), 1)
            
            beta = round(np.random.uniform(0.5, 1.2), 1)
            sharpe_ratio = round(np.random.uniform(1.8, 3.0), 1)
            market_correlation = round(np.random.uniform(0.3, 0.6), 2)
            carbon_footprint = round(np.random.uniform(5, 25), 1)
            carbon_reduction_target = round(np.random.uniform(30, 70), 1)
        
        # Calculate ESG score
        esg_score = round((environmental_score + social_score + governance_score) / 3, 1)
        
        # Portfolio data
        shares = np.random.randint(5, 1000)
        purchase_price = round(current_price * np.random.uniform(0.7, 1.3), 2)
        allocation = shares * current_price
        
        # SDG Alignment
        if esg_score > 70:
            sdg_alignment = random.sample(range(1, 18), k=random.randint(3, 6))
        elif esg_score > 50:
            sdg_alignment = random.sample(range(1, 18), k=random.randint(1, 3))
        else:
            sdg_alignment = random.sample(range(1, 18), k=random.randint(0, 2))
        
        # Create stock record
        stock = {
            "name": name,
            "ticker": ticker,
            "sector": sector,
            "asset_type": "Stock",
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "market_cap_b": market_cap_b,
            "roi_1y": roi_1y,
            "volatility": volatility,
            "environmental_score": environmental_score,
            "social_score": social_score,
            "governance_score": governance_score,
            "esg_score": esg_score,
            "beta": beta,
            "sharpe_ratio": sharpe_ratio,
            "market_correlation": market_correlation,
            "carbon_footprint": carbon_footprint,
            "carbon_reduction_target": carbon_reduction_target,
            "shares": shares,
            "purchase_price": purchase_price,
            "allocation": allocation,
            "sdg_alignment": sdg_alignment
        }
        
        stocks.append(stock)
    
    # Generate crypto data
    cryptos = []
    crypto_prefixes = ['Green', 'Eco', 'Sustain', 'Clean', 'Future', 'Smart', 'Renew', 
                      'Circular', 'Bio', 'Carbon', 'Solar', 'Wind', 'Hydro', 'Geo', 'Ocean']
    
    crypto_suffixes = ['Coin', 'Token', 'Chain', 'Credit', 'Cash', 'Pay', 'Money', 'Finance',
                      'Swap', 'Exchange', 'Trade', 'Market', 'Wallet', 'Asset', 'Fund']
    
    crypto_sectors = ['Green Technology', 'Renewable Energy', 'Sustainable Supply Chain',
                     'Carbon Credits', 'Future Tech', 'Sustainable Agriculture',
                     'Circular Economy', 'Marine Conservation', 'Biodiversity',
                     'DeFi', 'NFT', 'Metaverse', 'Gaming', 'Social Media', 'Content Creation']
    
    # Generate cryptos
    for i in range(num_crypto):
        # Progress indicator
        if i % 25 == 0 and i > 0:
            print(f"Generated {i} cryptocurrencies...")
        
        # Generate crypto name and ticker
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
            ticker = prefix[:1].upper() + suffix[:1].upper() + str(random.randint(10, 99))
        
        used_tickers.add(ticker)
        sector = random.choice(crypto_sectors)
        
        # Determine performance category
        if i < num_crypto * 0.3:  # 30% underperforming
            performance_category = "underperforming"
        elif i < num_crypto * 0.7:  # 40% average
            performance_category = "average"
        else:  # 30% high-performing
            performance_category = "high-performing"
        
        # Generate data based on performance category
        if performance_category == "underperforming":
            current_price = round(np.random.uniform(0.001, 5), 3)
            price_change_24h = round(np.random.uniform(-30, -5), 1)
            market_cap_b = round(np.random.uniform(0.01, 1), 2)
            roi_1y = round(np.random.uniform(-70, -10), 1)
            volatility = round(np.random.uniform(0.8, 1.5), 2)
            
            environmental_score = round(np.random.uniform(10, 30), 1)
            social_score = round(np.random.uniform(20, 50), 1)
            governance_score = round(np.random.uniform(10, 40), 1)
            
            beta = round(np.random.uniform(2.0, 4.0), 1)
            sharpe_ratio = round(np.random.uniform(0.1, 0.5), 1)
            market_correlation = round(np.random.uniform(0.1, 0.4), 2)
            
            energy_consumption = round(np.random.uniform(500, 2000), 1)
            renewable_energy_pct = round(np.random.uniform(1, 20), 1)
            carbon_footprint = round(np.random.uniform(80, 200), 1)
            carbon_reduction_target = round(np.random.uniform(0, 10), 1)
            
        elif performance_category == "average":
            current_price = round(np.random.uniform(1, 500), 2)
            price_change_24h = round(np.random.uniform(-15, 20), 1)
            market_cap_b = round(np.random.uniform(0.5, 20), 1)
            roi_1y = round(np.random.uniform(-30, 50), 1)
            volatility = round(np.random.uniform(0.5, 0.9), 2)
            
            environmental_score = round(np.random.uniform(30, 60), 1)
            social_score = round(np.random.uniform(40, 70), 1)
            governance_score = round(np.random.uniform(30, 70), 1)
            
            beta = round(np.random.uniform(1.2, 2.5), 1)
            sharpe_ratio = round(np.random.uniform(0.4, 1.2), 1)
            market_correlation = round(np.random.uniform(0.3, 0.6), 2)
            
            energy_consumption = round(np.random.uniform(100, 800), 1)
            renewable_energy_pct = round(np.random.uniform(20, 60), 1)
            carbon_footprint = round(np.random.uniform(40, 100), 1)
            carbon_reduction_target = round(np.random.uniform(10, 30), 1)
            
        else:  # high-performing
            current_price = round(np.random.uniform(100, 5000), 2)
            price_change_24h = round(np.random.uniform(5, 40), 1)
            market_cap_b = round(np.random.uniform(10, 200), 1)
            roi_1y = round(np.random.uniform(50, 300), 1)
            volatility = round(np.random.uniform(0.3, 0.6), 2)
            
            environmental_score = round(np.random.uniform(60, 90), 1)
            social_score = round(np.random.uniform(65, 90), 1)
            governance_score = round(np.random.uniform(70, 95), 1)
            
            beta = round(np.random.uniform(0.8, 1.8), 1)
            sharpe_ratio = round(np.random.uniform(1.2, 3.0), 1)
            market_correlation = round(np.random.uniform(0.5, 0.8), 2)
            
            energy_consumption = round(np.random.uniform(10, 200), 1)
            renewable_energy_pct = round(np.random.uniform(60, 100), 1)
            carbon_footprint = round(np.random.uniform(5, 40), 1)
            carbon_reduction_target = round(np.random.uniform(30, 80), 1)
        
        # Calculate ESG score
        esg_score = round((environmental_score + social_score + governance_score) / 3, 1)
        
        # Portfolio data
        shares = np.random.randint(10, 10000)
        purchase_price = round(current_price * np.random.uniform(0.5, 2.0), 2)
        allocation = shares * current_price
        
        # SDG Alignment
        if esg_score > 70:
            sdg_alignment = random.sample(range(1, 18), k=random.randint(2, 5))
        elif esg_score > 40:
            sdg_alignment = random.sample(range(1, 18), k=random.randint(1, 3))
        else:
            sdg_count = random.randint(0, 1)
            if sdg_count == 0:
                sdg_alignment = []
            else:
                sdg_alignment = random.sample(range(1, 18), k=1)
        
        # Create crypto record
        crypto = {
            "name": name,
            "ticker": ticker,
            "sector": sector,
            "asset_type": "Crypto",
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "market_cap_b": market_cap_b,
            "roi_1y": roi_1y,
            "volatility": volatility,
            "environmental_score": environmental_score,
            "social_score": social_score,
            "governance_score": governance_score,
            "esg_score": esg_score,
            "beta": beta,
            "sharpe_ratio": sharpe_ratio,
            "market_correlation": market_correlation,
            "carbon_footprint": carbon_footprint,
            "carbon_reduction_target": carbon_reduction_target,
            "shares": shares,
            "purchase_price": purchase_price,
            "allocation": allocation,
            "sdg_alignment": sdg_alignment,
            "energy_consumption": energy_consumption,
            "renewable_energy_pct": renewable_energy_pct
        }
        
        cryptos.append(crypto)
    
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
    
    elapsed_time = time.time() - start_time
    print(f"Portfolio dataset generation completed in {elapsed_time:.2f} seconds")
    
    return df

def generate_market_news(assets_df, num_news_items=200, seed=42):
    """
    Generate synthetic market news for assets in the dataset
    
    Args:
        assets_df: DataFrame of assets
        num_news_items: Number of news items to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame of news items
    """
    print(f"Generating {num_news_items} market news items...")
    start_time = time.time()
    
    np.random.seed(seed)
    random.seed(seed)
    
    # News templates
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
        "{company} achieves carbon neutrality ahead of schedule"
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
        "{company} CEO resigns amid controversy, stock plummets"
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
        "{company} releases sustainability report with mixed results"
    ]
    
    # News sources
    sources = ['Financial Times', 'Bloomberg', 'Reuters', 'CNBC', 'Wall Street Journal', 'MarketWatch',
              'The Economist', 'Forbes', 'Business Insider', 'Yahoo Finance']
    
    # Generate news items in batches for better performance
    news_items = []
    batch_size = 50
    num_batches = (num_news_items + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        batch_items = min(batch_size, num_news_items - batch * batch_size)
        
        # Progress indicator
        if batch > 0:
            print(f"Generated {batch * batch_size} news items...")
        
        for _ in range(batch_items):
            # Select a random asset
            asset = assets_df.sample(1).iloc[0]
            company = asset['name']
            ticker = asset['ticker']
            
            # Determine sentiment based on asset characteristics
            if asset['asset_type'] == 'Stock':
                if asset['roi_1y'] < 0:
                    sentiment_weights = [0.2, 0.5, 0.3]  # positive, negative, neutral
                elif asset['esg_score'] > 75:
                    sentiment_weights = [0.5, 0.2, 0.3]  # positive, negative, neutral
                elif asset['volatility'] > 0.6:
                    sentiment_weights = [0.3, 0.4, 0.3]  # positive, negative, neutral
                else:
                    sentiment_weights = [0.3, 0.3, 0.4]  # positive, negative, neutral
            else:  # Crypto
                if asset['roi_1y'] < 0:
                    sentiment_weights = [0.1, 0.6, 0.3]  # positive, negative, neutral
                elif abs(asset['price_change_24h']) > 20:
                    if asset['price_change_24h'] > 0:
                        sentiment_weights = [0.6, 0.2, 0.2]  # positive, negative, neutral
                    else:
                        sentiment_weights = [0.2, 0.6, 0.2]  # positive, negative, neutral
                elif asset['esg_score'] > 70:
                    sentiment_weights = [0.5, 0.2, 0.3]  # positive, negative, neutral
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
    
    news_df = pd.DataFrame(news_items)
    
    elapsed_time = time.time() - start_time
    print(f"Market news generation completed in {elapsed_time:.2f} seconds")
    
    return news_df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Starting dataset generation process...")
    
    # Generate portfolio dataset
    portfolio_df = generate_portfolio_dataset()
    portfolio_df.to_csv("data/portfolio_dataset.csv", index=False)
    print(f"Successfully generated portfolio dataset with {len(portfolio_df)} assets")
    
    # Generate market news
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
    print(portfolio_df[["name", "ticker", "asset_type", "current_price", "esg_score", "roi_1y"]].head())
    
    print("\nSample news data (first 5 rows):")
    print(news_df[["ticker", "headline", "sentiment", "publication_date"]].head())
    
    print("\nAll data files saved to the 'data' directory.")
