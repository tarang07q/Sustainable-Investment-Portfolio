import os
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import random

# Alpha Vantage API key (in production, this would be stored as an environment variable)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")

# List of sustainable companies to fetch data for
SUSTAINABLE_COMPANIES = [
    {"symbol": "MSFT", "name": "Microsoft", "sector": "Technology"},
    {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla", "sector": "Automotive"},
    {"symbol": "ENPH", "name": "Enphase Energy", "sector": "Energy"},
    {"symbol": "NEE", "name": "NextEra Energy", "sector": "Utilities"},
    {"symbol": "VWDRY", "name": "Vestas Wind Systems", "sector": "Energy"},
    {"symbol": "CSIQ", "name": "Canadian Solar", "sector": "Energy"},
    {"symbol": "SEDG", "name": "SolarEdge", "sector": "Energy"},
    {"symbol": "BEP", "name": "Brookfield Renewable", "sector": "Utilities"},
    {"symbol": "FSLR", "name": "First Solar", "sector": "Energy"},
    {"symbol": "CWEN", "name": "Clearway Energy", "sector": "Utilities"},
    {"symbol": "TPIC", "name": "TPI Composites", "sector": "Industrials"},
    {"symbol": "SPWR", "name": "SunPower", "sector": "Energy"},
    {"symbol": "AMRC", "name": "Ameresco", "sector": "Industrials"}
]

# List of sustainable cryptocurrencies
SUSTAINABLE_CRYPTO = [
    {"symbol": "SOL", "name": "Solana", "sector": "Smart Contracts"},
    {"symbol": "ADA", "name": "Cardano", "sector": "Smart Contracts"},
    {"symbol": "XLM", "name": "Stellar", "sector": "Payments"},
    {"symbol": "ALGO", "name": "Algorand", "sector": "Smart Contracts"},
    {"symbol": "HBAR", "name": "Hedera", "sector": "DLT"},
    {"symbol": "MATIC", "name": "Polygon", "sector": "Layer 2"},
    {"symbol": "NEAR", "name": "NEAR Protocol", "sector": "Smart Contracts"},
    {"symbol": "XTZ", "name": "Tezos", "sector": "Smart Contracts"}
]

# Get stock data from Alpha Vantage
def get_stock_data(symbol):
    try:
        # In a production environment, this would make a real API call
        # For the prototype, we'll simulate the API response
        
        # Check if we're using the demo key
        if ALPHA_VANTAGE_API_KEY == "demo":
            # Generate simulated data
            return generate_simulated_stock_data(symbol)
        
        # Make API call to Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            return {
                "symbol": symbol,
                "price": float(quote["05. price"]),
                "change": float(quote["09. change"]),
                "change_percent": float(quote["10. change percent"].replace("%", "")),
                "volume": int(quote["06. volume"]),
                "latest_trading_day": quote["07. latest trading day"]
            }
        else:
            # If API call fails, generate simulated data
            return generate_simulated_stock_data(symbol)
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {str(e)}")
        # If there's an error, generate simulated data
        return generate_simulated_stock_data(symbol)

# Generate simulated stock data
def generate_simulated_stock_data(symbol):
    # Set random seed based on symbol for consistency
    np.random.seed(hash(symbol) % 10000)
    
    # Generate realistic price based on symbol
    base_price = 50 + (hash(symbol) % 1000)
    price = np.random.uniform(0.9 * base_price, 1.1 * base_price)
    
    # Generate change and change percent
    change_percent = np.random.uniform(-3, 3)
    change = price * change_percent / 100
    
    # Generate volume
    volume = np.random.randint(100000, 10000000)
    
    # Get latest trading day
    latest_trading_day = datetime.now().strftime("%Y-%m-%d")
    
    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "change_percent": round(change_percent, 2),
        "volume": volume,
        "latest_trading_day": latest_trading_day
    }

# Get historical stock data
def get_historical_stock_data(symbol, days=180):
    try:
        # In a production environment, this would make a real API call
        # For the prototype, we'll simulate the API response
        
        # Check if we're using the demo key
        if ALPHA_VANTAGE_API_KEY == "demo":
            # Generate simulated historical data
            return generate_simulated_historical_data(symbol, days)
        
        # Make API call to Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ["open", "high", "low", "close", "volume"]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Filter to requested number of days
            df = df.tail(days)
            
            # Add date column
            df["date"] = df.index
            
            return df
        else:
            # If API call fails, generate simulated data
            return generate_simulated_historical_data(symbol, days)
    except Exception as e:
        print(f"Error fetching historical stock data for {symbol}: {str(e)}")
        # If there's an error, generate simulated data
        return generate_simulated_historical_data(symbol, days)

# Generate simulated historical stock data
def generate_simulated_historical_data(symbol, days=180):
    # Set random seed based on symbol for consistency
    np.random.seed(hash(symbol) % 10000)
    
    # Generate end date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    
    # Generate base price
    base_price = 50 + (hash(symbol) % 1000)
    
    # Generate price movement with some trend and volatility
    volatility = np.random.uniform(0.01, 0.03)
    trend = np.random.uniform(-0.0005, 0.001)
    
    # Generate returns
    returns = np.random.normal(trend, volatility, size=len(date_range))
    
    # Calculate prices
    price = base_price
    prices = [price]
    
    for r in returns:
        price = price * (1 + r)
        prices.append(price)
    
    prices = prices[:-1]  # Remove the extra price
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": date_range,
        "close": prices
    })
    
    # Generate other price columns
    df["open"] = df["close"] * np.random.uniform(0.99, 1.01, size=len(df))
    df["high"] = df[["open", "close"]].max(axis=1) * np.random.uniform(1.001, 1.02, size=len(df))
    df["low"] = df[["open", "close"]].min(axis=1) * np.random.uniform(0.98, 0.999, size=len(df))
    
    # Generate volume
    df["volume"] = np.random.randint(100000, 10000000, size=len(df))
    
    # Set index
    df.set_index("date", inplace=True)
    
    return df

# Get ESG data for a company
def get_esg_data(symbol):
    # In a real implementation, this would call an ESG data API
    # For the prototype, we'll generate simulated ESG data
    
    # Set random seed based on symbol for consistency
    np.random.seed(hash(symbol) % 10000)
    
    # Generate ESG scores
    # Companies with sustainable focus tend to have higher ESG scores
    sustainable_symbols = [company["symbol"] for company in SUSTAINABLE_COMPANIES]
    
    base_score = 50
    if symbol in sustainable_symbols:
        base_score = 70  # Higher base score for sustainable companies
    
    # Generate component scores
    environmental_score = np.random.uniform(base_score, base_score + 25)
    social_score = np.random.uniform(base_score - 10, base_score + 15)
    governance_score = np.random.uniform(base_score - 5, base_score + 20)
    
    # Calculate overall ESG score
    esg_score = (environmental_score + social_score + governance_score) / 3
    
    # Generate carbon footprint
    carbon_footprint = np.random.uniform(10, 100)
    if symbol in sustainable_symbols:
        carbon_footprint = carbon_footprint * 0.6  # Lower carbon footprint for sustainable companies
    
    # Generate carbon reduction target
    carbon_reduction_target = np.random.uniform(10, 50)
    
    # Generate SDG alignment
    sdg_count = np.random.randint(2, 6) if symbol in sustainable_symbols else np.random.randint(1, 4)
    sdg_alignment = random.sample(range(1, 18), k=sdg_count)
    
    # Generate controversy score (lower is better)
    controversy_score = np.random.uniform(0, 30) if symbol in sustainable_symbols else np.random.uniform(10, 50)
    
    return {
        "symbol": symbol,
        "environmental_score": round(environmental_score, 1),
        "social_score": round(social_score, 1),
        "governance_score": round(governance_score, 1),
        "esg_score": round(esg_score, 1),
        "carbon_footprint": round(carbon_footprint, 1),
        "carbon_reduction_target": round(carbon_reduction_target, 1),
        "sdg_alignment": sdg_alignment,
        "controversy_score": round(controversy_score, 1)
    }

# Get cryptocurrency data
def get_crypto_data(symbol):
    try:
        # In a production environment, this would make a real API call
        # For the prototype, we'll simulate the API response
        
        # Check if we're using the demo key
        if ALPHA_VANTAGE_API_KEY == "demo":
            # Generate simulated data
            return generate_simulated_crypto_data(symbol)
        
        # Make API call to Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={symbol}&to_currency=USD&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if "Realtime Currency Exchange Rate" in data:
            exchange_rate = data["Realtime Currency Exchange Rate"]
            return {
                "symbol": symbol,
                "price": float(exchange_rate["5. Exchange Rate"]),
                "change": 0,  # Alpha Vantage doesn't provide change in this endpoint
                "change_percent": 0,  # Alpha Vantage doesn't provide change percent in this endpoint
                "volume": 0,  # Alpha Vantage doesn't provide volume in this endpoint
                "latest_trading_day": exchange_rate["6. Last Refreshed"]
            }
        else:
            # If API call fails, generate simulated data
            return generate_simulated_crypto_data(symbol)
    except Exception as e:
        print(f"Error fetching crypto data for {symbol}: {str(e)}")
        # If there's an error, generate simulated data
        return generate_simulated_crypto_data(symbol)

# Generate simulated cryptocurrency data
def generate_simulated_crypto_data(symbol):
    # Set random seed based on symbol for consistency
    np.random.seed(hash(symbol) % 10000)
    
    # Generate realistic price based on symbol
    if symbol == "BTC":
        base_price = 30000
    elif symbol == "ETH":
        base_price = 2000
    else:
        base_price = 1 + (hash(symbol) % 100)
    
    price = np.random.uniform(0.9 * base_price, 1.1 * base_price)
    
    # Generate change and change percent
    change_percent = np.random.uniform(-5, 5)
    change = price * change_percent / 100
    
    # Generate volume
    volume = np.random.randint(1000000, 100000000)
    
    # Get latest trading day
    latest_trading_day = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "change_percent": round(change_percent, 2),
        "volume": volume,
        "latest_trading_day": latest_trading_day
    }

# Get all stock data for sustainable companies
def get_all_sustainable_stocks():
    stocks = []
    
    for company in SUSTAINABLE_COMPANIES:
        # Get stock data
        stock_data = get_stock_data(company["symbol"])
        
        # Get ESG data
        esg_data = get_esg_data(company["symbol"])
        
        # Combine data
        combined_data = {
            "symbol": company["symbol"],
            "name": company["name"],
            "sector": company["sector"],
            "asset_type": "Stock",
            "current_price": stock_data["price"],
            "price_change_24h": stock_data["change_percent"],
            "market_cap_b": round(stock_data["price"] * np.random.uniform(1000000000, 100000000000) / 1000000000, 2),
            "roi_1y": round(np.random.uniform(5, 30), 2),
            "volatility": round(np.random.uniform(0.1, 0.5), 2),
            "environmental_score": esg_data["environmental_score"],
            "social_score": esg_data["social_score"],
            "governance_score": esg_data["governance_score"],
            "esg_score": esg_data["esg_score"],
            "sdg_alignment": esg_data["sdg_alignment"],
            "carbon_footprint": esg_data["carbon_footprint"],
            "carbon_reduction_target": esg_data["carbon_reduction_target"],
            "ai_risk_score": round(100 - (esg_data["esg_score"] * 0.4 + (100 - np.random.uniform(0.1, 0.5) * 100) * 0.4 + np.random.uniform(5, 30) * 0.2), 1)
        }
        
        # Add AI recommendation
        if combined_data["ai_risk_score"] < 30:
            combined_data["ai_recommendation"] = "🟢 Strong Buy"
        elif combined_data["ai_risk_score"] < 50:
            combined_data["ai_recommendation"] = "🟡 Hold"
        else:
            combined_data["ai_recommendation"] = "🔴 Caution"
        
        stocks.append(combined_data)
    
    return stocks

# Get all cryptocurrency data for sustainable cryptocurrencies
def get_all_sustainable_crypto():
    cryptos = []
    
    for crypto in SUSTAINABLE_CRYPTO:
        # Get crypto data
        crypto_data = get_crypto_data(crypto["symbol"])
        
        # Generate ESG-like data for crypto
        # Set random seed based on symbol for consistency
        np.random.seed(hash(crypto["symbol"]) % 10000)
        
        # Environmental score is lower for proof-of-work cryptocurrencies
        environmental_score = np.random.uniform(60, 90)
        social_score = np.random.uniform(50, 85)
        governance_score = np.random.uniform(40, 80)
        esg_score = (environmental_score + social_score + governance_score) / 3
        
        # Energy consumption and renewable energy percentage
        energy_consumption = np.random.uniform(10, 1000)
        renewable_energy_pct = np.random.uniform(30, 90)
        
        # Carbon footprint based on energy consumption and renewable percentage
        carbon_footprint = energy_consumption * (1 - renewable_energy_pct/100) / 10
        carbon_reduction_target = np.random.uniform(5, 40)
        
        # SDG alignment
        sdg_count = np.random.randint(1, 4)
        sdg_alignment = random.sample(range(1, 18), k=sdg_count)
        
        # Combine data
        combined_data = {
            "symbol": crypto["symbol"],
            "name": crypto["name"],
            "sector": crypto["sector"],
            "asset_type": "Crypto",
            "current_price": crypto_data["price"],
            "price_change_24h": crypto_data["change_percent"],
            "market_cap_b": round(crypto_data["price"] * np.random.uniform(100000000, 10000000000) / 1000000000, 2),
            "roi_1y": round(np.random.uniform(10, 100), 2),
            "volatility": round(np.random.uniform(0.3, 0.8), 2),
            "environmental_score": round(environmental_score, 1),
            "social_score": round(social_score, 1),
            "governance_score": round(governance_score, 1),
            "esg_score": round(esg_score, 1),
            "sdg_alignment": sdg_alignment,
            "energy_consumption": round(energy_consumption, 1),
            "renewable_energy_pct": round(renewable_energy_pct, 1),
            "carbon_footprint": round(carbon_footprint, 1),
            "carbon_reduction_target": round(carbon_reduction_target, 1),
            "ai_risk_score": round(100 - (esg_score * 0.3 + (100 - np.random.uniform(0.3, 0.8) * 100) * 0.5 + np.random.uniform(10, 100) * 0.2 / 4), 1)
        }
        
        # Add AI recommendation
        if combined_data["ai_risk_score"] < 30:
            combined_data["ai_recommendation"] = "🟢 Strong Buy"
        elif combined_data["ai_risk_score"] < 50:
            combined_data["ai_recommendation"] = "🟡 Hold"
        else:
            combined_data["ai_recommendation"] = "🔴 Caution"
        
        cryptos.append(combined_data)
    
    return cryptos

# Get all assets (stocks and cryptocurrencies)
def get_all_assets():
    stocks = get_all_sustainable_stocks()
    cryptos = get_all_sustainable_crypto()
    return stocks + cryptos

# Get market trends
def get_market_trends():
    trends = [
        {
            "title": "Renewable Energy Growth",
            "description": "Renewable energy companies are showing strong growth potential due to increasing global commitments to carbon reduction.",
            "impact": "Positive for clean energy stocks and related cryptocurrencies",
            "confidence": 85
        },
        {
            "title": "ESG Regulation Strengthening",
            "description": "New ESG disclosure requirements are being implemented across major markets, affecting corporate reporting and compliance.",
            "impact": "Positive for companies with strong ESG practices, negative for laggards",
            "confidence": 92
        },
        {
            "title": "Green Technology Innovation",
            "description": "Breakthrough technologies in carbon capture and sustainable materials are creating new investment opportunities.",
            "impact": "Positive for green tech and sustainable material companies",
            "confidence": 78
        },
        {
            "title": "Crypto Energy Consumption Concerns",
            "description": "Increasing scrutiny of cryptocurrency energy usage is driving a shift toward more energy-efficient protocols.",
            "impact": "Positive for eco-friendly cryptocurrencies, negative for energy-intensive ones",
            "confidence": 88
        },
        {
            "title": "Sustainable Supply Chain Demand",
            "description": "Consumer and regulatory pressure is increasing demand for transparent and sustainable supply chains.",
            "impact": "Positive for companies with robust supply chain sustainability",
            "confidence": 82
        }
    ]
    return trends
