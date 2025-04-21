import pandas as pd
import numpy as np
import random

def generate_portfolio_dataset(num_assets=100):
    """Generate a simplified portfolio dataset."""
    np.random.seed(42)
    
    # Create company names and tickers
    companies = {
        'Tech': ['GreenTech', 'EcoSystems', 'SmartEnergy'],
        'Energy': ['SolarPower', 'WindEnergy', 'CleanGrid'],
        'Finance': ['SustainBank', 'GreenInvest', 'EcoFinance'],
        'Healthcare': ['BioHealth', 'EcoMed', 'GreenCare'],
        'Manufacturing': ['CleanMfg', 'EcoProduct', 'GreenBuild']
    }
    
    data = []
    for sector, names in companies.items():
        for name in names:
            # Basic metrics
            current_price = np.random.uniform(10, 500)
            volatility = np.random.uniform(0.1, 0.5)
            roi = np.random.uniform(5, 25)
            
            # ESG scores
            esg_score = np.random.uniform(50, 95)
            environmental = np.random.uniform(40, 100)
            social = np.random.uniform(40, 100)
            governance = np.random.uniform(40, 100)
            
            # Risk metrics
            beta = np.random.uniform(0.5, 1.5)
            sharpe = np.random.uniform(0.8, 2.5)
            
            data.append({
                'name': name,
                'ticker': name[:3].upper(),
                'sector': sector,
                'current_price': round(current_price, 2),
                'volatility': round(volatility, 3),
                'roi_1y': round(roi, 2),
                'esg_score': round(esg_score, 2),
                'environmental_score': round(environmental, 2),
                'social_score': round(social, 2),
                'governance_score': round(governance, 2),
                'beta': round(beta, 2),
                'sharpe_ratio': round(sharpe, 2),
                'market_cap_b': round(np.random.uniform(1, 100), 2),
                'recommendation': np.random.choice(['Strong Buy', 'Buy', 'Hold', 'Sell'])
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_portfolio_dataset()
    df.to_csv('portfolio_dataset.csv', index=False)
    print(f"Generated dataset with {len(df)} companies")
    print("\nSample data:")
    print(df.head()) 