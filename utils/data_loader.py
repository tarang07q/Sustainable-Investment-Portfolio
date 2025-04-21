"""
Data Loader Utility

This module provides functions to load and preprocess data for ML models.
"""

import pandas as pd
import numpy as np
import os

def load_portfolio_data():
    """
    Load portfolio data from CSV file.

    Returns:
        DataFrame: Portfolio data
    """
    try:
        # Check if the data file exists
        if os.path.exists('data/sample_portfolio_data.csv'):
            df = pd.read_csv('data/sample_portfolio_data.csv')

            # Calculate additional metrics if needed
            if 'current_value' not in df.columns:
                df['current_value'] = df['shares'] * df['current_price']

            if 'purchase_value' not in df.columns:
                df['purchase_value'] = df['shares'] * df['purchase_price']

            if 'gain_loss' not in df.columns:
                df['gain_loss'] = df['current_value'] - df['purchase_value']

            if 'gain_loss_pct' not in df.columns:
                df['gain_loss_pct'] = (df['current_price'] / df['purchase_price'] - 1) * 100

            # Fill missing values with appropriate defaults
            df = fill_missing_values(df)

            return df
        else:
            print("Sample portfolio data file not found. Using generated data.")
            return None
    except Exception as e:
        print(f"Error loading portfolio data: {str(e)}")
        return None

def load_market_news():
    """
    Load market news data from CSV file.

    Returns:
        DataFrame: Market news data
    """
    try:
        # Check if the data file exists
        if os.path.exists('data/sample_market_news.csv'):
            df = pd.read_csv('data/sample_market_news.csv')
            return df
        else:
            print("Sample market news file not found. Using generated data.")
            return None
    except Exception as e:
        print(f"Error loading market news data: {str(e)}")
        return None

def fill_missing_values(df):
    """
    Fill missing values in the DataFrame with appropriate defaults.

    Args:
        df: DataFrame to fill missing values in

    Returns:
        DataFrame: DataFrame with missing values filled
    """
    # Define default values for different columns
    default_values = {
        'volatility': 0.2,
        'beta': 1.0,
        'sharpe_ratio': 1.5,
        'market_correlation': 0.6,
        'environmental_score': 75.0,
        'social_score': 75.0,
        'governance_score': 75.0,
        'esg_score': 75.0,
        'carbon_footprint': 25.0,
        'esg_risk_score': 25.0,
        'sector_volatility': 0.2,
        'liquidity_ratio': 2.0
    }

    # Standardize column names (convert to lowercase)
    df.columns = [col.lower() for col in df.columns]

    # Map common column name variations
    column_mapping = {
        'environmental_score': 'environmental_score',
        'social_score': 'social_score',
        'governance_score': 'governance_score',
        'esg_score': 'esg_score',
        'carbon_footprint': 'carbon_footprint',
        'ticker': 'ticker',
        'name': 'name',
        'current_price': 'current_price',
        'purchase_price': 'purchase_price',
        'shares': 'shares',
        'asset_type': 'asset_type',
        'sector': 'sector',
        'allocation': 'allocation',
        'roi_1y': 'roi_1y',
        'volatility': 'volatility',
        'beta': 'beta',
        'sharpe_ratio': 'sharpe_ratio',
        'market_correlation': 'market_correlation',
        'esg_risk_score': 'esg_risk_score',
        'sector_volatility': 'sector_volatility',
        'liquidity_ratio': 'liquidity_ratio'
    }

    # Rename columns based on mapping
    for old_col in list(df.columns):
        for std_col, mapped_col in column_mapping.items():
            if old_col == std_col or old_col.replace('_', '').lower() == std_col.replace('_', '').lower():
                if old_col != mapped_col:
                    df.rename(columns={old_col: mapped_col}, inplace=True)
                break

    # Add missing columns with default values
    for col, default_val in default_values.items():
        if col not in df.columns:
            df[col] = default_val

    # Fill missing values with defaults or column means
    for col in df.columns:
        if col in default_values and df[col].isnull().any():
            df[col] = df[col].fillna(default_values[col])
        elif df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna('Unknown')

    return df

def get_asset_data(ticker, all_assets_df=None):
    """
    Get data for a specific asset by ticker.

    Args:
        ticker: Asset ticker symbol
        all_assets_df: DataFrame of all assets (optional)

    Returns:
        dict: Asset data
    """
    try:
        # Try to load from portfolio data first
        portfolio_df = load_portfolio_data()
        if portfolio_df is not None:
            asset_data = portfolio_df[portfolio_df['ticker'] == ticker]
            if not asset_data.empty:
                return asset_data.iloc[0].to_dict()

        # If not found in portfolio or portfolio data not available, check all assets
        if all_assets_df is not None:
            asset_data = all_assets_df[all_assets_df['ticker'] == ticker]
            if not asset_data.empty:
                return asset_data.iloc[0].to_dict()

        # If still not found, return None
        return None
    except Exception as e:
        print(f"Error getting asset data for {ticker}: {str(e)}")
        return None
