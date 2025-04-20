"""
Risk Assessment ML Model

This module implements a machine learning model for portfolio risk assessment
based on asset characteristics and market conditions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class RiskAssessmentModel:
    """Machine learning model for portfolio risk assessment."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.scaler = StandardScaler()
        self.features = [
            'volatility', 'beta', 'sharpe_ratio', 'market_correlation',
            'esg_risk_score', 'sector_volatility', 'liquidity_ratio'
        ]
        
        # Create model directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)
        
        # Try to load pre-trained model if it exists
        try:
            self.load_model()
            print("Pre-trained risk assessment model loaded successfully.")
        except:
            print("No pre-trained risk model found. Will train a new model when needed.")
    
    def preprocess_data(self, data):
        """Preprocess data for model training or prediction."""
        # Extract features
        X = data[self.features].copy()
        
        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def train(self, data, target_col='risk_category'):
        """Train the model on the given data."""
        # Preprocess data
        X = self.preprocess_data(data)
        y = data[target_col].values
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_preds)
        val_accuracy = accuracy_score(y_val, val_preds)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_preds))
        
        # Save model
        self.save_model()
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }
    
    def predict(self, data):
        """Predict risk categories based on the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess data
        X = self.preprocess_data(data)
        
        # Generate predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, 'models/saved/risk_assessment_model.pkl')
        joblib.dump(self.scaler, 'models/saved/risk_assessment_scaler.pkl')
    
    def load_model(self):
        """Load a trained model from disk."""
        self.model = joblib.load('models/saved/risk_assessment_model.pkl')
        self.scaler = joblib.load('models/saved/risk_assessment_scaler.pkl')

# Function to generate synthetic training data
def generate_training_data(n_samples=1000):
    """Generate synthetic data for model training."""
    np.random.seed(42)
    
    # Generate features
    volatility = np.random.uniform(0.05, 0.8, n_samples)
    beta = np.random.uniform(0.2, 2.0, n_samples)
    sharpe_ratio = np.random.uniform(-0.5, 3.0, n_samples)
    market_correlation = np.random.uniform(0.1, 0.9, n_samples)
    esg_risk_score = np.random.uniform(10, 90, n_samples)
    sector_volatility = np.random.uniform(0.1, 0.6, n_samples)
    liquidity_ratio = np.random.uniform(0.5, 5.0, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'volatility': volatility,
        'beta': beta,
        'sharpe_ratio': sharpe_ratio,
        'market_correlation': market_correlation,
        'esg_risk_score': esg_risk_score,
        'sector_volatility': sector_volatility,
        'liquidity_ratio': liquidity_ratio
    })
    
    # Generate risk score
    data['risk_score'] = (
        data['volatility'] * 30 +
        data['beta'] * 15 +
        (3.0 - data['sharpe_ratio']) * 10 +
        data['market_correlation'] * 10 +
        data['esg_risk_score'] * 0.2 +
        data['sector_volatility'] * 20 +
        (5.0 - data['liquidity_ratio']) * 5
    )
    
    # Categorize risk
    data['risk_category'] = pd.cut(
        data['risk_score'],
        bins=[0, 20, 40, 60, 100],
        labels=['Low', 'Moderate', 'High', 'Very High']
    )
    
    return data

# Function to assess portfolio risk
def assess_portfolio_risk(portfolio_assets):
    """
    Assess the risk of a portfolio using the ML model.
    
    Args:
        portfolio_assets: DataFrame of portfolio assets
    
    Returns:
        Dict with risk assessment results
    """
    # Create and train model if needed
    model = RiskAssessmentModel()
    
    # If model not loaded, train it on synthetic data
    if model.model is None:
        training_data = generate_training_data()
        model.train(training_data)
    
    # Prepare portfolio data
    portfolio_data = portfolio_assets.copy()
    
    # Calculate portfolio-level metrics
    portfolio_volatility = np.average(
        portfolio_data['volatility'],
        weights=portfolio_data['allocation']
    )
    
    # Generate synthetic data for other metrics we don't have
    np.random.seed(int(sum(portfolio_data['allocation'] * 100)))
    portfolio_beta = np.random.uniform(0.8, 1.2)
    portfolio_sharpe = np.random.uniform(0.5, 2.5)
    portfolio_market_corr = np.random.uniform(0.6, 0.9)
    portfolio_esg_risk = 100 - np.average(
        portfolio_data['esg_score'],
        weights=portfolio_data['allocation']
    )
    portfolio_sector_vol = np.random.uniform(0.1, 0.4)
    portfolio_liquidity = np.random.uniform(1.5, 4.0)
    
    # Create a DataFrame for the portfolio
    portfolio_metrics = pd.DataFrame({
        'volatility': [portfolio_volatility],
        'beta': [portfolio_beta],
        'sharpe_ratio': [portfolio_sharpe],
        'market_correlation': [portfolio_market_corr],
        'esg_risk_score': [portfolio_esg_risk],
        'sector_volatility': [portfolio_sector_vol],
        'liquidity_ratio': [portfolio_liquidity]
    })
    
    try:
        # Get risk prediction
        risk_category, risk_probabilities = model.predict(portfolio_metrics)
        
        # Calculate risk score (weighted average of probabilities)
        risk_levels = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
        risk_score = sum(
            prob * risk_levels[category] 
            for prob, category in zip(risk_probabilities[0], model.model.classes_)
        ) * 25  # Scale to 0-100
        
        # Prepare detailed risk breakdown
        risk_factors = {
            'Market Risk': portfolio_volatility * 100,
            'Systematic Risk': portfolio_beta * 50,
            'Return-Risk Ratio': (3.0 - portfolio_sharpe) * 33.3,
            'Market Correlation': portfolio_market_corr * 100,
            'ESG Risk': portfolio_esg_risk,
            'Sector Risk': portfolio_sector_vol * 100,
            'Liquidity Risk': (5.0 - portfolio_liquidity) * 20
        }
        
        # Normalize risk factors to 0-100 scale
        for factor in risk_factors:
            risk_factors[factor] = min(max(risk_factors[factor], 0), 100)
        
        return {
            'risk_category': risk_category[0],
            'risk_score': risk_score,
            'risk_probabilities': {
                category: prob for category, prob in zip(model.model.classes_, risk_probabilities[0])
            },
            'risk_factors': risk_factors,
            'portfolio_metrics': {
                'volatility': portfolio_volatility,
                'beta': portfolio_beta,
                'sharpe_ratio': portfolio_sharpe,
                'market_correlation': portfolio_market_corr,
                'esg_risk_score': portfolio_esg_risk,
                'sector_volatility': portfolio_sector_vol,
                'liquidity_ratio': portfolio_liquidity
            }
        }
    except Exception as e:
        print(f"Error in risk assessment: {str(e)}")
        # Fallback to simple risk calculation
        risk_score = portfolio_volatility * 100
        if risk_score < 25:
            risk_category = 'Low'
        elif risk_score < 50:
            risk_category = 'Moderate'
        elif risk_score < 75:
            risk_category = 'High'
        else:
            risk_category = 'Very High'
            
        return {
            'risk_category': risk_category,
            'risk_score': risk_score,
            'error': str(e)
        }
