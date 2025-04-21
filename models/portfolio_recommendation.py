"""
Portfolio Recommendation ML Model

This module implements a machine learning model for portfolio recommendations
based on user preferences, market data, and ESG criteria.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class PortfolioRecommendationModel:
    """Machine learning model for portfolio recommendations."""

    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.scaler = StandardScaler()
        self.features = [
            'esg_score', 'environmental_score', 'social_score', 'governance_score',
            'roi_1y', 'volatility', 'market_cap_b', 'price_change_24h'
        ]

        # Create model directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)

        # Try to load pre-trained model if it exists
        try:
            self.load_model()
            print("Pre-trained model loaded successfully.")
        except:
            print("No pre-trained model found. Will train a new model when needed.")

    def preprocess_data(self, data):
        """Preprocess data for model training or prediction."""
        # Extract features
        X = data[self.features].copy()

        # Handle missing values
        X.fillna(X.mean(), inplace=True)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled

    def train(self, data, target_col='custom_score'):
        """Train the model on the given data."""
        # Preprocess data
        X = self.preprocess_data(data)
        y = data[target_col].values

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Evaluate model
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

        train_r2 = r2_score(y_train, train_preds)
        val_r2 = r2_score(y_val, val_preds)

        print(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

        # Save model
        self.save_model()

        return {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }

    def predict(self, data):
        """Generate recommendations based on the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Preprocess data
        X = self.preprocess_data(data)

        # Generate predictions
        predictions = self.model.predict(X)

        return predictions

    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        joblib.dump(self.model, 'models/saved/portfolio_recommendation_model.pkl')
        joblib.dump(self.scaler, 'models/saved/portfolio_recommendation_scaler.pkl')

    def load_model(self):
        """Load a trained model from disk."""
        self.model = joblib.load('models/saved/portfolio_recommendation_model.pkl')
        self.scaler = joblib.load('models/saved/portfolio_recommendation_scaler.pkl')

# Function to generate synthetic training data
def generate_training_data(n_samples=1000):
    """Generate synthetic data for model training."""
    np.random.seed(42)

    # Generate features
    esg_score = np.random.uniform(30, 95, n_samples)
    environmental_score = np.random.uniform(20, 95, n_samples)
    social_score = np.random.uniform(30, 90, n_samples)
    governance_score = np.random.uniform(40, 95, n_samples)

    roi_1y = np.random.uniform(-10, 40, n_samples)
    volatility = np.random.uniform(0.05, 0.8, n_samples)
    market_cap_b = np.random.uniform(0.1, 500, n_samples)
    price_change_24h = np.random.uniform(-5, 5, n_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'esg_score': esg_score,
        'environmental_score': environmental_score,
        'social_score': social_score,
        'governance_score': governance_score,
        'roi_1y': roi_1y,
        'volatility': volatility,
        'market_cap_b': market_cap_b,
        'price_change_24h': price_change_24h
    })

    # Generate target variable (custom score)
    # Higher ESG scores, higher ROI, lower volatility, and stable price changes are preferred
    data['custom_score'] = (
        data['esg_score'] * 0.4 +
        data['roi_1y'] * 0.3 +
        (100 - data['volatility'] * 100) * 0.2 +
        (data['price_change_24h'] + 5) * 2 * 0.1
    )

    return data

# Function to get recommendations for a portfolio
def get_portfolio_recommendations(portfolio_assets, all_assets, user_preferences):
    """
    Generate portfolio recommendations based on user preferences.

    Args:
        portfolio_assets: DataFrame of current portfolio assets
        all_assets: DataFrame of all available assets
        user_preferences: Dict of user preferences

    Returns:
        DataFrame of recommended assets with scores
    """
    # Create and train model if needed
    model = PortfolioRecommendationModel()

    # If model not loaded, train it on synthetic data
    if model.model is None:
        training_data = generate_training_data()
        model.train(training_data)

    # Prepare data for prediction
    prediction_data = all_assets.copy()

    # Filter out assets already in portfolio
    if not portfolio_assets.empty:
        portfolio_tickers = portfolio_assets['ticker'].unique()
        prediction_data = prediction_data[~prediction_data['ticker'].isin(portfolio_tickers)]

    # Apply user preferences
    # Risk tolerance (1-10)
    risk_tolerance = user_preferences.get('risk_tolerance', 5)
    risk_weight = (11 - risk_tolerance) / 10

    # Sustainability focus (1-10)
    sustainability_focus = user_preferences.get('sustainability_focus', 5)
    esg_weight = sustainability_focus / 10

    # Investment horizon
    investment_horizon = user_preferences.get('investment_horizon', 'Medium-term (1-3 years)')

    # Adjust weights based on investment horizon
    if 'Short-term' in investment_horizon:
        # Short-term: More focus on recent performance and volatility
        roi_weight = 0.4
        volatility_weight = 0.4
        esg_weight = esg_weight * 0.5  # Reduce ESG importance for short-term
    elif 'Long-term' in investment_horizon:
        # Long-term: More focus on ESG and fundamentals
        roi_weight = 0.3
        volatility_weight = 0.2
        esg_weight = min(esg_weight * 1.5, 0.5)  # Increase ESG importance for long-term
    else:  # Medium-term
        roi_weight = 0.35
        volatility_weight = 0.3

    # Fine-tune weights based on exact risk tolerance value
    if risk_tolerance <= 3:  # Very risk-averse
        volatility_weight *= 1.5
        roi_weight *= 0.7
    elif risk_tolerance >= 8:  # Very risk-tolerant
        volatility_weight *= 0.7
        roi_weight *= 1.3

    # Fine-tune weights based on exact sustainability focus value
    if sustainability_focus >= 8:  # Very sustainability-focused
        environmental_weight = 0.5  # Higher weight on environmental score
        social_weight = 0.3        # Medium weight on social score
        governance_weight = 0.2    # Lower weight on governance score
    elif sustainability_focus <= 3:  # Very returns-focused
        environmental_weight = 0.2  # Lower weight on environmental score
        social_weight = 0.3        # Medium weight on social score
        governance_weight = 0.5    # Higher weight on governance score
    else:  # Balanced
        environmental_weight = 0.33  # Equal weights
        social_weight = 0.33
        governance_weight = 0.34

    # Normalize weights to ensure they sum to 1
    total_weight = roi_weight + volatility_weight + esg_weight
    roi_weight /= total_weight
    volatility_weight /= total_weight
    esg_weight /= total_weight

    # Calculate custom score based on user preferences
    prediction_data['custom_score'] = (
        # ESG component with sub-weights
        esg_weight * (
            prediction_data['environmental_score'] * environmental_weight +
            prediction_data['social_score'] * social_weight +
            prediction_data['governance_score'] * governance_weight
        ) +
        # Financial return component
        prediction_data['roi_1y'] * roi_weight +
        # Risk component
        (100 - prediction_data['volatility'] * 100) * volatility_weight
    )

    # Get model predictions
    try:
        ml_scores = model.predict(prediction_data)
        prediction_data['ml_score'] = ml_scores

        # Combine custom score and ML score
        prediction_data['final_score'] = prediction_data['custom_score'] * 0.4 + prediction_data['ml_score'] * 0.6

        # Sort by final score
        recommendations = prediction_data.sort_values('final_score', ascending=False)

        # Add recommendation strength
        recommendations['recommendation_strength'] = pd.cut(
            recommendations['final_score'],
            bins=[0, 40, 60, 100],
            labels=['🔴 Consider with Caution', '🟡 Moderate Opportunity', '🟢 Strong Recommendation']
        )

        return recommendations
    except Exception as e:
        print(f"Error generating ML recommendations: {str(e)}")
        # Fallback to simple scoring if ML fails
        recommendations = prediction_data.sort_values('custom_score', ascending=False)
        recommendations['recommendation_strength'] = pd.cut(
            recommendations['custom_score'],
            bins=[0, 40, 60, 100],
            labels=['🔴 Consider with Caution', '🟡 Moderate Opportunity', '🟢 Strong Recommendation']
        )
        return recommendations
