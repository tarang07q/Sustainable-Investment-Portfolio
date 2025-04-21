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
import matplotlib.pyplot as plt
import seaborn as sns

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
        os.makedirs('models', exist_ok=True)
    
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
        
        # Plot feature importance
        self.plot_feature_importance()
        
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
        
        joblib.dump(self.model, 'models/portfolio_recommendation_model.pkl')
        joblib.dump(self.scaler, 'models/portfolio_recommendation_scaler.pkl')
    
    def load_model(self):
        """Load a trained model from disk."""
        self.model = joblib.load('models/portfolio_recommendation_model.pkl')
        self.scaler = joblib.load('models/portfolio_recommendation_scaler.pkl')
    
    def plot_feature_importance(self):
        """Plot feature importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for plotting
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        plt.close()

# Function to generate synthetic training data
def generate_training_data(data, user_preferences=None):
    """
    Generate training data for model training based on real data.
    
    Args:
        data: DataFrame of real assets
        user_preferences: Dict of user preferences (optional)
    
    Returns:
        DataFrame with training data
    """
    # Make a copy of the data
    training_data = data.copy()
    
    # Default user preferences if not provided
    if user_preferences is None:
        user_preferences = {
            'risk_tolerance': 5,  # 1-10
            'sustainability_focus': 5  # 1-10
        }
    
    # Risk tolerance (1-10)
    risk_weight = (11 - user_preferences.get('risk_tolerance', 5)) / 10
    
    # Sustainability focus (1-10)
    esg_weight = user_preferences.get('sustainability_focus', 5) / 10
    
    # Calculate custom score based on user preferences
    training_data['custom_score'] = (
        training_data['esg_score'] * esg_weight +
        training_data['roi_1y'] * (1 - risk_weight - esg_weight/2) +
        (100 - training_data['volatility'] * 100) * risk_weight
    )
    
    return training_data

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
    
    # Prepare training data
    training_data = generate_training_data(all_assets, user_preferences)
    
    # Train the model
    model.train(training_data)
    
    # Prepare data for prediction
    prediction_data = all_assets.copy()
    
    # Filter out assets already in portfolio
    if not portfolio_assets.empty:
        portfolio_tickers = portfolio_assets['ticker'].unique()
        prediction_data = prediction_data[~prediction_data['ticker'].isin(portfolio_tickers)]
    
    # Apply user preferences
    # Risk tolerance (1-10)
    risk_weight = (11 - user_preferences.get('risk_tolerance', 5)) / 10
    
    # Sustainability focus (1-10)
    esg_weight = user_preferences.get('sustainability_focus', 5) / 10
    
    # Calculate custom score based on user preferences
    prediction_data['custom_score'] = (
        prediction_data['esg_score'] * esg_weight +
        prediction_data['roi_1y'] * (1 - risk_weight - esg_weight/2) +
        (100 - prediction_data['volatility'] * 100) * risk_weight
    )
    
    # Get model predictions
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
        labels=['Consider with Caution', 'Moderate Opportunity', 'Strong Recommendation']
    )
    
    return recommendations

# Function to visualize recommendations
def visualize_recommendations(recommendations, top_n=10):
    """
    Visualize the top recommendations.
    
    Args:
        recommendations: DataFrame of recommendations
        top_n: Number of top recommendations to visualize
    """
    # Get top N recommendations
    top_recommendations = recommendations.head(top_n)
    
    # Plot ESG score vs ROI
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        top_recommendations['roi_1y'],
        top_recommendations['esg_score'],
        c=top_recommendations['final_score'],
        s=100,
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Final Score')
    
    # Add labels for each point
    for i, row in top_recommendations.iterrows():
        plt.annotate(
            row['ticker'],
            (row['roi_1y'], row['esg_score']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add labels and title
    plt.xlabel('1-Year ROI (%)')
    plt.ylabel('ESG Score')
    plt.title('Top Recommendations: ESG Score vs ROI')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('models/top_recommendations.png')
    plt.close()
    
    # Plot recommendation strength distribution
    plt.figure(figsize=(10, 6))
    strength_counts = recommendations['recommendation_strength'].value_counts()
    strength_counts.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Distribution of Recommendation Strengths')
    plt.xlabel('Recommendation Strength')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('models/recommendation_strength_distribution.png')
    plt.close()
