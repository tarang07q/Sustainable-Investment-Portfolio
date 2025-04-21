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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        os.makedirs('models', exist_ok=True)
    
    def preprocess_data(self, data):
        """Preprocess data for model training or prediction."""
        # Extract features
        X = data[self.features].copy()
        
        # Handle missing values - first check if any columns are completely missing
        for feature in self.features:
            if feature not in X.columns:
                X[feature] = np.nan
        
        # Fill missing values with appropriate defaults
        default_values = {
            'volatility': 0.2,
            'beta': 1.0,
            'sharpe_ratio': 1.5,
            'market_correlation': 0.6,
            'esg_risk_score': 25.0,
            'sector_volatility': 0.2,
            'liquidity_ratio': 2.0
        }
        
        for feature in self.features:
            if X[feature].isnull().any():
                if feature in default_values:
                    X[feature] = X[feature].fillna(default_values[feature])
                else:
                    X[feature] = X[feature].fillna(X[feature].mean() if not X[feature].isnull().all() else 0.5)
        
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
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_val, val_preds)
        
        # Plot feature importance
        self.plot_feature_importance()
        
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
        
        joblib.dump(self.model, 'models/risk_assessment_model.pkl')
        joblib.dump(self.scaler, 'models/risk_assessment_scaler.pkl')
    
    def load_model(self):
        """Load a trained model from disk."""
        self.model = joblib.load('models/risk_assessment_model.pkl')
        self.scaler = joblib.load('models/risk_assessment_scaler.pkl')
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=self.model.classes_)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.model.classes_, 
                    yticklabels=self.model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.close()
    
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
        plt.title('Feature Importance for Risk Assessment')
        plt.tight_layout()
        plt.savefig('models/risk_feature_importance.png')
        plt.close()

# Function to generate synthetic training data
def generate_training_data(data=None, n_samples=1000):
    """
    Generate synthetic data for model training.
    
    Args:
        data: Real data to base synthetic data on (optional)
        n_samples: Number of samples to generate if no real data provided
        
    Returns:
        DataFrame with training data
    """
    if data is not None:
        # Use real data as a base
        training_data = data.copy()
        
        # Ensure all required columns exist
        required_columns = [
            'volatility', 'beta', 'sharpe_ratio', 'market_correlation',
            'esg_risk_score', 'sector_volatility', 'liquidity_ratio'
        ]
        
        for col in required_columns:
            if col not in training_data.columns:
                if col == 'esg_risk_score' and 'esg_score' in training_data.columns:
                    training_data[col] = 100 - training_data['esg_score']
                elif col == 'sector_volatility' and 'sector' in training_data.columns and 'volatility' in training_data.columns:
                    training_data[col] = training_data.groupby('sector')['volatility'].transform('mean')
                elif col == 'liquidity_ratio':
                    training_data[col] = np.random.uniform(0.5, 5.0, len(training_data))
                else:
                    # Generate random values for missing columns
                    if col == 'volatility':
                        training_data[col] = np.random.uniform(0.05, 0.8, len(training_data))
                    elif col == 'beta':
                        training_data[col] = np.random.uniform(0.2, 2.0, len(training_data))
                    elif col == 'sharpe_ratio':
                        training_data[col] = np.random.uniform(-0.5, 3.0, len(training_data))
                    elif col == 'market_correlation':
                        training_data[col] = np.random.uniform(0.1, 0.9, len(training_data))
                    else:
                        training_data[col] = np.random.uniform(0, 100, len(training_data))
    else:
        # Generate completely synthetic data
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
        training_data = pd.DataFrame({
            'volatility': volatility,
            'beta': beta,
            'sharpe_ratio': sharpe_ratio,
            'market_correlation': market_correlation,
            'esg_risk_score': esg_risk_score,
            'sector_volatility': sector_volatility,
            'liquidity_ratio': liquidity_ratio
        })
    
    # Generate risk score
    training_data['risk_score'] = (
        training_data['volatility'] * 30 +
        training_data['beta'] * 15 +
        (3.0 - training_data['sharpe_ratio']) * 10 +
        training_data['market_correlation'] * 10 +
        training_data['esg_risk_score'] * 0.2 +
        training_data['sector_volatility'] * 20 +
        (5.0 - training_data['liquidity_ratio']) * 5
    )
    
    # Categorize risk
    training_data['risk_category'] = pd.cut(
        training_data['risk_score'],
        bins=[0, 20, 40, 60, 100],
        labels=['Low', 'Moderate', 'High', 'Very High']
    )
    
    return training_data

# Function to assess portfolio risk
def assess_portfolio_risk(portfolio_assets):
    """
    Assess the risk of a portfolio using the ML model.
    
    Args:
        portfolio_assets: DataFrame of portfolio assets
    
    Returns:
        Dict with risk assessment results
    """
    try:
        # Create and train model if needed
        model = RiskAssessmentModel()
        
        # Generate training data
        training_data = generate_training_data(portfolio_assets)
        
        # Train the model
        model.train(training_data)
        
        # Prepare portfolio data
        if isinstance(portfolio_assets, pd.DataFrame):
            portfolio_data = portfolio_assets.copy()
        else:
            # Convert to DataFrame if it's a list of dictionaries
            portfolio_data = pd.DataFrame(portfolio_assets)
        
        # Calculate portfolio-level metrics
        portfolio_volatility = portfolio_data['volatility'].mean()
        portfolio_beta = portfolio_data['beta'].mean() if 'beta' in portfolio_data.columns else 1.0
        portfolio_sharpe = portfolio_data['sharpe_ratio'].mean() if 'sharpe_ratio' in portfolio_data.columns else 1.5
        portfolio_market_corr = portfolio_data['market_correlation'].mean() if 'market_correlation' in portfolio_data.columns else 0.6
        
        if 'esg_score' in portfolio_data.columns:
            portfolio_esg_risk = 100 - portfolio_data['esg_score'].mean()
        elif 'esg_risk_score' in portfolio_data.columns:
            portfolio_esg_risk = portfolio_data['esg_risk_score'].mean()
        else:
            portfolio_esg_risk = 25.0
            
        portfolio_sector_vol = portfolio_data['sector_volatility'].mean() if 'sector_volatility' in portfolio_data.columns else 0.2
        portfolio_liquidity = portfolio_data['liquidity_ratio'].mean() if 'liquidity_ratio' in portfolio_data.columns else 2.5
        
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
        
        # Visualize risk factors
        visualize_risk_factors(risk_factors, risk_category[0], risk_score)
        
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
        return {
            'risk_category': 'Moderate',
            'risk_score': 50,
            'error': str(e)
        }

# Function to visualize risk factors
def visualize_risk_factors(risk_factors, risk_category, risk_score):
    """
    Visualize risk factors.
    
    Args:
        risk_factors: Dict of risk factors
        risk_category: Risk category
        risk_score: Risk score
    """
    # Create DataFrame for plotting
    risk_df = pd.DataFrame({
        'Factor': list(risk_factors.keys()),
        'Score': list(risk_factors.values())
    })
    
    # Sort by score descending
    risk_df = risk_df.sort_values('Score', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(risk_df['Factor'], risk_df['Score'], color=plt.cm.RdYlGn_r(risk_df['Score']/100))
    
    # Add labels
    plt.xlabel('Risk Score (0-100)')
    plt.title(f'Risk Factor Analysis - Category: {risk_category}, Overall Score: {risk_score:.1f}')
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{risk_df["Score"].iloc[i]:.1f}', 
                va='center')
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Set x-axis limit
    plt.xlim(0, 105)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('models/risk_factors.png')
    plt.close()
    
    # Create risk gauge chart
    plt.figure(figsize=(10, 6))
    
    # Create gauge
    gauge_colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']  # Green to Red
    
    # Create a horizontal bar for the gauge
    plt.barh(['Risk'], [100], color='#E0E0E0', height=0.3)
    
    # Add colored segments
    plt.barh(['Risk'], [25], color=gauge_colors[0], height=0.3, left=0)
    plt.barh(['Risk'], [25], color=gauge_colors[1], height=0.3, left=25)
    plt.barh(['Risk'], [25], color=gauge_colors[2], height=0.3, left=50)
    plt.barh(['Risk'], [25], color=gauge_colors[3], height=0.3, left=75)
    
    # Add marker for the risk score
    plt.scatter([risk_score], ['Risk'], color='black', s=300, zorder=5, marker='v')
    
    # Add labels
    plt.text(12.5, 0.85, 'Low', ha='center', fontsize=12)
    plt.text(37.5, 0.85, 'Moderate', ha='center', fontsize=12)
    plt.text(62.5, 0.85, 'High', ha='center', fontsize=12)
    plt.text(87.5, 0.85, 'Very High', ha='center', fontsize=12)
    
    # Add risk score text
    plt.text(risk_score, 1.15, f'{risk_score:.1f}', ha='center', fontsize=14, fontweight='bold')
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title(f'Portfolio Risk Score: {risk_category}', fontsize=16, pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('models/risk_gauge.png')
    plt.close()
