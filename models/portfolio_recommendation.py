"""
Portfolio Recommendation ML Model

This module implements a machine learning model for portfolio recommendations
based on user preferences, market data, and ESG criteria.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from typing import Dict, Any, List, Tuple
import joblib
import os

class PortfolioRecommender:
    def __init__(self):
        """Initialize the Portfolio Recommendation model."""
        self.model = XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = [
            'market_cap', 'volatility', 'roi_1y',
            'esg_score', 'environmental_score',
            'social_score', 'governance_score', 'price_change_24h'
        ]

        # Create model directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)

        # Try to load pre-trained model if it exists
        try:
            self.load_model()
            print("Pre-trained model loaded successfully.")
        except:
            print("No pre-trained model found. Will train a new model when needed.")

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input data for the model.

        Args:
            data (pd.DataFrame): Raw input data

        Returns:
            np.ndarray: Preprocessed features
        """
        # Normalize column names (case-insensitive)
        data_normalized = data.copy()
        data_normalized.columns = [col.lower() for col in data_normalized.columns]

        # Create a DataFrame with the required features
        features = pd.DataFrame(index=data_normalized.index)

        # Map of possible column names for each feature
        column_mappings = {
            'market_cap': ['market_cap', 'market_cap_b', 'cap', 'size'],
            'volatility': ['volatility', 'vol', 'risk'],
            'roi_1y': ['roi_1y', 'roi', 'return_1y', 'annual_return'],
            'esg_score': ['esg_score', 'esg', 'sustainability_score'],
            'environmental_score': ['environmental_score', 'environmental', 'env_score'],
            'social_score': ['social_score', 'social', 'soc_score'],
            'governance_score': ['governance_score', 'governance', 'gov_score'],
            'price_change_24h': ['price_change_24h', 'price_change', 'daily_change', 'change_24h']
        }

        # Find and map columns
        for feature, possible_names in column_mappings.items():
            # Try to find a matching column
            found = False
            for name in possible_names:
                if name in data_normalized.columns:
                    features[feature] = data_normalized[name]
                    found = True
                    break

            # If not found in lowercase, try original case
            if not found:
                for name in possible_names:
                    capitalized_name = name.capitalize()
                    if capitalized_name in data.columns:
                        features[feature] = data[capitalized_name]
                        found = True
                        break

            # If still not found, use a default value
            if not found:
                if feature == 'market_cap':
                    features[feature] = 10  # Default market cap
                elif feature == 'volatility':
                    features[feature] = 0.3  # Default volatility
                elif feature == 'roi_1y':
                    features[feature] = 10  # Default ROI
                elif feature == 'esg_score':
                    features[feature] = 50  # Default ESG score
                elif feature in ['environmental_score', 'social_score', 'governance_score']:
                    features[feature] = 50  # Default component scores
                elif feature == 'price_change_24h':
                    features[feature] = 0.0  # Default price change

        # Handle missing values
        features = features.fillna(features.mean())

        # Scale features
        scaled_features = self.scaler.fit_transform(features)

        return scaled_features

    def train(self, data: pd.DataFrame, target: str = 'performance_score'):
        """
        Train the model on historical data.

        Args:
            data (pd.DataFrame): Training data
            target (str): Target variable name
        """
        X = self.preprocess_data(data)
        y = data[target]

        self.model.fit(X, y)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            np.ndarray: Predicted scores
        """
        X = self.preprocess_data(data)
        return self.model.predict(X)

    def generate_recommendations(
        self,
        all_assets: pd.DataFrame,
        user_preferences: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized investment recommendations.

        Args:
            all_assets (pd.DataFrame): Available assets data
            user_preferences (Dict[str, Any]): User preferences
            top_n (int): Number of recommendations to generate

        Returns:
            List[Dict[str, Any]]: Ranked recommendations
        """
        # Get base predictions
        base_scores = self.predict(all_assets)

        # Adjust scores based on user preferences
        adjusted_scores = self._adjust_scores(
            base_scores,
            all_assets,
            user_preferences
        )

        # Rank and select top recommendations
        recommendations = self._rank_recommendations(
            all_assets,
            adjusted_scores,
            top_n
        )

        return recommendations

    def _adjust_scores(
        self,
        base_scores: np.ndarray,
        assets: pd.DataFrame,
        preferences: Dict[str, Any]
    ) -> np.ndarray:
        """
        Adjust prediction scores based on user preferences.

        Args:
            base_scores (np.ndarray): Base model predictions
            assets (pd.DataFrame): Asset data
            preferences (Dict[str, Any]): User preferences

        Returns:
            np.ndarray: Adjusted scores
        """
        # Get preference weights
        risk_weight = (11 - preferences['risk_tolerance']) / 10
        esg_weight = preferences['sustainability_focus'] / 10
        return_weight = 1 - risk_weight - esg_weight/2

        # Calculate component scores
        risk_scores = 1 - assets['volatility']
        esg_scores = assets['esg_score'] / 100
        return_scores = assets['roi_1y'] / 100

        # Combine scores
        adjusted_scores = (
            base_scores * 0.4 +
            risk_scores * risk_weight * 0.2 +
            esg_scores * esg_weight * 0.2 +
            return_scores * return_weight * 0.2
        )

        return adjusted_scores

    def _rank_recommendations(
        self,
        assets: pd.DataFrame,
        scores: np.ndarray,
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Rank assets and create recommendation objects.

        Args:
            assets (pd.DataFrame): Asset data
            scores (np.ndarray): Adjusted prediction scores
            top_n (int): Number of recommendations to return

        Returns:
            List[Dict[str, Any]]: Ranked recommendations
        """
        # Create DataFrame with scores
        ranked_assets = assets.copy()
        ranked_assets['score'] = scores

        # Sort by score
        ranked_assets = ranked_assets.sort_values('score', ascending=False)

        # Select top N recommendations
        top_recommendations = []
        for _, asset in ranked_assets.head(top_n).iterrows():
            recommendation = {
                'ticker': asset['Ticker'],
                'name': asset['Name'],
                'sector': asset['Sector'],
                'current_price': asset['Current_Price'],
                'price_change': asset['Price_Change_24h'],
                'esg_score': asset['ESG_Score'],
                'roi_1y': asset['ROI_1Y'],
                'volatility': asset['Volatility'],
                'recommendation_score': asset['score'],
                'recommendation_strength': self._get_recommendation_strength(asset['score'])
            }
            top_recommendations.append(recommendation)

        return top_recommendations

    def _get_recommendation_strength(self, score: float) -> str:
        """
        Convert score to recommendation strength category.

        Args:
            score (float): Recommendation score

        Returns:
            str: Recommendation strength category
        """
        if score >= 0.8:
            return 'Strong Buy'
        elif score >= 0.6:
            return 'Buy'
        elif score >= 0.4:
            return 'Hold'
        elif score >= 0.2:
            return 'Consider with Caution'
        else:
            return 'Not Recommended'

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dict[str, float]: Feature importance scores
        """
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance_scores))

    def evaluate_model(
        self,
        test_data: pd.DataFrame,
        target: str = 'performance_score'
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            test_data (pd.DataFrame): Test data
            target (str): Target variable name

        Returns:
            Dict[str, float]: Performance metrics
        """
        from sklearn.metrics import mean_squared_error, r2_score

        # Generate predictions
        X_test = self.preprocess_data(test_data)
        y_test = test_data[target]
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }

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
    model = PortfolioRecommender()

    # If model not loaded, train it on synthetic data
    if model.model is None:
        training_data = generate_training_data()
        model.train(training_data)

    # Prepare data for prediction
    prediction_data = all_assets.copy()

    # Normalize column names for consistent access
    prediction_data_normalized = prediction_data.copy()
    prediction_data_normalized.columns = [col.lower() for col in prediction_data_normalized.columns]

    # Ensure we have a ticker column
    ticker_col = None
    for col_name in ['ticker', 'symbol', 'id']:
        if col_name in prediction_data_normalized.columns:
            ticker_col = col_name
            break

    # If no ticker column found in lowercase, try original case
    if ticker_col is None:
        for col_name in ['Ticker', 'Symbol', 'ID']:
            if col_name in prediction_data.columns:
                ticker_col = col_name
                break

    # If still no ticker column, create a dummy one
    if ticker_col is None:
        prediction_data['ticker'] = [f'ASSET_{i}' for i in range(len(prediction_data))]
        ticker_col = 'ticker'

    # Filter out assets already in portfolio
    if not portfolio_assets.empty:
        # Normalize portfolio column names
        portfolio_normalized = portfolio_assets.copy()
        portfolio_normalized.columns = [col.lower() for col in portfolio_normalized.columns]

        # Find ticker column in portfolio
        portfolio_ticker_col = None
        for col_name in ['ticker', 'symbol', 'id']:
            if col_name in portfolio_normalized.columns:
                portfolio_ticker_col = col_name
                break

        # If no ticker column found in lowercase, try original case
        if portfolio_ticker_col is None:
            for col_name in ['Ticker', 'Symbol', 'ID']:
                if col_name in portfolio_assets.columns:
                    portfolio_ticker_col = col_name
                    break

        # Filter only if we found ticker columns in both dataframes
        if portfolio_ticker_col is not None:
            portfolio_tickers = portfolio_assets[portfolio_ticker_col].unique()
            prediction_data = prediction_data[~prediction_data[ticker_col].isin(portfolio_tickers)]

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
