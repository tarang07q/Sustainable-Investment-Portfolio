"""
Risk Assessment Model

This module provides risk assessment functionality for investment portfolios
based on various risk factors including market volatility, ESG risks,
sector-specific risks, and other relevant metrics.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

class RiskAssessor:
    def __init__(self):
        """Initialize the Risk Assessment model."""
        # Risk factor weights (can be adjusted based on requirements)
        self.risk_weights = {
            'market_volatility': 0.3,
            'esg_risk': 0.3,
            'sector_risk': 0.2,
            'liquidity_risk': 0.2
        }

        # Risk thresholds for categorization
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }

    def assess_portfolio_risk(
        self,
        portfolio_data: pd.DataFrame,
        market_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Assess the overall risk of a portfolio.

        Args:
            portfolio_data (pd.DataFrame): Portfolio holdings and metrics
            market_data (Dict[str, Any], optional): Additional market context

        Returns:
            Dict[str, Any]: Risk assessment results
        """
        try:
            # Calculate individual risk components
            market_risk = self._calculate_market_risk(portfolio_data)
            esg_risk = self._calculate_esg_risk(portfolio_data)
            sector_risk = self._calculate_sector_risk(portfolio_data)
            liquidity_risk = self._calculate_liquidity_risk(portfolio_data)

            # Calculate weighted risk score
            risk_score = (
                market_risk * self.risk_weights['market_volatility'] +
                esg_risk * self.risk_weights['esg_risk'] +
                sector_risk * self.risk_weights['sector_risk'] +
                liquidity_risk * self.risk_weights['liquidity_risk']
            )

            # Determine risk category
            risk_category = self._categorize_risk(risk_score)

            # Generate risk breakdown
            risk_breakdown = {
                'Market Risk': market_risk,
                'ESG Risk': esg_risk,
                'Sector Risk': sector_risk,
                'Liquidity Risk': liquidity_risk
            }

            return {
                'risk_score': risk_score,
                'risk_category': risk_category,
                'risk_breakdown': risk_breakdown,
                'risk_factors': self._identify_risk_factors(portfolio_data),
                'recommendations': self._generate_risk_recommendations(risk_breakdown)
            }

        except Exception as e:
            print(f"Error in risk assessment: {str(e)}")
            return self._generate_default_risk_assessment()

    def _calculate_market_risk(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate market risk based on volatility and beta."""
        try:
            # Normalize column names (case-insensitive)
            portfolio_data_normalized = portfolio_data.copy()
            portfolio_data_normalized.columns = [col.lower() for col in portfolio_data_normalized.columns]

            # Check for volatility column with different possible names
            volatility_col = None
            for col_name in ['volatility', 'vol', 'risk']:
                if col_name in portfolio_data_normalized.columns:
                    volatility_col = col_name
                    break

            # If no volatility column found, try to use uppercase version
            if volatility_col is None and 'Volatility' in portfolio_data.columns:
                volatilities = portfolio_data['Volatility'].fillna(0)
            elif volatility_col is not None:
                volatilities = portfolio_data_normalized[volatility_col].fillna(0)
            else:
                # No volatility column found, use default values
                return 0.5

            # Check for weight column
            weight_col = None
            for col_name in ['weight', 'allocation', 'portfolio_weight']:
                if col_name in portfolio_data_normalized.columns:
                    weight_col = col_name
                    break

            # If weight column found, use it; otherwise, assign equal weights
            if weight_col is not None:
                weights = portfolio_data_normalized[weight_col].fillna(1/len(portfolio_data))
            else:
                weights = pd.Series([1/len(portfolio_data)] * len(portfolio_data))

            # Calculate weighted average volatility
            portfolio_volatility = np.sum(volatilities * weights)

            # Normalize to 0-1 scale
            normalized_risk = min(portfolio_volatility / 0.5, 1.0)

            return normalized_risk

        except Exception as e:
            print(f"Error calculating market risk: {str(e)}")
            return 0.5  # Return moderate risk as fallback

    def _calculate_esg_risk(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate ESG risk based on ESG scores."""
        try:
            # Normalize column names (case-insensitive)
            portfolio_data_normalized = portfolio_data.copy()
            portfolio_data_normalized.columns = [col.lower() for col in portfolio_data_normalized.columns]

            # Check for ESG score column with different possible names
            esg_col = None
            for col_name in ['esg_score', 'esg', 'sustainability_score', 'environmental_score']:
                if col_name in portfolio_data_normalized.columns:
                    esg_col = col_name
                    break

            # If no ESG column found, try to use uppercase version
            if esg_col is None and 'ESG_Score' in portfolio_data.columns:
                esg_scores = portfolio_data['ESG_Score'].fillna(50)
            elif esg_col is not None:
                esg_scores = portfolio_data_normalized[esg_col].fillna(50)
            else:
                # No ESG column found, use default values
                return 0.5

            # Check for weight column
            weight_col = None
            for col_name in ['weight', 'allocation', 'portfolio_weight']:
                if col_name in portfolio_data_normalized.columns:
                    weight_col = col_name
                    break

            # If weight column found, use it; otherwise, assign equal weights
            if weight_col is not None:
                weights = portfolio_data_normalized[weight_col].fillna(1/len(portfolio_data))
            else:
                weights = pd.Series([1/len(portfolio_data)] * len(portfolio_data))

            # Calculate weighted average ESG risk
            esg_risk = np.sum((100 - esg_scores) / 100 * weights)

            return min(esg_risk, 1.0)

        except Exception as e:
            print(f"Error calculating ESG risk: {str(e)}")
            return 0.5

    def _calculate_sector_risk(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate sector concentration risk."""
        try:
            # Normalize column names (case-insensitive)
            portfolio_data_normalized = portfolio_data.copy()
            portfolio_data_normalized.columns = [col.lower() for col in portfolio_data_normalized.columns]

            # Check for sector column with different possible names
            sector_col = None
            for col_name in ['sector', 'industry', 'category']:
                if col_name in portfolio_data_normalized.columns:
                    sector_col = col_name
                    break

            # If no sector column found, try to use uppercase version
            if sector_col is None and 'Sector' in portfolio_data.columns:
                sector_col = 'Sector'
                sector_data = portfolio_data
            elif sector_col is not None:
                sector_data = portfolio_data_normalized
            else:
                # No sector column found, use default values
                return 0.5

            # Check for weight column
            weight_col = None
            for col_name in ['weight', 'allocation', 'portfolio_weight']:
                if col_name in portfolio_data_normalized.columns:
                    weight_col = col_name
                    break

            # If weight column found, use it; otherwise, assign equal weights
            if weight_col is None:
                # Create a weight column with equal weights
                sector_data = sector_data.copy()
                weight_col = 'temp_weight'
                sector_data[weight_col] = 1.0 / len(sector_data)

            # Calculate sector concentrations
            try:
                sector_weights = sector_data.groupby(sector_col)[weight_col].sum()

                # Higher concentration = higher risk
                if not sector_weights.empty:
                    max_sector_weight = sector_weights.max()

                    # Normalize to 0-1 scale (0.5 is considered diversified)
                    sector_risk = min(max_sector_weight / 0.5, 1.0)

                    return sector_risk
                else:
                    return 0.5
            except Exception as e:
                print(f"Error in sector groupby: {str(e)}")
                return 0.5

        except Exception as e:
            print(f"Error calculating sector risk: {str(e)}")
            return 0.5

    def _calculate_liquidity_risk(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate liquidity risk based on market cap and volume."""
        try:
            # Normalize column names (case-insensitive)
            portfolio_data_normalized = portfolio_data.copy()
            portfolio_data_normalized.columns = [col.lower() for col in portfolio_data_normalized.columns]

            # Check for market cap column with different possible names
            market_cap_col = None
            for col_name in ['market_cap_b', 'market_cap', 'cap', 'size']:
                if col_name in portfolio_data_normalized.columns:
                    market_cap_col = col_name
                    break

            # If no market cap column found, try to use uppercase version
            if market_cap_col is None and 'Market_Cap_B' in portfolio_data.columns:
                market_caps = portfolio_data['Market_Cap_B'].fillna(1)
            elif market_cap_col is not None:
                market_caps = portfolio_data_normalized[market_cap_col].fillna(1)
            else:
                # No market cap column found, use default values
                return 0.5

            # Check for weight column
            weight_col = None
            for col_name in ['weight', 'allocation', 'portfolio_weight']:
                if col_name in portfolio_data_normalized.columns:
                    weight_col = col_name
                    break

            # If weight column found, use it; otherwise, assign equal weights
            if weight_col is not None:
                weights = portfolio_data_normalized[weight_col].fillna(1/len(portfolio_data))
            else:
                weights = pd.Series([1/len(portfolio_data)] * len(portfolio_data))

            # Calculate weighted average liquidity risk
            # Smaller market cap = higher risk
            liquidity_risks = 1 / (1 + market_caps)  # Transform to 0-1 scale
            portfolio_liquidity_risk = np.sum(liquidity_risks * weights)

            return min(portfolio_liquidity_risk, 1.0)

        except Exception as e:
            print(f"Error calculating liquidity risk: {str(e)}")
            return 0.5

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into risk level."""
        if risk_score <= self.risk_thresholds['low']:
            return '🟢 Low Risk'
        elif risk_score <= self.risk_thresholds['medium']:
            return '🟡 Medium Risk'
        else:
            return '🔴 High Risk'

    def _identify_risk_factors(self, portfolio_data: pd.DataFrame) -> List[str]:
        """Identify specific risk factors in the portfolio."""
        risk_factors = []

        try:
            # Normalize column names (case-insensitive)
            portfolio_data_normalized = portfolio_data.copy()
            portfolio_data_normalized.columns = [col.lower() for col in portfolio_data_normalized.columns]

            # Check sector concentration if sector and weight columns exist
            sector_col = None
            weight_col = None

            # Find sector column
            for col_name in ['sector', 'industry', 'category']:
                if col_name in portfolio_data_normalized.columns:
                    sector_col = col_name
                    break
            if sector_col is None and 'Sector' in portfolio_data.columns:
                sector_col = 'Sector'
                sector_data = portfolio_data
            else:
                sector_data = portfolio_data_normalized

            # Find weight column
            for col_name in ['weight', 'allocation', 'portfolio_weight']:
                if col_name in portfolio_data_normalized.columns:
                    weight_col = col_name
                    break

            # Check sector concentration if both columns exist
            if sector_col is not None:
                if weight_col is None:
                    # Create temporary weight column
                    sector_data = sector_data.copy()
                    weight_col = 'temp_weight'
                    sector_data[weight_col] = 1.0 / len(sector_data)

                try:
                    sector_weights = sector_data.groupby(sector_col)[weight_col].sum()
                    if not sector_weights.empty and sector_weights.max() > 0.3:
                        risk_factors.append(f"High concentration in {sector_weights.idxmax()} sector")
                except Exception as e:
                    print(f"Error checking sector concentration: {str(e)}")

            # Check volatility if column exists
            vol_col = None
            for col_name in ['volatility', 'vol', 'risk']:
                if col_name in portfolio_data_normalized.columns:
                    vol_col = col_name
                    break
            if vol_col is None and 'Volatility' in portfolio_data.columns:
                vol_col = 'Volatility'
                vol_data = portfolio_data
            else:
                vol_data = portfolio_data_normalized

            if vol_col is not None:
                try:
                    high_vol_assets = vol_data[vol_data[vol_col] > 0.3]
                    if not high_vol_assets.empty:
                        risk_factors.append("High volatility assets present")
                except Exception as e:
                    print(f"Error checking volatility: {str(e)}")

            # Check ESG risks if column exists
            esg_col = None
            for col_name in ['esg_score', 'esg', 'sustainability_score']:
                if col_name in portfolio_data_normalized.columns:
                    esg_col = col_name
                    break
            if esg_col is None and 'ESG_Score' in portfolio_data.columns:
                esg_col = 'ESG_Score'
                esg_data = portfolio_data
            else:
                esg_data = portfolio_data_normalized

            if esg_col is not None:
                try:
                    low_esg_assets = esg_data[esg_data[esg_col] < 50]
                    if not low_esg_assets.empty:
                        risk_factors.append("Assets with low ESG scores")
                except Exception as e:
                    print(f"Error checking ESG scores: {str(e)}")

            # Check market cap (liquidity) if column exists
            cap_col = None
            for col_name in ['market_cap_b', 'market_cap', 'cap', 'size']:
                if col_name in portfolio_data_normalized.columns:
                    cap_col = col_name
                    break
            if cap_col is None and 'Market_Cap_B' in portfolio_data.columns:
                cap_col = 'Market_Cap_B'
                cap_data = portfolio_data
            else:
                cap_data = portfolio_data_normalized

            if cap_col is not None:
                try:
                    small_cap_assets = cap_data[cap_data[cap_col] < 1]
                    if not small_cap_assets.empty:
                        risk_factors.append("Small-cap assets present")
                except Exception as e:
                    print(f"Error checking market cap: {str(e)}")

            # If no risk factors identified, add a default one
            if not risk_factors:
                risk_factors.append("No specific risk factors identified")

        except Exception as e:
            print(f"Error identifying risk factors: {str(e)}")
            risk_factors.append("Unable to analyze risk factors")

        return risk_factors

    def _generate_risk_recommendations(self, risk_breakdown: Dict[str, float]) -> List[str]:
        """Generate recommendations based on risk assessment."""
        recommendations = []

        # Check each risk component
        if risk_breakdown['Market Risk'] > 0.6:
            recommendations.append("Consider reducing exposure to high-volatility assets")

        if risk_breakdown['ESG Risk'] > 0.6:
            recommendations.append("Review holdings with poor ESG scores")

        if risk_breakdown['Sector Risk'] > 0.6:
            recommendations.append("Increase sector diversification")

        if risk_breakdown['Liquidity Risk'] > 0.6:
            recommendations.append("Consider increasing allocation to more liquid assets")

        if not recommendations:
            recommendations.append("Portfolio risk levels are within acceptable ranges")

        return recommendations

    def _generate_default_risk_assessment(self) -> Dict[str, Any]:
        """Generate a default risk assessment when calculation fails."""
        return {
            'risk_score': 0.5,
            'risk_category': '🟡 Medium Risk',
            'risk_breakdown': {
                'Market Risk': 0.5,
                'ESG Risk': 0.5,
                'Sector Risk': 0.5,
                'Liquidity Risk': 0.5
            },
            'risk_factors': ['Unable to calculate specific risk factors'],
            'recommendations': ['Please review portfolio data for accurate risk assessment']
        }

def calculate_portfolio_risk(portfolio_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to calculate portfolio risk.

    Args:
        portfolio_data (pd.DataFrame): Portfolio holdings and metrics

    Returns:
        Dict[str, Any]: Risk assessment results
    """
    risk_assessor = RiskAssessor()
    return risk_assessor.assess_portfolio_risk(portfolio_data)


def assess_portfolio_risk(portfolio_data: pd.DataFrame, user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Assess portfolio risk with user preferences.

    Args:
        portfolio_data (pd.DataFrame): Portfolio holdings and metrics
        user_preferences (Dict[str, Any], optional): User preferences including risk_tolerance and sustainability_focus

    Returns:
        Dict[str, Any]: Risk assessment results with risk score and category
    """
    try:
        # Create risk assessor
        risk_assessor = RiskAssessor()

        # Get base risk assessment
        risk_assessment = risk_assessor.assess_portfolio_risk(portfolio_data)

        # Adjust risk assessment based on user preferences if provided
        if user_preferences is not None:
            # Get risk tolerance (1-10 scale)
            risk_tolerance = user_preferences.get('risk_tolerance', 5)

            # Get sustainability focus (1-10 scale)
            sustainability_focus = user_preferences.get('sustainability_focus', 5)

            # Adjust risk score based on risk tolerance
            # Lower risk tolerance = higher perceived risk
            risk_tolerance_factor = (11 - risk_tolerance) / 5  # 1.0-2.0 range

            # Adjust risk score based on sustainability focus
            # Higher sustainability focus = higher sensitivity to ESG risks
            sustainability_factor = sustainability_focus / 5  # 0.2-2.0 range

            # Calculate ESG risk component
            esg_risk = risk_assessment['risk_breakdown']['ESG Risk']

            # Apply adjustments
            adjusted_risk_score = risk_assessment['risk_score'] * (
                1 + (risk_tolerance_factor - 1) * 0.3 +  # Risk tolerance adjustment
                (sustainability_factor - 1) * 0.2 * esg_risk  # Sustainability adjustment
            )

            # Ensure risk score stays within 0-100 range
            adjusted_risk_score = max(0, min(100, adjusted_risk_score))

            # Update risk category based on adjusted score
            if adjusted_risk_score <= 30:
                risk_category = '🟢 Low Risk'
            elif adjusted_risk_score <= 60:
                risk_category = '🟡 Medium Risk'
            else:
                risk_category = '🔴 High Risk'

            # Update risk assessment
            risk_assessment['risk_score'] = adjusted_risk_score
            risk_assessment['risk_category'] = risk_category

        # Create risk factors dictionary for ML integration
        risk_factors = {
            'Market Risk': risk_assessment['risk_breakdown']['Market Risk'] * 100,
            'ESG Risk': risk_assessment['risk_breakdown']['ESG Risk'] * 100,
            'Sector Risk': risk_assessment['risk_breakdown']['Sector Risk'] * 100,
            'Liquidity Risk': risk_assessment['risk_breakdown']['Liquidity Risk'] * 100
        }

        # Add risk factors to assessment
        risk_assessment['risk_factors'] = risk_factors

        return risk_assessment

    except Exception as e:
        print(f"Error in assess_portfolio_risk: {str(e)}")
        # Return default risk assessment
        return {
            'risk_score': 50,
            'risk_category': '🟡 Medium Risk',
            'risk_breakdown': {
                'Market Risk': 0.5,
                'ESG Risk': 0.5,
                'Sector Risk': 0.5,
                'Liquidity Risk': 0.5
            },
            'risk_factors': {
                'Market Risk': 50,
                'ESG Risk': 50,
                'Sector Risk': 50,
                'Liquidity Risk': 50
            },
            'recommendations': ['Unable to calculate specific risk factors']
        }
