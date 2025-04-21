# ML Portfolio Manager

A comprehensive investment portfolio management system with AI-powered recommendations, ESG (Environmental, Social, and Governance) analysis, and machine learning models for sustainable investing.

## Project Overview

The ML Portfolio Manager is a sophisticated web application built with Streamlit that combines traditional financial analysis with machine learning and ESG considerations to provide personalized investment recommendations. The system uses multiple ML models to analyze assets, assess risks, and generate sustainable investment strategies.

## Key Features

### 1. Portfolio Management
- Real-time portfolio tracking and analysis
- Asset allocation visualization
- Performance metrics and historical data
- Multi-currency support
- Custom watchlists and alerts

### 2. AI-Powered Recommendations
Three main ML models power the recommendation system:

#### a. Portfolio Recommendation Model
- **Algorithm**: XGBoost
- **Features**:
  - Financial metrics (ROI, volatility, market cap)
  - ESG scores
  - Market trends
  - Risk metrics
- **Output**: Personalized asset recommendations with confidence scores
- **Training Data**: Historical market data, ESG ratings, financial performance

#### b. Risk Assessment Model
- **Algorithm**: Random Forest Classifier
- **Features**:
  - Market volatility
  - ESG risk factors
  - Industry-specific risks
  - Economic indicators
- **Output**: Risk scores and categorization (Low/Medium/High)
- **Training Data**: Historical risk events, market volatility data, ESG incidents

#### c. Sentiment Analysis Model
- **Algorithm**: BERT-based model with TF-IDF
- **Features**:
  - News headlines
  - Social media sentiment
  - Market reports
  - Company announcements
- **Output**: Sentiment scores and market sentiment trends
- **Training Data**: Financial news, social media data, market sentiment labels

### 3. ESG Integration
- Comprehensive ESG scoring system
- UN SDG (Sustainable Development Goals) alignment
- Carbon footprint tracking
- Sustainability trend analysis
- ESG risk assessment

### 4. Market Explorer
- Real-time market data visualization
- Technical analysis tools
- Fundamental analysis metrics
- ESG performance indicators
- Sector-specific insights

### 5. User Profile & Preferences
- Risk tolerance assessment
- Investment horizon setting
- Sustainability focus customization
- Sector preferences
- Asset type preferences

## Technical Architecture

### Frontend
- **Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **UI Components**: Custom CSS, Responsive Design
- **Interactive Elements**: Dynamic filters, Real-time updates

### Backend
- **Language**: Python 3.8+
- **ML Framework**: Scikit-learn, XGBoost, TensorFlow
- **Data Processing**: Pandas, NumPy
- **API Integration**: RESTful APIs, WebSocket

### Database
- **Type**: SQL/NoSQL (configurable)
- **Data Models**:
  - User profiles
  - Portfolio data
  - Market data
  - ESG metrics
  - ML model results

### ML Models Integration
```python
# Example of ML model integration
class PortfolioRecommender:
    def __init__(self):
        self.model = XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100
        )
        
    def preprocess_data(self, data):
        # Feature engineering
        features = [
            'market_cap', 'volatility', 'roi_1y',
            'esg_score', 'environmental_score',
            'social_score', 'governance_score'
        ]
        return data[features]
        
    def generate_recommendations(self, user_preferences, market_data):
        # Process data and generate recommendations
        processed_data = self.preprocess_data(market_data)
        scores = self.model.predict(processed_data)
        return self.rank_recommendations(scores, user_preferences)
```

## Data Sources

### Market Data
- Real-time price data
- Historical performance
- Trading volumes
- Market indicators

### ESG Data
- Company ESG ratings
- Sustainability reports
- Carbon emissions data
- SDG alignment metrics

### News and Sentiment
- Financial news APIs
- Social media feeds
- Market analysis reports
- Company announcements

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-portfolio-manager.git
cd ml-portfolio-manager
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

4. Run the application:
```bash
python -m streamlit run app.py
```

## Project Structure
```
ml-portfolio-manager/
├── app.py                 # Main Streamlit application
├── pages/                 # Streamlit pages
│   ├── market_explorer.py
│   ├── portfolio_manager.py
│   ├── ai_recommendations.py
│   └── esg_education.py
├── models/               # ML models
│   ├── portfolio_recommendation.py
│   ├── risk_assessment.py
│   └── sentiment_analysis.py
├── utils/               # Utility functions
│   ├── data_loader.py
│   ├── esg_calculator.py
│   └── market_analysis.py
├── data/                # Data storage
│   ├── market_data/
│   ├── esg_data/
│   └── user_data/
└── tests/              # Unit tests
```

## ML Model Details

### Portfolio Recommendation Model
```python
def train_portfolio_model(data, user_preferences):
    """
    Train the portfolio recommendation model using historical data
    and user preferences.
    """
    features = [
        'market_cap', 'volatility', 'roi_1y',
        'esg_score', 'environmental_score',
        'social_score', 'governance_score'
    ]
    
    target = 'performance_score'
    
    X = data[features]
    y = data[target]
    
    model = XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100
    )
    
    model.fit(X, y)
    return model
```

### Risk Assessment Model
```python
def assess_risk(portfolio_data):
    """
    Assess portfolio risk using multiple factors.
    """
    risk_factors = [
        'market_volatility',
        'esg_risk_score',
        'sector_risk',
        'liquidity_risk'
    ]
    
    weights = {
        'market_volatility': 0.3,
        'esg_risk_score': 0.3,
        'sector_risk': 0.2,
        'liquidity_risk': 0.2
    }
    
    risk_score = calculate_weighted_risk(portfolio_data, weights)
    return categorize_risk(risk_score)
```

### Sentiment Analysis Model
```python
def analyze_sentiment(news_data):
    """
    Analyze market sentiment using news and social media data.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10
    )
    
    X = vectorizer.fit_transform(news_data['text'])
    sentiment_scores = classifier.predict_proba(X)
    return process_sentiment_scores(sentiment_scores)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial data providers
- ESG rating agencies
- Open-source ML libraries
- Streamlit community
