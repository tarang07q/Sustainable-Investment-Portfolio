# Sustainable Investment Portfolio App

An AI-powered platform that helps users create sustainable investment portfolios by balancing profitability with environmental, social, and governance (ESG) criteria. The application features a modern dark-themed interface and provides educational resources on sustainable investing that are accessible without requiring user authentication.

## Features

### Market Explorer
- Browse stocks and cryptocurrencies with ESG ratings
- Filter assets by sector, ESG score, and other criteria
- View detailed asset information including ESG breakdown
- Track price history and performance metrics

### Portfolio Manager
- Create and manage custom investment portfolios
- Track portfolio performance against benchmarks
- Analyze asset allocation and risk metrics
- Generate impact reports showing environmental and social contributions

### AI Recommendations
- Receive personalized investment recommendations based on your preferences
- Set your risk tolerance, investment horizon, and sustainability focus
- Get insights on market trends and ESG developments
- Understand the reasoning behind each recommendation

### ESG Education Center
- Learn about sustainable investing principles
- Explore the UN Sustainable Development Goals (SDGs)
- Understand impact measurement methodologies
- Access educational resources and a sustainable investing glossary
- **Publicly accessible** without requiring login or account creation
- Features a modern dark-themed interface for comfortable reading

## Setup

1. Clone the repository:
```bash
git clone https://github.com/tarang07q/Sustainable-Investment-Portfolio.git
cd Sustainable-Investment-Portfolio
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Supabase credentials:
   ```
   SUPABASE_URL=your-supabase-url
   SUPABASE_KEY=your-supabase-anon-key
   ```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Start with the home page to understand the platform's capabilities
2. Visit the ESG Education Center to learn about sustainable investing (no login required)
3. Create an account or sign in through the Authentication page to access premium features
4. Explore the Market Explorer to discover sustainable investment opportunities
5. Use the AI Recommendations page to get personalized investment advice
6. Create and manage your portfolio in the Portfolio Manager

## Data

The application uses simulated data for demonstration purposes. In a production environment, this would be replaced with real-time data from financial APIs and ESG rating providers.

## Technologies Used

- **Frontend**: Streamlit for the web interface
- **Authentication**: Supabase for user management
- **Data Sources**: Alpha Vantage API for financial data (simulated)
- **Data Processing**: Pandas and NumPy for data manipulation
- **Visualization**: Plotly for interactive charts
- **Theme**: Dark mode interface with optimized color schemes for better readability and reduced eye strain

## Project Structure

```
sustainable-investment-portfolio/
├── app.py                  # Main application file
├── update_pages.py         # Utility script for page updates
├── pages/                  # Streamlit pages
│   ├── 0_Authentication.py # User authentication
│   ├── 1_Market_Explorer.py # Asset exploration
│   ├── 2_Portfolio_Manager.py # Portfolio management
│   ├── 3_AI_Recommendations.py # AI recommendations
│   ├── 4_ESG_Education.py # Educational content
│   └── 5_User_Profile.py  # User profile management
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── financial_data.py   # Financial data handling
│   ├── quotes.py           # Finance quotes
│   ├── supabase.py         # Authentication utilities
│   ├── auth_redirect.py    # Authentication redirection
│   ├── sustainability_data.py # Sustainability data handling
│   └── theme.py           # Theme management
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not in git)
└── README.md               # Project documentation
```

## Recent Updates

- **Improved Authentication**: Enhanced the authentication page with a more intuitive interface and fixed HTML rendering issues
- **Public ESG Education**: Made the ESG Education Center accessible without requiring login
- **Dark Theme**: Implemented a dark-colored theme for better readability and reduced eye strain
- **Enhanced UI**: Improved visual hierarchy, spacing, and component styling throughout the application
- **Welcome Banner**: Added informative welcome banners for non-authenticated users
- **Navigation**: Updated navigation to provide clearer access to educational resources

## Future Enhancements

- Integration with real financial data APIs
- Machine learning models for more sophisticated recommendations
- Portfolio optimization algorithms
- Mobile application version
- Additional educational content and interactive tutorials
