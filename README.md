# Sustainable Investment Portfolio Dashboard

A Streamlit-based dashboard that helps users create sustainable investment portfolios by balancing profitability and environmental sustainability.

## Features

- Sector-based investment selection
- Profitability vs Sustainability weighting
- Interactive visualizations of company performance
- ESG and ROI-based company rankings
- Top recommendations based on combined scores
- Portfolio impact summary

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run sustainable_dashboard.py
```

## Usage

1. Select sectors you're interested in from the sidebar
2. Enter your investment amount
3. Adjust the Profitability vs Sustainability slider to set your preferences
4. View the interactive visualizations and recommendations
5. Analyze the company performance table for detailed metrics

## Data

The application uses dummy data for demonstration purposes. In a production environment, this would be replaced with real company data from financial APIs and ESG rating providers.

## Technologies Used

- Streamlit
- Pandas
- Plotly
- NumPy # Sustainable-Investment-Portfolio
