# ML Models for Sustainable Investment Portfolio - Sample Output

This document shows sample outputs from the ML models to demonstrate what you'll see when running the code in Google Colab.

## 1. Portfolio Recommendation Model

### Sample Portfolio

| Name | Ticker | Asset Type | Sector | Current Price | ESG Score | Volatility | ROI (1Y) |
|------|--------|------------|--------|--------------|-----------|------------|----------|
| GreenTech Solutions | GRNT | Stock | Technology | $145.75 | 85.2 | 0.15 | 18.5% |
| Renewable Power | RNWP | Stock | Energy | $52.30 | 92.1 | 0.18 | 15.2% |
| EcoToken | ECO | Crypto | Renewable Energy | $32.45 | 76.5 | 0.45 | 42.5% |
| Sustainable Finance | SUFN | Stock | Finance | $82.15 | 79.8 | 0.12 | 10.8% |
| WaterTech | WTRT | Stock | Utilities | $74.50 | 88.7 | 0.16 | 14.3% |

### User Preferences
- Risk Tolerance: 7/10
- Sustainability Focus: 8/10

### Top Recommendations

| Name | Ticker | Asset Type | Sector | ESG Score | ROI (1Y) | Volatility | Final Score | Recommendation |
|------|--------|------------|--------|-----------|----------|------------|-------------|----------------|
| BiodiversityCoin | BIOC | Crypto | Biodiversity | 78.5 | 35.5% | 0.52 | 87.3 | Strong Recommendation |
| CarbonOffset | CRBO | Crypto | Carbon Management | 80.2 | 42.8% | 0.48 | 86.9 | Strong Recommendation |
| Sustainable Forestry | SFST | Stock | Forestry | 90.5 | 10.2% | 0.16 | 85.4 | Strong Recommendation |
| Green Hydrogen | GRHD | Stock | Clean Energy | 89.8 | 17.5% | 0.24 | 84.7 | Strong Recommendation |
| Ocean Conservation | OCNC | Stock | Marine Conservation | 87.3 | 10.5% | 0.18 | 83.9 | Strong Recommendation |

### Feature Importance
![Feature Importance](https://i.imgur.com/JKL3m7n.png)

### Top Recommendations Visualization
![Top Recommendations](https://i.imgur.com/NOP4q5r.png)

## 2. Risk Assessment Model

### Portfolio Risk Assessment

**Risk Category:** Moderate  
**Risk Score:** 42.8/100

### Risk Probabilities
- Low: 35.2%
- Moderate: 48.7%
- High: 14.5%
- Very High: 1.6%

### Top Risk Factors
| Factor | Score |
|--------|-------|
| Market Correlation | 65.3 |
| ESG Risk | 18.7 |
| Sector Risk | 17.5 |
| Market Risk | 16.2 |
| Systematic Risk | 12.5 |

### Risk Visualization
![Risk Factors](https://i.imgur.com/QRS7t8u.png)
![Risk Gauge](https://i.imgur.com/TUV9w0x.png)

## 3. Sentiment Analysis Model

### Sentiment Analysis for GRNT

**Overall Sentiment:** Somewhat Bullish  
**Sentiment Score:** 25.0 (-100 to 100)

### Sentiment Distribution
- Positive: 5 (50%)
- Neutral: 3 (30%)
- Negative: 2 (20%)

### Recent News Headlines
1. "GreenTech Solutions reports strong quarterly earnings, exceeding analyst expectations"  
   Source: Financial Times, Date: 2023-04-01  
   Sentiment: positive

2. "GreenTech Solutions announces new sustainable initiative to reduce carbon footprint"  
   Source: Bloomberg, Date: 2023-04-05  
   Sentiment: positive

3. "GreenTech Solutions faces regulatory scrutiny over environmental practices"  
   Source: Reuters, Date: 2023-04-10  
   Sentiment: negative

4. "Analysts maintain 'Hold' rating for GreenTech Solutions stock"  
   Source: CNBC, Date: 2023-04-15  
   Sentiment: neutral

5. "GreenTech Solutions partners with tech giant for next-generation solutions"  
   Source: Wall Street Journal, Date: 2023-04-20  
   Sentiment: positive

### Sentiment Visualizations
![Sentiment Distribution](https://i.imgur.com/WXY5z6a.png)
![Sentiment Gauge](https://i.imgur.com/Z1A2b3c.png)
![Word Cloud - Positive](https://i.imgur.com/DEF7g8h.png)

## Summary

The ML models have successfully analyzed the portfolio data and provided:

1. Personalized investment recommendations based on ESG criteria and user preferences
2. Comprehensive risk assessment with detailed risk factor breakdown
3. Market sentiment analysis for selected assets

These insights can help investors make more informed decisions that align with both their financial goals and sustainability values.

---

*Note: The images shown above are placeholders. When you run the actual code in Google Colab, you'll see real visualizations generated from your data.*
