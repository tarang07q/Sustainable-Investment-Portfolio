import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Sustainable Development Goals data
SDG_DATA = {
    1: {
        'name': 'No Poverty',
        'description': 'End poverty in all its forms everywhere',
        'related_sectors': ['Finance', 'Healthcare', 'Education'],
        'key_metrics': ['Jobs created', 'Income growth', 'Financial inclusion'],
        'investment_themes': ['Microfinance', 'Affordable housing', 'Financial inclusion technologies']
    },
    2: {
        'name': 'Zero Hunger',
        'description': 'End hunger, achieve food security and improved nutrition and promote sustainable agriculture',
        'related_sectors': ['Agriculture', 'Food', 'Retail'],
        'key_metrics': ['Sustainable farming area', 'Food waste reduction', 'Nutrition improvement'],
        'investment_themes': ['Sustainable agriculture', 'Food waste reduction', 'Nutrition technologies']
    },
    3: {
        'name': 'Good Health and Well-being',
        'description': 'Ensure healthy lives and promote well-being for all at all ages',
        'related_sectors': ['Healthcare', 'Pharmaceuticals', 'Insurance'],
        'key_metrics': ['Healthcare access', 'Disease prevention', 'Mental health support'],
        'investment_themes': ['Affordable healthcare', 'Preventive medicine', 'Health technology']
    },
    4: {
        'name': 'Quality Education',
        'description': 'Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all',
        'related_sectors': ['Education', 'Technology', 'Publishing'],
        'key_metrics': ['Education access', 'Digital literacy', 'Teacher training'],
        'investment_themes': ['EdTech', 'Affordable education', 'Vocational training']
    },
    5: {
        'name': 'Gender Equality',
        'description': 'Achieve gender equality and empower all women and girls',
        'related_sectors': ['All'],
        'key_metrics': ['Gender pay gap', 'Women in leadership', 'Gender-based violence reduction'],
        'investment_themes': ['Women-led businesses', 'Gender lens investing', 'Workplace equality']
    },
    6: {
        'name': 'Clean Water and Sanitation',
        'description': 'Ensure availability and sustainable management of water and sanitation for all',
        'related_sectors': ['Utilities', 'Water Management', 'Construction'],
        'key_metrics': ['Clean water access', 'Wastewater treatment', 'Water efficiency'],
        'investment_themes': ['Water purification', 'Sanitation infrastructure', 'Water conservation']
    },
    7: {
        'name': 'Affordable and Clean Energy',
        'description': 'Ensure access to affordable, reliable, sustainable and modern energy for all',
        'related_sectors': ['Energy', 'Clean Energy', 'Renewable Energy', 'Utilities'],
        'key_metrics': ['Renewable energy generation', 'Energy efficiency', 'Energy access'],
        'investment_themes': ['Solar power', 'Wind energy', 'Energy storage', 'Smart grid']
    },
    8: {
        'name': 'Decent Work and Economic Growth',
        'description': 'Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all',
        'related_sectors': ['All'],
        'key_metrics': ['Job creation', 'Labor rights', 'Economic productivity'],
        'investment_themes': ['Fair trade', 'Ethical supply chains', 'Worker-owned enterprises']
    },
    9: {
        'name': 'Industry, Innovation and Infrastructure',
        'description': 'Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation',
        'related_sectors': ['Technology', 'Manufacturing', 'Construction', 'Transportation'],
        'key_metrics': ['R&D investment', 'Infrastructure development', 'Industrial efficiency'],
        'investment_themes': ['Sustainable infrastructure', 'Green manufacturing', 'Innovation hubs']
    },
    10: {
        'name': 'Reduced Inequalities',
        'description': 'Reduce inequality within and among countries',
        'related_sectors': ['Finance', 'Healthcare', 'Education', 'Technology'],
        'key_metrics': ['Income inequality', 'Financial inclusion', 'Digital divide reduction'],
        'investment_themes': ['Inclusive finance', 'Accessible technology', 'Community development']
    },
    11: {
        'name': 'Sustainable Cities and Communities',
        'description': 'Make cities and human settlements inclusive, safe, resilient and sustainable',
        'related_sectors': ['Construction', 'Transportation', 'Utilities', 'Real Estate'],
        'key_metrics': ['Affordable housing', 'Public transportation', 'Green spaces'],
        'investment_themes': ['Smart cities', 'Green buildings', 'Urban mobility']
    },
    12: {
        'name': 'Responsible Consumption and Production',
        'description': 'Ensure sustainable consumption and production patterns',
        'related_sectors': ['Retail', 'Manufacturing', 'Circular Economy', 'Consumer Goods'],
        'key_metrics': ['Waste reduction', 'Resource efficiency', 'Sustainable sourcing'],
        'investment_themes': ['Circular economy', 'Sustainable packaging', 'Product lifecycle management']
    },
    13: {
        'name': 'Climate Action',
        'description': 'Take urgent action to combat climate change and its impacts',
        'related_sectors': ['All'],
        'key_metrics': ['Carbon emissions', 'Climate resilience', 'Climate risk management'],
        'investment_themes': ['Carbon capture', 'Climate adaptation', 'Emissions reduction']
    },
    14: {
        'name': 'Life Below Water',
        'description': 'Conserve and sustainably use the oceans, seas and marine resources for sustainable development',
        'related_sectors': ['Marine Conservation', 'Fisheries', 'Tourism', 'Shipping'],
        'key_metrics': ['Marine protection', 'Sustainable fishing', 'Ocean pollution reduction'],
        'investment_themes': ['Sustainable fisheries', 'Ocean cleanup', 'Blue economy']
    },
    15: {
        'name': 'Life on Land',
        'description': 'Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss',
        'related_sectors': ['Forestry', 'Biodiversity', 'Agriculture', 'Land Management'],
        'key_metrics': ['Forest conservation', 'Biodiversity protection', 'Land restoration'],
        'investment_themes': ['Sustainable forestry', 'Conservation finance', 'Regenerative agriculture']
    },
    16: {
        'name': 'Peace, Justice and Strong Institutions',
        'description': 'Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels',
        'related_sectors': ['Finance', 'Technology', 'Government'],
        'key_metrics': ['Transparency', 'Anti-corruption', 'Inclusive governance'],
        'investment_themes': ['Ethical governance', 'Legal empowerment', 'Peace technology']
    },
    17: {
        'name': 'Partnerships for the Goals',
        'description': 'Strengthen the means of implementation and revitalize the global partnership for sustainable development',
        'related_sectors': ['All'],
        'key_metrics': ['Cross-sector partnerships', 'Development financing', 'Technology transfer'],
        'investment_themes': ['Impact investing', 'Public-private partnerships', 'Collaborative platforms']
    }
}

# ESG Metrics by Sector
ESG_METRICS_BY_SECTOR = {
    'Technology': {
        'environmental': ['Energy efficiency', 'E-waste management', 'Carbon footprint'],
        'social': ['Data privacy', 'Digital inclusion', 'Workforce diversity'],
        'governance': ['Cybersecurity governance', 'Ethical AI', 'Board diversity']
    },
    'Energy': {
        'environmental': ['Renewable energy transition', 'Emissions reduction', 'Water usage'],
        'social': ['Community engagement', 'Worker safety', 'Energy access'],
        'governance': ['Climate risk disclosure', 'Executive compensation', 'Lobbying transparency']
    },
    'Healthcare': {
        'environmental': ['Medical waste management', 'Facility efficiency', 'Sustainable packaging'],
        'social': ['Healthcare access', 'Drug pricing', 'Clinical trial ethics'],
        'governance': ['Quality management', 'Ethical marketing', 'Patient data protection']
    },
    'Finance': {
        'environmental': ['Climate finance', 'Environmental risk assessment', 'Green bonds'],
        'social': ['Financial inclusion', 'Community investment', 'Responsible lending'],
        'governance': ['Risk management', 'Executive compensation', 'Anti-corruption measures']
    },
    'Manufacturing': {
        'environmental': ['Resource efficiency', 'Pollution prevention', 'Circular design'],
        'social': ['Labor practices', 'Supply chain management', 'Product safety'],
        'governance': ['Quality control', 'Responsible sourcing', 'Compliance systems']
    },
    'Retail': {
        'environmental': ['Packaging reduction', 'Energy efficiency', 'Sustainable sourcing'],
        'social': ['Labor practices', 'Product safety', 'Community engagement'],
        'governance': ['Supply chain transparency', 'Anti-corruption', 'Data protection']
    },
    'Transportation': {
        'environmental': ['Emissions reduction', 'Fuel efficiency', 'Alternative fuels'],
        'social': ['Safety standards', 'Accessibility', 'Labor relations'],
        'governance': ['Regulatory compliance', 'Risk management', 'Lobbying transparency']
    },
    'Utilities': {
        'environmental': ['Renewable energy', 'Emissions reduction', 'Water management'],
        'social': ['Service reliability', 'Community engagement', 'Affordability'],
        'governance': ['Regulatory compliance', 'Infrastructure resilience', 'Emergency response']
    },
    'Agriculture': {
        'environmental': ['Sustainable farming', 'Water conservation', 'Biodiversity protection'],
        'social': ['Fair labor practices', 'Food safety', 'Community relations'],
        'governance': ['Land rights', 'Sustainable sourcing', 'Certification compliance']
    },
    'Construction': {
        'environmental': ['Green building', 'Material efficiency', 'Waste reduction'],
        'social': ['Worker safety', 'Community impact', 'Affordable housing'],
        'governance': ['Building code compliance', 'Anti-corruption', 'Quality assurance']
    },
    'Marine Conservation': {
        'environmental': ['Ocean protection', 'Sustainable fishing', 'Pollution prevention'],
        'social': ['Coastal community support', 'Education programs', 'Indigenous rights'],
        'governance': ['Marine protected areas', 'Sustainable management', 'Transparency']
    },
    'Biodiversity': {
        'environmental': ['Habitat protection', 'Species conservation', 'Ecosystem restoration'],
        'social': ['Indigenous rights', 'Community engagement', 'Education'],
        'governance': ['Conservation management', 'Anti-poaching measures', 'Transparency']
    },
    'Clean Air': {
        'environmental': ['Emissions reduction', 'Air quality monitoring', 'Clean technology'],
        'social': ['Public health impact', 'Community engagement', 'Education'],
        'governance': ['Regulatory compliance', 'Emissions disclosure', 'Policy advocacy']
    },
    'Materials': {
        'environmental': ['Recycled content', 'Lifecycle assessment', 'Waste reduction'],
        'social': ['Product safety', 'Labor practices', 'Community impact'],
        'governance': ['Responsible sourcing', 'Quality control', 'Compliance systems']
    },
    'Carbon Management': {
        'environmental': ['Carbon capture', 'Emissions reduction', 'Climate impact'],
        'social': ['Just transition', 'Community engagement', 'Education'],
        'governance': ['Carbon accounting', 'Climate risk disclosure', 'Policy advocacy']
    },
    'Forestry': {
        'environmental': ['Sustainable forestry', 'Biodiversity protection', 'Carbon sequestration'],
        'social': ['Indigenous rights', 'Community forestry', 'Worker safety'],
        'governance': ['Certification compliance', 'Illegal logging prevention', 'Transparency']
    },
    'Clean Energy': {
        'environmental': ['Renewable generation', 'Emissions avoidance', 'Land use'],
        'social': ['Energy access', 'Community benefits', 'Just transition'],
        'governance': ['Project development standards', 'Stakeholder engagement', 'Transparency']
    },
    'Waste Management': {
        'environmental': ['Recycling rates', 'Landfill diversion', 'Hazardous waste'],
        'social': ['Community impact', 'Worker safety', 'Environmental justice'],
        'governance': ['Regulatory compliance', 'Facility management', 'Transparency']
    },
    'Green Technology': {
        'environmental': ['Resource efficiency', 'Emissions reduction', 'Lifecycle impact'],
        'social': ['Accessibility', 'Digital inclusion', 'Education'],
        'governance': ['Intellectual property', 'Ethical standards', 'Transparency']
    },
    'Renewable Energy': {
        'environmental': ['Carbon displacement', 'Land/water impact', 'Biodiversity'],
        'social': ['Community benefits', 'Energy access', 'Just transition'],
        'governance': ['Project standards', 'Stakeholder engagement', 'Transparency']
    },
    'Sustainable Supply Chain': {
        'environmental': ['Carbon footprint', 'Resource efficiency', 'Waste reduction'],
        'social': ['Labor practices', 'Human rights', 'Community impact'],
        'governance': ['Traceability', 'Supplier standards', 'Compliance verification']
    },
    'Carbon Credits': {
        'environmental': ['Additionality', 'Permanence', 'Co-benefits'],
        'social': ['Community benefits', 'Indigenous rights', 'Stakeholder engagement'],
        'governance': ['Verification standards', 'Registry systems', 'Transparency']
    },
    'Future Tech': {
        'environmental': ['Energy efficiency', 'Material usage', 'Lifecycle impact'],
        'social': ['Accessibility', 'Digital divide', 'Privacy'],
        'governance': ['Ethical standards', 'Risk management', 'Transparency']
    },
    'Sustainable Agriculture': {
        'environmental': ['Soil health', 'Water conservation', 'Biodiversity'],
        'social': ['Farmer livelihoods', 'Food security', 'Rural development'],
        'governance': ['Certification standards', 'Land rights', 'Transparency']
    },
    'Circular Economy': {
        'environmental': ['Material recovery', 'Waste reduction', 'Design for recycling'],
        'social': ['Job creation', 'Community engagement', 'Accessibility'],
        'governance': ['Product stewardship', 'Extended producer responsibility', 'Transparency']
    },
    'Water Management': {
        'environmental': ['Water efficiency', 'Quality protection', 'Ecosystem health'],
        'social': ['Access to clean water', 'Community engagement', 'Public health'],
        'governance': ['Water rights', 'Regulatory compliance', 'Stakeholder engagement']
    },
    'Ethical Technology': {
        'environmental': ['Energy efficiency', 'E-waste', 'Resource use'],
        'social': ['Digital rights', 'Accessibility', 'Privacy protection'],
        'governance': ['Ethical AI principles', 'Data governance', 'Transparency']
    },
    'Solar Energy': {
        'environmental': ['Carbon displacement', 'Land use', 'End-of-life recycling'],
        'social': ['Energy access', 'Job creation', 'Community benefits'],
        'governance': ['Project standards', 'Supply chain ethics', 'Transparency']
    }
}

# Climate Risk Data
CLIMATE_RISK_DATA = {
    'physical_risks': {
        'acute': ['Extreme weather events', 'Floods', 'Wildfires', 'Hurricanes/cyclones', 'Heatwaves'],
        'chronic': ['Sea level rise', 'Changing precipitation patterns', 'Rising temperatures', 'Water stress', 'Biodiversity loss']
    },
    'transition_risks': {
        'policy_legal': ['Carbon pricing', 'Emissions regulations', 'Climate litigation', 'Mandatory disclosures', 'Subsidies phase-out'],
        'technology': ['Clean tech disruption', 'Stranded assets', 'Energy storage breakthroughs', 'Electrification', 'Renewable cost decline'],
        'market': ['Changing consumer preferences', 'Increased cost of raw materials', 'Energy price volatility', 'Shifting investor preferences'],
        'reputation': ['Stigmatization of sector', 'Increased stakeholder concern', 'Public perception shifts']
    },
    'opportunities': {
        'resource_efficiency': ['Energy efficiency', 'Water efficiency', 'Waste reduction', 'Circular economy', 'Material substitution'],
        'energy_source': ['Renewable energy', 'Energy storage', 'Grid modernization', 'Distributed generation'],
        'products_services': ['Low-emission products', 'Climate adaptation solutions', 'Green building', 'Sustainable transportation'],
        'markets': ['New markets access', 'Public sector incentives', 'Green bonds', 'Sustainable finance'],
        'resilience': ['Resource diversification', 'Supply chain resilience', 'Climate adaptation measures']
    },
    'sector_exposure': {
        'high_risk': ['Oil & Gas', 'Coal', 'Transportation', 'Agriculture', 'Construction'],
        'medium_risk': ['Manufacturing', 'Retail', 'Technology', 'Healthcare', 'Finance'],
        'low_risk': ['Renewable Energy', 'Green Technology', 'Water Management', 'Circular Economy', 'Sustainable Agriculture']
    }
}

# Sustainability Trends
SUSTAINABILITY_TRENDS = [
    {
        'title': 'Renewable Energy Growth',
        'description': 'Renewable energy companies are showing strong growth potential due to increasing global commitments to carbon reduction.',
        'impact': 'Positive for clean energy stocks and related cryptocurrencies',
        'confidence': 85,
        'related_sectors': ['Energy', 'Clean Energy', 'Renewable Energy']
    },
    {
        'title': 'ESG Regulation Strengthening',
        'description': 'New ESG disclosure requirements are being implemented across major markets, affecting corporate reporting and compliance.',
        'impact': 'Positive for companies with strong ESG practices, negative for laggards',
        'confidence': 92,
        'related_sectors': ['All']
    },
    {
        'title': 'Green Technology Innovation',
        'description': 'Breakthrough technologies in carbon capture and sustainable materials are creating new investment opportunities.',
        'impact': 'Positive for green tech and sustainable material companies',
        'confidence': 78,
        'related_sectors': ['Technology', 'Materials', 'Carbon Management']
    },
    {
        'title': 'Crypto Energy Consumption Concerns',
        'description': 'Increasing scrutiny of cryptocurrency energy usage is driving a shift toward more energy-efficient protocols.',
        'impact': 'Positive for eco-friendly cryptocurrencies, negative for energy-intensive ones',
        'confidence': 88,
        'related_sectors': ['Green Technology', 'Renewable Energy']
    },
    {
        'title': 'Sustainable Supply Chain Demand',
        'description': 'Consumer and regulatory pressure is increasing demand for transparent and sustainable supply chains.',
        'impact': 'Positive for companies with robust supply chain sustainability',
        'confidence': 82,
        'related_sectors': ['Retail', 'Manufacturing', 'Sustainable Supply Chain']
    },
    {
        'title': 'Biodiversity Conservation Focus',
        'description': 'Growing awareness of biodiversity loss is driving investment in conservation and sustainable resource management.',
        'impact': 'Positive for companies focused on biodiversity preservation and sustainable land use',
        'confidence': 79,
        'related_sectors': ['Biodiversity', 'Forestry', 'Marine Conservation']
    },
    {
        'title': 'Water Scarcity Solutions',
        'description': 'Increasing water scarcity is creating demand for water conservation technologies and sustainable water management.',
        'impact': 'Positive for water technology companies and utilities with strong water management practices',
        'confidence': 84,
        'related_sectors': ['Utilities', 'Water Management']
    },
    {
        'title': 'Carbon Markets Expansion',
        'description': 'Carbon markets are growing rapidly as more countries implement carbon pricing mechanisms and net-zero commitments.',
        'impact': 'Positive for carbon credit providers and companies with carbon reduction technologies',
        'confidence': 86,
        'related_sectors': ['Carbon Management', 'Carbon Credits']
    },
    {
        'title': 'Circular Economy Acceleration',
        'description': 'Circular business models are gaining traction as resource scarcity and waste concerns drive innovation.',
        'impact': 'Positive for companies with circular business models and waste reduction technologies',
        'confidence': 81,
        'related_sectors': ['Circular Economy', 'Manufacturing', 'Waste Management']
    },
    {
        'title': 'Green Hydrogen Momentum',
        'description': 'Green hydrogen is emerging as a key solution for hard-to-decarbonize sectors and energy storage.',
        'impact': 'Positive for companies developing hydrogen technologies and infrastructure',
        'confidence': 77,
        'related_sectors': ['Clean Energy', 'Energy', 'Transportation']
    },
    {
        'title': 'Nature-Based Solutions Growth',
        'description': 'Nature-based solutions for climate mitigation and adaptation are attracting increased investment.',
        'impact': 'Positive for forestry, agriculture, and conservation-focused companies',
        'confidence': 75,
        'related_sectors': ['Forestry', 'Agriculture', 'Biodiversity', 'Carbon Credits']
    },
    {
        'title': 'Sustainable Finance Expansion',
        'description': 'Green bonds, sustainability-linked loans, and other sustainable finance instruments are growing rapidly.',
        'impact': 'Positive for financial institutions with strong sustainability capabilities',
        'confidence': 89,
        'related_sectors': ['Finance', 'All']
    }
]

# Function to generate sector-specific recommendations
def generate_sector_recommendations(sector, risk_tolerance, sustainability_focus):
    """
    Generate sector-specific investment recommendations based on risk tolerance and sustainability focus
    
    Parameters:
    sector (str): The sector to generate recommendations for
    risk_tolerance (str or int): Risk tolerance level (can be numeric 1-10 or string like "Low", "Moderate", "High")
    sustainability_focus (str or int): Sustainability focus level (can be numeric 1-10 or string like "Financial Returns First", "Balanced", "Impact First")
    
    Returns:
    dict: Recommendation details including rationale, key metrics, and specific assets
    """
    
    # Convert string risk tolerance to numeric if needed
    if isinstance(risk_tolerance, str):
        risk_map = {"Very Low": 1, "Low": 3, "Moderate": 5, "High": 7, "Very High": 9}
        risk_tolerance = risk_map.get(risk_tolerance, 5)
    
    # Convert string sustainability focus to numeric if needed
    if isinstance(sustainability_focus, str):
        focus_map = {"Financial Returns First": 3, "Balanced Approach": 5, "Impact First": 8}
        sustainability_focus = focus_map.get(sustainability_focus, 5)
    
    # Get sector-specific ESG metrics
    esg_metrics = ESG_METRICS_BY_SECTOR.get(sector, ESG_METRICS_BY_SECTOR['Technology'])
    
    # Find related SDGs
    related_sdgs = []
    for sdg_num, sdg_data in SDG_DATA.items():
        if sector in sdg_data['related_sectors'] or 'All' in sdg_data['related_sectors']:
            related_sdgs.append(f"SDG {sdg_num}: {sdg_data['name']}")
    
    # Generate recommendation based on risk and sustainability preferences
    if risk_tolerance < 4:  # Low risk
        if sustainability_focus > 7:  # High sustainability focus
            approach = "Conservative sustainability-focused"
            rationale = f"For the {sector} sector, we recommend a conservative approach with strong ESG integration, focusing on established companies with proven sustainability practices."
        else:
            approach = "Conservative balanced"
            rationale = f"For the {sector} sector, we recommend a conservative approach balancing financial stability with basic ESG risk management."
    elif risk_tolerance > 7:  # High risk
        if sustainability_focus > 7:  # High sustainability focus
            approach = "Growth-oriented impact"
            rationale = f"For the {sector} sector, we recommend a growth-oriented approach targeting innovative companies driving sustainability transformation."
        else:
            approach = "Growth-oriented financial"
            rationale = f"For the {sector} sector, we recommend a growth-oriented approach focusing on financial returns with baseline ESG risk management."
    else:  # Moderate risk
        if sustainability_focus > 7:  # High sustainability focus
            approach = "Balanced sustainability-focused"
            rationale = f"For the {sector} sector, we recommend a balanced approach with strong ESG integration and moderate growth potential."
        else:
            approach = "Balanced conventional"
            rationale = f"For the {sector} sector, we recommend a balanced approach with standard ESG risk management and solid financial fundamentals."
    
    # Find relevant trends
    relevant_trends = []
    for trend in SUSTAINABILITY_TRENDS:
        if sector in trend['related_sectors'] or 'All' in trend['related_sectors']:
            relevant_trends.append(trend['title'])
    
    # Compile recommendation
    recommendation = {
        'sector': sector,
        'approach': approach,
        'rationale': rationale,
        'key_esg_metrics': {
            'environmental': random.sample(esg_metrics['environmental'], min(2, len(esg_metrics['environmental']))),
            'social': random.sample(esg_metrics['social'], min(2, len(esg_metrics['social']))),
            'governance': random.sample(esg_metrics['governance'], min(2, len(esg_metrics['governance'])))
        },
        'related_sdgs': related_sdgs[:3],  # Top 3 related SDGs
        'relevant_trends': relevant_trends[:2],  # Top 2 relevant trends
        'climate_risk_exposure': 'Low' if sector in CLIMATE_RISK_DATA['sector_exposure']['low_risk'] else 
                               'High' if sector in CLIMATE_RISK_DATA['sector_exposure']['high_risk'] else 'Medium'
    }
    
    return recommendation

# Function to generate portfolio impact metrics
def generate_portfolio_impact_metrics(portfolio_assets):
    """
    Generate impact metrics for a portfolio based on its assets
    
    Parameters:
    portfolio_assets (list): List of portfolio assets with sector and allocation information
    
    Returns:
    dict: Impact metrics including carbon, water, social impact, and SDG alignment
    """
    
    # Calculate weighted average ESG score
    total_allocation = sum(asset.get('allocation', 0) for asset in portfolio_assets)
    weighted_esg = sum(asset.get('esg_score', 75) * asset.get('allocation', 0) for asset in portfolio_assets) / total_allocation if total_allocation > 0 else 0
    
    # Calculate carbon metrics
    carbon_intensity = sum(asset.get('carbon_footprint', 30) * asset.get('allocation', 0) for asset in portfolio_assets) / total_allocation if total_allocation > 0 else 0
    carbon_avoided = carbon_intensity * 0.2 * sum(asset.get('current_value', 1000) for asset in portfolio_assets) / 1000000
    
    # Identify sectors in portfolio
    sectors = set(asset.get('sector', 'Technology') for asset in portfolio_assets)
    
    # Find SDGs aligned with portfolio sectors
    aligned_sdgs = {}
    for sdg_num, sdg_data in SDG_DATA.items():
        for sector in sectors:
            if sector in sdg_data['related_sectors'] or 'All' in sdg_data['related_sectors']:
                # Calculate contribution level based on allocations to relevant sectors
                sector_allocation = sum(asset.get('allocation', 0) for asset in portfolio_assets if asset.get('sector') == sector)
                contribution = 'High' if sector_allocation > 0.2 else 'Medium' if sector_allocation > 0.1 else 'Low'
                aligned_sdgs[sdg_num] = {
                    'name': sdg_data['name'],
                    'contribution': contribution
                }
                break
    
    # Generate water and waste metrics based on sectors
    water_intensive_sectors = ['Agriculture', 'Utilities', 'Manufacturing', 'Energy']
    water_usage = 40 + sum(5 for asset in portfolio_assets if asset.get('sector') in water_intensive_sectors)
    
    waste_intensive_sectors = ['Manufacturing', 'Retail', 'Construction', 'Technology']
    waste_reduction = 25 + sum(2 for asset in portfolio_assets if asset.get('sector') in waste_intensive_sectors)
    
    # Generate renewable energy percentage based on energy sector exposure
    renewable_energy = 40
    for asset in portfolio_assets:
        if asset.get('sector') in ['Renewable Energy', 'Clean Energy']:
            renewable_energy += 10 * asset.get('allocation', 0) / 0.1
        elif asset.get('sector') == 'Energy':
            renewable_energy += 5 * asset.get('allocation', 0) / 0.1
    renewable_energy = min(95, renewable_energy)
    
    # Generate social impact metrics
    jobs_supported = int(sum(asset.get('current_value', 1000) for asset in portfolio_assets) / 1000)
    
    diversity_focused_sectors = ['Technology', 'Healthcare', 'Finance', 'Retail']
    diversity_score = 65 + sum(3 for asset in portfolio_assets if asset.get('sector') in diversity_focused_sectors)
    diversity_score = min(95, diversity_score)
    
    # Compile impact metrics
    impact_metrics = {
        'esg_score': round(weighted_esg, 1),
        'carbon': {
            'intensity': round(carbon_intensity, 1),
            'avoided': round(carbon_avoided, 1),
            'reduction_potential': round(carbon_intensity * 0.3, 1)
        },
        'water': {
            'usage': round(water_usage, 1),
            'reduction_potential': round(water_usage * 0.25, 1)
        },
        'waste': {
            'reduction': round(waste_reduction, 1),
            'circular_economy_alignment': 'Medium' if waste_reduction > 30 else 'Low'
        },
        'energy': {
            'renewable_percentage': round(renewable_energy, 1)
        },
        'social': {
            'jobs_supported': jobs_supported,
            'diversity_score': round(diversity_score, 1)
        },
        'sdg_alignment': {
            'aligned_sdgs': aligned_sdgs,
            'count': len(aligned_sdgs)
        }
    }
    
    return impact_metrics
