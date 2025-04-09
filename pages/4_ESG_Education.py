import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page config
st.set_page_config(
    page_title="ESG Education - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Get theme from session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Apply theme-specific styles
theme_bg_color = "#0e1117" if st.session_state.theme == "dark" else "#ffffff"
theme_text_color = "#ffffff" if st.session_state.theme == "dark" else "#0e1117"
theme_secondary_bg = "#1e2530" if st.session_state.theme == "dark" else "#f0f2f6"
theme_card_bg = "#262730" if st.session_state.theme == "dark" else "white"

# Set Plotly theme based on app theme
plotly_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

# Custom CSS with dynamic theming
st.markdown(f"""
    <style>
    .main {{
        padding: 2rem;
        background-color: {theme_bg_color};
        color: {theme_text_color};
    }}
    .stButton>button {{
        width: 100%;
    }}
    .education-card {{
        background-color: {theme_secondary_bg};
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: {theme_text_color};
    }}
    .sdg-card {{
        background-color: {theme_card_bg};
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin: 0.5rem 0;
        display: flex;
        flex-direction: column;
        height: 100%;
        color: {theme_text_color};
    }}
    .sdg-card img {{
        max-width: 80px;
        margin: 0 auto;
    }}
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📚 ESG Education Center")
st.markdown("*Learn about sustainable investing principles and ESG criteria*")

# Main navigation tabs
tabs = st.tabs(["ESG Basics", "SDG Framework", "Impact Measurement", "Resources"])

# ESG Basics tab
with tabs[0]:
    st.markdown("## Understanding ESG Investing")

    st.markdown("""
    ESG investing is an approach that incorporates Environmental, Social, and Governance factors
    into investment decisions, alongside traditional financial analysis. This approach aims to
    generate long-term competitive financial returns while creating positive societal impact.
    """)

    # ESG components
    st.markdown("### ESG Components")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="education-card" style="border-left: 5px solid #4CAF50;">
            <h4>Environmental</h4>
            <p>Factors related to a company's impact on the natural environment:</p>
            <ul>
                <li>Climate change and carbon emissions</li>
                <li>Resource depletion and waste management</li>
                <li>Pollution and environmental degradation</li>
                <li>Renewable energy and energy efficiency</li>
                <li>Biodiversity and ecosystem protection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="education-card" style="border-left: 5px solid #2196F3;">
            <h4>Social</h4>
            <p>Factors related to a company's relationships with people and society:</p>
            <ul>
                <li>Labor standards and working conditions</li>
                <li>Human rights and community relations</li>
                <li>Diversity, equity, and inclusion</li>
                <li>Customer satisfaction and product safety</li>
                <li>Data privacy and security</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="education-card" style="border-left: 5px solid #9C27B0;">
            <h4>Governance</h4>
            <p>Factors related to a company's leadership and oversight:</p>
            <ul>
                <li>Board composition and independence</li>
                <li>Executive compensation and incentives</li>
                <li>Business ethics and anti-corruption</li>
                <li>Transparency and disclosure</li>
                <li>Shareholder rights and engagement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ESG integration approaches
    st.markdown("### ESG Integration Approaches")

    approaches = {
        'Approach': [
            'Negative/Exclusionary Screening',
            'Positive/Best-in-Class Screening',
            'Norms-Based Screening',
            'ESG Integration',
            'Sustainability Themed Investing',
            'Impact Investing',
            'Corporate Engagement & Shareholder Action'
        ],
        'Description': [
            'Excluding companies or sectors involved in controversial activities (e.g., tobacco, weapons)',
            'Investing in companies with better ESG performance relative to peers',
            'Screening investments against minimum standards of business practice',
            'Systematically including ESG factors in financial analysis',
            'Investing in themes or assets specifically related to sustainability',
            'Investing with the intention to generate positive, measurable social and environmental impact',
            'Using shareholder power to influence corporate behavior'
        ],
        'Complexity': [
            2, 3, 3, 4, 4, 5, 5
        ]
    }

    approaches_df = pd.DataFrame(approaches)

    fig = px.bar(
        approaches_df,
        x='Approach',
        y='Complexity',
        color='Complexity',
        hover_data=['Description'],
        title='ESG Investment Approaches by Complexity',
        color_continuous_scale='Viridis',
        template=plotly_template
    )

    fig.update_layout(xaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    for _, row in approaches_df.iterrows():
        st.markdown(f"**{row['Approach']}**: {row['Description']}")

    # ESG ratings
    st.markdown("### Understanding ESG Ratings")

    st.markdown("""
    ESG ratings are assessments of a company's exposure to ESG risks and its management of those risks
    relative to industry peers. These ratings are provided by specialized agencies that analyze various
    data points from company disclosures, public sources, and proprietary research.

    **Key ESG rating providers include:**
    - MSCI ESG Ratings
    - Sustainalytics
    - S&P Global ESG Scores
    - Bloomberg ESG Disclosure Scores
    - ISS ESG Corporate Rating

    **Challenges with ESG ratings:**
    - Lack of standardization across rating providers
    - Data quality and availability issues
    - Different methodologies and weightings
    - Focus on risk rather than impact

    It's important to understand the methodology behind ESG ratings and use multiple sources when
    evaluating a company's ESG performance.
    """)

    # ESG performance and financial returns
    st.markdown("### ESG Performance and Financial Returns")

    st.markdown("""
    Research on the relationship between ESG performance and financial returns has shown:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="education-card">
            <h4>Potential Benefits</h4>
            <ul>
                <li>Better risk management and lower volatility</li>
                <li>Improved operational efficiency and cost savings</li>
                <li>Enhanced brand reputation and customer loyalty</li>
                <li>Attraction and retention of talent</li>
                <li>Innovation and new market opportunities</li>
                <li>Reduced regulatory and legal risks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="education-card">
            <h4>Potential Challenges</h4>
            <ul>
                <li>Short-term performance may vary</li>
                <li>ESG implementation costs</li>
                <li>Data quality and comparability issues</li>
                <li>Sector and regional biases</li>
                <li>Greenwashing risks</li>
                <li>Evolving regulatory landscape</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Meta-analysis visualization
    st.markdown("#### Meta-Analysis of ESG Studies")

    study_data = {
        'Finding': [
            'Positive Relationship',
            'Neutral Relationship',
            'Negative Relationship',
            'Mixed Results'
        ],
        'Percentage': [58, 25, 8, 9]
    }

    study_df = pd.DataFrame(study_data)

    fig = px.pie(
        study_df,
        values='Percentage',
        names='Finding',
        title='Meta-Analysis of Studies on ESG and Financial Performance',
        color_discrete_map={
            'Positive Relationship': '#4CAF50',
            'Neutral Relationship': '#FFC107',
            'Negative Relationship': '#F44336',
            'Mixed Results': '#9E9E9E'
        },
        template=plotly_template
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    *Note: This chart represents a simplified view of meta-analyses of academic studies on the relationship
    between ESG factors and financial performance. Actual results vary by industry, time period, and methodology.*
    """)

# SDG Framework tab
with tabs[1]:
    st.markdown("## The UN Sustainable Development Goals (SDGs)")

    st.markdown("""
    The United Nations Sustainable Development Goals (SDGs) are a collection of 17 interlinked global
    goals designed to be a "blueprint to achieve a better and more sustainable future for all."
    The SDGs were set in 2015 by the United Nations General Assembly and are intended to be achieved by 2030.

    Investors can use the SDG framework to align their portfolios with specific sustainability objectives
    and measure their contribution to global sustainability challenges.
    """)

    # SDG overview
    st.markdown("### The 17 Sustainable Development Goals")

    # Create a grid of SDGs
    col1, col2, col3, col4 = st.columns(4)

    sdgs = [
        {
            "number": 1,
            "name": "No Poverty",
            "description": "End poverty in all its forms everywhere",
            "color": "#E5243B"
        },
        {
            "number": 2,
            "name": "Zero Hunger",
            "description": "End hunger, achieve food security and improved nutrition",
            "color": "#DDA63A"
        },
        {
            "number": 3,
            "name": "Good Health and Well-being",
            "description": "Ensure healthy lives and promote well-being for all",
            "color": "#4C9F38"
        },
        {
            "number": 4,
            "name": "Quality Education",
            "description": "Ensure inclusive and equitable quality education",
            "color": "#C5192D"
        },
        {
            "number": 5,
            "name": "Gender Equality",
            "description": "Achieve gender equality and empower all women and girls",
            "color": "#FF3A21"
        },
        {
            "number": 6,
            "name": "Clean Water and Sanitation",
            "description": "Ensure availability and sustainable management of water",
            "color": "#26BDE2"
        },
        {
            "number": 7,
            "name": "Affordable and Clean Energy",
            "description": "Ensure access to affordable, reliable, sustainable energy",
            "color": "#FCC30B"
        },
        {
            "number": 8,
            "name": "Decent Work and Economic Growth",
            "description": "Promote sustained, inclusive and sustainable economic growth",
            "color": "#A21942"
        },
        {
            "number": 9,
            "name": "Industry, Innovation and Infrastructure",
            "description": "Build resilient infrastructure and foster innovation",
            "color": "#FD6925"
        },
        {
            "number": 10,
            "name": "Reduced Inequalities",
            "description": "Reduce inequality within and among countries",
            "color": "#DD1367"
        },
        {
            "number": 11,
            "name": "Sustainable Cities and Communities",
            "description": "Make cities inclusive, safe, resilient and sustainable",
            "color": "#FD9D24"
        },
        {
            "number": 12,
            "name": "Responsible Consumption and Production",
            "description": "Ensure sustainable consumption and production patterns",
            "color": "#BF8B2E"
        },
        {
            "number": 13,
            "name": "Climate Action",
            "description": "Take urgent action to combat climate change and its impacts",
            "color": "#3F7E44"
        },
        {
            "number": 14,
            "name": "Life Below Water",
            "description": "Conserve and sustainably use oceans, seas and marine resources",
            "color": "#0A97D9"
        },
        {
            "number": 15,
            "name": "Life on Land",
            "description": "Protect, restore and promote sustainable use of terrestrial ecosystems",
            "color": "#56C02B"
        },
        {
            "number": 16,
            "name": "Peace, Justice and Strong Institutions",
            "description": "Promote peaceful and inclusive societies for sustainable development",
            "color": "#00689D"
        },
        {
            "number": 17,
            "name": "Partnerships for the Goals",
            "description": "Strengthen the means of implementation and revitalize partnerships",
            "color": "#19486A"
        }
    ]

    cols = [col1, col2, col3, col4]

    for i, sdg in enumerate(sdgs):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="sdg-card" style="border-top: 5px solid {sdg['color']}">
                <h4>SDG {sdg['number']}: {sdg['name']}</h4>
                <p>{sdg['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    # SDG investment alignment
    st.markdown("### Aligning Investments with SDGs")

    st.markdown("""
    Investors can align their portfolios with the SDGs in several ways:

    1. **Thematic investing**: Focusing on specific themes related to one or more SDGs
    2. **Impact investing**: Directing capital to projects and companies that contribute to SDG targets
    3. **Engagement**: Using shareholder influence to encourage companies to address SDG challenges
    4. **Exclusions**: Avoiding investments in activities that negatively impact SDG progress

    When evaluating SDG alignment, consider both positive contributions and potential negative impacts.
    """)

    # SDG investment opportunities
    st.markdown("### Investment Opportunities by SDG")

    # Sample data on investment opportunities by SDG
    sdg_opportunities = {
        'SDG': [
            'SDG 7: Affordable and Clean Energy',
            'SDG 6: Clean Water and Sanitation',
            'SDG 9: Industry, Innovation and Infrastructure',
            'SDG 3: Good Health and Well-being',
            'SDG 11: Sustainable Cities and Communities',
            'SDG 2: Zero Hunger',
            'SDG 13: Climate Action'
        ],
        'Market_Size_B': [2800, 1000, 3500, 2200, 1800, 1200, 2500]
    }

    sdg_opp_df = pd.DataFrame(sdg_opportunities)

    fig = px.bar(
        sdg_opp_df,
        x='SDG',
        y='Market_Size_B',
        title='Estimated Market Size of SDG Investment Opportunities (Billions USD)',
        color='Market_Size_B',
        color_continuous_scale='Viridis',
        template=plotly_template
    )

    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    *Note: Market size estimates are illustrative and based on various industry reports.
    Actual market sizes may vary and continue to evolve as sustainability challenges and
    opportunities develop.*
    """)

# Impact Measurement tab
with tabs[2]:
    st.markdown("## Measuring Investment Impact")

    st.markdown("""
    Impact measurement is the process of assessing the social and environmental effects of investments.
    It helps investors understand how their capital contributes to sustainability goals and provides
    accountability for impact claims.
    """)

    # Impact measurement framework
    st.markdown("### Impact Measurement Framework")

    st.markdown("""
    A comprehensive impact measurement approach typically includes:

    1. **Setting objectives**: Defining the intended social and environmental outcomes
    2. **Selecting metrics**: Choosing appropriate indicators to track progress
    3. **Collecting data**: Gathering information on impact performance
    4. **Analyzing results**: Evaluating the data against objectives and benchmarks
    5. **Reporting findings**: Communicating impact performance to stakeholders
    6. **Learning and improving**: Using insights to enhance impact strategies
    """)

    # Impact metrics
    st.markdown("### Common Impact Metrics")

    impact_metrics = {
        'Category': [
            'Environmental',
            'Environmental',
            'Environmental',
            'Social',
            'Social',
            'Social',
            'Governance',
            'Governance'
        ],
        'Metric': [
            'Carbon Emissions',
            'Water Usage',
            'Waste Generation',
            'Gender Diversity',
            'Job Creation',
            'Health & Safety',
            'Board Independence',
            'Executive Compensation'
        ],
        'Unit': [
            'tCO2e',
            'Cubic meters',
            'Metric tons',
            '% of women',
            'Number of jobs',
            'Incident rate',
            '% independent',
            'CEO-worker pay ratio'
        ],
        'Example': [
            '15 tCO2e per $1M invested',
            '8,500 m³ per $1M revenue',
            '120 tons per $1M revenue',
            '42% women in workforce',
            '500 jobs created',
            '1.2 incidents per 100 workers',
            '75% independent directors',
            '45:1 CEO-worker pay ratio'
        ]
    }

    metrics_df = pd.DataFrame(impact_metrics)

    st.dataframe(metrics_df, use_container_width=True)

    # Impact measurement challenges
    st.markdown("### Challenges in Impact Measurement")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="education-card">
            <h4>Data Challenges</h4>
            <ul>
                <li>Limited availability of consistent, comparable data</li>
                <li>Varying quality and reliability of reported information</li>
                <li>Lack of standardized metrics across companies</li>
                <li>Difficulty in measuring intangible impacts</li>
                <li>Attribution challenges (linking investments to outcomes)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="education-card">
            <h4>Methodological Challenges</h4>
            <ul>
                <li>Balancing quantitative and qualitative assessments</li>
                <li>Accounting for both positive and negative impacts</li>
                <li>Addressing time lags between investment and impact</li>
                <li>Comparing impacts across different contexts</li>
                <li>Avoiding impact washing (exaggerated impact claims)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Impact reporting frameworks
    st.markdown("### Impact Reporting Frameworks")

    frameworks = {
        'Framework': [
            'Impact Management Project (IMP)',
            'IRIS+',
            'Global Reporting Initiative (GRI)',
            'Sustainability Accounting Standards Board (SASB)',
            'Task Force on Climate-related Financial Disclosures (TCFD)',
            'UN Principles for Responsible Investment (PRI)'
        ],
        'Focus': [
            'Comprehensive impact assessment',
            'Impact metrics catalog',
            'Sustainability reporting',
            'Industry-specific standards',
            'Climate-related risks and opportunities',
            'ESG integration in investment'
        ],
        'Adoption': [65, 70, 85, 75, 60, 90]
    }

    frameworks_df = pd.DataFrame(frameworks)

    fig = px.bar(
        frameworks_df,
        x='Framework',
        y='Adoption',
        color='Adoption',
        hover_data=['Focus'],
        title='Adoption of Impact Reporting Frameworks (%)',
        color_continuous_scale='Viridis',
        template=plotly_template
    )

    st.plotly_chart(fig, use_container_width=True)

    for _, row in frameworks_df.iterrows():
        st.markdown(f"**{row['Framework']}**: {row['Focus']}")

# Resources tab
with tabs[3]:
    st.markdown("## Educational Resources")

    st.markdown("""
    Explore these resources to deepen your understanding of sustainable investing and ESG principles.
    """)

    # Books
    st.markdown("### Recommended Books")

    books = [
        {
            "title": "Sustainable Investing: The Art of Long-Term Performance",
            "author": "Cary Krosinsky and Nick Robins",
            "description": "Explores the link between sustainability and financial performance"
        },
        {
            "title": "ESG Investing For Dummies",
            "author": "Brendan Bradley",
            "description": "A beginner-friendly guide to ESG investing principles and practices"
        },
        {
            "title": "Values at Work: Sustainable Investing and ESG Reporting",
            "author": "Daniel C. Esty and Todd Cort",
            "description": "Examines how ESG factors are reshaping corporate behavior and investment strategies"
        },
        {
            "title": "Responsible Investing: An Introduction to Environmental, Social, and Governance Investments",
            "author": "Matthew W. Sherwood and Julia Pollard",
            "description": "Provides a comprehensive overview of responsible investing approaches"
        }
    ]

    for book in books:
        st.markdown(f"""
        <div class="education-card">
            <h4>{book['title']}</h4>
            <p><strong>Author:</strong> {book['author']}</p>
            <p>{book['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Online courses
    st.markdown("### Online Courses")

    courses = [
        {
            "title": "Sustainable Finance and Investment Strategies",
            "provider": "Coursera",
            "duration": "6 weeks",
            "description": "Learn how to integrate ESG factors into investment analysis and decision-making"
        },
        {
            "title": "ESG Investing: A Comprehensive Guide",
            "provider": "edX",
            "duration": "4 weeks",
            "description": "Understand the fundamentals of ESG investing and impact measurement"
        },
        {
            "title": "Sustainable Business Strategy",
            "provider": "Harvard Business School Online",
            "duration": "3 weeks",
            "description": "Explore how businesses can thrive while addressing social and environmental challenges"
        },
        {
            "title": "Climate Change and Financial Markets",
            "provider": "LinkedIn Learning",
            "duration": "2 weeks",
            "description": "Examine the financial implications of climate change and transition risks"
        }
    ]

    col1, col2 = st.columns(2)

    for i, course in enumerate(courses):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="education-card">
                <h4>{course['title']}</h4>
                <p><strong>Provider:</strong> {course['provider']}</p>
                <p><strong>Duration:</strong> {course['duration']}</p>
                <p>{course['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Organizations and initiatives
    st.markdown("### Organizations and Initiatives")

    organizations = [
        {
            "name": "Principles for Responsible Investment (PRI)",
            "website": "https://www.unpri.org/",
            "description": "UN-supported network of investors working to promote sustainable investment"
        },
        {
            "name": "Global Sustainable Investment Alliance (GSIA)",
            "website": "http://www.gsi-alliance.org/",
            "description": "Collaboration of sustainable investment organizations around the world"
        },
        {
            "name": "Sustainability Accounting Standards Board (SASB)",
            "website": "https://www.sasb.org/",
            "description": "Develops industry-specific sustainability accounting standards"
        },
        {
            "name": "Climate Disclosure Project (CDP)",
            "website": "https://www.cdp.net/",
            "description": "Global disclosure system for environmental impacts"
        },
        {
            "name": "Global Impact Investing Network (GIIN)",
            "website": "https://thegiin.org/",
            "description": "Organization dedicated to increasing the scale and effectiveness of impact investing"
        }
    ]

    for org in organizations:
        st.markdown(f"""
        <div class="education-card">
            <h4>{org['name']}</h4>
            <p><strong>Website:</strong> <a href="{org['website']}" target="_blank">{org['website']}</a></p>
            <p>{org['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Glossary
    st.markdown("### Sustainable Investing Glossary")

    with st.expander("View Glossary"):
        terms = {
            "Active Ownership": "Using shareholder rights to influence company behavior",
            "Carbon Footprint": "Total greenhouse gas emissions caused by an individual, organization, or product",
            "Corporate Social Responsibility (CSR)": "Self-regulating business model that holds companies accountable to social and environmental standards",
            "Divestment": "Selling shares of a company for ethical or political reasons",
            "Double Bottom Line": "Measuring both financial returns and social/environmental impact",
            "ESG Integration": "Incorporating ESG factors into investment analysis and decision-making",
            "Greenwashing": "Making misleading claims about environmental practices or benefits",
            "Impact Investing": "Investments made with the intention to generate positive social and environmental impact alongside financial return",
            "Materiality": "The relevance of an ESG issue to a company's financial performance",
            "Net Zero": "Achieving a balance between carbon emissions produced and carbon emissions removed from the atmosphere",
            "Shareholder Advocacy": "Using shareholder position to promote corporate change",
            "Socially Responsible Investing (SRI)": "Investment strategy that considers both financial return and social/environmental good",
            "Stranded Assets": "Assets that have suffered from unanticipated or premature write-downs due to environmental regulations or market shifts",
            "Sustainable Development": "Development that meets present needs without compromising future generations' ability to meet their needs",
            "Triple Bottom Line": "Accounting framework that incorporates social, environmental, and financial performance"
        }

        for term, definition in terms.items():
            st.markdown(f"**{term}**: {definition}")

    # Quiz
    st.markdown("### Test Your Knowledge")

    st.markdown("""
    Take this quick quiz to test your understanding of sustainable investing concepts.
    """)

    quiz_questions = [
        {
            "question": "Which of the following is NOT typically considered an environmental factor in ESG analysis?",
            "options": [
                "Carbon emissions",
                "Executive compensation",
                "Waste management",
                "Biodiversity impact"
            ],
            "correct": 1
        },
        {
            "question": "What does the 'G' in ESG stand for?",
            "options": [
                "Growth",
                "Global",
                "Governance",
                "Green"
            ],
            "correct": 2
        },
        {
            "question": "Which investment approach involves excluding companies involved in controversial activities?",
            "options": [
                "Impact investing",
                "Negative screening",
                "ESG integration",
                "Thematic investing"
            ],
            "correct": 1
        }
    ]

    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_completed = False

    if not st.session_state.quiz_completed:
        for i, question in enumerate(quiz_questions):
            st.markdown(f"**Question {i+1}:** {question['question']}")
            answer = st.radio(
                f"Select your answer for question {i+1}:",
                question['options'],
                key=f"question_{i}"
            )

            if answer == question['options'][question['correct']]:
                st.session_state.quiz_score += 1

        if st.button("Submit Quiz"):
            st.session_state.quiz_completed = True
            st.success(f"You scored {st.session_state.quiz_score} out of {len(quiz_questions)}!")

            if st.session_state.quiz_score == len(quiz_questions):
                st.balloons()
                st.markdown("**Perfect score! You're an ESG expert!**")
            elif st.session_state.quiz_score >= len(quiz_questions) / 2:
                st.markdown("**Good job! You have a solid understanding of ESG concepts.**")
            else:
                st.markdown("**Keep learning! Review the materials to strengthen your ESG knowledge.**")
    else:
        st.success(f"You scored {st.session_state.quiz_score} out of {len(quiz_questions)}!")

        if st.button("Retake Quiz"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_completed = False
            st.rerun()
