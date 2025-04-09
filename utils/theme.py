import streamlit as st

def get_theme():
    """Always return dark theme as requested"""
    # Set theme to dark permanently
    st.session_state.theme = 'dark'
    return 'dark'

def toggle_theme():
    """Toggle the theme between light and dark"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

def get_theme_colors():
    """Get the theme colors based on the current theme"""
    theme = get_theme()
    return {
        'bg_color': "#0e1117" if theme == "dark" else "#ffffff",
        'text_color': "#ffffff" if theme == "dark" else "#0e1117",
        'secondary_bg': "#1e2530" if theme == "dark" else "#f0f2f6",
        'card_bg': "#262730" if theme == "dark" else "#f8f9fa",
        'accent_color': "#4CAF50",
        'plotly_template': "plotly_dark" if theme == "dark" else "plotly_white"
    }

def get_theme_toggle_button():
    """Create an elegant theme toggle button with sun/moon icon"""
    theme = get_theme()
    icon = "☀️" if theme == "dark" else "🌙"

    # Create a container for the toggle button with custom styling
    toggle_container = st.container()

    with toggle_container:
        # Use HTML/CSS for a more elegant toggle button
        st.markdown(f"""
        <div class="theme-toggle-container">
            <button id="theme-toggle" class="theme-toggle-btn" onclick="handleThemeToggle()">{icon}</button>
        </div>
        <script>
            function handleThemeToggle() {{
                // This is handled by the Streamlit event handler below
                window.parent.postMessage({{type: "streamlit:buttonClicked", label: "__theme_toggle__"}}, "*");
            }}
        </script>
        <style>
            .theme-toggle-container {{
                position: fixed;
                top: 15px;
                right: 80px;
                z-index: 1000;
            }}
            .theme-toggle-btn {{
                background-color: {"#1e2530" if theme == "dark" else "#e6e6e6"};
                border: 1px solid {"#4d4d4d" if theme == "dark" else "#d1d1d1"};
                font-size: 20px;
                cursor: pointer;
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                padding: 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            .theme-toggle-btn:hover {{
                transform: scale(1.1);
            }}
        </style>
        """, unsafe_allow_html=True)

        # Create a small empty column to place the button in the corner
        col1, col2 = st.columns([0.97, 0.03])
        with col2:
            # Hidden button to capture the click event (with empty label)
            if st.button(" ", key="theme_toggle_btn", help="Toggle between light and dark theme"):
                toggle_theme()
                st.rerun()

def apply_theme_css():
    """Apply theme CSS to the current page"""
    colors = get_theme_colors()

    # Custom CSS with dynamic theming
    st.markdown(f"""
        <style>
        /* Base styles */
        .main {{
            padding: 2rem;
            background-color: {colors['bg_color']};
            color: {colors['text_color']};
        }}

        /* Button styles */
        .stButton>button {{
            width: 100%;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: 1px solid {colors['secondary_bg']};
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}

        /* Card styles with better alignment and consistency */
        .recommendation-card, .metric-card, .asset-card, .portfolio-card {{
            background-color: {colors['secondary_bg']};
            padding: 1.2rem;
            border-radius: 8px;
            margin: 0.8rem 0;
            color: {colors['text_color']};
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            height: calc(100% - 1.6rem); /* Consistent height accounting for margins */
            display: flex;
            flex-direction: column;
        }}
        .recommendation-card:hover, .metric-card:hover, .asset-card:hover, .portfolio-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        }}

        /* Fix background colors for various Streamlit elements */
        .css-1v3fvcr, .css-18e3th9, .css-1d391kg, .css-1wrcr25, .css-ocqkz7, .css-1y4p8pa {{
            background-color: {colors['bg_color']} !important;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background-color: {colors['bg_color']};
            border-bottom: 1px solid {colors['secondary_bg']};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: {colors['secondary_bg']};
            color: {colors['text_color']};
            border-radius: 8px 8px 0 0;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: {colors['accent_color']};
            color: white;
            opacity: 0.8;
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            background-color: {colors['bg_color']};
            color: {colors['text_color']};
            padding-top: 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {colors['accent_color']} !important;
            color: white !important;
        }}

        /* Typography */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {colors['text_color']};
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}

        /* Quote container */
        .quote-container {{
            background-color: {colors['secondary_bg']};
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            border-left: 5px solid {colors['accent_color']};
            color: {colors['text_color']};
            font-style: italic;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        /* Auth container */
        .auth-container {{
            max-width: 500px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: {colors['card_bg']};
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
            color: {colors['text_color']};
        }}

        /* Auth banner */
        .auth-banner {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: {colors['secondary_bg']};
            padding: 1.2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        /* Feature cards */
        .feature-card {{
            background-color: {colors['card_bg']};
            padding: 1.8rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }}
        .feature-icon {{
            font-size: 2.5rem;
            margin-bottom: 1.2rem;
            color: {colors['accent_color']};
        }}

        /* Table styling */
        .stDataFrame table {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {colors['secondary_bg']};
        }}
        .stDataFrame th {{
            background-color: {colors['secondary_bg']};
            color: {colors['text_color']};
            font-weight: 600;
            padding: 0.75rem 1rem;
        }}
        .stDataFrame td {{
            padding: 0.6rem 1rem;
            border-bottom: 1px solid {colors['secondary_bg']};
        }}

        /* Metric styling */
        .metric-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 1rem;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {colors['accent_color']};
            margin: 0.5rem 0;
        }}
        .metric-label {{
            font-size: 1rem;
            font-weight: 500;
            color: {colors['text_color']};
            opacity: 0.8;
        }}

        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background-color: {colors['card_bg']} !important;
            border-right: 1px solid {colors['secondary_bg']};
        }}
        </style>
    """, unsafe_allow_html=True)

    # Theme toggle button removed as requested

    return colors
