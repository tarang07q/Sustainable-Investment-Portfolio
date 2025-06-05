import streamlit as st
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.supabase import is_authenticated, get_current_user, update_profile, sign_out
from utils.quotes import get_random_finance_quote

# Set page config
st.set_page_config(
    page_title="User Profile - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Set theme to dark mode
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
else:
    # Force dark theme for this page
    st.session_state.theme = 'dark'

# Apply theme-specific styles
theme_bg_color = "#0e1117" if st.session_state.theme == "dark" else "#ffffff"
theme_text_color = "#ffffff" if st.session_state.theme == "dark" else "#0e1117"
theme_secondary_bg = "#1e2530" if st.session_state.theme == "dark" else "#f0f2f6"
theme_card_bg = "#262730" if st.session_state.theme == "dark" else "#f8f9fa"
theme_border_color = "#555" if st.session_state.theme == "dark" else "#ddd"

# Add authentication check
from utils.auth_redirect import check_authentication
check_authentication()

# Custom CSS with dynamic theming
st.markdown(f"""
    <style>
    .main {{
        padding: 2rem;
        background-color: {theme_bg_color};
        color: {theme_text_color};
    }}
    .profile-container {{
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
        background-color: {theme_card_bg};
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        color: {theme_text_color};
        border: 1px solid rgba(76, 175, 80, 0.2);
        margin-top: 0;
    }}
    .profile-header {{
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }}
    .profile-avatar {{
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4CAF50, #2196F3);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        font-weight: bold;
        margin-right: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        border: 4px solid rgba(255, 255, 255, 0.2);
    }}
    .profile-avatar:hover {{
        transform: scale(1.05);
    }}
    .profile-info {{
        flex: 1;
    }}
    .profile-info h2 {{
        margin-bottom: 0.5rem;
        color: #4CAF50;
    }}
    .profile-info p {{
        margin: 0.25rem 0;
        opacity: 0.8;
    }}
    .settings-section {{
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid {theme_border_color};
    }}
    .stButton>button {{
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #3e8e41;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .quote-container {{
        background-color: rgba(76, 175, 80, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0 0 2rem 0;
        border-left: 5px solid #4CAF50;
        color: {theme_text_color};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        font-family: 'Georgia', serif;
    }}
    /* Form styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div,
    .stMultiselect>div>div>div {{
        background-color: #1e2530 !important;
        border: 1px solid #555 !important;
        border-radius: 5px;
        padding: 0.5rem;
        color: #ffffff !important;
        max-width: 100%;
    }}

    /* Dropdown menu styling */
    .stSelectbox ul,
    .stMultiselect ul {{
        background-color: #1e2530 !important;
        border: 1px solid #555 !important;
    }}

    .stSelectbox ul li,
    .stMultiselect ul li {{
        color: #ffffff !important;
    }}

    .stSelectbox ul li:hover,
    .stMultiselect ul li:hover {{
        background-color: #2d3748 !important;
    }}

    /* Form container styling */
    form {{
        background-color: #1e2530 !important;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(76, 175, 80, 0.1);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }}

    /* Form label styling */
    label {{
        font-weight: 500;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
        display: block;
    }}

    /* Form field spacing */
    .stTextInput, .stSelectbox, .stMultiselect, .stSlider {{
        margin-bottom: 1rem;
    }}

    /* Form field width control */
    .stTextInput, .stSelectbox, .stMultiselect {{
        max-width: 600px;
    }}

    /* Select slider styling */
    .stSlider {{
        max-width: 600px;
        padding: 0.5rem 0;
    }}

    /* Form section styling */
    h4 {{
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
        color: #4CAF50;
        font-size: 1.2rem;
        border-bottom: 1px solid rgba(76, 175, 80, 0.2);
        padding-bottom: 0.5rem;
        font-weight: 600;
    }}

    h5 {{
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        color: #4CAF50;
        font-size: 1rem;
        font-weight: 500;
    }}

    /* Form submit button styling */
    .stButton>button[kind="formSubmit"] {{
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 600;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }}

    .stButton>button[kind="formSubmit"]:hover {{
        background-color: #3e8e41;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {{
        border-color: #4CAF50;
        box-shadow: 0 0 0 1px #4CAF50;
    }}
    /* Section headers */
    h3 {{
        color: #4CAF50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(76, 175, 80, 0.3);
    }}
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: #1e2530 !important;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.1);
    }}
    /* Toggle styling */
    .stToggle>div {{
        background-color: rgba(76, 175, 80, 0.2);
    }}
    /* Success message styling */
    .stSuccess {{
        background-color: rgba(76, 175, 80, 0.1);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 5px;
    }}
    /* Additional styling to fix layout issues and remove blank cards */
    div[data-testid="stVerticalBlock"] > div:first-child {{
        margin-top: 0;
    }}
    div.block-container {{
        padding-top: 0 !important;
        max-width: 100%;
    }}
    section[data-testid="stSidebar"] + div {{
        padding-top: 0 !important;
    }}
    /* Remove any empty containers */
    div:empty {{
        display: none !important;
    }}
    /* Remove extra padding from main container */
    .main .block-container {{
        padding: 0 !important;
        margin-top: 0 !important;
    }}
    /* Custom header styling */
    .custom-header {{
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }}
    .custom-header h1 {{
        margin: 0;
        font-size: 2rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Demo stubs for authentication and user management
def is_authenticated():
    return 'user' in st.session_state and st.session_state.user is not None

def get_current_user():
    return st.session_state.get('user', None)

def update_profile(full_name, user_profile_data=None):
    if 'user' in st.session_state:
        st.session_state.user['full_name'] = full_name
        if user_profile_data:
            if 'preferences' not in st.session_state.user:
                st.session_state.user['preferences'] = {}
            for key, value in user_profile_data.items():
                if key != 'full_name':
                    st.session_state.user['preferences'][key] = value
        return True, "Demo Mode: Profile updated successfully!"
    else:
        return False, "No user is currently signed in."

def sign_out():
    if 'user' in st.session_state:
        del st.session_state.user
    return True, "Successfully signed out!"

# Check if user is authenticated
if not is_authenticated():
    st.warning("Please sign in to access your profile.")

    if st.button("Go to Sign In"):
        st.switch_page("pages/0_Authentication.py")
else:
    # Get current user
    user = get_current_user()

    # Custom header with no extra spacing
    st.markdown("""
    <div class="custom-header">
        <h1>👤 User Profile</h1>
    </div>
    <p style="margin-top: 0; font-style: italic; margin-bottom: 1.5rem;">Manage your account settings and preferences</p>
    """, unsafe_allow_html=True)

    # Get a random finance quote
    quote = get_random_finance_quote()

    # No extra spacing needed

    # Profile container
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)

    # Display quote at the top of the profile container
    st.markdown(f"""
    <div class="quote-container">
        <p style="font-style: italic; font-size: 1.1rem;">"{quote['quote']}"</p>
        <p style="text-align: right; font-weight: bold;">— {quote['author']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Profile header
    st.markdown(f"""
    <div class="profile-header">
        <div class="profile-avatar">{user['full_name'][0].upper()}</div>
        <div class="profile-info">
            <h2>{user['full_name']}</h2>
            <p>{user['email']}</p>
            <p>Member since: {user['created_at'].split('T')[0] if isinstance(user['created_at'], str) else user['created_at'].strftime('%Y-%m-%d')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Profile settings
    st.markdown("### Profile Settings")

    # Personal information
    with st.form("profile_form"):
        st.markdown("#### Personal Information")

        # Create two columns for personal information
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input("Full Name", value=user['full_name'])
            phone = st.text_input("Phone Number", value="")

        with col2:
            country = st.selectbox("Country", ["India", "United States", "Canada", "United Kingdom", "Australia", "Germany", "France", "Japan", "China", "Singapore", "United Arab Emirates", "Other"])

            # Add language preference with Indian languages
            language = st.selectbox(
                "Preferred Language",
                options=["English", "Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi", "Other"]
            )

        # Add tax residency status for Indian users
        tax_residency = st.selectbox(
            "Tax Residency Status",
            options=["Resident Indian", "Non-Resident Indian (NRI)", "Foreign National", "Overseas Citizen of India (OCI)", "Person of Indian Origin (PIO)", "Other"]
        )

        st.markdown("#### Investment Preferences")

        # Currency preference removed as requested

        # Create two columns for investment preferences
        col1, col2 = st.columns(2)

        with col1:
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Very Low", "Low", "Moderate", "High", "Very High"],
                value="Moderate"
            )

            sustainability_focus = st.select_slider(
                "Sustainability Focus",
                options=["Financial Returns First", "Balanced Approach", "Impact First"],
                value="Balanced Approach"
            )

        with col2:
            investment_horizon = st.selectbox(
                "Investment Horizon",
                options=["Short-term (< 1 year)", "Medium-term (1-3 years)", "Long-term (> 3 years)"],
                index=1
            )

        # Full width for multiselect fields
        st.markdown("##### Preferred Investment Areas")
        preferred_sectors = st.multiselect(
            "Sectors",
            options=[
                "Technology", "Energy", "Healthcare", "Utilities", "Finance",
                "Consumer Goods", "Real Estate", "Industrials", "IT Services",
                "Renewable Energy", "Pharmaceuticals", "Automotive", "Agriculture",
                "Textiles", "Steel", "Infrastructure", "E-commerce", "Education"
            ],
            default=["Energy", "Technology"]
        )

        # Add Indian stock exchange preferences
        preferred_exchanges = st.multiselect(
            "Stock Exchanges",
            options=["NSE (National Stock Exchange of India)", "BSE (Bombay Stock Exchange)",
                    "NYSE (New York Stock Exchange)", "NASDAQ",
                    "LSE (London Stock Exchange)", "TSE (Tokyo Stock Exchange)",
                    "SSE (Shanghai Stock Exchange)", "HKEX (Hong Kong Exchange)"],
            default=["NSE (National Stock Exchange of India)", "BSE (Bombay Stock Exchange)"]
        )

        submit_button = st.form_submit_button("Save Changes")

        if submit_button:
            # Create a user profile data dictionary with all fields
            user_profile_data = {
                'full_name': full_name,
                'phone': phone,
                'country': country,
                'language': language,
                'tax_residency': tax_residency,
                'risk_tolerance': risk_tolerance,
                'investment_horizon': investment_horizon,
                'sustainability_focus': sustainability_focus,
                'preferred_sectors': preferred_sectors,
                'preferred_exchanges': preferred_exchanges
            }

            # Save all user profile data
            success, message = update_profile(full_name, user_profile_data)

            if success:
                st.success("Profile updated successfully! All your preferences have been saved.")

                # Display a summary of the updated profile
                with st.expander("View Updated Profile Summary", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Personal Information**")
                        st.write(f"**Name:** {full_name}")
                        st.write(f"**Phone:** {phone}")
                        st.write(f"**Country:** {country}")
                        st.write(f"**Language:** {language}")
                        st.write(f"**Tax Status:** {tax_residency}")

                    with col2:
                        st.markdown("**Investment Preferences**")
                        st.write(f"**Risk Tolerance:** {risk_tolerance}")
                        st.write(f"**Investment Horizon:** {investment_horizon}")
                        st.write(f"**Sustainability Focus:** {sustainability_focus}")
                        st.write(f"**Sectors:** {', '.join(preferred_sectors)}")
                        st.write(f"**Exchanges:** {', '.join(preferred_exchanges)}")
            else:
                st.error(message)

    # Security settings
    st.markdown("### Security Settings")

    with st.expander("Change Password"):
        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")

            password_submit = st.form_submit_button("Update Password")

            if password_submit:
                if not current_password or not new_password or not confirm_password:
                    st.error("Please fill in all password fields.")
                elif new_password != confirm_password:
                    st.error("New passwords do not match.")
                else:
                    # In a real implementation, this would call the Supabase API
                    st.success("Password updated successfully!")

    with st.expander("Two-Factor Authentication"):
        st.markdown("Two-factor authentication adds an extra layer of security to your account.")

        enable_2fa = st.toggle("Enable Two-Factor Authentication", value=False)

        if enable_2fa:
            st.info("In a production version, this would guide you through setting up 2FA.")

    # Notification settings
    st.markdown("### Notification Settings")

    with st.form("notification_form"):
        st.markdown("Choose which notifications you want to receive:")

        email_notifications = st.toggle("Email Notifications", value=True)

        if email_notifications:
            st.checkbox("Portfolio Updates", value=True)
            st.checkbox("Market Alerts", value=True)
            st.checkbox("ESG News", value=True)
            st.checkbox("Recommendation Updates", value=True)

        notification_submit = st.form_submit_button("Save Notification Settings")

        if notification_submit:
            st.success("Notification settings updated successfully!")

    # Account actions
    st.markdown("### Account Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sign Out"):
            success, message = sign_out()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with col2:
        if st.button("Delete Account", type="primary"):
            st.warning("This action cannot be undone. All your data will be permanently deleted.")

            confirm_delete = st.text_input("Type 'DELETE' to confirm account deletion:")

            if confirm_delete == "DELETE":
                # In a real implementation, this would call the Supabase API
                st.success("Account deleted successfully. You will be redirected to the home page.")
                sign_out()
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
