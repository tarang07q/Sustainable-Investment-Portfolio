import streamlit as st
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.supabase import is_authenticated, get_current_user, update_profile, sign_out
from utils.quotes import get_random_finance_quote

# Set page config
st.set_page_config(
    page_title="User Profile - Sustainable Investment Portfolio",
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
theme_card_bg = "#262730" if st.session_state.theme == "dark" else "#f8f9fa"
theme_border_color = "#555" if st.session_state.theme == "dark" else "#ddd"

# Custom CSS with dynamic theming
st.markdown(f"""
    <style>
    .main {{
        padding: 2rem;
        background-color: {theme_bg_color};
        color: {theme_text_color};
    }}
    .profile-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background-color: {theme_card_bg};
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: {theme_text_color};
    }}
    .profile-header {{
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }}
    .profile-avatar {{
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background-color: #4CAF50;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        margin-right: 1.5rem;
    }}
    .profile-info {{
        flex: 1;
    }}
    .settings-section {{
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid {theme_border_color};
    }}
    .stButton>button {{
        width: 100%;
    }}
    .quote-container {{
        background-color: {theme_secondary_bg};
        padding: 1.5rem;
        border-radius: 10px;
        margin: 2rem 0;
        border-left: 5px solid #4CAF50;
        color: {theme_text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Check if user is authenticated
if not is_authenticated():
    st.warning("Please sign in to access your profile.")

    if st.button("Go to Sign In"):
        st.switch_page("pages/0_Authentication.py")
else:
    # Get current user
    user = get_current_user()

    # Header
    st.title("👤 User Profile")
    st.markdown("*Manage your account settings and preferences*")

    # Display a random finance quote
    quote = get_random_finance_quote()
    st.markdown(f"""
    <div class="quote-container">
        <p style="font-style: italic; font-size: 1.1rem;">"{quote['quote']}"</p>
        <p style="text-align: right; font-weight: bold;">— {quote['author']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Profile container
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)

    # Profile header
    st.markdown(f"""
    <div class="profile-header">
        <div class="profile-avatar">{user['full_name'][0].upper()}</div>
        <div class="profile-info">
            <h2>{user['full_name']}</h2>
            <p>{user['email']}</p>
            <p>Member since: {user['created_at'][:10]}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Profile settings
    st.markdown("### Profile Settings")

    # Personal information
    with st.form("profile_form"):
        st.markdown("#### Personal Information")

        full_name = st.text_input("Full Name", value=user['full_name'])

        # In a real implementation, these fields would be populated with actual user data
        phone = st.text_input("Phone Number", value="")
        country = st.selectbox("Country", ["United States", "Canada", "United Kingdom", "Australia", "Germany", "France", "Japan", "Other"])

        st.markdown("#### Investment Preferences")

        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Very Low", "Low", "Moderate", "High", "Very High"],
            value="Moderate"
        )

        investment_horizon = st.selectbox(
            "Investment Horizon",
            options=["Short-term (< 1 year)", "Medium-term (1-3 years)", "Long-term (> 3 years)"],
            index=1
        )

        sustainability_focus = st.select_slider(
            "Sustainability Focus",
            options=["Financial Returns First", "Balanced Approach", "Impact First"],
            value="Balanced Approach"
        )

        preferred_sectors = st.multiselect(
            "Preferred Sectors",
            options=["Technology", "Energy", "Healthcare", "Utilities", "Finance", "Consumer Goods", "Real Estate", "Industrials"],
            default=["Energy", "Technology"]
        )

        submit_button = st.form_submit_button("Save Changes")

        if submit_button:
            success, message = update_profile(full_name)
            if success:
                st.success(message)
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
