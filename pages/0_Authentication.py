import streamlit as st
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.supabase import sign_up, sign_in, sign_in_with_google, is_authenticated, get_current_user
from utils.quotes import get_random_sustainable_quote
from utils.theme import apply_theme_css

# Set page config
st.set_page_config(
    page_title="Authentication - Sustainable Investment Portfolio",
    page_icon="🌱",
    layout="wide"
)

# Apply theme-specific styles
theme_colors = apply_theme_css()

# Check if user is already authenticated
if is_authenticated():
    user = get_current_user()
    st.success(f"You are already signed in as {user['email']}.")
    st.write("You can now access all features of the Sustainable Investment Portfolio App.")

    if st.button("Go to Home Page"):
        st.switch_page("app.py")
else:
    # Header
    st.title("🔐 Authentication")
    st.markdown("*Sign in or create an account to access the Sustainable Investment Portfolio App*")

    # Display a random sustainable investing quote
    quote = get_random_sustainable_quote()
    st.markdown(f"""
    <div class="quote-container">
        <p style="font-style: italic; font-size: 1.1rem;">"{quote['quote']}"</p>
        <p style="text-align: right; font-weight: bold;">— {quote['author']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Authentication container
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    # Tabs for sign in and sign up
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    # Sign In tab
    with tab1:
        st.markdown("### Sign In to Your Account")

        email = st.text_input("Email", key="signin_email")
        password = st.text_input("Password", type="password", key="signin_password")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Sign In", key="signin_button"):
                if email and password:
                    success, message = sign_in(email, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both email and password.")

        with col2:
            if st.button("Forgot Password?", key="forgot_password"):
                st.info("Password reset functionality would be implemented in a production version.")

        st.markdown("### Or Sign In With")

        if st.button("Sign In with Google", key="google_signin"):
            success, message = sign_in_with_google()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    # Sign Up tab
    with tab2:
        st.markdown("### Create a New Account")

        full_name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")

        # Terms and conditions checkbox
        terms_accepted = st.checkbox("I accept the Terms of Service and Privacy Policy", key="terms")

        if st.button("Sign Up", key="signup_button"):
            if not full_name or not email or not password or not confirm_password:
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif not terms_accepted:
                st.error("You must accept the Terms of Service and Privacy Policy.")
            else:
                success, message = sign_up(email, password, full_name)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.markdown("### Or Sign Up With")

        if st.button("Sign Up with Google", key="google_signup"):
            success, message = sign_in_with_google()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    By signing up, you'll gain access to:
    - Personalized investment recommendations
    - Portfolio tracking and management
    - ESG analysis and impact reporting
    - Educational resources on sustainable investing
    """)
