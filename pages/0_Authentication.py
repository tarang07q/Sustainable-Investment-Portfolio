import streamlit as st
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.supabase import sign_up, sign_in, is_authenticated, get_current_user

# --- Demo authentication stubs (no supabase required) ---
def is_authenticated():
    return 'user' in st.session_state and st.session_state.user is not None

def get_current_user():
    return st.session_state.get('user', None)

def sign_in(email, password):
    # Accept any email/password for demo
    user = {
        "id": "demo-user-id",
        "email": email,
        "full_name": email.split('@')[0].title(),
        "created_at": "2023-04-10T00:00:00.000Z"
    }
    st.session_state.user = user
    return True, "Demo Mode: Successfully signed in!"

def sign_up(email, password, full_name):
    user = {
        "id": "demo-user-id",
        "email": email,
        "full_name": full_name,
        "created_at": "2023-04-10T00:00:00.000Z"
    }
    st.session_state.user = user
    return True, "Demo Mode: Successfully signed up!"

# Set page config
st.set_page_config(
    page_title="Sustainable Investment Portfolio - Authentication",
    page_icon="🌱",
    layout="wide"
)

# Add Font Awesome for icons
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0e1117;
        background-image: url('https://images.pexels.com/photos/7788009/pexels-photo-7788009.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Overlay for better readability - reduced opacity to 20% of current */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.06); /* Reduced from 0.3 to 0.06 (20% of 0.3) */
        z-index: -1;
    }

    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Increased font sizes */
    p, div, span, li {
        font-size: 1.2rem !important; /* Increased base font size */
    }

    h1 {
        font-size: 2.5rem !important;
    }

    h2 {
        font-size: 2.2rem !important;
    }

    h3 {
        font-size: 1.8rem !important;
    }

    h4 {
        font-size: 1.5rem !important;
    }

    .stButton button {
        font-size: 1.2rem !important;
    }

    input, select, textarea {
        font-size: 1.2rem !important;
    }

    /* Header */
    .header {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }

    .header-logo {
        font-size: 1.8rem;
        margin-right: 10px;
    }

    .header-title {
        font-size: 1.8rem;
        margin: 0;
    }

    /* Feature card */
    .feature-card {
        background-color: rgba(30, 37, 48, 0.14); /* Reduced to 20% of current (0.7 * 0.2 = 0.14) */
        border-radius: 10px;
        padding: 1.8rem; /* Increased padding */
        margin-bottom: 1.5rem; /* Increased margin */
        text-align: center;
        backdrop-filter: blur(5px); /* Adds a slight blur effect */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* Added subtle shadow */
    }

    .feature-icon {
        font-size: 3.5rem; /* Increased from 2.5rem */
        color: #4CAF50;
        margin-bottom: 1.5rem; /* Increased from 1rem */
    }

    .feature-title {
        font-size: 1.6rem; /* Increased from 1.2rem */
        margin-bottom: 0.8rem; /* Increased from 0.5rem */
        font-weight: 600;
    }

    .feature-text {
        color: #aaa;
        line-height: 1.6; /* Increased from 1.5 */
        font-size: 1.3rem; /* Explicitly set font size */
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem; /* Increased from 3rem */
        color: #888;
        font-size: 1.1rem; /* Increased from 0.9rem */
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Check if user is already authenticated
if is_authenticated():
    user = get_current_user()

    # Success message
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem;">
        <i class="fas fa-check-circle" style="font-size: 4rem; color: #4CAF50; margin-bottom: 1.5rem;"></i>
        <h2 style="margin-bottom: 1rem;">You're already signed in!</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">Welcome back, <strong>{user['full_name']}</strong>. You have full access to all features.</p>
    </div>
    """, unsafe_allow_html=True)

    # Redirect button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Go to Dashboard", type="primary", use_container_width=True):
            st.switch_page("app.py")
else:
    # Header
    st.markdown("""
    <div class="header">
        <span class="header-logo">🌱</span>
        <h1 class="header-title">Sustainable Investment Portfolio</h1>
    </div>
    """, unsafe_allow_html=True)

    # Welcome message
    st.markdown("""
    <h2 style="text-align: center; margin-bottom: 2rem; font-size: 2.5rem;">Welcome to Sustainable Investment Portfolio</h2>
    <p style="text-align: center; margin-bottom: 3rem; max-width: 700px; margin-left: auto; margin-right: auto; font-size: 1.4rem;">
        Sign in or create an account to access our sustainable investment tools and resources.
    </p>
    """, unsafe_allow_html=True)

    # Create tabs for sign in and sign up
    tab1, tab2 = st.tabs(["Sign In", "Create Account"])

    with tab1:
        # Sign In Form
        with st.form("signin_form", clear_on_submit=False):
            st.subheader("Sign In")
            email = st.text_input("Email", key="signin_email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", key="signin_password", placeholder="Your password")

            col1, col2 = st.columns(2)
            with col1:
                remember_me = st.checkbox("Remember me", key="remember_me")
            with col2:
                st.markdown("<div style='text-align: right;'><a href='#' style='color: #4CAF50;'>Forgot password?</a></div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                if email and password:
                    success, message = sign_in(email, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both email and password.")

        # Social login (outside the form)
        st.markdown("<p style='text-align: center; margin-top: 1.5rem; color: #888;'>Or sign in with</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Google", key="google_signin", use_container_width=True):
                st.info("Google sign-in functionality would be implemented here.")

    with tab2:
        # Sign Up Form
        with st.form("signup_form", clear_on_submit=False):
            st.subheader("Create Account")
            full_name = st.text_input("Full Name", key="signup_name", placeholder="John Doe")
            email = st.text_input("Email", key="signup_email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", key="signup_password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Confirm your password")

            terms_accepted = st.checkbox("I accept the Terms of Service and Privacy Policy", key="terms")

            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
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

        # Social signup (outside the form)
        st.markdown("<p style='text-align: center; margin-top: 1.5rem; color: #888;'>Or sign up with</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Google", key="google_signup", use_container_width=True):
                st.info("Google sign-up functionality would be implemented here.")

    # Features section
    st.markdown("<h3 style='text-align: center; margin: 4rem 0 3rem 0; font-size: 2.2rem; font-weight: 600;'>Why Choose Sustainable Investment Portfolio?</h3>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-leaf"></i></div>
            <h4 class="feature-title">ESG-Focused Investing</h4>
            <p class="feature-text">Invest in companies that align with your environmental and social values.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-chart-line"></i></div>
            <h4 class="feature-title">Data-Driven Insights</h4>
            <p class="feature-text">Make informed decisions with comprehensive analytics and market trends.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-robot"></i></div>
            <h4 class="feature-title">AI Recommendations</h4>
            <p class="feature-text">Receive personalized investment suggestions based on your goals and values.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-shield-alt"></i></div>
            <h4 class="feature-title">Secure & Private</h4>
            <p class="feature-text">Your data is protected with enterprise-grade security and encryption.</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        &copy; 2023 Sustainable Investment Portfolio. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
