import streamlit as st
from utils.supabase import is_authenticated

def check_authentication():
    """
    Check if user is authenticated and redirect to authentication page if not.
    This should be called at the beginning of each page that requires authentication.
    """
    if not is_authenticated():
        st.warning("You need to sign in to access this page.")
        st.markdown("""
        <meta http-equiv="refresh" content="3;url=/Authentication">
        <script>
            setTimeout(function() {
                window.location.href = "/Authentication";
            }, 3000);
        </script>
        """, unsafe_allow_html=True)
        st.stop()
