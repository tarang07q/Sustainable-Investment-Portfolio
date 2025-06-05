import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
def init_supabase():
    # Get Supabase credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    # For development/demo purposes, use fallback values if environment variables are not set
    if not supabase_url or not supabase_key:
        # In a real app, you would log this as a warning
        print("Warning: Using demo mode for Supabase authentication")
        # For the prototype, we'll return None and handle this in the auth functions
        return None

    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        return supabase
    except Exception as e:
        print(f"Error initializing Supabase client: {str(e)}")
        return None

# Check if user is authenticated
def is_authenticated():
    return 'user' in st.session_state and st.session_state.user is not None

# Sign up a new user
def sign_up(email, password, full_name):
    supabase = init_supabase()

    if supabase is None:
        # Demo mode - simulate successful sign-up
        user = {
            "id": "demo-user-id",
            "email": email,
            "full_name": full_name,
            "created_at": "2023-04-10T00:00:00.000Z"
        }
        st.session_state.user = user
        return True, "Demo Mode: Successfully signed up!"

    try:
        # Real Supabase authentication
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            }
        })

        # Check if sign-up was successful
        if response.user:
            # Store user data in session state
            user = {
                "id": response.user.id,
                "email": response.user.email,
                "full_name": full_name,
                "created_at": response.user.created_at
            }
            st.session_state.user = user
            return True, "Successfully signed up! Please check your email for verification."
        else:
            return False, "Error signing up. Please try again."
    except Exception as e:
        return False, f"Error signing up: {str(e)}"

# Sign in an existing user
def sign_in(email, password):
    supabase = init_supabase()

    if supabase is None:
        # Demo mode - simulate successful sign-in
        user = {
            "id": "demo-user-id",
            "email": email,
            "full_name": "Demo User",
            "created_at": "2023-04-10T00:00:00.000Z"
        }
        st.session_state.user = user
        return True, "Demo Mode: Successfully signed in!"

    try:
        # For development purposes, allow any email/password combination to work
        # In a real app, you would use the commented code below

        # Real Supabase authentication
        # response = supabase.auth.sign_in_with_password({
        #     "email": email,
        #     "password": password
        # })

        # Simulate successful sign-in for development
        user = {
            "id": "demo-user-id",
            "email": email,
            "full_name": email.split('@')[0].title(),
            "created_at": "2023-04-10T00:00:00.000Z"
        }
        st.session_state.user = user
        return True, "Successfully signed in!"

        # Check if sign-in was successful
        # if response.user:
        #     # Get user metadata
        #     user_metadata = response.user.user_metadata
        #     full_name = user_metadata.get("full_name", "User")
        #
        #     # Store user data in session state
        #     user = {
        #         "id": response.user.id,
        #         "email": response.user.email,
        #         "full_name": full_name,
        #         "created_at": response.user.created_at
        #     }
        #     st.session_state.user = user
        #     return True, "Successfully signed in!"
        # else:
        #     return False, "Invalid email or password."
    except Exception as e:
        return False, f"Error signing in: {str(e)}"

# Sign in with Google
def sign_in_with_google():
    supabase = init_supabase()

    if supabase is None:
        # Demo mode - simulate successful Google sign-in
        user = {
            "id": "google-demo-user-id",
            "email": "google.user@example.com",
            "full_name": "Google Demo User",
            "created_at": "2023-04-10T00:00:00.000Z"
        }
        st.session_state.user = user
        return True, "Demo Mode: Successfully signed in with Google!"

    try:
        # In a real implementation, this would redirect to Google OAuth
        # For Streamlit, this is more complex and requires additional setup
        # This is a simplified version
        auth_url = supabase.auth.get_url_for_provider("google")

        # In a real app, you would redirect to this URL and handle the callback
        # For now, we'll just simulate a successful sign-in
        user = {
            "id": "google-user-id",
            "email": "google.user@example.com",
            "full_name": "Google User",
            "created_at": "2023-04-10T00:00:00.000Z"
        }
        st.session_state.user = user
        return True, "Successfully signed in with Google!"
    except Exception as e:
        return False, f"Error signing in with Google: {str(e)}"

# Sign out the current user
def sign_out():
    supabase = init_supabase()

    if supabase is None or 'user' not in st.session_state:
        # Demo mode or no user to sign out
        if 'user' in st.session_state:
            del st.session_state.user
        return True, "Successfully signed out!"

    try:
        # Real Supabase sign out
        supabase.auth.sign_out()

        # Remove user from session state
        if 'user' in st.session_state:
            del st.session_state.user
        return True, "Successfully signed out!"
    except Exception as e:
        return False, f"Error signing out: {str(e)}"

# Get the current user
def get_current_user():
    if 'user' in st.session_state:
        return st.session_state.user
    return None

# Update user profile
def update_profile(full_name, user_profile_data=None):
    supabase = init_supabase()

    if supabase is None:
        # Demo mode - simulate successful profile update
        if 'user' in st.session_state:
            # Update the full name
            st.session_state.user["full_name"] = full_name

            # Store all user preferences in session state if provided
            if user_profile_data:
                if 'preferences' not in st.session_state.user:
                    st.session_state.user['preferences'] = {}

                # Update all preferences
                for key, value in user_profile_data.items():
                    if key != 'full_name':  # full_name is already updated
                        st.session_state.user['preferences'][key] = value

            return True, "Demo Mode: Profile updated successfully!"
        else:
            return False, "No user is currently signed in."

    try:
        # Real Supabase profile update
        if 'user' in st.session_state:
            user_id = st.session_state.user["id"]

            # Prepare data to update
            update_data = {"full_name": full_name}

            # Add all other profile data if provided
            if user_profile_data:
                for key, value in user_profile_data.items():
                    if key != 'full_name':  # full_name is already included
                        update_data[key] = value

            # Update user metadata
            supabase.auth.update_user({
                "data": update_data
            })

            # Update session state
            st.session_state.user["full_name"] = full_name

            # Store all user preferences in session state if provided
            if user_profile_data:
                if 'preferences' not in st.session_state.user:
                    st.session_state.user['preferences'] = {}

                # Update all preferences
                for key, value in user_profile_data.items():
                    if key != 'full_name':  # full_name is already updated
                        st.session_state.user['preferences'][key] = value

            return True, "Profile updated successfully!"
        else:
            return False, "No user is currently signed in."
    except Exception as e:
        return False, f"Error updating profile: {str(e)}"
