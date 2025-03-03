import streamlit as st
import pickle
import numpy as np

# ---- USER CREDENTIALS ----
USERNAME = "princeraw"
PASSWORD = "12345"

# ---- SESSION STATE ----
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ---- LOGIN FUNCTION ----
def login():
    """Renders a simple login page"""
    st.title("🔐 Login")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Login Successful! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")

# ---- MAIN FUNCTION ----
def main():
    """Main function for healthcare anomaly detection"""
    st.title("🏥 Healthcare Anomaly Detection")

    # Sidebar Description
    st.sidebar.header("📋 Enter Patient Data")
    st.sidebar.markdown("Provide the following medical details to check for anomalies.")

    # Input fields for the user
    DRG_Definition = st.sidebar.number_input("DRG Definition", min_value=0, step=1)
    Provider_Id = st.sidebar.number_input("Provider ID", min_value=0, step=1)
    Total_Discharges = st.sidebar.number_input("Total Discharges", min_value=0, step=1)
    Average_Covered_Charges = st.sidebar.number_input("Average Covered Charges ($)", min_value=0.0, step=0.01)
    Average_Total_Payments = st.sidebar.number_input("Average Total Payments ($)", min_value=0.0, step=0.01)
    Average_Medicare_Payments = st.sidebar.number_input("Average Medicare Payments ($)", min_value=0.0, step=0.01)

    # Load trained model
    with open("ifm1.pkl", "rb") as file:
        model = pickle.load(file)

    # Predict Button
    if st.sidebar.button("🔍 Predict"):
        # Convert input data into NumPy array for prediction
        features = np.array([
            DRG_Definition,
            Provider_Id,
            Total_Discharges,
            Average_Covered_Charges,
            Average_Total_Payments,
            Average_Medicare_Payments
        ]).reshape(1, -1)

        # Predict using Isolation Forest
        prediction = model.predict(features)
        score = decision_function(features)[0]

        # Interpretation: -1 = anomaly, 1 = normal
        result = "🚨 Anomalous" if prediction[0] == -1 else "✅ Normal"

        # Display Result
        st.subheader("📊 Prediction Result")
        st.write(f"**Result:** {result}")
        st.write(f"Anamoly Score: {score:.3f}")

    # Add footer
    st.markdown("---")
    st.markdown("Developed by **[Your Name]** | Healthcare Anomaly Detection App")

# ---- CHECK LOGIN STATUS ----
if not st.session_state.authenticated:
    login()
else:
    main()
