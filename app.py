import streamlit as st
import joblib
import numpy as np

# load model (TOP ONLY)
model = joblib.load("model.pkl")

st.caption("ML model predicts customer churn based on tenure and monthly changes ")

tenure = st.slider("Tenure", 0, 72)
monthly = st.slider("Monthly Charges", 0, 150)

if st.button("Predict"):
    input_data = np.array([[tenure, monthly]])

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prob > 0.5:
        st.error(f"Churn Probability: {prob:.2f}")

        # 👉 explanation
        if tenure < 12:
            st.write("⚠️ Low tenure increases churn risk")
        if monthly > 80:
            st.write("⚠️ High monthly charges increase churn risk")

    else:
        st.success(f"Churn Probability: {prob:.2f}")
  