import streamlit as st
import joblib
import numpy as np

# ── Page Setup 
st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")

# ── Load Model 
model = joblib.load("model.pkl")

# ── Header 
st.title("📡 Customer Churn Predictor")
st.caption("Telco Dataset · Random Forest Model")
st.divider()

# ── Inputs 
st.subheader("Customer Details")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges ($)", 0, 150, 65)

st.divider()

# ── Predict 
if st.button("Predict Churn", use_container_width=True):

    prob = model.predict_proba(np.array([[tenure, monthly]]))[0][1]
    pct = round(prob * 100, 1)

    st.subheader("Result")
    if prob > 0.5:
        st.error(f"⚠️ High Churn Risk — **{pct}%**")
    else:
        st.success(f"✅ Low Churn Risk — **{pct}%**")

    st.progress(int(prob * 100))

    # Risk Factors
    st.subheader("Risk Factors")
    if tenure < 12:
        st.warning("⚡ Low tenure — new customers churn more often")
    else:
        st.info("✓ Good tenure — customer shows loyalty")

    if monthly > 80:
        st.warning("⚡ High monthly charges — increases churn risk")
    else:
        st.info("✓ Reasonable monthly charges")

    # Recommended Action
    st.subheader("Recommended Action")
    if prob > 0.7:
        st.error("🚨 Urgent retention offer — give discount or upgrade incentive")
    elif prob > 0.5:
        st.warning("📞 Proactive outreach — check satisfaction and offer bundling")
    elif prob > 0.3:
        st.info("📧 Monitor closely — send loyalty rewards")
    else:
        st.success("✅ Stable customer — standard engagement is fine")
