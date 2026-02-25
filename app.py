import streamlit as st

# ------------------ PAGE CONFIG (ONLY ONCE) ------------------
st.set_page_config(
    page_title="Customer Churn AI Platform",
    page_icon="📊",
    layout="wide"
)

# ------------------ CUSTOM UI CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1, h2, h3 {
    color: #1f4e79;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("Customer Churn Intelligence Platform")

st.write("""
Welcome to the **Customer Churn AI Dashboard**.

This platform provides:

✅ Churn prediction using machine learning  
✅ Feature importance insights  
✅ Customer analytics visualization  
✅ Bulk churn prediction capability  
""")

# ------------------ INFO CARDS ------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📌 Predict churn probability using AI model")

with col2:
    st.info("📊 Analyze key churn drivers")

with col3:
    st.info("📂 Upload CSV for bulk prediction")

# ------------------ HERO METRICS ------------------
st.subheader("Platform Highlights")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Model Accuracy", "96%")

with m2:
    st.metric("ROC AUC Score", "0.97")

with m3:
    st.metric("Deployment", "Live Dashboard")

# ------------------ FOOTER ------------------
st.divider()

st.caption("Built with Streamlit + XGBoost | Customer Churn Prediction Project")