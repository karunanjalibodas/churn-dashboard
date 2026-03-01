import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Retention AI", layout="wide")

# -----------------------------
# CUSTOM CSS (Modern SaaS Look)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    font-size: 3rem !important;
}
.metric-box {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
# 🚀 RetentionAI  
### Predict • Analyze • Prevent Customer Churn
""")

st.markdown("---")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("churn_model.pkl")

# -----------------------------
# LOAD SAMPLE DATA FOR DEMO
# -----------------------------
try:
    df = pd.read_csv("bulk_demo.csv")
    X = df[model.feature_names_in_]
    df["risk"] = model.predict_proba(X)[:, 1]
    df["prediction"] = model.predict(X)
except:
    df = None

# -----------------------------
# KPI SECTION
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-box"><h2>Customers</h2><h1>1,000+</h1></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box"><h2>High Risk</h2><h1>142</h1></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box"><h2>Avg Risk</h2><h1>0.38</h1></div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-box"><h2>Retention Rate</h2><h1>86%</h1></div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# CHART SECTION
# -----------------------------
st.subheader("📊 Churn Risk Distribution")

if df is not None:
    fig = px.histogram(df, x="risk", nbins=20, title="Customer Risk Score Distribution")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload or create bulk_demo.csv to enable analytics.")

st.markdown("---")

# -----------------------------
# HIGH RISK TABLE
# -----------------------------
st.subheader("⚠️ High Risk Customers")

if df is not None:
    high = df.sort_values("risk", ascending=False).head(5)
    st.dataframe(high)
else:
    st.info("No data available")

st.markdown("---")

st.markdown("Built with ❤️ using Streamlit + XGBoost")
