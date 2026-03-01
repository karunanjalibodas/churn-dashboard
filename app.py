import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="RetentionAI", layout="wide")

# ------------------------------------------------
# CUSTOM STRIPE-LIKE DARK THEME
# ------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: white;
}
.metric-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    color: white;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
}
.stDataFrame {
    background-color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOGIN SYSTEM
# ------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 RetentionAI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown("# 🚀 RetentionAI Dashboard")
st.markdown("### Predict • Analyze • Prevent Customer Churn")

st.markdown("---")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("churn_model.pkl")

# ------------------------------------------------
# LOAD DEMO DATA
# ------------------------------------------------
try:
    df = pd.read_csv("bulk_demo.csv")
    X = df[model.feature_names_in_]
    df["risk"] = model.predict_proba(X)[:, 1]
    df["prediction"] = model.predict(X)
except:
    st.error("bulk_demo.csv not found")
    st.stop()

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------
st.sidebar.header("⚙️ Filters")

risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

filtered_df = df[df["risk"] >= risk_threshold]

# ------------------------------------------------
# ANIMATED KPI COUNTERS
# ------------------------------------------------
def animated_metric(label, value):
    placeholder = st.empty()
    for i in range(int(value)):
        placeholder.markdown(
            f'<div class="metric-card"><h3>{label}</h3><h1>{i}</h1></div>',
            unsafe_allow_html=True
        )
        time.sleep(0.01)
    placeholder.markdown(
        f'<div class="metric-card"><h3>{label}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True
    )

col1, col2, col3 = st.columns(3)

with col1:
    animated_metric("Total Customers", len(df))

with col2:
    animated_metric("High Risk Customers", len(filtered_df))

with col3:
    animated_metric("Avg Risk", round(df["risk"].mean(), 2))

st.markdown("---")

# ------------------------------------------------
# INTERACTIVE RISK CHART
# ------------------------------------------------
st.subheader("📊 Churn Risk Distribution")

fig = px.histogram(df, x="risk", nbins=20)
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# HIGH RISK TABLE
# ------------------------------------------------
st.subheader("⚠️ High Risk Customers")

st.dataframe(filtered_df.sort_values("risk", ascending=False))

# ------------------------------------------------
# EXPORT BUTTON
# ------------------------------------------------
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Export Risk Report",
    data=csv,
    file_name="churn_risk_report.csv",
    mime="text/csv"
)

st.markdown("---")
