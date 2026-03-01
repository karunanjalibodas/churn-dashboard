import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Customer Retention Intelligence Platform",
    layout="wide"
)

# ------------------------------------------------
# BLACK PROFESSIONAL THEME
# ------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b0f19 !important;
    color: #f5f5f5 !important;
}

section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

h1, h2, h3 {
    color: #ffffff !important;
}

.stMetric {
    background-color: #111827 !important;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 30px !important;
    font-weight: 600;
}

div[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
    font-size: 14px !important;
}

.stDataFrame {
    background-color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.title("Customer Retention Intelligence Platform")
st.markdown(
    "<p style='color:#9ca3af; font-size:18px;'>AI-powered churn risk monitoring and decision support</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
try:
    model = joblib.load("churn_model.pkl")
except:
    st.error("Model file 'churn_model.pkl' not found.")
    st.stop()

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload processed customer dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except:
        st.error("Unable to read the uploaded file.")
        st.stop()

    # ------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ------------------------------------------------
    required_columns = model.feature_names_in_

    if not set(required_columns).issubset(df.columns):
        st.error("Uploaded file does not contain required model features.")
        st.write("Required columns:", list(required_columns))
        st.stop()

    # ------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------
    X = df[required_columns]

    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["prediction"] = model.predict(X)

    # ------------------------------------------------
    # SIDEBAR FILTER
    # ------------------------------------------------
    st.sidebar.header("Filters")

    threshold = st.sidebar.slider(
        "Churn Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    filtered_df = df[df["churn_probability"] >= threshold]

    # ------------------------------------------------
    # KPI SECTION
    # ------------------------------------------------
    total_customers = len(df)
    high_risk = len(filtered_df)
    avg_risk = round(df["churn_probability"].mean(), 2)
    retention_rate = round((1 - high_risk / total_customers) * 100, 1)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("High Risk Customers", f"{high_risk:,}")
    col3.metric("Average Risk Score", avg_risk)
    col4.metric("Estimated Retention Rate", f"{retention_rate}%")

    st.markdown("---")

    # ------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------
    st.subheader("Executive Summary")

    st.info(
        f"""
        • {high_risk} customers exceed the selected risk threshold of {threshold}.  
        • Portfolio average churn probability is {avg_risk}.  
        • Estimated retention rate stands at {retention_rate}%.  

        Recommended Action: Prioritize retention campaigns for high-risk customers to reduce churn exposure.
        """
    )

    st.markdown("---")

    # ------------------------------------------------
    # CHURN DISTRIBUTION CHART
    # ------------------------------------------------
    st.subheader("Churn Probability Distribution")

    fig = px.histogram(
        df,
        x="churn_probability",
        nbins=25
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers",
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------
    # HIGH RISK TABLE
    # ------------------------------------------------
    st.subheader("High Risk Customers")

    st.dataframe(
        filtered_df.sort_values("churn_probability", ascending=False),
        use_container_width=True
    )

    # ------------------------------------------------
    # EXPORT BUTTON
    # ------------------------------------------------
    st.download_button(
        label="Download Risk Report",
        data=filtered_df.to_csv(index=False),
        file_name="churn_risk_report.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a processed dataset to begin churn risk analysis.")

st.markdown("---")
st.caption("Built with Streamlit + XGBoost | Customer Churn Prediction System")
