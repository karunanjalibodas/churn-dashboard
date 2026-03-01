import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

st.set_page_config(page_title="Customer Churn Intelligence Platform", layout="wide")

# ---------------------------
# TITLE
# ---------------------------
st.title("🚀 AI Customer Retention Intelligence Platform")
st.markdown("### Predict • Analyze • Prevent Customer Churn")

st.markdown("---")

# ---------------------------
# LOAD MODEL & DATA
# ---------------------------
model = joblib.load("churn_model.pkl")

try:
    df = pd.read_csv("final_processed_data.csv")
except:
    st.error("final_processed_data.csv not found")
    st.stop()

# ---------------------------
# RUN PREDICTIONS ON DATA
# ---------------------------
X = df[model.feature_names_in_]
df["churn_probability"] = model.predict_proba(X)[:, 1]
df["prediction"] = model.predict(X)

# ---------------------------
# KPI SECTION
# ---------------------------
total_customers = len(df)
high_risk = len(df[df["prediction"] == 1])
avg_risk = round(df["churn_probability"].mean(), 2)
retention_rate = round((1 - high_risk/total_customers) * 100, 1)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("High Risk Customers", high_risk)
col3.metric("Average Churn Risk", avg_risk)
col4.metric("Estimated Retention Rate", f"{retention_rate}%")

st.markdown("---")

# ---------------------------
# RISK DISTRIBUTION
# ---------------------------
st.subheader("📊 Churn Risk Distribution")

fig, ax = plt.subplots()
df["churn_probability"].hist(bins=20, ax=ax)
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

st.markdown("---")

# ---------------------------
# HIGH RISK CUSTOMER TABLE
# ---------------------------
st.subheader("⚠️ Top 10 High Risk Customers")

high_risk_df = df.sort_values("churn_probability", ascending=False).head(10)
st.dataframe(high_risk_df)

st.markdown("---")

# ---------------------------
# BUSINESS RECOMMENDATIONS
# ---------------------------
st.subheader("📈 Strategic Recommendations")

st.markdown("""
Based on current churn risk analysis:

- Focus retention campaigns on high-risk segment
- Offer loyalty discounts to low-engagement users
- Improve customer experience for cancelled subscriptions
- Monitor churn probability weekly for proactive intervention
""")

st.markdown("---")

st.markdown("Built with Streamlit + XGBoost | End-to-End ML Deployment")
