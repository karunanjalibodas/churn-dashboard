import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    layout="wide"
)

# -------------------------------
# TITLE SECTION
# -------------------------------

st.title("AI-Powered Customer Retention Intelligence Platform")
st.markdown("### Predict • Analyze • Prevent Customer Churn Using Machine Learning")

st.markdown("---")

# -------------------------------
# EXECUTIVE OVERVIEW
# -------------------------------

st.markdown("""
## Executive Overview

This AI-driven platform enables proactive customer retention strategies by:

- Identifying high-risk churn customers
- Analyzing behavioral drivers of churn
- Performing real-time individual predictions
- Executing bulk churn risk assessment at scale
""")

st.markdown("---")

# -------------------------------
# KPI SECTION
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "96%")

with col2:
    st.metric("ROC-AUC Score", "0.97")

with col3:
    st.metric("Deployment Status", "Live", delta="Active")

st.markdown("---")

# -------------------------------
# BUSINESS IMPACT
# -------------------------------

st.markdown("""
## Business Impact

✔ Reduce customer churn through early risk detection  
✔ Improve retention campaign targeting  
✔ Enable data-driven subscription decisions  
✔ Automate churn monitoring using AI  
""")

st.markdown("---")

# -------------------------------
# CHURN DISTRIBUTION CHART
# -------------------------------

st.subheader("Training Data Churn Distribution")

try:
    df = pd.read_csv("final_processed_data.csv")
    if "is_churn" in df.columns:
        churn_counts = df["is_churn"].value_counts()

        fig, ax = plt.subplots()
        churn_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Churn (0 = No, 1 = Yes)")
        ax.set_ylabel("Count")
        ax.set_title("Churn Distribution")

        st.pyplot(fig)
    else:
        st.info("is_churn column not found in dataset.")

except:
    st.info("Upload or generate final_processed_data.csv to show distribution.")

st.markdown("---")

# -------------------------------
# FOOTER
# -------------------------------

st.markdown("""
Built with **Streamlit + XGBoost**  
End-to-End Machine Learning Deployment Pipeline
""")
