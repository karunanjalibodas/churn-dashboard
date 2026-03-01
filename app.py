import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Retention Intelligence", layout="wide")

# ----------------------------
# TITLE
# ----------------------------
st.title("Customer Retention Intelligence Platform")
st.markdown("Predict and monitor customer churn risk")

st.markdown("---")

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("churn_model.pkl")

# ----------------------------
# FILE UPLOAD (Professional)
# ----------------------------
uploaded_file = st.file_uploader("Upload processed customer dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Ensure correct columns
    X = df[model.feature_names_in_]

    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["prediction"] = model.predict(X)

    # ----------------------------
    # FILTER SECTION
    # ----------------------------
    st.sidebar.header("Filters")

    threshold = st.sidebar.slider(
        "Churn Risk Threshold",
        0.0, 1.0, 0.5, 0.05
    )

    filtered_df = df[df["churn_probability"] >= threshold]

    # ----------------------------
    # KPI SECTION
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("High Risk Customers", len(filtered_df))
    col3.metric("Average Risk", round(df["churn_probability"].mean(), 2))

    st.markdown("---")

    # ----------------------------
    # CHART
    # ----------------------------
    st.subheader("Churn Probability Distribution")

    fig = px.histogram(df, x="churn_probability", nbins=25)
    fig.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers",
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # TABLE
    # ----------------------------
    st.subheader("High Risk Customers")

    st.dataframe(
        filtered_df.sort_values("churn_probability", ascending=False),
        use_container_width=True
    )

    # ----------------------------
    # EXPORT
    # ----------------------------
    st.download_button(
        "Download Risk Report",
        filtered_df.to_csv(index=False),
        "churn_risk_report.csv",
        "text/csv"
    )

else:
    st.info("Upload a processed dataset to begin analysis.")

st.markdown("---")
st.caption("Built with Streamlit + XGBoost")
