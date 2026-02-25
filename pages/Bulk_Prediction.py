import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("churn_model.pkl")

st.title("Bulk Prediction")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    # ---------------- DATA CLEANING ----------------
    # remove target if present
    df = df.drop(columns=["Exited", "Churn", "churn"], errors="ignore")

    # ---------------- ENCODING ----------------
    # convert categorical to one hot
    obj_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=obj_cols)

    # ---------------- FEATURE ALIGNMENT ----------------
    model_features = model.get_booster().feature_names

    # add missing columns
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # remove extra columns
    df = df[model_features]

    # ---------------- PREDICTION ----------------
    df["churn_prob"] = model.predict_proba(df)[:, 1]
    df["prediction"] = model.predict(df)

    # ---------------- RISK SEGMENTATION ----------------
    def risk_segment(p):
        if p > 0.7:
            return "High Risk"
        elif p > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_segment"] = df["churn_prob"].apply(risk_segment)

    # ---------------- RETENTION ACTION ----------------
    def retention_action(r):
        if r == "High Risk":
            return "Offer discount + proactive call"
        elif r == "Medium Risk":
            return "Send engagement email"
        else:
            return "Loyalty rewards"

    df["retention_action"] = df["risk_segment"].apply(retention_action)

    # ---------------- UI OUTPUT ----------------
    st.success("Bulk prediction completed")

    st.subheader("Prediction Results")
    st.write(df.head())

    # ---------------- DOWNLOAD ----------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        csv,
        "bulk_predictions.csv",
        "text/csv",
        key="download"
    )
