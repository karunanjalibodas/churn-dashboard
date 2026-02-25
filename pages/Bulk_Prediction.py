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

    # ---------------- REMOVE USELESS COLUMNS ----------------
    df = df.drop(
        columns=["RowNumber", "CustomerId", "Surname", "Exited", "Churn", "churn"],
        errors="ignore"
    )

    # ---------------- ENCODE CATEGORICAL ----------------
    obj_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=obj_cols)

    # ---------------- ALIGN FEATURES ----------------
    model_features = model.get_booster().feature_names

    # Add missing columns
    missing_cols = set(model_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Remove extra columns
    df = df[model_features]

    # ---------------- PREDICT ----------------
    prob = model.predict_proba(df)[:, 1]
    pred = model.predict(df)

    df["churn_prob"] = prob
    df["prediction"] = pred

    # ---------------- RISK SEGMENT ----------------
    def risk(p):
        if p > 0.7:
            return "High Risk"
        elif p > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_segment"] = df["churn_prob"].apply(risk)

    # ---------------- RETENTION ACTION ----------------
    def retention(r):
        if r == "High Risk":
            return "Offer discount + proactive call"
        elif r == "Medium Risk":
            return "Send engagement email"
        else:
            return "Loyalty rewards"

    df["retention_action"] = df["risk_segment"].apply(retention)

    st.success("Bulk prediction completed")
    st.write(df.head())

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        "bulk_predictions.csv"
    )
