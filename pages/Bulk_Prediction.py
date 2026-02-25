import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("📂 Bulk Customer Churn Prediction")

# ---------------- LOAD MODEL ----------------
model = joblib.load("churn_model.pkl")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV dataset")

if file:
    df_original = pd.read_csv(file)
    df = df_original.copy()

    st.write("📊 Uploaded data preview")
    st.dataframe(df.head())

    # ---------------- AUTO ENCODING ----------------
    # encode binary gender safely
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(
            {"Male":1, "Female":0, "male":1, "female":0}
        ).fillna(0)

    # encode object columns automatically
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=obj_cols)

    # ---------------- FEATURE ALIGNMENT ----------------
    model_features = model.get_booster().feature_names

    df = df.reindex(columns=model_features, fill_value=0)

    # ---------------- PREDICTIONS ----------------
    prob = model.predict_proba(df)[:,1]
    pred = model.predict(df)

    df_original["churn_prob"] = prob
    df_original["prediction"] = pred

    # ---------------- RISK SEGMENTATION ----------------
    def risk_segment(p):
        if p >= 0.7:
            return "High Risk"
        elif p >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df_original["risk_segment"] = df_original["churn_prob"].apply(risk_segment)

    # ---------------- RETENTION RECOMMENDATION ----------------
    def retention_action(row):
        if row["risk_segment"] == "High Risk":
            return "Offer discount + proactive call"
        elif row["risk_segment"] == "Medium Risk":
            return "Send personalized offer"
        else:
            return "Loyalty reward / engagement email"

    df_original["retention_action"] = df_original.apply(retention_action, axis=1)

    # ---------------- RESULTS ----------------
    st.success("✅ Bulk prediction completed")
    st.dataframe(df_original.head())

    # ---------------- DOWNLOAD ----------------
    csv = df_original.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Predictions",
        csv,
        "churn_predictions.csv",
        "text/csv"
    )
