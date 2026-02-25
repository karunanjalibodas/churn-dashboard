import streamlit as st
import pandas as pd
import joblib

st.title("Bulk Churn Prediction")

# ---------------- LOAD MODEL ----------------
model = joblib.load("churn_model.pkl")
model_features = model.get_booster().feature_names

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # ---------------- DROP NON-USEFUL COLS ----------------
    drop_cols = ["CustomerId", "Surname", "RowNumber"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ---------------- ENCODE CATEGORICAL ----------------
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ---------------- ALIGN FEATURES ----------------
    df = df.reindex(columns=model_features, fill_value=0)

    # ---------------- PREDICTIONS ----------------
    df["churn_prob"] = model.predict_proba(df)[:, 1]
    df["prediction"] = model.predict(df)

    # ---------------- RISK SEGMENTATION ----------------
    def risk(p):
        if p >= 0.7:
            return "High Risk"
        elif p >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_segment"] = df["churn_prob"].apply(risk)

    # ---------------- RETENTION ACTION ----------------
    def action(r):
        if r == "High Risk":
            return "Offer discount + proactive call"
        elif r == "Medium Risk":
            return "Send loyalty email"
        else:
            return "No action"

    df["retention_action"] = df["risk_segment"].apply(action)

    st.success("Bulk prediction completed")
    st.dataframe(df)

    st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        "bulk_predictions.csv",
        "text/csv"
    )
