import streamlit as st
import pandas as pd
import joblib

st.title("Bulk Churn Prediction")

model = joblib.load("churn_model.pkl")

file = st.file_uploader("Upload CSV for Bulk Prediction", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Get expected model features
    expected_cols = list(model.feature_names_in_)

    # Add missing columns with 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Remove extra columns
    df = df[expected_cols]

    # Convert everything to numeric (VERY IMPORTANT)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["prediction"] = preds
    df["churn_probability"] = probs

    st.subheader("Prediction Results")
    st.write(df.head())

    st.success("Bulk prediction completed successfully!")
