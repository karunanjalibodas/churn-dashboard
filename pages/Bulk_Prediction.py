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

    expected_cols = list(model.feature_names_in_)

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Add missing columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep only required columns
    df = df[expected_cols]

    # Force numeric conversion
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    st.write("Final columns used for prediction:", df.columns)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["prediction"] = preds
    df["churn_probability"] = probs

    st.success("Prediction successful!")
    st.write(df.head())
