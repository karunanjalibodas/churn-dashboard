import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

st.title("Bulk Prediction")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    # ⭐ ENCODING FIX
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    if "Geography" in df.columns:
        df = pd.get_dummies(df, columns=["Geography"])

    # ⭐ FEATURE ALIGNMENT
    cols = model.get_booster().feature_names
    df = df.reindex(columns=cols, fill_value=0)

    df["churn_prob"] = model.predict_proba(df)[:,1]
    df["prediction"] = model.predict(df)

    st.write(df.head())
