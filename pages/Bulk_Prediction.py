import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

st.title("Bulk Prediction")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    # ⭐ get model feature list
    cols = model.get_booster().feature_names

    # ⭐ create model input dataframe
    df_model = df.reindex(columns=cols, fill_value=0)

    # ⭐ predictions using df_model (IMPORTANT)
    df["churn_prob"] = model.predict_proba(df_model)[:,1]
    df["prediction"] = model.predict(df_model)

    st.write(df.head())
