import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

model = joblib.load("churn_model.pkl")

st.title("Feature Importance")

feat_imp = pd.Series(
    model.feature_importances_,
    index=model.get_booster().feature_names
).sort_values(ascending=False).head(10)

fig = px.bar(
    feat_imp,
    orientation='h',
    title="Top Churn Drivers"
)

st.plotly_chart(fig, use_container_width=True)
