import streamlit as st
import pandas as pd

st.title("Customer Analytics")

st.write("Upload dataset to visualize churn analytics")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    st.write(df.head())
    st.bar_chart(df.select_dtypes(include='number'))
