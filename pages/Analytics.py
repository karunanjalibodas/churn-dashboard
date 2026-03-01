import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Customer Analytics")

# Upload CSV
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    if "is_churn" in df.columns:
        churn_counts = df["is_churn"].value_counts()

        fig, ax = plt.subplots()
        churn_counts.plot(kind="bar", ax=ax)

        plt.title("Churn Distribution")
        plt.xlabel("Churn (0 = No, 1 = Yes)")
        plt.ylabel("Count")

        st.pyplot(fig)
    else:
        st.error("Column 'is_churn' not found in dataset.")
