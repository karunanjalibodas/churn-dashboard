import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

st.title("Churn Prediction")

payment_plan_days = st.sidebar.number_input("Payment Plan Days", 1, 365, 30)
actual_amount_paid = st.sidebar.number_input("Actual Amount Paid", 0, 5000, 100)
is_cancel = st.sidebar.selectbox("Cancelled?", [0,1])
engagement = st.sidebar.number_input("Engagement Score", 0, 1000, 100)

cols = model.get_booster().feature_names
input_df = pd.DataFrame(columns=cols)
input_df.loc[0] = 0

input_df['payment_plan_days'] = payment_plan_days
input_df['actual_amount_paid'] = actual_amount_paid
input_df['is_cancel'] = is_cancel
input_df['engagement'] = engagement

if st.button("Predict"):
    prob = model.predict_proba(input_df)[:,1][0]
    pred = model.predict(input_df)[0]

    # ⭐ risk segmentation
    if prob < 0.3:
        st.success(f"Low Risk ({prob:.2f})")
    elif prob < 0.7:
        st.warning(f"Medium Risk ({prob:.2f})")
    else:
        st.error(f"High Risk ({prob:.2f})")