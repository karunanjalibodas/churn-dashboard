import matplotlib.pyplot as plt

churn_counts = df["is_churn"].value_counts()

fig, ax = plt.subplots()
churn_counts.plot(kind="bar", ax=ax)

plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")

st.pyplot(fig)
