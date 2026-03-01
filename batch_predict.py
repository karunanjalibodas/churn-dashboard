import pandas as pd
import joblib
import time

# ========== LOAD MODEL ==========
print("Loading model...")
model = joblib.load("churn_model.pkl")

print("\nModel expects these columns:")
print(model.feature_names_in_)

# ========== LOAD DATA ==========
print("\nLoading dataset...")

df = pd.read_csv(
    r"C:\Users\91939\Downloads\kkbox-churn-prediction-challenge\data\churn_comp_refresh\train.csv"
)

print("\nDataset columns:")
print(df.columns)

# ========== CHECK COLUMN MATCH ==========
expected_cols = list(model.feature_names_in_)
dataset_cols = list(df.columns)

if set(expected_cols) != set(dataset_cols):
    print("\n❌ ERROR: Column mismatch!")
    print("Model expects:", expected_cols)
    print("Dataset has:", dataset_cols)
    print("\nYou must use processed dataset used during training.")
    exit()

# ========== REORDER COLUMNS ==========
df = df[expected_cols]

# ========== PREDICT ==========
print("\nStarting prediction...")
start = time.time()

preds = model.predict(df)
probs = model.predict_proba(df)[:, 1]

df["prediction"] = preds
df["churn_probability"] = probs

end = time.time()

print(f"\nPrediction completed in {round(end - start, 2)} seconds")

df.to_csv("predictions_output.csv", index=False)

print("\n✅ Saved as predictions_output.csv")