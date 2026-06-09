import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")
df = pd.read_csv("preprocessed10_dataset.csv")
X_base = df.drop('label', axis=1) if 'label' in df.columns else df
X_base_num = X_base.select_dtypes(include=['int64', 'float64'])

print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(X_base_num)

print("Saving scaler and feature column names...")
joblib.dump((scaler, X_base_num.columns.tolist()), "scaler_and_features.pkl")
print("Done! Saved scaler_and_features.pkl successfully.")
