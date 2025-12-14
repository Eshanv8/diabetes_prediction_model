import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🚀 Loading Big Data...")
df = pd.read_csv('big_diabetes.csv')
print(f"✅ Data Loaded! Rows: {df.shape[0]}, Columns: {df.shape[1]}")

target_col = 'Diabetes_binary'
if target_col not in df.columns:
    print(f"⚠️ Column '{target_col}' not found. Using the first column as target.")
    target_col = df.columns[0]

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'big_scaler.pkl')

print("🧠 Training Random Forest on 200,000+ patients...")
print("(This may take 1-3 minutes. Please wait...)")

model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("🔍 Testing model accuracy...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🏆 Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- Detailed Report ---")
print(classification_report(y_test, predictions))

print("\n📊 Top 5 Risk Factors found by AI:")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(5))

joblib.dump(model, 'big_diabetes_model.pkl')
print("\n💾 Success! Saved 'big_diabetes_model.pkl' and 'big_scaler.pkl'")