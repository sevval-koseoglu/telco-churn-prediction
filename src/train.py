import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import os

# ---- Veri Yukleme ve Temizleme ----
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

feature_columns = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Model 1: Logistic Regression ----
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
lr_preds = log_reg.predict(X_test)

print("=" * 50)
print("Model 1: Logistic Regression")
print(f"  Accuracy  : {accuracy_score(y_test, lr_preds):.4f}")
print(f"  Precision : {precision_score(y_test, lr_preds):.4f}")
print(f"  Recall    : {recall_score(y_test, lr_preds):.4f}")
print(f"  F1 Score  : {f1_score(y_test, lr_preds):.4f}")

# ---- Model 2: Random Forest ----
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("=" * 50)
print("Model 2: Random Forest")
print(f"  Accuracy  : {accuracy_score(y_test, rf_preds):.4f}")
print(f"  Precision : {precision_score(y_test, rf_preds):.4f}")
print(f"  Recall    : {recall_score(y_test, rf_preds):.4f}")
print(f"  F1 Score  : {f1_score(y_test, rf_preds):.4f}")
print("=" * 50)

# ---- En iyi modeli F1 Score'a gore sec ----
lr_f1 = f1_score(y_test, lr_preds)
rf_f1 = f1_score(y_test, rf_preds)

if rf_f1 >= lr_f1:
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = log_reg
    best_name = "Logistic Regression"

print(f"Kazanan model: {best_name} (F1: {max(lr_f1, rf_f1):.4f})")

# ---- Modelleri Kaydet ----
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print(f"'{best_name}' modeli 'models/model.pkl' olarak kaydedildi.")